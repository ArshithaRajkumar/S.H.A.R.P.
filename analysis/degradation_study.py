#!/usr/bin/env python3
"""
Phase 1: Degradation Study
===========================
Loads ALL matched GT/NoisyLR pairs and characterises the degradation model.

Reports:
  1. Array shapes, dtypes, value ranges -- confirms x2 scale factor and GT range.
  2. Resampling kernel identification (area, bilinear, bicubic, lanczos, subsample +/- AA).
  3. Noise magnitude: ratio r = x / (A*y), mean(r) and std(r) per file and aggregate.
  4. Noise ordering: kernel-comparison test + 2-D autocorrelation of residual.
  5. Blur sweep: pre-blur sigma that best reproduces the degraded array.
  6. Intensity-dependent noise analysis.

All I/O is .npy float32.  No PIL, no OpenCV, no 8-bit image format anywhere.
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter
from scipy.signal import fftconvolve
import time, sys, warnings

warnings.filterwarnings("ignore")

# -- Paths -----------------------------------------------------------------
GT_DIR  = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\GT")
LR_DIR  = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\NoisyLR")
MASK_THR = 0.02          # mask out near-zero GT to avoid ratio blow-up
OUT_DIR  = Path(__file__).resolve().parent   # save results next to this script

# -- Downsampling kernels --------------------------------------------------

def ds_area(img):
    """Area-average x2 -- physically: detector pixel integration."""
    H, W = img.shape
    return img.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))

def ds_bilinear(img):
    return zoom(img, 0.5, order=1, prefilter=True)

def ds_bicubic(img):
    return zoom(img, 0.5, order=3, prefilter=True)

def ds_lanczos(img):
    return zoom(img, 0.5, order=5, prefilter=True)

def ds_subsample(img):
    return img[::2, ::2]

# Anti-aliased variants (Gaussian sigma=0.5 before downsampling)
def ds_bilinear_aa(img):
    return ds_bilinear(gaussian_filter(img, 0.5))

def ds_bicubic_aa(img):
    return ds_bicubic(gaussian_filter(img, 0.5))

def ds_subsample_aa(img):
    return ds_subsample(gaussian_filter(img, 0.5))


KERNELS = {
    "area":           ds_area,
    "bilinear":       ds_bilinear,
    "bicubic":        ds_bicubic,
    "lanczos5":       ds_lanczos,
    "subsample":      ds_subsample,
    "bilinear_aa":    ds_bilinear_aa,
    "bicubic_aa":     ds_bicubic_aa,
    "subsample_aa":   ds_subsample_aa,
}

# Map from kernel name -> its base (without AA blur) for blur sweep
BASE_NAME = {k: k.replace("_aa", "") for k in KERNELS}

# -- Helpers ---------------------------------------------------------------

def ratio_stats(lr, ds_gt, thr=MASK_THR):
    """Return (mean(r), std(r), n_valid) for r = lr / ds_gt on valid pixels."""
    mask = ds_gt > thr
    nv = int(mask.sum())
    if nv < 100:
        return np.nan, np.nan, nv
    r = lr[mask] / ds_gt[mask]
    return float(np.mean(r)), float(np.std(r)), nv


def masked_autocorrelation(noise, mask, maxlag=10):
    """Properly normalised 2-D autocorrelation with a validity mask."""
    n = noise.copy().astype(np.float64)
    if mask.sum() < 100:
        return np.zeros((2 * maxlag + 1, 2 * maxlag + 1))
    mu = noise[mask].mean()
    n[mask] -= mu
    n[~mask] = 0.0
    # numerator: cross-correlation via FFT
    acf = fftconvolve(n, n[::-1, ::-1], mode="full")
    # denominator: count of valid-pixel pairs at each lag
    mf = mask.astype(np.float64)
    pair_cnt = fftconvolve(mf, mf[::-1, ::-1], mode="full")
    pair_cnt = np.maximum(pair_cnt, 1.0)
    acf /= pair_cnt
    # normalise so centre = 1
    cy, cx = acf.shape[0] // 2, acf.shape[1] // 2
    cv = acf[cy, cx]
    if abs(cv) < 1e-15:
        return np.zeros((2 * maxlag + 1, 2 * maxlag + 1))
    acf /= cv
    return acf[cy - maxlag: cy + maxlag + 1, cx - maxlag: cx + maxlag + 1]


# -- Main ------------------------------------------------------------------

def main():
    t0 = time.time()

    # -- find matched pairs --
    gt_names = sorted(f.name for f in GT_DIR.glob("*.npy"))
    lr_names = sorted(f.name for f in LR_DIR.glob("*.npy"))
    matched  = sorted(set(gt_names) & set(lr_names))
    N = len(matched)
    print(f"GT files:  {len(gt_names)}")
    print(f"LR files:  {len(lr_names)}")
    print(f"Matched:   {N}")
    if N == 0:
        print("ERROR: no matched pairs"); sys.exit(1)

    # ==================================================================
    #  PASS 1 -- basic statistics  +  kernel comparison (no blur)
    # ==================================================================
    print()
    print("=" * 72)
    print(" PASS 1: basic stats + kernel comparison (no blur)")
    print("=" * 72)

    # accumulators -- basic stats
    gt_mins, gt_maxs, gt_means, gt_stds = [], [], [], []
    lr_mins, lr_maxs, lr_means, lr_stds = [], [], [], []
    lr_pgt1, lr_plt0 = [], []
    gt_shapes, lr_shapes = set(), set()
    gt_dtypes, lr_dtypes = set(), set()

    # accumulators -- kernel comparison
    knames = list(KERNELS.keys())
    k_rmean = {k: [] for k in knames}
    k_rstd  = {k: [] for k in knames}

    for i, fn in enumerate(matched):
        gt = np.load(GT_DIR / fn)
        lr = np.load(LR_DIR / fn)

        gt_shapes.add(gt.shape); gt_dtypes.add(str(gt.dtype))
        lr_shapes.add(lr.shape); lr_dtypes.add(str(lr.dtype))
        gt_mins.append(gt.min()); gt_maxs.append(gt.max())
        gt_means.append(gt.mean()); gt_stds.append(gt.std())
        lr_mins.append(lr.min()); lr_maxs.append(lr.max())
        lr_means.append(lr.mean()); lr_stds.append(lr.std())
        lr_pgt1.append((lr > 1.0).mean() * 100)
        lr_plt0.append((lr < 0.0).mean() * 100)

        for kn in knames:
            ds = KERNELS[kn](gt)
            rm, rs, _ = ratio_stats(lr, ds)
            k_rmean[kn].append(rm)
            k_rstd[kn].append(rs)

        if (i + 1) % 400 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  {time.time()-t0:.0f}s")

    # convert
    gt_mins  = np.array(gt_mins);  gt_maxs  = np.array(gt_maxs)
    gt_means = np.array(gt_means); gt_stds  = np.array(gt_stds)
    lr_mins  = np.array(lr_mins);  lr_maxs  = np.array(lr_maxs)
    lr_means = np.array(lr_means); lr_stds  = np.array(lr_stds)
    lr_pgt1  = np.array(lr_pgt1);  lr_plt0  = np.array(lr_plt0)
    for kn in knames:
        k_rmean[kn] = np.array(k_rmean[kn])
        k_rstd[kn]  = np.array(k_rstd[kn])

    # -- Section 1: basic stats --
    print()
    print("-" * 72)
    print(" SECTION 1 -- Array shapes, dtypes, value ranges")
    print("-" * 72)
    print(f"\n  {'':>25} {'GT':>20} {'NoisyLR':>20}")
    print(f"  {'Shapes':>25} {str(list(gt_shapes)):>20} {str(list(lr_shapes)):>20}")
    print(f"  {'Dtypes':>25} {str(list(gt_dtypes)):>20} {str(list(lr_dtypes)):>20}")

    if len(gt_shapes) == 1 and len(lr_shapes) == 1:
        sf = list(gt_shapes)[0][0] / list(lr_shapes)[0][0]
        print(f"  {'Scale factor':>25} {f'{sf:.0f}x uniform':>20}")

    print(f"\n  {'':>25} {'GT':>20} {'NoisyLR':>20}")
    print(f"  {'Global min':>25} {gt_mins.min():>20.6f} {lr_mins.min():>20.6f}")
    print(f"  {'Global max':>25} {gt_maxs.max():>20.6f} {lr_maxs.max():>20.6f}")
    print(f"  {'Mean of means':>25} {gt_means.mean():>20.6f} {lr_means.mean():>20.6f}")
    print(f"  {'Std of means':>25} {gt_means.std():>20.6f} {lr_means.std():>20.6f}")
    print(f"  {'Mean of stds':>25} {gt_stds.mean():>20.6f} {lr_stds.mean():>20.6f}")
    print(f"  {'Mean % > 1.0':>25} {'0':>20} {lr_pgt1.mean():>20.4f}")
    print(f"  {'Max  % > 1.0':>25} {'0':>20} {lr_pgt1.max():>20.4f}")
    print(f"  {'Mean % < 0.0':>25} {'0':>20} {lr_plt0.mean():>20.4f}")

    mean_diffs = np.abs(gt_means - lr_means)
    print(f"\n  Mean preservation  |mean(GT) - mean(LR)|")
    print(f"    Average: {mean_diffs.mean():.6f}    Max: {mean_diffs.max():.6f}")

    gt_gmin = gt_mins.min(); gt_gmax = gt_maxs.max()
    if gt_gmin == 0.0 and gt_gmax == 1.0:
        print(f"  [OK] GT range exactly [0.0, 1.0]")
    else:
        print(f"  GT range: [{gt_gmin}, {gt_gmax}]")

    # -- Section 2: kernel comparison --
    print()
    print("-" * 72)
    print(" SECTION 2 -- Resampling kernel comparison")
    print("-" * 72)
    print("  r = LR / A(GT)")

    print(f"\n  {'Kernel':<16} {'mean(r)':>10} {'+-std':>10}"
          f"  {'std(r)':>10} {'+-std':>10}")
    print("  " + "-" * 60)
    scores = {}
    for kn in knames:
        rm = k_rmean[kn]; rs = k_rstd[kn]
        v = ~np.isnan(rm)
        rm_c = rm[v]; rs_c = rs[v]
        bias  = abs(rm_c.mean() - 1.0)
        noise = rs_c.mean()
        scores[kn] = bias + noise
        print(f"  {kn:<16} {rm_c.mean():>10.6f} {rm_c.std():>10.6f}"
              f"  {rs_c.mean():>10.6f} {rs_c.std():>10.6f}")

    ranked = sorted(scores, key=scores.get)
    winner = ranked[0]
    print(f"\n  Winner: '{winner}'  (bias+noise = {scores[winner]:.6f})")
    print(f"  Ranking: {', '.join(ranked)}")

    # -- Noise-ordering test via kernel comparison --
    ss = k_rstd["subsample"][~np.isnan(k_rstd["subsample"])].mean()
    ar = k_rstd["area"][~np.isnan(k_rstd["area"])].mean()
    ratio_ss_ar = ss / ar if ar > 0 else float("inf")
    print(f"\n  Noise-ordering test (subsample vs area):")
    print(f"    subsample std(r) = {ss:.6f}")
    print(f"    area      std(r) = {ar:.6f}")
    print(f"    ratio              = {ratio_ss_ar:.4f}")
    if ratio_ss_ar > 1.3:
        ordering_kernel = "BEFORE downsampling"
    elif ratio_ss_ar < 0.85:
        ordering_kernel = "UNCLEAR"
    else:
        ordering_kernel = "AFTER downsampling"
    print(f"    -> {ordering_kernel}")

    # ==================================================================
    #  PASS 2 -- blur sigma sweep  (winning base kernel only)
    # ==================================================================
    base_kn = BASE_NAME[winner]
    base_fn = KERNELS[base_kn]

    print()
    print("=" * 72)
    print(f" PASS 2: blur sigma sweep  (kernel = '{base_kn}')")
    print("=" * 72)

    blur_sigmas = np.round(np.arange(0, 2.55, 0.05), 2)
    blur_stdr  = {s: [] for s in blur_sigmas}

    t2 = time.time()
    for i, fn in enumerate(matched):
        gt = np.load(GT_DIR / fn)
        lr = np.load(LR_DIR / fn)
        for sigma in blur_sigmas:
            gt_b = gaussian_filter(gt, sigma) if sigma > 0 else gt
            ds = base_fn(gt_b)
            _, rs, _ = ratio_stats(lr, ds)
            blur_stdr[sigma].append(rs)
        if (i + 1) % 400 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  {time.time()-t2:.0f}s")

    for s in blur_sigmas:
        blur_stdr[s] = np.array(blur_stdr[s])

    avg_stdr = {s: float(np.nanmean(blur_stdr[s])) for s in blur_sigmas}
    best_blur = min(avg_stdr, key=avg_stdr.get)

    print(f"\n  {'sigma':>8}  {'mean std(r)':>14}")
    print("  " + "-" * 30)
    for s in blur_sigmas:
        flag = " <-- best" if s == best_blur else ""
        # print every 0.1 step + the best
        if abs(s * 10 - round(s * 10)) < 0.01 or s == best_blur:
            print(f"  {s:>8.2f}  {avg_stdr[s]:>14.6f}{flag}")

    print(f"\n  Best blur sigma = {best_blur:.2f}")
    print(f"  std(r) at sigma=0.00 : {avg_stdr[0.0]:.6f}")
    print(f"  std(r) at sigma={best_blur:.2f} : {avg_stdr[best_blur]:.6f}")
    improv = (avg_stdr[0.0] - avg_stdr[best_blur]) / avg_stdr[0.0] * 100
    print(f"  Improvement:         {improv:.2f}%")

    # ==================================================================
    #  PASS 3 -- autocorrelation + intensity-dependent noise
    # ==================================================================
    print()
    print("=" * 72)
    print(f" PASS 3: noise structure  (kernel='{base_kn}', blur={best_blur:.2f})")
    print("=" * 72)

    maxlag = 10
    acf_sum   = np.zeros((2 * maxlag + 1, 2 * maxlag + 1))
    acf_count = 0

    ibins = np.linspace(0, 1.0, 11)
    bin_stds  = [[] for _ in range(10)]
    bin_means = [[] for _ in range(10)]

    # Also collect per-file noise sigma at winning params
    pf_sigma = []

    t3 = time.time()
    for i, fn in enumerate(matched):
        gt = np.load(GT_DIR / fn)
        lr = np.load(LR_DIR / fn)
        gt_b = gaussian_filter(gt, best_blur) if best_blur > 0 else gt
        ds = base_fn(gt_b)

        mask = ds > MASK_THR
        if mask.sum() < 100:
            pf_sigma.append(np.nan)
            continue

        ratio = np.ones_like(lr)
        ratio[mask] = lr[mask] / ds[mask]
        noise = ratio - 1.0
        pf_sigma.append(float(noise[mask].std()))

        # autocorrelation
        acf = masked_autocorrelation(noise, mask, maxlag)
        acf_sum += acf
        acf_count += 1

        # intensity-dependent noise
        for b in range(10):
            bm = mask & (ds >= ibins[b]) & (ds < ibins[b + 1])
            if bm.sum() > 30:
                bin_stds[b].append(float(noise[bm].std()))
                bin_means[b].append(float(noise[bm].mean()))

        if (i + 1) % 400 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  {time.time()-t3:.0f}s")

    pf_sigma = np.array(pf_sigma)
    pf_clean = pf_sigma[~np.isnan(pf_sigma)]
    acf_avg = acf_sum / max(acf_count, 1)

    # -- autocorrelation printout --
    c = maxlag
    print(f"\n  2-D autocorrelation of noise residual (centre +-5 lags):")
    hdr_row = "         " + "".join(f"{dx:>8}" for dx in range(-5, 6))
    print(hdr_row)
    for dy in range(-5, 6):
        vals = "".join(f"{acf_avg[c + dy, c + dx]:>8.4f}" for dx in range(-5, 6))
        print(f"  dy={dy:+2d}:  {vals}")

    n4 = np.mean([acf_avg[c-1,c], acf_avg[c+1,c], acf_avg[c,c-1], acf_avg[c,c+1]])
    n8 = np.mean([acf_avg[c-1,c-1], acf_avg[c-1,c+1], acf_avg[c+1,c-1], acf_avg[c+1,c+1]])
    print(f"\n  centre:           1.0000")
    print(f"  4-neighbour mean: {n4:.6f}")
    print(f"  diagonal mean:    {n8:.6f}")
    if abs(n4) < 0.05 and abs(n8) < 0.05:
        ordering_acf = "AFTER downsampling"
    elif n4 > 0.05:
        ordering_acf = "BEFORE downsampling"
    else:
        ordering_acf = "UNCLEAR"
    print(f"  -> {ordering_acf}")

    # -- intensity-dependent noise --
    print(f"\n  Intensity-dependent noise (ratio model r = 1 + n):")
    print(f"  {'Intensity':>15} {'noise mean':>12} {'noise sigma':>12} {'N_files':>8}")
    print("  " + "-" * 50)
    valid_bin_stds = []
    for b in range(10):
        lo, hi = ibins[b], ibins[b + 1]
        if len(bin_stds[b]) > 10:
            ms = np.mean(bin_stds[b])
            mm = np.mean(bin_means[b])
            valid_bin_stds.append(ms)
            print(f"  [{lo:.1f} , {hi:.1f})      {mm:>12.6f} {ms:>12.6f} {len(bin_stds[b]):>8}")
        else:
            print(f"  [{lo:.1f} , {hi:.1f})             ---          ---        {len(bin_stds[b])}")

    if len(valid_bin_stds) >= 3:
        cv = np.std(valid_bin_stds) / np.mean(valid_bin_stds)
        print(f"\n  CV of noise sigma across bins: {cv:.4f}")
        if cv < 0.15:
            print("  [OK] sigma approximately constant -> multiplicative model confirmed")
        else:
            print(f"  [!!] sigma varies across intensity (CV = {cv:.2f})")

    # ==================================================================
    #  GATE SUMMARY
    # ==================================================================
    # consensus on ordering
    if ordering_kernel == ordering_acf:
        final_ord = ordering_kernel
    elif "UNCLEAR" in ordering_kernel:
        final_ord = ordering_acf
    elif "UNCLEAR" in ordering_acf:
        final_ord = ordering_kernel
    else:
        final_ord = f"AMBIGUOUS  kernel-test -> {ordering_kernel} | ACF -> {ordering_acf}"

    sigma_at_best = avg_stdr[best_blur]
    sigma_no_blur = avg_stdr[0.0]

    winner_rm = k_rmean[winner][~np.isnan(k_rmean[winner])]
    winner_rs = k_rstd[winner][~np.isnan(k_rstd[winner])]

    print()
    print()
    print("#" * 72)
    print("##  PHASE 1 GATE SUMMARY -- FITTED DEGRADATION PARAMETERS")
    print("#" * 72)
    print()
    print("  DATA FORMAT")
    print(f"    GT:      {str(list(gt_shapes)[0]):>12}  float32   range [{gt_gmin:.4f}, {gt_gmax:.4f}]")
    print(f"    LR:      {str(list(lr_shapes)[0]):>12}  float32   range [{lr_mins.min():.4f}, {lr_maxs.max():.4f}]")
    print(f"    Scale:   x2 uniform   ({N} pairs)")
    print(f"    |d_mean|: {mean_diffs.mean():.6f} avg,  {mean_diffs.max():.6f} max")
    print()
    print("  RESAMPLING KERNEL")
    print(f"    Winner:    {winner}")
    print(f"    mean(r):   {winner_rm.mean():.6f}  +-  {winner_rm.std():.6f}")
    print(f"    Ranking:   {', '.join(ranked[:5])}")
    print()
    print("  NOISE MODEL")
    print(f"    Type:      Multiplicative  x = A(G_s(y)) * (1 + n)")
    print(f"    Bias:      mean(r) = {winner_rm.mean():.6f}   (unbiased ~ 1.0)")
    print(f"    sigma:     {sigma_at_best:.6f}  (aggregate mean std(r) at best blur)")
    print(f"    Per-file:  mean {pf_clean.mean():.6f}   std {pf_clean.std():.6f}")
    print(f"    Range:     [{pf_clean.min():.4f} , {pf_clean.max():.4f}]")
    print(f"    Ordering:  {final_ord}")
    print()
    print("  PRE-BLUR")
    print(f"    Best sigma:    {best_blur:.2f}")
    print(f"    std(r) no blur:  {sigma_no_blur:.6f}")
    print(f"    std(r) w/ blur:  {sigma_at_best:.6f}")
    print(f"    Improvement:     {improv:.2f}%")
    print()
    print("  FORWARD MODEL")
    print()
    print(f"        x  =  A( G_s(y) )  *  (1 + n)")
    print()
    print(f"    y   in  [0, 1]          ground truth")
    print(f"    G_s =  Gaussian blur   sigma_blur = {best_blur:.2f}")
    print(f"    A   =  {base_kn:<18} x2 downsample")
    print(f"    n   ~  multiplicative  E[n] = 0,  std(n) = {sigma_at_best:.4f}")
    print()

    # per-file sigma distribution
    print(f"  Per-file noise sigma distribution  (N = {len(pf_clean)}):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:02d}:  {np.percentile(pf_clean, p):.6f}")
    print(f"    Mean: {pf_clean.mean():.6f}")
    print(f"    Std:  {pf_clean.std():.6f}")

    # save results for Phase 2
    out_path = OUT_DIR / "phase1_results.npz"
    np.savez(
        out_path,
        matched_files=np.array(matched),
        gt_means=gt_means, gt_stds=gt_stds,
        lr_means=lr_means, lr_stds=lr_stds,
        winner_kernel=np.array(winner),
        base_kernel=np.array(base_kn),
        best_blur=np.array(best_blur),
        pf_sigma=pf_sigma,
        acf_avg=acf_avg,
        blur_sigmas=blur_sigmas,
        blur_avg_stdr=np.array([avg_stdr[s] for s in blur_sigmas]),
    )
    print(f"\n  Results saved to {out_path}")

    total = time.time() - t0
    print(f"  Total analysis time: {total:.0f}s  ({total / 60:.1f} min)")

    print()
    print("#" * 72)
    print("##  STOP -- Review this summary before proceeding to Phase 2.")
    print("#" * 72)


if __name__ == "__main__":
    main()
