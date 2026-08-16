"""
Sweep with larger, more realistic defect sizes (radius=4-5) and
also try a local-anomaly approach.
"""
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, label, center_of_mass

GT_DIR = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\GT")

def inject_blobs(img, contrast=0.30, num=10, radius=4):
    out = img.copy()
    h, w = out.shape
    locs = []
    for _ in range(num):
        y = np.random.randint(radius+2, h-radius-2)
        x = np.random.randint(radius+2, w-radius-2)
        sign = 1 if np.random.rand() > 0.5 else -1
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = np.exp(-(xx**2 + yy**2) / (2*(radius/2.5)**2))
        out[y-radius:y+radius+1, x-radius:x+radius+1] += sign * contrast * mask
        locs.append((y, x))
    return np.clip(out, 0, 1), locs

def detect(img, s1, s2, thresh):
    dog = np.abs(gaussian_filter(img, s1) - gaussian_filter(img, s2))
    mask = dog > thresh
    labeled, nf = label(mask)
    if nf == 0:
        return []
    return center_of_mass(mask, labeled, range(1, nf+1))

def main():
    files = sorted(GT_DIR.glob("*.npy"))
    np.random.seed(42)
    sample = [files[i] for i in np.random.choice(len(files), 30, replace=False)]
    clean_imgs = [np.load(f) for f in sample]

    for radius in [3, 4, 5]:
        print(f"\n=== Defect radius = {radius} (diameter = {2*radius+1}) ===")
        
        defected = []
        all_locs = []
        for img in clean_imgs:
            d, l = inject_blobs(img, contrast=0.30, radius=radius)
            defected.append(d)
            all_locs.append(l)

        print(f"{'s1':>4} {'s2':>4} {'thresh':>7} | {'Clean FP/img':>12} {'TPR@0.30':>10}")
        print("-" * 55)

        best = None
        for s1 in [0.5, 1.0, 1.5]:
            for s2 in [2.0, 3.0, 4.0, 5.0]:
                if s2 <= s1:
                    continue
                for thresh in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
                    clean_fps = []
                    tps = []
                    for i in range(len(clean_imgs)):
                        dets_clean = detect(clean_imgs[i], s1, s2, thresh)
                        clean_fps.append(len(dets_clean))
                        
                        dets_def = detect(defected[i], s1, s2, thresh)
                        tp = 0
                        matched = set()
                        for dy, dx in dets_def:
                            if np.isnan(dy) or np.isnan(dx):
                                continue
                            for j, (ty, tx) in enumerate(all_locs[i]):
                                if j not in matched and np.sqrt((dy-ty)**2 + (dx-tx)**2) <= 6:
                                    tp += 1
                                    matched.add(j)
                                    break
                        tps.append(tp / len(all_locs[i]))
                    
                    mean_fp = np.mean(clean_fps)
                    mean_tpr = np.mean(tps)
                    
                    if mean_fp < 15:
                        print(f"{s1:4.1f} {s2:4.1f} {thresh:7.2f} | {mean_fp:12.1f} {mean_tpr:10.3f}")
                        if best is None or mean_tpr > best[0]:
                            best = (mean_tpr, s1, s2, thresh, mean_fp, radius)
        
        if best:
            print(f"Best at r={radius}: s1={best[1]}, s2={best[2]}, thresh={best[3]} -> TPR={best[0]:.3f}, FP={best[4]:.1f}")

if __name__ == "__main__":
    main()
