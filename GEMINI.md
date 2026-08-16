# Agent Build Brief — KLA PS01 Image Restoration

---

## PROMPT BEGINS

You are implementing a semiconductor image-restoration system for a hackathon submission judged by KLA. Work through the phases below in order. **Do not skip ahead — each phase has a verification gate that must pass before you proceed.**

### Context

The task is joint denoising + deblurring + ×2 super-resolution on single-channel float arrays from semiconductor inspection tooling. Scoring is on PSNR, SSIM, LPIPS, and inference latency on an H100. Half the test set is out-of-distribution (structure types not present in training).

Data location: `C:\Users\LENOVO\Downloads\KLA\train\train\NoisyLR` and `C:\Users\LENOVO\Downloads\KLA\train\train\GT`. Files are matched by filename.

### Non-negotiable constraints

Violating any of these invalidates the submission. Treat them as hard requirements, not preferences.

1. **All I/O is `.npy` float32.** Never route arrays through PIL, OpenCV, `imread`, `imwrite`, or any 8-bit image format at any point in the train or inference path. Doing so destroys sub-integer precision and out-of-range values. Write a round-trip unit test that loads a real degraded file, saves it, reloads it, and asserts bit-exact equality. Run it before anything else.
2. **`evaluate.py` must run with exactly two arguments** — `--input_dir` and `--output_dir` — on a freshly cloned repository, in a fresh virtualenv, with no manual edits, no environment variables, and no working-directory assumptions. Model weights load from a path resolved relative to the script file itself.
3. **Never fabricate a measurement.** If a number is not yet measured, write `TBD` and say so. Do not fill benchmark tables with plausible-looking values.
4. **No GAN training and no adversarial loss.** Hallucinated structure is a domain-level failure here, not a stylistic one.
5. **Deterministic inference.** The shipped inference path must produce identical output for identical input. Seed everything; no sampling on the default path.

### Phase 1 — Measure the data before modelling it

Do not write any model code in this phase.

Build `analysis/degradation_study.py`. It must load at least 50 matched pairs and report:

- Array shapes, dtypes, and value ranges for both sets. Confirm the scale factor is uniformly ×2. Confirm the ground-truth range.
- **Resampling kernel.** Downsample each GT with bicubic, bilinear, area-average, and Lanczos, antialiased and not. Report which minimizes residual against the paired degraded array.
- **Noise magnitude.** Compute the residual `r = x / (A·y)` using the winning kernel `A`. Report `mean(r)` and `std(r)` per file and in aggregate. A mean near 1.0 confirms unbiased multiplicative noise. `std(r)` is σ. Report its distribution across files.
- **Noise structure.** Compute the 2-D autocorrelation of `r`. A delta-like peak means noise is applied after downsampling; measurable width means it is applied before and averaged down.
- **Blur.** Sweep pre-blur σ and report which value, combined with the winning kernel, best reproduces the degraded array.

Prior analysis of one pair suggests: multiplicative speckle with σ somewhere in 0.15–0.22 (two estimators disagree — resolve this), GT range exactly [0, 1], mean-preserving downsampling. **Treat these as hypotheses to test, not as facts.** If your measurements contradict them, trust your measurements and say so explicitly.

**Gate:** print a summary block with the fitted degradation parameters and their spread across files. Stop and report before continuing.

### Phase 2 — Data pipeline

- `src/normalization.py` — invertible per-image robust normalization. Map the 1st/99th percentile of the input to [0,1]; return the transform parameters; provide an exact inverse. Percentiles, not min/max. Unit-test that `inverse(forward(x)) == x` to float32 tolerance.
- `src/noise_estimate.py` — scalar noise-level estimator σ̂ from a degraded array (Laplacian convolution + median-absolute-deviation). Must run in well under a millisecond.
- `src/degradation.py` — synthetic pair generator implementing the forward model measured in Phase 1, with each parameter randomized over a range centred on the measured value and widened roughly ±50%. Blur σ, resampling kernel, speckle σ, optional additive Gaussian, optional gamma shift.
- `src/dataset.py` — paired loader. Random crops of 64×64 LR / 128×128 HR. Full D4 augmentation (flips + 90° rotations). CutBlur. Configurable mixing ratio between provided pairs and synthetic pairs.
- **Validation split by source type, not randomly.** Infer structure categories from filenames or directory layout; if that is not possible, cluster on image statistics and report what you did. Hold out at least one category entirely. Every metric is reported twice thereafter: in-distribution and held-out-source.

**Gate:** save a figure showing real degraded/GT pairs beside synthetic degraded/GT pairs generated from the same GT. They should be visually indistinguishable. If they are not, return to Phase 1.

### Phase 3 — Model

`src/model.py`, a NAFNet-style encoder–decoder:

- Input: 2 channels — the normalized array and `log(normalized + 1e-3)`, clamped at zero.
- All blocks operate at **LR resolution**. This is the primary latency decision; do not upsample early.
- FiLM conditioning on the scalar σ̂ from `noise_estimate.py`.
- Output head: PixelShuffle ×2, producing **2 output channels** — the residual and a predicted log-variance.
- Global residual: add a bicubic ×2 upsample of the input to the predicted residual.
- Final: inverse-normalize, then clip to the GT range measured in Phase 1.
- Target 3–8 M parameters. Pure convolution — no attention, no dynamic shapes, so it exports cleanly to ONNX.

Make the uncertainty head switchable via a constructor flag so a single-output baseline is available for ablation.

### Phase 4 — Losses

`src/losses.py`, each independently toggleable and weighted by config:

- Charbonnier (primary reconstruction)
- MS-SSIM
- L1 on FFT magnitude
- LPIPS (replicate grayscale to 3 channels for the VGG extractor)
- Gradient-consistency L1
- Gaussian NLL, for the uncertainty head. **Warm up with Charbonnier for the first 10k iterations before transitioning, and clamp predicted log-variance** — NLL is unstable from a cold start.
- Forward-consistency: re-degrade the prediction with the measured operator `A` and penalize disagreement with a local-mean-filtered version of the input. Since the noise is unbiased, the local mean of `x` is an unbiased estimate of `A·y`.

### Phase 5 — Training

`train.py`, config-driven (YAML or dataclass — no hardcoded paths):

AdamW at 2e-4 with cosine decay to 1e-6, batch 32–64, bf16 autocast, `channels_last`, EMA at 0.999, progressive patch size, checkpointing with resume, and logging of both validation splits.

**Gate:** train the smallest viable configuration to convergence and confirm it beats bicubic on both splits before scaling up.

### Phase 6 — Evaluation

`evaluate.py` — see constraint 2. Reads `.npy`, writes `.npy`, preserves filenames exactly, `torch.inference_mode()`, bf16, batched, prints total time and mean ms/image. Optional flags `--tta` and `--uncertainty`, both defaulting to off and neither required for a scored run.

`src/metrics.py` — PSNR, SSIM, LPIPS, computed on float arrays in the GT range. Do not quantize to 8-bit before computing metrics.

`src/defect_eval.py` — **this is the differentiating deliverable; do not treat it as optional.** Inject synthetic defects into GT arrays (small bright/dark blobs, line breaks, line bridges) at controlled contrast levels. Apply the measured degradation. Run a fixed, simple, non-learned detector — blob detection or thresholded local difference — on the degraded array, the restored output, and the GT. Report detection rate and false-positive rate at matched thresholds as a function of defect contrast.

**Fix the detector's parameters before you look at any results**, and report the GT row as a ceiling. The point is to measure whether restoration recovers detectability, not to tune a detector until it flatters the model.

### Phase 7 — Reporting

Produce, as files in `analysis/`:

- The ablation table, one row per component, with in-distribution and held-out-source columns plus latency. Every row is a single change from the row above.
- The defect-detection table from Phase 6.
- Before/after/GT visual triptychs for both in-distribution and OOD samples.
- A three-panel figure: restored output, predicted uncertainty map, absolute error against GT. The uncertainty map should correlate with the error map; report the Spearman correlation.
- A quality-versus-latency scatter across at least three model configurations.

### Optional Phase 8 — only if Phases 1–7 are complete and time remains

Train a conditional diffusion model and sample it 8–16 times per input to obtain a **posterior variance map**. This is not the shipped restorer — the deterministic model ships. The purpose is to cross-check the learned uncertainty head against an independent estimate of what was genuinely unrecoverable. Report the correlation between the two maps.

Do not begin this phase at the expense of a working `evaluate.py`.

### Explicitly out of scope

- Adversarial or GAN training.
- Attention-based backbones (SwinIR, Restormer) as the primary model — latency is scored.
- Any dependency not installable via `pip install -r requirements.txt` on a clean Ubuntu image.
- TTA enabled by default.
- Jupyter notebooks anywhere in the inference path. Notebooks are permitted for analysis only.

### Definition of done

- [ ] `.npy` float32 round-trip test passes.
- [ ] Degradation parameters measured, reported, and used by the synthetic generator.
- [ ] Fresh clone into a fresh virtualenv → `pip install -r requirements.txt` → `python evaluate.py --input_dir X --output_dir Y` produces correct output arrays with zero manual intervention. Verified on a machine that has never seen the project.
- [ ] Every metric reported for both the in-distribution and held-out-source splits.
- [ ] Defect-detection evaluation complete, with the detector fixed in advance.
- [ ] Latency measured and reported in ms/image, with the hardware named.
- [ ] `README.md` allows a reviewer to run inference without contacting the authors.
- [ ] `requirements.txt` is a genuine `pip freeze`.
- [ ] No fabricated numbers anywhere in the repository or the report.

### Working style

Report measurements as you obtain them rather than batching everything to the end. When a measurement contradicts an assumption in this brief, say so directly and propose the correction — the brief was written from a single data pair and is expected to be wrong in places. Prefer a smaller model that demonstrably works over a larger one that has not been validated end to end.

## PROMPT ENDS

---
