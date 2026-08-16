# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Technical Solution Report — SEMICON India Hackathon 2026, Track 1 (KLA), PS01**

*Version 2 — incorporates ground-truth pair analysis and differentiation strategy*

---

> **Status of this document.** This is a *solution design* report, written before training. All quantitative result tables are marked `[TBD]` and must be filled with measured numbers before submission. Section 2 reports real measurements from provided data, with the confidence level of each claim stated explicitly.

---

## 1. Problem Framing

### 1.1 What the task reduces to

Stripped of framing, PS01 is a **joint restoration problem**: simultaneous denoising, deblurring, and ×2 super-resolution on single-channel float arrays, judged on a mixed distortion/perceptual metric set plus inference latency, with a test set containing out-of-distribution samples.

| Aspect | Specification |
|---|---|
| Input | Degraded array, 128×128 or 256×256 |
| Output | Restored array, 256×256 or 512×512 |
| Scale factor | **Always ×2** in both cases |
| Channels | 1 (grayscale) |
| Scored on | SSIM, PSNR, LPIPS, inference time |
| Test set | In-distribution + out-of-distribution samples |

The constant ×2 scale factor is a significant simplification: a single fully-convolutional model handles both resolution regimes with no scale conditioning and no architectural branching. **Verify this holds across the full dataset before committing.**

### 1.2 Ambiguity in the problem statement

The published degradation table lists "Gaussian Noise" but describes it as *"image appears soft and hazy — edges and fine structures lose sharpness."* That describes **blur**, not additive noise. This solution treats that row as a blur/PSF term and handles additive Gaussian noise separately as a robustness measure. Raise this in the hackathon Q&A for confirmation.

### 1.3 Why this matters industrially

Process control depends on measurements taken from inspection imagery. Two failure modes carry real cost:

- **False positives** — noise read as a defect triggers unnecessary tool downtime, wafer scrap, and engineering investigation.
- **Escapes** — a real defect hidden under noise or lost to downsampling propagates to packaged product, where detection costs orders of magnitude more.

Restoration that raises effective SNR *without inventing structure* lets the fleet scan faster — lower magnification, shorter dwell time — while holding defect sensitivity constant. That is a direct throughput gain. This framing drives the architectural decision in §5.3 and the evaluation strategy in §7.1.

---

## 2. Data Characterization

Analysis performed on a matched pair: `000068.npy` from both the degraded and ground-truth sets.

### 2.1 Format — confirmed

| Property | Degraded | Ground truth |
|---|---|---|
| Container | `.npy` — **not an image format** | `.npy` |
| dtype | `float32` | `float32` |
| Shape | `(128, 128)` | `(256, 256)` |
| Observed range | −0.0040 to 1.3438 | **exactly 0.0 to 1.0** |
| Mean | 0.5375 | 0.5378 |
| Std | 0.2509 | 0.2369 |
| Pixels > 1.0 | 1.51 % | 0 |

Two conclusions follow immediately.

**Output clipping to [0, 1] is correct and free.** The GT range is exactly [0, 1], with true 0.0 and 1.0 values present in the data. Clamping model output to this interval is guaranteed-safe and recovers a small amount of PSNR at zero cost. This open item is now closed.

**The evaluation script must read and write `float32 .npy`.** Any pipeline routing through PIL, `imread`/`imwrite`, or 8-bit PNG silently destroys both the sub-integer precision and the out-of-range input values. This remains the single most likely cause of a well-trained submission scoring near zero.

### 2.2 Noise model — multiplicative speckle, confirmed

**Mean preservation.** Degraded mean (0.5375) and GT mean (0.5378) agree to 3×10⁻⁴. This confirms two things at once: the noise is **unbiased multiplicative** (E[1+n] = 1, so E[x] = E[y]), and the downsampling operator is mean-preserving. Additive-noise or biased models are excluded.

**Intensity dependence.** Local noise σ was estimated using an Immerkær-style Laplacian kernel with a median-absolute-deviation estimator over 8×8 blocks, binned against local mean:

| Local mean μ | Estimated σ | σ / μ |
|---|---|---|
| 0.00 – 0.15 | 0.032 | 0.28 |
| 0.15 – 0.30 | 0.048 | 0.24 |
| 0.30 – 0.45 | 0.078 | 0.23 |
| 0.45 – 0.60 | 0.112 | 0.21 |
| 0.60 – 0.75 | 0.141 | 0.21 |
| 0.75 – 0.90 | 0.176 | 0.22 |

σ/μ is flat. The three candidate models predict distinguishable signatures:

| Model | Predicted signature | Fits? |
|---|---|---|
| Additive Gaussian | σ constant, independent of μ | **No** — σ varies 5.5× |
| Poisson / shot noise | σ ∝ √μ ⟹ σ/μ decreasing | **No** — σ/μ is flat |
| Multiplicative speckle | σ ∝ μ ⟹ σ/μ constant | **Yes** |

The degradation is therefore:

```
x = A·y · (1 + n),    n ~ N(0, σ²),    E[n] = 0
```

where `A` is the ×2 downsampling operator. This also explains the range overshoot in §2.1 with no additional mechanism: pixels near y = 1.0 scale up to ~1.34 in the tail, near-zero pixels dip marginally negative.

### 2.3 Open item — σ magnitude is not yet settled

Two independent estimators disagree:

| Method | Implied σ |
|---|---|
| Local Laplacian/MAD on the degraded array | **0.22** |
| Global variance budget, GT ↔ degraded | **0.15** |

The variance-budget calculation: given `Var(x) = (Var(Ay) + E[Ay]²)(1 + σ²) − E[Ay]²`, the observed degraded variance of 0.0630 against an area-downsampled GT variance of 0.0552 implies σ ≈ 0.15. Achieving σ = 0.22 would require the clean LR image to have std ≈ 0.216, which is below what any plausible pre-blur produces from this GT.

The likely explanation is that the Laplacian estimator inflates on this image because dense fine wafer texture bleeds into the high-frequency band it measures. **The true value is probably 0.15–0.18.**

**Action:** compute the residual `r = x / (A·y)` directly across ≥50 real pairs and read σ off `std(r)`. This is definitive and takes minutes. **Do not build the synthetic generator until this is done** — an error here biases every synthetic training pair.

### 2.4 Open item — degradation ordering

Two orderings are plausible and produce different synthetic data:

- **(A)** `blur → downsample → speckle` — noise is added by the sensor after optical and sampling losses; grain is white at LR scale.
- **(B)** `blur → speckle → downsample` — HR noise is averaged down by resampling; grain is correlated and lower-amplitude.

Measured noise appears spatially white at LR resolution, weakly favouring (A). **Resolve by computing the residual autocorrelation of `x / (A·y)`** — under (A) it is a delta function; under (B) it has measurable width.

### 2.5 Open item — resampling kernel and blur

Downsample the GT with bicubic, bilinear, area-average, and Lanczos (antialiased and not), sweep pre-blur σ, and select the combination minimizing residual against the paired degraded array. One-line sweep, removes a major train/test mismatch. Preliminary indication is that some pre-blur is present, but the estimate is confounded with the unresolved σ above — settle §2.3 first.

---

## 3. Method Overview

```
 .npy float32 (128×128)
        │
        ├──► robust percentile normalization ──► transform params (s, b) ────┐
        │                                                                    │
        ├──► noise-level estimator σ̂ ──────────────────────────────────────┐ │
        │                                                                  │ │
        ▼                                                                  │ │
 [ x_norm , log(x_norm + ε) ]   (2 channels)                               │ │
        │                                                                  │ │
        ▼                                                                  │ │
 ┌────────────────────────────────────────────────┐                        │ │
 │  NAFNet encoder–decoder @ LR resolution        │ ◄── FiLM conditioning ─┘ │
 │  (all heavy compute at 128×128)                │                          │
 └────────────────────────────────────────────────┘                          │
        │                                                                    │
        ▼                                                                    │
 PixelShuffle ×2 head ──┬──► residual ──(+)── bicubic ×2 of input            │
                        │                                                    │
                        └──► log-variance channel  ──► uncertainty map        │
        │                                                                    │
        ▼                                                                    │
 inverse normalization ◄─────────────────────────────────────────────────────┘
        │
        ├──► clip to [0, 1]                    (§2.1)
        ├──► optional back-projection step     (§7.3)
        ▼
 .npy float32 (256×256)  +  trust map (256×256)
```

---

## 4. Input Handling

### 4.1 Robust per-image normalization

Fixed global normalization fails when test images come from different sources with different brightness and contrast statistics — precisely the OOD condition specified.

Instead: compute the 1st and 99th percentile of each input, map that range to [0, 1], run the model, invert the identical affine transform on output.

- **Percentiles, not min/max** — speckle outliers occupy the extreme tails (1.5% of pixels above 1.0). Min/max would let a handful of noise spikes set the scale for the whole image.
- **Invertible and deterministic** — parameters are carried through and applied in reverse; no absolute-intensity information is lost.
- **Effect** — the network sees contrast-normalized input regardless of source. The cheapest available generalization mechanism.

Apply the same wrapper during training so train and test distributions match exactly.

### 4.2 Log-domain representation

Because the noise is multiplicative (§2.2), the log transform makes it additive:

```
log(x) = log(A·y) + log(1 + n)
```

The transformed noise is approximately additive and, critically, **signal-independent** — homoscedastic rather than heteroscedastic. Convolutional networks with spatially-shared weights are structurally better suited to signal-independent noise: a single filter response applies uniformly rather than needing to modulate with local brightness. This is the classical homomorphic filtering argument applied as a representation choice rather than as a filter.

**Bias correction matters.** E[log(1 + n)] ≠ 0 — Jensen's inequality gives a negative bias of approximately −σ²/2 for small σ. Naive log-domain training will produce a systematic intensity offset. Either correct analytically or let the network learn it, but state which; most implementations get this silently wrong.

Implementation: feed both channels, `[x_norm, log(x_norm + ε)]`, ε ≈ 1e-3, input clamped at zero for the small negative tail. Providing both lets the network use whichever representation suits each region.

### 4.3 Noise-level conditioning

The §2.2 estimator costs microseconds and returns a scalar σ̂. Inject it via FiLM layers — per-channel affine modulation of intermediate features conditioned on σ̂.

The rationale is specific to the OOD requirement. The test set includes samples from different sources, and the parallel Applied Materials track explicitly states its test data is noisier than training data; it is reasonable to expect the same here. A model with a hard-coded implicit noise assumption will over-smooth clean inputs and under-denoise noisy ones. Conditioning on measured σ̂ lets one set of weights span the range.

---

## 5. Architecture

### 5.1 Backbone

**NAFNet-style encoder–decoder operating entirely at LR resolution, with a PixelShuffle ×2 head, plus a global residual from a bicubic ×2 upsample of the input.**

| Decision | Justification |
|---|---|
| All computation at LR resolution | A network that upsamples early does ~4× the work per layer. Latency is scored; deferring upsampling to one PixelShuffle layer at the end is the highest-leverage speed decision available. |
| NAFNet blocks (SimpleGate, no activations, no attention) | Near the compute/quality frontier for restoration. Pure convolution — predictable latency, clean ONNX export, good TensorRT behaviour on H100. |
| PixelShuffle over transposed convolution | Avoids checkerboard artifacts, which on a wafer image would be misread as periodic structure. |
| Global residual from bicubic | The network learns only the correction. Faster convergence and a graceful failure mode: output collapse degrades to bicubic, not to noise. |

Target: 3–8 M parameters, sub-100 ms per image on H100 at 256×256 output.

### 5.2 Alternatives considered

| Candidate | Disposition |
|---|---|
| SwinIR / Restormer | Strong quality; attention is substantially slower and harder to export. Rejected on latency. |
| GAN-based (ESRGAN family) | Rejected — see §5.3. |
| EDSR | Retain as an ablation baseline row. |
| SAFMN | Benchmark as the speed-optimized fallback if NAFNet misses the latency target. |

### 5.3 No adversarial training — a domain-driven decision

State this explicitly in the submission.

GAN-based restoration produces sharpness by *synthesizing* plausible high-frequency detail. In natural images that is desirable. In metrology it is a defect: a hallucinated particle is a false defect triggering an unnecessary investigation; a hallucinated-away particle is an escape. No downstream consumer can distinguish invented structure from recovered structure.

Adversarial training also degrades PSNR and SSIM — two of three scored quality metrics — for LPIPS gains. The rubric votes 2-to-1 against it.

---

## 6. Loss Function

```
L  =  L_Charbonnier
    + 0.20 · (1 − MS-SSIM)
    + 0.05 · L1( FFT(ŷ), FFT(y) )
    + 0.05 · LPIPS(ŷ, y)
    + 0.10 · L_gradient
    + 0.10 · L_consistency          (§7.3)
    +        L_NLL  (replaces Charbonnier when the uncertainty head is enabled, §7.2)
```

| Term | Purpose |
|---|---|
| **Charbonnier** | Smooth-L1 primary reconstruction loss. Less over-smoothing than MSE; differentiable everywhere unlike pure L1. |
| **MS-SSIM** | SSIM is scored directly; optimize a differentiable multi-scale form of it. |
| **Frequency-domain L1** | Penalizes Fourier magnitude discrepancy. This is what enforces "do not blur it to remove noise" — a blurred prediction has a distinctly deficient high-frequency spectrum even when pixel-space L1 is competitive. |
| **LPIPS** | Scored directly, so include it — at low weight. Grayscale replicated to 3 channels for the VGG extractor. Over-weighting trades away PSNR (perception–distortion tradeoff, §8.1). |
| **Gradient consistency** | L1 on spatial gradients, optionally weighted by local structure-tensor magnitude. In inspection imagery, fidelity at feature boundaries carries the measurement information; flat-region accuracy matters less. Reallocates loss budget accordingly. |
| **Forward consistency** | See §7.3. |

Weights are starting points; sweep and report the ablation.

---

## 7. Differentiation Strategy

The architecture will not differentiate this submission. A large fraction of teams will submit a U-Net or NAFNet trained with L1 + SSIM and flip augmentation, and the PSNR spread between a competent and an excellent implementation is small enough that a jury cannot perceive it from a slide.

Differentiation comes from three other places: **what the system outputs, how success is measured, and what can be guaranteed.** Four contributions, ranked by differentiation × feasibility × relevance to the sponsor.

### 7.1 Task-level evaluation — does restoration improve defect detection?

**The primary differentiator.**

Every team will report PSNR/SSIM/LPIPS, which measure whether the image *looks* like ground truth. KLA does not sell images; it sells defect detection. Build a second evaluation axis:

1. Inject synthetic defects into GT images — particles, line bridges, line breaks — a few pixels each, at controlled contrast levels.
2. Apply the measured degradation model (§2.2).
3. Run a fixed, simple, non-learned defect detector — blob detection or die-to-die differencing — on three inputs: the degraded array, the restored output, and the GT.
4. Report **detection rate and false-positive rate at matched operating thresholds**, as a function of defect contrast.

The resulting claim is of a different kind: *"restoration recovers 87% of defects that were undetectable in the degraded image, at no increase in false-positive rate"* rather than *"we reached 32.4 dB."* It reframes the submission from an image-quality exercise into a yield exercise, in the language the jury speaks.

Cost: roughly one day. No additional training required. The detector must be fixed and simple, so that any measured improvement is attributable to restoration rather than to detector tuning.

### 7.2 Per-pixel uncertainty — a trust map

Add one output channel predicting per-pixel log-variance, trained with Gaussian negative log-likelihood in place of plain Charbonnier. Cost: one extra convolution channel, effectively zero latency.

The output is a map distinguishing regions where the model confidently recovered real structure from regions where information was genuinely destroyed and the output is interpolation.

This answers the hallucination objection with a deliverable rather than an argument. An inspection engineer can gate decisions on it — do not call a defect in a high-uncertainty region. No competing team is likely to hand the jury a restoration model that reports when it should not be trusted.

Implementation note: the NLL loss is unstable early in training. Warm up with Charbonnier for the first ~10k iterations, then transition, and clamp predicted log-variance to a sane interval.

### 7.3 Physics-consistent restoration

The forward operator is now known almost exactly (§2.2): blur, ×2 downsample, unbiased multiplicative speckle. This is unusual — most super-resolution work is blind. Exploit it.

**Forward-consistency loss:** re-degrade the output with the known operator and require agreement with the input.

```
L_consistency = ‖ A·ŷ − x/(1 + n̂) ‖₁     ≈   ‖ A·ŷ − E[x] ‖₁
```

Since the noise is unbiased, `E[x] = A·y`, so a local-mean-filtered `x` is an unbiased estimate of `A·y` and the constraint is directly computable without knowing the noise realization.

Optionally enforce it as a hard projection at inference — one iterative back-projection step, sub-millisecond.

The resulting claim: *"our output is provably consistent with the measurement — re-imaging it through the known degradation reproduces the observed input."* That is a guarantee rather than a hope, and it is the kind of statement a metrology organization responds to. It also structurally suppresses hallucination: invented structure inconsistent with the measurement is penalized.

### 7.4 A quality–latency Pareto frontier

Most teams will submit one checkpoint. Submit three operating points — fast / balanced / accurate — with measured H100 latency for each, exported to ONNX/TensorRT, plus the frontier plot. Then state which point you would deploy on an inline tool versus an offline review station, and why.

That is a fleet-deployment answer rather than a benchmark answer, and it signals an understanding that KLA ships tools, not checkpoints.

### 7.5 The unglamorous differentiator

Of everything above, the contribution with the highest expected value remains **a repository that runs correctly on the first attempt.** The problem statement states twice, in bold, that an unrunnable evaluation script cannot be scored. A meaningful fraction of submissions will fail on this alone. Prioritize §7.1 and §7.2, ship a flawless repository, and the position is strong before any of the more ambitious ideas land.

---

## 8. Role of Diffusion Models

### 8.1 Rejected as the primary restorer

Diffusion-based restoration was evaluated and rejected as the primary model — for reasons specific to this competition's scoring function, not to the method's quality.

**Latency.** Standard diffusion sampling requires 20–1000 network evaluations per image. Even aggressive DDIM schedules at 20 steps cost ~20× a single-pass regression model of comparable capacity. Inference time is explicitly scored.

**Metric alignment.** The perception–distortion tradeoff (Blau & Michaeli, 2018) establishes that distortion metrics (PSNR, SSIM) and perceptual quality are in fundamental tension beyond a certain point. Diffusion sits at the perceptual extreme and typically concedes 1–3 dB PSNR to a regression model trained on L1/Charbonnier. The scored set is two distortion metrics to one perceptual metric.

**Determinism.** Diffusion sampling is stochastic: two runs on identical input produce different outputs. Metrology requires measurements reproducible across time and across tools — that is the premise of process control. A non-deterministic restoration step breaks measurement comparability. This objection is stronger than the metric argument and applies even where a diffusion model would score well.

**Hallucination under distribution shift.** A diffusion prior encodes the training distribution. On OOD structures it does not merely fail — it actively pulls output toward structures it has seen. Regression models degrade toward blur, a conservative and recognizable failure mode. With half the test set explicitly OOD, this asymmetry matters.

**Compute budget.** Diffusion needs roughly an order of magnitude more training compute to surpass a well-tuned regression baseline.

### 8.2 Deployed as a posterior sampler for uncertainty auditing

Diffusion is nonetheless included — placed where it is genuinely superior rather than where it is fashionable.

A diffusion model conditioned on a degraded input is, properly understood, a **sampler from the posterior p(y | x)** of an inverse problem. That capability is wasted when used to emit a single point estimate; the point estimate is exactly where it loses to regression.

Instead: run the conditional diffusion model k times (k ≈ 8–16) on the same input and compute the **pixel-wise variance across samples**. Regions where samples agree are regions where the measurement genuinely determines the answer. Regions where samples diverge are regions where information was destroyed and *any* restoration method — including ours — is interpolating.

This yields:

- **Zero cost on the scored path.** The deterministic regression model is what ships and what gets benchmarked. The diffusion ensemble runs offline or behind a flag; measured latency is unaffected.
- **An independent cross-check on §7.2.** If the cheap learned uncertainty head agrees with the diffusion posterior spread, that is validation. If it disagrees, that is a finding worth reporting.
- **A defensible narrative.** *"We evaluated diffusion for restoration, measured that it costs N dB PSNR and 20× latency on this data, and redeployed it where it is actually superior — quantifying what was unrecoverable."* Stronger than either adopting diffusion for fashion or dismissing it without measurement.
- **The strongest visual in the deck.** A three-panel figure — restored output, learned uncertainty, diffusion posterior variance — with the two uncertainty maps illuminating the same regions.

Relevant literature for the framing: DPS, DDRM, and ΠGDM on diffusion-based posterior sampling for inverse problems.

### 8.3 If a generative restorer is pursued anyway

Only the **residual / few-step** family is compatible with the latency constraint:

- **ResShift** (Yue et al., 2023) — diffuses the residual between LR and HR rather than from Gaussian noise; ~15 steps.
- **SinSR** (Wang et al., 2024) — distills ResShift to a **single step**, latency-competitive with regression.

Expose single-step refinement behind a `--refine` flag, disabled by default, and report both configurations.

### 8.4 Reporting

| Model | PSNR | SSIM | LPIPS | ms/img (H100) |
|---|---|---|---|---|
| Bicubic ×2 (floor) | [TBD] | [TBD] | [TBD] | [TBD] |
| NAFNet-×2 (regression, shipped) | [TBD] | [TBD] | [TBD] | [TBD] |
| + 1-step residual diffusion refinement | [TBD] | [TBD] | [TBD] | [TBD] |
| 15-step ResShift | [TBD] | [TBD] | [TBD] | [TBD] |

A measured table demonstrating the tradeoff was quantified *on this data* is worth more than either model alone. It converts a fashion decision into an engineering decision.

---

## 9. Data Strategy

### 9.1 Synthetic pair generation

Once §2.3–2.5 are settled, the degradation is fully specified as a forward process, so unlimited training pairs can be generated from the provided GT images:

```
y  = ground truth (clean, HR, ∈ [0,1])
    → random blur, σ_blur ~ U(0.3, 1.5)          [range pending §2.5]
    → downsample ×2, kernel ~ {bicubic, bilinear, area, Lanczos} × {AA, no-AA}
    → multiplicative speckle, σ ~ U(0.08, 0.35)  [centre pending §2.3]
    → optional additive Gaussian, σ_add ~ U(0, 0.02)
    → optional gamma / intensity shift
    → x
```

Ranges are **centred on measured values and deliberately widened**. This is the primary mechanism for OOD generalization, following the degradation-modelling strategy of BSRGAN and Real-ESRGAN: a model trained across a broad randomized degradation space generalizes to unseen degradations far better than one trained on a single fixed operator.

Train on a mixture of provided and synthetic pairs — 50/50 as a starting ratio, tuned by ablation.

### 9.2 Geometric augmentation

The full D4 dihedral group: horizontal flip, vertical flip, 90° rotations. Semiconductor layouts have strong axis-aligned structure, making these both label-preserving and distribution-preserving. Free capacity gain.

### 9.3 CutBlur

CutBlur (Yoo et al., 2020) pastes a rectangular region of the upsampled LR input into the HR image and vice versa, training the model to determine *where* and *how much* restoration to apply rather than applying a uniform transformation. Designed specifically to improve super-resolution robustness on unseen degradation levels — directly aligned with the OOD requirement.

### 9.4 Validation protocol

**Split by source type, not randomly.** The dataset spans multiple structure types; hold out at least one category entirely.

Report two numbers throughout:

- **In-distribution validation** — random split; predicts the in-distribution test half.
- **Held-out-source validation** — predicts the OOD half.

A random split overstates OOD performance and gives false confidence during development. Reporting both, including where the held-out number is worse, reads as rigor to a technical jury.

---

## 10. Training Protocol

| Parameter | Value |
|---|---|
| Patch size | 64×64 LR → 128×128 HR (progressive to 96×96 LR late) |
| Batch size | 32–64 |
| Optimizer | AdamW, lr 2e-4, cosine decay to 1e-6 |
| Precision | bf16 autocast, `channels_last` |
| Weight averaging | EMA, decay 0.999 |
| NLL warm-up | Charbonnier for first 10k iters, then transition (§7.2) |
| Iterations | 100–200 k (budget-dependent) |
| Hardware | [TBD — record GPU type, platform, wall-clock training time] |

EMA is consistently worth 0.1–0.3 dB at zero inference cost; do not skip it.

---

## 11. Inference and Deliverables

### 11.1 Evaluation script

The problem statement is explicit that this script is executed as-is by KLA's benchmarking team and that an unrunnable script cannot be scored. It is scored infrastructure, not an afterthought.

- `argparse` interface: `--input_dir`, `--output_dir`. No other mandatory arguments.
- Reads `.npy` float32. Writes `.npy` float32. Filenames preserved exactly.
- Output clipped to **[0, 1]** (confirmed §2.1).
- Device auto-detection with CPU fallback.
- `torch.inference_mode()`, bf16 autocast, batched processing.
- Weights loaded from a path **relative to the script file**, not an absolute path and not dependent on the working directory.
- Prints total wall-clock time and mean ms/image.
- Optional flags — `--tta`, `--refine`, `--uncertainty` — all default off, none required for a scored run.

**Acceptance test:** `git clone` into a fresh virtual environment on a machine that has never seen the project, `pip install -r requirements.txt`, run the script. If any manual step is required, it has failed.

### 11.2 Test-time augmentation

×8 geometric self-ensemble yields roughly +0.1 to +0.2 dB at 8× latency. Given explicit latency scoring, TTA is disabled by default and exposed behind `--tta`. Report both configurations and state the tradeoff — the reasoning is itself evidence of engineering judgment.

### 11.3 Repository structure

```
├── README.md                     # clone → run inference, no external contact needed
├── requirements.txt              # full pip freeze
├── evaluate.py                   # standalone; --input_dir --output_dir
├── train.py                      # reproduces training from scratch
├── src/
│   ├── model.py                  # NAFNet-×2 + FiLM + uncertainty head
│   ├── normalization.py          # robust percentile transform (invertible)
│   ├── degradation.py            # synthetic pair generator
│   ├── losses.py                 # Charbonnier, MS-SSIM, FFT, gradient, NLL, consistency
│   ├── noise_estimate.py         # Laplacian + MAD σ̂ estimator
│   └── defect_eval.py            # §7.1 task-level evaluation harness
├── weights/
│   └── model.pt                  # Git LFS or documented external link
├── outputs/                      # restored test set outputs
└── analysis/
    ├── degradation_study.ipynb   # §2 characterization, reproducible
    └── uncertainty_audit.ipynb   # §8.2 diffusion posterior comparison
```

### 11.4 Mapping to required submission slides

| Slide | Source section |
|---|---|
| 2 — Problem statement | §1.3 |
| 3 — Idea description | §3, §5.1 |
| 4 — Proposed solution | §4, §5, §6, §9 + pipeline diagram from §3 |
| 5 — Innovation & uniqueness | §7.1, §7.2, §7.3, §8.2 |
| 6 — Results | §12 tables, §7.1 defect-detection curves, before/after triptychs, uncertainty three-panel |
| 7 — Technology & feasibility | §10 hardware row, §7.4 Pareto frontier, §11.1 latency |
| 9 — References | §14 |

File naming: `TeamName_KLA_PS01.pdf`. Maximum 8–9 slides. Remove the template instruction slide.

---

## 12. Evaluation Plan

### 12.1 Pixel-metric ablation

Each row is a single change from the row above.

| Variant | PSNR (in-dist) | SSIM | LPIPS | PSNR (held-out source) | ms/img |
|---|---|---|---|---|---|
| Bicubic ×2 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| U-Net baseline | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| → NAFNet blocks | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + robust normalization | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + log-domain channel | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + σ̂ FiLM conditioning | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + synthetic degradation pairs | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + CutBlur | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + FFT loss | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + forward-consistency loss | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + EMA | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

The held-out-source column carries the argument. Components introduced for generalization — normalization, σ̂ conditioning, synthetic degradation, CutBlur — should show larger gains there than in-distribution. If they do not, report that honestly rather than burying it.

### 12.2 Task-level evaluation (§7.1)

| Input | Defect detection rate @ contrast 0.05 | @ 0.10 | @ 0.20 | False positives / image |
|---|---|---|---|---|
| Degraded | [TBD] | [TBD] | [TBD] | [TBD] |
| Bicubic ×2 | [TBD] | [TBD] | [TBD] | [TBD] |
| Restored (ours) | [TBD] | [TBD] | [TBD] | [TBD] |
| Ground truth (ceiling) | [TBD] | [TBD] | [TBD] | [TBD] |

### 12.3 Uncertainty calibration (§7.2, §8.2)

- Reliability diagram: predicted σ versus realized absolute error, binned.
- Spearman correlation between the learned uncertainty map and the diffusion posterior variance map (§8.2).

---

## 13. Risks and Open Items

| Risk | Severity | Mitigation |
|---|---|---|
| `.npy` I/O handled as images in the eval script | **Critical** | Explicit float32 round-trip test in CI; verify bit-exact reload of an unmodified file |
| σ magnitude unresolved (§2.3) | **High** | Direct residual computation on ≥50 real pairs *before* building the generator |
| Degradation ordering guessed wrong (§2.4) | High | Residual autocorrelation test on real pairs |
| Scale factor not uniformly ×2 | Medium | Audit array shapes across the entire dataset |
| NLL loss destabilizes training | Medium | Charbonnier warm-up; clamp predicted log-variance; keep a no-uncertainty checkpoint as fallback |
| Latency target missed | Medium | Fall back to SAFMN or reduce channel width; profile early, not at the end |
| Task-level eval seen as self-serving | Low | Fix the detector before seeing results; publish the detector code; report GT ceiling alongside |
| Eval script fails on reviewer's machine | **Critical** | Fresh-clone acceptance test on a different machine before submission |

---

## 14. References

1. Blau, Y. & Michaeli, T. *The Perception-Distortion Tradeoff.* CVPR 2018.
2. Chen, L. et al. *Simple Baselines for Image Restoration* (NAFNet). ECCV 2022.
3. Zhang, K. et al. *Designing a Practical Degradation Model for Deep Blind Image Super-Resolution* (BSRGAN). ICCV 2021.
4. Wang, X. et al. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCVW 2021.
5. Yoo, J., Ahn, N. & Sohn, K.-A. *Rethinking Data Augmentation for Image Super-resolution* (CutBlur). CVPR 2020.
6. Cho, S.-J. et al. *Rethinking Coarse-to-Fine Approach in Single Image Deblurring* (MIMO-UNet, frequency loss). ICCV 2021.
7. Jiang, L. et al. *Focal Frequency Loss for Image Reconstruction and Synthesis.* ICCV 2021.
8. Zhang, R. et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS). CVPR 2018.
9. Wang, Z., Simoncelli, E. P. & Bovik, A. C. *Multiscale Structural Similarity for Image Quality Assessment.* Asilomar 2003.
10. Shi, W. et al. *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN* (PixelShuffle). CVPR 2016.
11. Lim, B. et al. *Enhanced Deep Residual Networks for Single Image Super-Resolution* (EDSR; geometric self-ensemble). CVPRW 2017.
12. Perez, E. et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018.
13. Kendall, A. & Gal, Y. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS 2017.
14. Nix, D. A. & Weigend, A. S. *Estimating the Mean and Variance of the Target Probability Distribution.* ICNN 1994.
15. Immerkær, J. *Fast Noise Variance Estimation.* CVIU, 1996.
16. Goodman, J. W. *Some Fundamental Properties of Speckle.* JOSA, 1976.
17. Lee, J.-S. *Digital Image Enhancement and Noise Filtering by Use of Local Statistics.* IEEE TPAMI, 1980.
18. Irani, M. & Peleg, S. *Improving Resolution by Image Registration* (iterative back-projection). CVGIP, 1991.
19. Chung, H. et al. *Diffusion Posterior Sampling for General Noisy Inverse Problems* (DPS). ICLR 2023.
20. Kawar, B. et al. *Denoising Diffusion Restoration Models* (DDRM). NeurIPS 2022.
21. Song, J. et al. *Pseudoinverse-Guided Diffusion Models for Inverse Problems* (ΠGDM). ICLR 2023.
22. Yue, Z., Wang, J. & Loy, C. C. *ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting.* NeurIPS 2023.
23. Wang, Y. et al. *SinSR: Diffusion-Based Image Super-Resolution in a Single Step.* CVPR 2024.
24. Zhang, K. et al. *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising* (DnCNN). IEEE TIP, 2017.

---

## 15. Development Timeline

| Days | Deliverable |
|---|---|
| 1–2 | Full-dataset degradation study (§2). **Close §2.3, §2.4, §2.5.** Paired dataloader with `.npy` round-trip test. |
| 3–4 | Bicubic and U-Net baselines. End-to-end train → evaluate → metrics loop operational. |
| 5–8 | NAFNet-×2 with normalization, log channel, σ̂ conditioning, uncertainty head. Synthetic degradation pipeline. Primary training run. |
| 9 | **Defect-injection evaluation harness (§7.1).** Runs against existing checkpoints — no retraining. |
| 10 | Loss ablations, consistency term, EMA, latency profiling and Pareto points. |
| 11 | Optional: diffusion posterior-sampling audit (§8.2). Skip without penalty if behind. |
| 12 | Freeze weights. Generate test outputs. Write and harden `evaluate.py`. Fresh-machine acceptance test. |
| 13–14 | README, `pip freeze`, slide deck, demo video. |

Round 1 (deadline 16 Aug 2026) is evaluated primarily on the deck and repository. A working baseline with a clearly-argued method and a flawless repository outscores a partially-trained sophisticated model with a broken evaluation script. **Reach a submittable state by day 10, then improve against the remaining time.**
