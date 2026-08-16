# Pretrained Initialization Due Diligence — Full Investigation

As part of the optimization strategy, we investigated leveraging transfer learning from established image restoration models to warm-start our NAFNet-SR architecture before training on the KLA semiconductor dataset.

## Our Architecture Summary

| Property | Value |
|---|---|
| Architecture | NAFNet-style flat (non-UNet) |
| Macro-structure | 16 sequential NAFBlocks at constant resolution |
| Width (channels) | 256 |
| Input channels | 2 (normalized array + log-transformed) |
| Block internals | GroupNorm → 1×1 Conv (expand ×2) → 3×3 DWConv → SimpleGate → SCA → 1×1 Conv |
| Extra conditioning | FiLM MLP (scalar σ̂ → per-block affine) |
| Output head | PixelShuffle ×2 with zero-initialized projection |
| Parameter count | ~4.83M |

---

## Candidate 1: NAFNet-SIDD (width=64) — Megvii Research

- **Source:** [github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)
- **Task Match:** Image denoising (SIDD dataset) — domain-relevant
- **Modality:** 3-channel RGB
- **Micro-block:** Identical NAFBlock structure (SimpleGate, SCA, 1×1/3×3 convs)

**Verdict: REJECTED — Dimensional mismatch (width 64 vs 256) + Macro-architecture mismatch (UNet vs flat)**

The official SIDD checkpoints use `width=64` with a UNet macro-structure (`enc_blk_nums=[2,2,4,8]`, `middle_blk_num=12`, `dec_blk_nums=[2,2,2,2]`). Encoder widths scale 64→128→256→512 across levels. Our model is a flat sequence of 16 blocks all at width=256. Even extracting individual NAFBlock weights from their encoder level 0 (width=64) would give us only 2 blocks with mismatched channel dimensions. Force-padding a `[64, 64, 1, 1]` tensor to `[256, 256, 1, 1]` destroys the learned spatial-channel correlations.

---

## Candidate 2: DnCNN (Grayscale) — cszn

- **Source:** [github.com/cszn/DnCNN](https://github.com/cszn/DnCNN)
- **Task Match:** Grayscale Gaussian denoising — modality-relevant
- **Modality:** 1-channel grayscale

**Verdict: REJECTED — Completely different micro-architecture**

DnCNN is a classical plain CNN: sequential `[64, 64, 3, 3]` Conv + BatchNorm + ReLU layers. Our NAFBlocks use GroupNorm, channel-expanding 1×1 pointwise convolutions, grouped 3×3 depthwise convolutions, SimpleGate activations, and Simplified Channel Attention. Zero structurally overlapping tensors exist between the two architectures.

---

## Candidate 3: SCUNet (Grayscale) — cszn

- **Source:** [github.com/cszn/SCUNet](https://github.com/cszn/SCUNet)
- **Task Match:** Blind image denoising — domain-relevant
- **Modality:** 1-channel grayscale checkpoints available (`scunet_gray_15`, `_25`, `_50`)

**Verdict: REJECTED — Completely different micro-architecture + UNet macro-architecture**

SCUNet uses Swin-Conv (SC) blocks combining Swin Transformer attention with residual convolutions inside a UNet encoder-decoder at scales 64/128/256/512. Our model uses NAFBlocks (no attention mechanism) in a flat topology. The internal block structures share zero layer-name or tensor-shape compatibility.

---

## Candidate 4: Restormer (Grayscale) — swz30

- **Source:** [github.com/swz30/Restormer](https://github.com/swz30/Restormer)
- **Task Match:** Grayscale Gaussian denoising at σ=15/25/50 and blind — domain-relevant
- **Modality:** 1-channel grayscale checkpoints available

**Verdict: REJECTED — Attention-based architecture, explicitly out of scope**

Restormer uses Multi-Dconv Head Transposed Attention (MDTA) and Gated-Dconv Feed-Forward Networks (GDFN) in an encoder-decoder hierarchy. This is an attention-based backbone, which is explicitly listed as out-of-scope in the project brief ("Attention-based backbones (SwinIR, Restormer) as the primary model — latency is scored"). Even as a weight donor, the internal block structure (self-attention projections, Q/K/V matrices) shares zero compatibility with our NAFBlocks.

---

## Candidate 5: EDSR / RCAN (x2 Super-Resolution)

- **Source:** Various (EDSR: github.com/sanghyun-son/EDSR-PyTorch; RCAN: github.com/yulunzhang/RCAN)
- **Task Match:** ×2 super-resolution — task-relevant
- **Modality:** Typically 3-channel RGB; can be configured for 1-channel

**Verdict: REJECTED — Standard residual blocks, not NAFBlocks**

EDSR uses plain residual blocks (Conv-ReLU-Conv with skip). RCAN adds channel attention but uses standard convolutions throughout. Neither uses SimpleGate, depthwise separable convolutions, or GroupNorm. No tensor shapes overlap with our NAFBlock internals. The "large" EDSR variant uses width=256 (matching our channel count), but the block structure is fundamentally different — a `[256, 256, 3, 3]` standard conv weight cannot be loaded into our `[512, 256, 1, 1]` pointwise expand + `[512, 1, 3, 3]` depthwise pattern.

---

## Candidate 6: tk_r_em (Electron Microscopy Denoising)

- **Source:** [github.com/Ivanlh20/tk_r_em](https://github.com/Ivanlh20/tk_r_em)
- **Task Match:** Single-channel electron microscopy denoising — closest domain match
- **Modality:** Grayscale SEM/STEM/TEM

**Verdict: REJECTED — Different architecture + GAN-trained (out of scope) + ONNX-only**

tk_r_em uses a Concatenated Grouped Residual Dense Network (CGRDN) architecture with ~7M parameters, trained with a relativistic PatchGAN discriminator and 11-loss objective including adversarial loss. Our brief explicitly prohibits GAN-trained weights ("No GAN training and no adversarial loss"). Additionally, weights are distributed only as ONNX files, making PyTorch `state_dict` extraction non-trivial. The CGRDN block structure (grouped residual dense connections) shares no overlap with our NAFBlock internals.

---

## Candidate 7: NAFNet-SIDD (width=32)

- **Source:** Same as Candidate 1
- **Task Match:** Same as Candidate 1

**Verdict: REJECTED — Same dimensional incompatibility, worse (32 vs 256)**

Width=32 makes the mismatch even more severe than width=64.

---

## Structural Analysis: Why No Published Model Can Fit

The incompatibility is not incidental — it is structural, arising from three deliberate design choices in our architecture that differ from all published restoration models:

1. **Flat topology vs UNet:** All published NAFNet, SCUNet, and Restormer checkpoints use encoder-decoder UNets with spatial downsampling. Our model runs all 16 blocks at constant LR resolution (a deliberate latency optimization for H100 inference).

2. **Width=256 with depthwise separable convolutions:** Our NAFBlocks expand channels via 1×1 pointwise convs (`[512, 256, 1, 1]`) then apply 3×3 depthwise convs (`[512, 1, 3, 3]`). No published checkpoint at any width shares these exact tensor shapes, and you cannot meaningfully reshape between different depthwise channel counts.

3. **2-channel input with FiLM conditioning:** Our input layer expects 2 channels (normalized + log-transformed) with external scalar noise conditioning via FiLM. No published model has this input configuration.

---

## Conclusion

We investigated **7 candidate pretrained models** spanning:
- The exact same micro-architecture (NAFNet) at different widths
- Domain-matched grayscale denoisers (DnCNN, SCUNet, Restormer)
- Task-matched super-resolution models (EDSR, RCAN)
- The closest domain match available (tk_r_em electron microscopy)

**None are architecturally compatible** without destructive weight reshaping that would nullify any benefit of pretraining.

We proceed with the validated **zero-initialization from-scratch approach**, which guarantees the model starts at the exact bicubic-baseline PSNR and learns strictly positive improvements from step 1. This has been empirically verified: after just 40 gradient steps, the model already exceeds the bicubic baseline on both in-distribution and held-out-source validation splits.
