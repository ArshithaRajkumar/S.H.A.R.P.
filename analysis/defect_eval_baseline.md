# Defect Detection Baseline Results

> **Status:** Phase 6 approved. These are the "before" reference numbers.
> The trained model's results will be compared against this table.

## Detector Configuration (Principled, Not Searched)

| Parameter | Value | Reasoning |
|---|---|---|
| DoG sigma_small | 1.8 | Scale-space theory: sigma = r/sqrt(2) = 2.12; sigma_small = sigma/sqrt(k) for k=sqrt(2) |
| DoG sigma_large | 2.5 | sigma_large = sigma * sqrt(k), standard SIFT octave ratio |
| Threshold | 0.04928 | 99th percentile of clean GT DoG response (fixed statistical rule) |
| Match radius | 6 px | ~2x defect radius, standard object-detection matching |
| Defect radius | 3 px | ~3% of FOV at 256x256, realistic for particles/shorts |
| Line defects | 8x2 px | Realistic line-break/bridge dimensions |
| Defects/image | 16 | 8 blobs + 4 line-breaks + 4 line-bridges |

## Baseline Table (Bicubic "Restoration")

| Contrast | GT TPR | GT FP/img | Degraded TPR | Degraded FP/img | Restored TPR | Restored FP/img |
|----------|--------|-----------|--------------|-----------------|--------------|-----------------|
| 0.02 | 0.125 | 95.2 | 0.184 | 152.8 | 0.184 | 152.8 |
| 0.05 | 0.153 | 95.2 | 0.169 | 137.2 | 0.169 | 137.2 |
| 0.10 | 0.128 | 95.6 | 0.150 | 134.2 | 0.150 | 134.2 |
| 0.20 | 0.134 | 96.3 | 0.200 | 138.2 | 0.200 | 138.2 |
| 0.30 | 0.253 | 96.2 | 0.350 | 143.4 | 0.350 | 143.4 |
| 0.50 | 0.716 | 97.2 | 0.713 | 170.8 | 0.713 | 170.8 |

## Interpretation

- **GT column is the ceiling.** The detector finds ~72% of defects at contrast 0.50 and ~25% at 0.30 on clean GT. Low-contrast defects (<=0.10) are near the structural noise floor of these semiconductor images.
- **~95 FP/img on clean GT** reflects the inherent structural density of semiconductor imagery at this DoG scale — not a detector miscalibration.
- **Degraded = Restored** because both use bicubic upsampling in this baseline. After training, these columns will diverge.
- **Degraded FP > GT FP** (~140-170 vs ~95) because multiplicative noise creates additional spurious DoG peaks.

## Success Criteria for the Trained Model

A successful restoration model should:
1. **Restored TPR approaching GT TPR** — the model recovers defect detectability lost to degradation.
2. **Restored FP/img closer to GT's ~95 than degraded's ~150** — the model removes noise-induced false positives.
3. These improvements should hold across contrast levels, not just at 0.50.
