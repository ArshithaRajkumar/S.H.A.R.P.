# Final Degradation Model Summary

Based on the Phase 1 degradation study and full dataset noise analysis, the final parameters for the synthetic degradation model are:

## Degradation Process

**Forward Model Equation:**
`LR = A( G_blur(HR) ) * (1 + N_mult) + N_add`

**Parameters:**
- **GT Range:** Exactly `[0.0, 1.0]` (confirmed physically bounded).
- **Scale Factor:** x2 uniform downsampling.
- **Resampling Kernel (`A`):** `bicubic` (Winner of the ratio bias + noise tests).
- **Noise Ordering:** Multiplicative noise is applied **BEFORE** downsampling (confirmed by 4-neighbor spatial autocorrelation and kernel-comparison tests).
- **Pre-Blur (`G_blur`):** Gaussian blur with `sigma = 0.3` (minimizes residual variance compared to no-blur).

## Noise Distribution (`N_mult`)
- **Type:** Multiplicative (variance explicitly scales with local brightness).
- **Aggregate Std (σ):** `0.1735`
- **Mean:** `-0.000385` (Effectively unbiased, `E[N] = 0`).
- **Excess Kurtosis (Fisher):** `1.46` (measured across full 3,200 images, heavily-tailed).
- **Sampling Strategy:** We directly sample from the measured empirical heavy-tailed distribution rather than fitting a Gaussian, scaling to the target sigma dynamically during training.

*(Note: Synthetic training incorporates ±50% randomized sweeps on these core values to prevent overfitting.)*
