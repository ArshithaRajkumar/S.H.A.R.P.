import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path
import os

_EMPIRICAL_NOISE_PATH = Path(__file__).resolve().parent.parent / "analysis" / "empirical_noise.npy"
_EMPIRICAL_NOISE = None

def load_empirical_noise():
    global _EMPIRICAL_NOISE
    if _EMPIRICAL_NOISE is None:
        if _EMPIRICAL_NOISE_PATH.exists():
            _EMPIRICAL_NOISE = np.load(_EMPIRICAL_NOISE_PATH)
        else:
            # Fallback for CI/CD or if analysis wasn't run
            _EMPIRICAL_NOISE = np.random.randn(10000).astype(np.float32)

def generate_synthetic_pair(gt, 
                            blur_range=(0.15, 0.45),
                            noise_sigma_range=(0.10, 0.22), # Centered on measured 0.16
                            add_gaussian_range=(0.005, 0.025), # Centered on measured 0.015
                            gamma_range=(0.8, 1.2)):
    """
    Applies the measured forward degradation model to a GT image.
    Measured ordering: Pre-blur -> Multiplicative Noise -> Bicubic Downsample
    
    Args:
        gt: 2D numpy array (float32) in range [0, 1]. Size (256, 256).
        
    Returns:
        lr, gt_out: Degraded (128x128) and original GT (256x256).
    """
    load_empirical_noise()
    
    # 1. Optional Gamma shift (applied to GT)
    gamma = np.random.uniform(*gamma_range)
    if gamma != 1.0:
        # Avoid zero/negative values for gamma power
        gt_shifted = np.clip(gt, 1e-6, 1.0) ** gamma
    else:
        gt_shifted = gt.copy()
        
    # 2. Pre-blur (Gaussian)
    blur_sigma = np.random.uniform(*blur_range)
    if blur_sigma > 0:
        hr_blurred = gaussian_filter(gt_shifted, blur_sigma)
    else:
        hr_blurred = gt_shifted
        
    # 3. Multiplicative Noise (BEFORE downsampling, as per Phase 1 conclusion)
    # Sample noise from the empirical heavy-tailed distribution
    noise_sigma = np.random.uniform(*noise_sigma_range)
    
    # The empirical noise has std ~0.1726, we scale it to match our target sigma
    empirical_std = _EMPIRICAL_NOISE.std()
    scale = noise_sigma / empirical_std
    
    # Randomly draw from empirical distribution
    n_samples = hr_blurred.size
    noise_base = np.random.choice(_EMPIRICAL_NOISE, size=n_samples, replace=True).reshape(hr_blurred.shape)
    
    # Apply multiplicative noise
    hr_noisy = hr_blurred * (1.0 + noise_base * scale)
    
    # 4. Downsample (Bicubic x2)
    lr = zoom(hr_noisy, 0.5, order=3, prefilter=True)
    
    # 5. Additive Gaussian Noise (Measured sensor read noise floor)
    add_sigma = np.random.uniform(*add_gaussian_range)
    if add_sigma > 0:
        lr += np.random.randn(*lr.shape).astype(np.float32) * add_sigma
        
    return lr.astype(np.float32), gt.astype(np.float32)

if __name__ == "__main__":
    # Test generator
    gt = np.ones((256, 256), dtype=np.float32) * 0.5
    lr, gt_out = generate_synthetic_pair(gt)
    print(f"LR shape: {lr.shape}, LR mean: {lr.mean():.4f}, LR std: {lr.std():.4f}")
