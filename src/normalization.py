import numpy as np

def robust_normalize(x, p_low=1.0, p_high=99.0, eps=1e-6):
    """
    Robustly normalizes an array mapping the p_low and p_high percentiles to [0, 1].
    
    Args:
        x: Input array (numpy).
        p_low: Lower percentile.
        p_high: Upper percentile.
        eps: Small value to avoid division by zero.
        
    Returns:
        x_norm: Normalized array.
        params: Tuple (low_val, scale) needed for exact inversion.
    """
    low_val = np.percentile(x, p_low)
    high_val = np.percentile(x, p_high)
    
    scale = high_val - low_val
    if scale < eps:
        scale = eps
        
    x_norm = (x - low_val) / scale
    return x_norm, (low_val, scale)

def robust_denormalize(x_norm, params):
    """
    Exact inverse of robust_normalize.
    
    Args:
        x_norm: Normalized array.
        params: Tuple (low_val, scale) returned by robust_normalize.
        
    Returns:
        x: Original unnormalized array.
    """
    low_val, scale = params
    x = (x_norm * scale) + low_val
    return x

if __name__ == "__main__":
    # Unit test
    x = np.random.randn(128, 128).astype(np.float32) * 2.5 - 0.5
    
    x_norm, params = robust_normalize(x)
    x_recon = robust_denormalize(x_norm, params)
    
    max_err = np.abs(x - x_recon).max()
    print(f"Max reconstruction error: {max_err}")
    assert max_err < 1e-5, "Inversion is not exact!"
    print("Normalization unit test passed.")
