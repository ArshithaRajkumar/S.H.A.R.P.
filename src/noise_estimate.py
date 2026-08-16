import numpy as np
import scipy.signal

# The standard 3x3 Laplacian kernel for noise estimation
# Sum of squares = 36
_LAPLACIAN_KERNEL = np.array([
    [1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1]
], dtype=np.float32)

def estimate_noise_sigma(img):
    """
    Estimates the global scalar noise level (sigma) of a single-channel image.
    Uses convolution with a Laplacian kernel followed by Median Absolute Deviation (MAD).
    Runs in <1ms for 128x128 images.
    
    Args:
        img: 2D numpy array (float32).
        
    Returns:
        sigma_hat: Scalar float noise estimate.
    """
    # Fast 2D convolution
    filtered = scipy.signal.convolve2d(img, _LAPLACIAN_KERNEL, mode='valid')
    
    # MAD of the filtered image
    mad = np.median(np.abs(filtered))
    
    # Scale factor for Gaussian noise and the specific kernel:
    # 1.4826 is the scaling factor for MAD to standard deviation
    # sqrt(36) = 6 is the l2 norm of the kernel
    # scale = 1.4826 / 6.0 ~ 0.2471
    sigma_hat = mad * 0.2471
    
    return float(sigma_hat)

if __name__ == "__main__":
    import time
    # Test performance and accuracy
    np.random.seed(42)
    test_img = np.random.randn(128, 128).astype(np.float32) * 0.15 + 0.5
    
    t0 = time.time()
    for _ in range(100):
        est = estimate_noise_sigma(test_img)
    t1 = time.time()
    
    print(f"Estimated sigma: {est:.4f} (Expected ~0.15)")
    print(f"Average time per call: {(t1 - t0) * 10:.2f} ms")
