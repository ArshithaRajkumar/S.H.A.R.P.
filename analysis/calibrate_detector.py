"""
Calibrate the DoG detector threshold using CLEAN GT images (no defects).
The threshold is set so that the expected FP rate on clean GT is < 5 per image.
This script looks ONLY at GT data — never at degraded or restored outputs.
"""
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

GT_DIR = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\GT")

def compute_dog_response(img, sigma_small=0.5, sigma_large=2.0):
    blur_small = gaussian_filter(img, sigma=sigma_small)
    blur_large = gaussian_filter(img, sigma=sigma_large)
    return np.abs(blur_small - blur_large)

def main():
    files = sorted(GT_DIR.glob("*.npy"))
    np.random.seed(42)
    sample = [files[i] for i in np.random.choice(len(files), 50, replace=False)]
    
    all_dog_vals = []
    per_image_stats = []
    
    for fn in sample:
        img = np.load(fn)
        dog = compute_dog_response(img)
        all_dog_vals.append(dog.flatten())
        per_image_stats.append({
            "file": fn.name,
            "dog_max": dog.max(),
            "dog_99.9": np.percentile(dog, 99.9),
            "dog_99.99": np.percentile(dog, 99.99),
            "dog_mean": dog.mean(),
            "dog_std": dog.std(),
        })
    
    all_vals = np.concatenate(all_dog_vals)
    
    print("DoG response on CLEAN GT images (no defects):")
    print(f"  Total pixels analyzed: {len(all_vals):,}")
    print(f"  Mean:   {all_vals.mean():.5f}")
    print(f"  Std:    {all_vals.std():.5f}")
    print(f"  Median: {np.median(all_vals):.5f}")
    print(f"  P95:    {np.percentile(all_vals, 95):.5f}")
    print(f"  P99:    {np.percentile(all_vals, 99):.5f}")
    print(f"  P99.9:  {np.percentile(all_vals, 99.9):.5f}")
    print(f"  P99.99: {np.percentile(all_vals, 99.99):.5f}")
    print(f"  Max:    {all_vals.max():.5f}")
    
    # Test candidate thresholds
    print("\nCandidate thresholds vs expected FP per 256x256 image:")
    for thresh in [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        fp_frac = (all_vals > thresh).mean()
        fp_per_img = fp_frac * 256 * 256  # approximate
        print(f"  threshold={thresh:.2f}  -> {fp_frac*100:.4f}% pixels above -> ~{fp_per_img:.0f} FP pixels/img")

    # Per-image max and 99.99th percentile
    print("\nPer-image 99.99th percentile of DoG (top 5):")
    per_image_stats.sort(key=lambda x: x["dog_99.99"], reverse=True)
    for s in per_image_stats[:5]:
        print(f"  {s['file']}: P99.99={s['dog_99.99']:.4f}, max={s['dog_max']:.4f}")

if __name__ == "__main__":
    main()
