import numpy as np
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter
from scipy.stats import kurtosis
import time

GT_DIR  = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\GT")
LR_DIR  = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\NoisyLR")

def ds_bicubic(img):
    return zoom(img, 0.5, order=3, prefilter=True)

def main():
    gt_names = sorted(f.name for f in GT_DIR.glob("*.npy"))
    
    all_noise = []
    
    t0 = time.time()
    for i, fn in enumerate(gt_names):
        gt = np.load(GT_DIR / fn)
        lr = np.load(LR_DIR / fn)
        
        gt_b = gaussian_filter(gt, 0.3)
        ds = ds_bicubic(gt_b)
        
        mask = ds > 0.02
        if mask.sum() > 100:
            noise = (lr[mask] / ds[mask]) - 1.0
            # Clip extreme outliers for stability of standard deviation measurement,
            # but leave enough tail to check kurtosis.
            noise = np.clip(noise, -5, 5)
            all_noise.append(noise)
            
        if (i + 1) % 400 == 0:
            print(f"Processed {i+1}/{len(gt_names)} files in {time.time()-t0:.1f}s")
            
    n_arr = np.concatenate(all_noise)
    
    mean = n_arr.mean()
    std = n_arr.std()
    kurt = kurtosis(n_arr, fisher=True) # Fisher's kurtosis: normal == 0
    
    print("\n--- Full Dataset Results ---")
    print(f"Total valid pixels: {len(n_arr):,}")
    print(f"Noise mean: {mean:.6f}")
    print(f"Noise std:  {std:.6f}")
    print(f"Excess Kurtosis (Fisher): {kurt:.4f}")

if __name__ == "__main__":
    main()
