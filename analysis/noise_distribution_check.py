import numpy as np
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

GT_DIR  = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\GT")
LR_DIR  = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\NoisyLR")
OUT_DIR = Path(__file__).resolve().parent

def ds_bicubic(img):
    return zoom(img, 0.5, order=3, prefilter=True)

def main():
    gt_names = sorted(f.name for f in GT_DIR.glob("*.npy"))
    np.random.seed(42)
    # Sample 50 random files for the distribution check
    sample = np.random.choice(gt_names, 50, replace=False)
    
    all_noise = []
    
    for fn in sample:
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
            
    n_arr = np.concatenate(all_noise)
    
    mean = n_arr.mean()
    std = n_arr.std()
    kurt = kurtosis(n_arr, fisher=True) # Fisher's kurtosis: normal == 0
    
    print(f"Noise mean: {mean:.6f}")
    print(f"Noise std:  {std:.6f}")
    print(f"Excess Kurtosis: {kurt:.2f} (0 = Gaussian, >0 = heavy tailed)")
    
    # Save the empirical distribution as a 1D array for sampling later
    # To save space, we can sort it and save a subset of quantiles or a subsample
    n_sorted = np.sort(n_arr)
    subsample = n_sorted[::10] # take 10% of the data to keep file size reasonable
    np.save(OUT_DIR / "empirical_noise.npy", subsample)
    print(f"Saved empirical noise distribution to {OUT_DIR / 'empirical_noise.npy'} ({len(subsample)} samples)")
    
    # Plot histogram vs Gaussian
    plt.figure(figsize=(10, 6))
    plt.hist(n_arr, bins=100, density=True, alpha=0.6, color='b', label='Empirical Noise', range=(-1, 1))
    
    x = np.linspace(-1, 1, 1000)
    g = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    plt.plot(x, g, 'r', linewidth=2, label=f'Gaussian (σ={std:.3f})')
    
    plt.yscale('log') # Log scale to see tails clearly
    plt.title(f'Noise Distribution (Excess Kurtosis: {kurt:.2f})')
    plt.xlabel('Noise Value')
    plt.ylabel('Density (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_DIR / "noise_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to {OUT_DIR / 'noise_distribution.png'}")

if __name__ == "__main__":
    main()
