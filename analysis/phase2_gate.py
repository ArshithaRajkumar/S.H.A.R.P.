import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.degradation import generate_synthetic_pair

GT_DIR = BASE_DIR / "train" / "train" / "GT"
LR_DIR = BASE_DIR / "train" / "train" / "NoisyLR"
OUT_DIR = BASE_DIR / "analysis"

def main():
    np.random.seed(42)
    
    # Pick 5 random files
    gt_names = sorted(f.name for f in GT_DIR.glob("*.npy"))
    sample_files = np.random.choice(gt_names, 5, replace=False)
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle("Phase 2 Gate: Real vs Synthetic Degradation", fontsize=16)
    
    for i, fn in enumerate(sample_files):
        gt = np.load(GT_DIR / fn)
        real_lr = np.load(LR_DIR / fn)
        
        # Generate a synthetic pair
        # Force a specific noise sigma from the mid-range so it's comparable
        syn_lr, syn_gt = generate_synthetic_pair(gt, 
                                                 blur_range=(0.3, 0.3), 
                                                 noise_sigma_range=(0.17, 0.17),
                                                 add_gaussian_std=0.0,
                                                 gamma_range=(1.0, 1.0))
        
        # Crop a 64x64 patch from the center of LR to see noise detail
        h, w = real_lr.shape
        cy, cx = h//2, w//2
        s = 32
        
        real_lr_crop = real_lr[cy-s:cy+s, cx-s:cx+s]
        syn_lr_crop = syn_lr[cy-s:cy+s, cx-s:cx+s]
        gt_crop = gt[(cy-s)*2:(cy+s)*2, (cx-s)*2:(cx+s)*2]
        
        axes[i, 0].imshow(gt_crop, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f"GT (Crop) - {fn}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(real_lr_crop, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Real LR")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(syn_lr_crop, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Synthetic LR")
        axes[i, 2].axis('off')
        
        # Residual difference between real and synthetic isn't meaningful pixel-wise
        # due to random noise, but we can compare the local standard deviation (rough visual noise proxy)
        diff_real = real_lr_crop - np.mean(real_lr_crop)
        diff_syn = syn_lr_crop - np.mean(syn_lr_crop)
        
        axes[i, 3].imshow(np.abs(diff_real - diff_syn), cmap='magma', vmin=0, vmax=0.5)
        axes[i, 3].set_title(f"Noise structural diff")
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    out_file = OUT_DIR / "phase2_gate_comparison.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Phase 2 Gate figure saved to {out_file}")

if __name__ == "__main__":
    main()
