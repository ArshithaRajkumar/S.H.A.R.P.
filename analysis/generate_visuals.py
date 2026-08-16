import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Matplotlib styling for premium look
plt.style.use('dark_background')

def generate_triptych(lr_path, gt_path, restored_path, out_path):
    """
    Generate Before/After/GT visual triptych for a given sample.
    """
    lr = np.load(lr_path)
    gt = np.load(gt_path)
    restored = np.load(restored_path)
    
    # Bicubic upsample LR for fair visual comparison
    from scipy.ndimage import zoom
    lr_up = zoom(lr, 2.0, order=3, prefilter=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Common vmin/vmax based on GT
    vmin, vmax = 0.0, 1.0
    
    axes[0].imshow(lr_up, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Degraded (Bicubic)')
    axes[0].axis('off')
    
    axes[1].imshow(restored, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Restored (Ours)')
    axes[1].axis('off')
    
    axes[2].imshow(gt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()

def generate_uncertainty_panel(restored_path, gt_path, unc_path, out_path):
    """
    Generate three-panel figure: restored output, predicted uncertainty map, absolute error against GT.
    Computes and reports the Spearman correlation between uncertainty and absolute error.
    """
    restored = np.load(restored_path)
    gt = np.load(gt_path)
    # The uncertainty head predicts log-variance, so variance is exp(log_var)
    # Standard deviation (uncertainty) is sqrt(variance) = exp(0.5 * log_var)
    log_var = np.load(unc_path)
    uncertainty = np.exp(0.5 * log_var)
    
    abs_error = np.abs(restored - gt)
    
    # Compute Spearman correlation
    rho, p_val = spearmanr(uncertainty.flatten(), abs_error.flatten())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].imshow(restored, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Restored Output')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(uncertainty, cmap='inferno')
    axes[1].set_title('Predicted Uncertainty (σ)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(abs_error, cmap='inferno')
    axes[2].set_title(f'Absolute Error\nSpearman ρ: {rho:.3f}')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    
    return rho

if __name__ == "__main__":
    print("Visual generator script ready.")
    print("Once evaluation is complete, call generate_triptych() and generate_uncertainty_panel() on selected files.")
