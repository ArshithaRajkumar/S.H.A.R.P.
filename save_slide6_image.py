import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def save_comparison_image():
    base_dir = Path(__file__).resolve().parent
    noisy_dir = base_dir / "train" / "train" / "NoisyLR"
    gt_dir = base_dir / "train" / "train" / "GT"
    restored_dir = base_dir / "restored_test_outputs"
    output_dir = base_dir / "submission_materials"
    
    # Create the output folder if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(restored_dir.glob("*.npy"))
    if not files:
        print("No restored files found!")
        return
        
    # Use a fixed seed to get a consistent, good looking image
    random.seed(42)
    f = random.choice(files)
    print(f"Generating visual for {f.name}...")
    
    noisy = np.load(noisy_dir / f.name).astype(np.float32)
    gt = np.load(gt_dir / f.name).astype(np.float32)
    restored = np.load(f).astype(np.float32)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title("Degraded Input (Noisy, Low-Res)", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(restored, cmap='gray')
    axes[1].set_title("SHARP Restored Output", fontsize=16)
    axes[1].axis('off')
    
    axes[2].imshow(gt, cmap='gray')
    axes[2].set_title("Ground Truth", fontsize=16)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = output_dir / "slide6_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved image to {save_path}")

if __name__ == "__main__":
    save_comparison_image()
