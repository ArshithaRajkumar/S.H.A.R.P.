import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def show_random_comparison():
    base_dir = Path(__file__).resolve().parent
    noisy_dir = base_dir / "train" / "train" / "NoisyLR"
    gt_dir = base_dir / "train" / "train" / "GT"
    restored_dir = base_dir / "restored_test_outputs"
    
    files = list(restored_dir.glob("*.npy"))
    if not files:
        print("No restored files found!")
        return
        
    f = random.choice(files)
    print(f"Visualizing {f.name}...")
    
    noisy = np.load(noisy_dir / f.name).astype(np.float32)
    gt = np.load(gt_dir / f.name).astype(np.float32)
    restored = np.load(f).astype(np.float32)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title("Degraded Input (Noisy, Low-Res)")
    axes[0].axis('off')
    
    axes[1].imshow(restored, cmap='gray')
    axes[1].set_title("SHARP Restored Output")
    axes[1].axis('off')
    
    axes[2].imshow(gt, cmap='gray')
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_random_comparison()
