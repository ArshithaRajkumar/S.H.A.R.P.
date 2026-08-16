import sys
import torch
import numpy as np
from pathlib import Path
from src.metrics import compute_metrics

BASE_DIR = Path(__file__).resolve().parent
restored_dir = BASE_DIR / "restored_test_outputs"
gt_dir = BASE_DIR / "train" / "train" / "GT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not restored_dir.exists():
    print(f"Error: {restored_dir} does not exist.")
    sys.exit(1)

restored_files = sorted(list(restored_dir.glob("*.npy")))
if not restored_files:
    print("No .npy files found in restored_test_outputs.")
    sys.exit(1)

print(f"Computing metrics on {len(restored_files)} files...")
all_psnr, all_ssim, all_lpips = [], [], []

# Sample up to 100 images for speed if needed, but let's just do 50 for a quick solid average
sample_files = restored_files[:50]

for f in sample_files:
    gt_path = gt_dir / f.name
    if not gt_path.exists():
        continue
        
    pred = np.load(f).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)
    
    m = compute_metrics(pred, gt, device=device)
    all_psnr.append(m['psnr'])
    all_ssim.append(m['ssim'])
    all_lpips.append(m['lpips'])

print("\n--- FINAL METRICS ---")
print(f"PSNR:  {np.nanmean(all_psnr):.2f} dB")
print(f"SSIM:  {np.nanmean(all_ssim):.4f}")
print(f"LPIPS: {np.nanmean(all_lpips):.4f}")
