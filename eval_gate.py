import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.model import NAFNetSR
from src.dataset import KLAImageDataset
from src.metrics import compute_metrics

def eval_checkpoint(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NAFNetSR(width=64, num_blocks=4, uncertainty=False)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['ema_shadow'] if 'ema_shadow' in ckpt else ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    for split_name in ["val_in_dist", "val_ood"]:
        ds = KLAImageDataset(split=split_name, crop_size=128) # Val images are 256x256 GT, 128x128 LR
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        
        psnr_pred, ssim_pred = [], []
        psnr_bic, ssim_bic = [], []
        
        print(f"\nEvaluating {split_name} (N={len(ds)})...")
        with torch.inference_mode():
            for i, batch in enumerate(dl):
                if i >= 10:
                    break
                lr = batch["lr"].to(device)
                hr = batch["hr"].to(device)
                norm_params = batch["norm_params"].to(device)
                
                # Sigma is needed. Use 0.173 as a placeholder based on our findings
                sigma = torch.full((1, 1), 0.173, device=device)
                
                with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=torch.bfloat16):
                    out = model(lr, sigma, norm_params)
                    pred_hr = out[0] if isinstance(out, tuple) else out
                
                # Reconstruct unnormalized LR for bicubic baseline
                low_val = norm_params[:, 0].view(1, 1, 1, 1)
                scale = norm_params[:, 1].view(1, 1, 1, 1)
                lr_unnorm = lr[:, 0:1] * scale + low_val
                bic_hr = torch.nn.functional.interpolate(lr_unnorm, scale_factor=2, mode='bicubic', align_corners=False)
                bic_hr = torch.clamp(bic_hr, 0.0, 1.0)
                
                pred_np = pred_hr.squeeze().cpu().numpy()
                hr_np = hr.squeeze().cpu().numpy()
                bic_np = bic_hr.squeeze().cpu().numpy()
                
                m_pred = compute_metrics(pred_np, hr_np, device='cpu')
                m_bic = compute_metrics(bic_np, hr_np, device='cpu')
                
                psnr_pred.append(m_pred['psnr'])
                ssim_pred.append(m_pred['ssim'])
                psnr_bic.append(m_bic['psnr'])
                ssim_bic.append(m_bic['ssim'])
                
                if (i+1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(ds)}")
                    
        print(f"  Model   - PSNR: {np.mean(psnr_pred):.2f} dB, SSIM: {np.mean(ssim_pred):.4f}")
        print(f"  Bicubic - PSNR: {np.mean(psnr_bic):.2f} dB, SSIM: {np.mean(ssim_bic):.4f}")

if __name__ == "__main__":
    ckpt = Path("checkpoints/baseline/checkpoint_2.pt")
    if ckpt.exists():
        eval_checkpoint(ckpt)
    else:
        print(f"Checkpoint not found: {ckpt}")
