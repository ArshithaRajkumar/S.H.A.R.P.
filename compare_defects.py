import sys
import torch
import numpy as np
import warnings
from pathlib import Path

# Filter out specific warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.defect_eval import run_defect_eval, bicubic_restore
from evaluate import load_model
from src.normalization import robust_normalize
from src.noise_estimate import estimate_noise_sigma

gt_dir = BASE_DIR / "train" / "train" / "GT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_restore_fn(model):
    def restore(degraded_lr):
        lr_np = degraded_lr.astype(np.float32)
        sigma_hat = estimate_noise_sigma(lr_np)
        lr_norm, params = robust_normalize(lr_np)
        
        lr_ch2 = np.log(np.maximum(lr_norm + 1e-3, 1e-6))
        lr_ch2 = np.maximum(lr_ch2, 0.0)
        
        batch_tensors = [np.stack([lr_norm, lr_ch2], axis=0)]
        batch_sigmas = [[sigma_hat]]
        batch_params = [params]
        
        lr_tensor = torch.from_numpy(np.array(batch_tensors)).float().to(device, memory_format=torch.channels_last)
        sigma_tensor = torch.tensor(batch_sigmas, dtype=torch.float32, device=device)
        params_tensor = torch.tensor(batch_params, dtype=torch.float32, device=device)
        
        with torch.inference_mode():
            pred = model(lr_tensor, sigma_tensor, params_tensor)
            if isinstance(pred, tuple):
                pred = pred[0]
        return pred.cpu().numpy()[0, 0].astype(np.float32)
    return restore

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Baseline Model (50 epochs, Charb only)")
    print("="*50)
    baseline_path = BASE_DIR / "checkpoints" / "baseline" / "latest.pt"
    model_baseline = load_model(baseline_path, device, width=64, num_blocks=4)
    run_defect_eval(make_restore_fn(model_baseline), gt_dir, num_samples=20)
    
    print("\n" + "="*50)
    print("Testing V2 Model (50 epochs, Multi-loss, deeper)")
    print("="*50)
    v2_path = BASE_DIR / "checkpoints" / "v2_deeper_multiloss" / "latest.pt"
    model_v2 = load_model(v2_path, device, width=64, num_blocks=12)
    run_defect_eval(make_restore_fn(model_v2), gt_dir, num_samples=20)
