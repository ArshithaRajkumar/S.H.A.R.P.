import os
import sys
import numpy as np
import torch
from pathlib import Path

# Fix path to import src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.model import NAFNetSR
from src.normalization import robust_normalize
from src.noise_estimate import estimate_noise_sigma

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
        
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model architecture parameters for SHARP V2 (Width=64, Blocks=12)
    width = 64
    num_blocks = 12
    
    # Load model from the required models/ directory
    model_path = BASE_DIR / "models" / "model.pt"
    if not model_path.exists():
        print(f"FATAL: Model weights not found at {model_path}")
        sys.exit(1)
        
    model = NAFNetSR(width=width, num_blocks=num_blocks, uncertainty=False)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    state_dict = checkpoint.get('ema_shadow', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()
    
    files = sorted(list(input_dir.glob("*.npy")))
    if not files:
        print(f"No .npy files found in {input_dir}")
        return
        
    batch_size = 16
    print(f"Starting inference on {len(files)} files...")
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        
        batch_tensors = []
        batch_sigmas = []
        batch_params = []
        
        for lr_path in batch_files:
            lr_np = np.load(lr_path).astype(np.float32)
            sigma_hat = estimate_noise_sigma(lr_np)
            lr_norm, params = robust_normalize(lr_np)
            
            lr_ch2 = np.log(np.maximum(lr_norm + 1e-3, 1e-6))
            lr_ch2 = np.maximum(lr_ch2, 0.0)
            
            batch_tensors.append(np.stack([lr_norm, lr_ch2], axis=0))
            batch_sigmas.append([sigma_hat])
            batch_params.append(params)
            
        lr_tensor = torch.from_numpy(np.array(batch_tensors)).float().to(device, memory_format=torch.channels_last)
        sigma_tensor = torch.tensor(batch_sigmas, dtype=torch.float32, device=device)
        params_tensor = torch.tensor(batch_params, dtype=torch.float32, device=device)
        
        with torch.inference_mode():
            out = model(lr_tensor, sigma_tensor, params_tensor)
            if isinstance(out, tuple):
                pred_hr = out[0]
            else:
                pred_hr = out
                
        pred_hr_np = pred_hr.cpu().numpy()
        
        for j, lr_path in enumerate(batch_files):
            out_path = output_dir / lr_path.name
            img = pred_hr_np[j, 0].astype(np.float32)
            
            # CRITICAL: Clean NaN/Inf and clip strictly to [0, 1] as required by the email
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            img = np.clip(img, 0.0, 1.0)
            
            np.save(out_path, img)

if __name__ == "__main__":
    main()
