import os
import sys
import time
import argparse
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path

# Fix path to import src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.model import NAFNetSR
from src.normalization import robust_normalize, robust_denormalize
from src.noise_estimate import estimate_noise_sigma

def load_model(model_path, device, width, num_blocks, uncertainty=False):
    """Load model with explicit architecture params. Fatal error if checkpoint missing."""
    model = NAFNetSR(width=width, num_blocks=num_blocks, uncertainty=uncertainty)
    
    if not os.path.exists(model_path):
        print(f"FATAL: Model weights not found at {model_path}")
        print(f"Available files in checkpoint dir:")
        ckpt_dir = Path(model_path).parent
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.glob("*.pt")):
                print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
        sys.exit(1)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # If we trained with EMA, load the ema_shadow (these are the better weights)
    if 'ema_shadow' in checkpoint:
        state_dict = checkpoint['ema_shadow']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Use strict=True so architecture mismatches are caught immediately
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {model_path}")
    if 'epoch' in checkpoint:
        print(f"  Trained epoch: {checkpoint['epoch'] + 1}")
    
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()
    return model

def process_batch(model, batch_files, output_dir, device, use_tta=False, return_uncertainty=False):
    # 1. Load float32 and prepare batch
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
    
    # 5. Inference
    with torch.inference_mode():
        if use_tta:
            preds = []
            log_vars = []
            for k in range(4):
                lr_rot = torch.rot90(lr_tensor, k, [2, 3])
                out = model(lr_rot, sigma_tensor, params_tensor)
                if isinstance(out, tuple):
                    pred, log_var = out
                    log_vars.append(torch.rot90(log_var, -k, [2, 3]))
                else:
                    pred = out
                preds.append(torch.rot90(pred, -k, [2, 3]))
                
            pred_hr = torch.stack(preds).mean(0)
            if len(log_vars) > 0:
                log_var_out = torch.stack(log_vars).mean(0)
        else:
            out = model(lr_tensor, sigma_tensor, params_tensor)
            if isinstance(out, tuple):
                pred_hr, log_var_out = out
            else:
                pred_hr = out
                log_var_out = None
                    
    # 6. Save outputs
    pred_hr_np = pred_hr.cpu().numpy()
    if log_var_out is not None:
        unc_np = log_var_out.cpu().numpy()
        
    for i, lr_path in enumerate(batch_files):
        out_path = output_dir / lr_path.name
        np.save(out_path, pred_hr_np[i, 0].astype(np.float32))
        
        if return_uncertainty and log_var_out is not None:
            unc_path = str(out_path).replace(".npy", "_unc.npy")
            np.save(unc_path, unc_np[i, 0].astype(np.float32))

def main():
    parser = argparse.ArgumentParser(description="KLA Image Restoration Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .npy files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file. Defaults to checkpoints/baseline/latest.pt")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--uncertainty", action="store_true", help="Output uncertainty maps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Resolve checkpoint path (default: latest.pt, relative to this script)
    if args.checkpoint is None:
        model_path = Path(BASE_DIR / "checkpoints" / "v2_deeper_multiloss" / "latest.pt")
    else:
        model_path = Path(args.checkpoint)
    
    # Read architecture params from the training config saved alongside the checkpoint
    config_path = model_path.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            train_cfg = yaml.safe_load(f)
        width = train_cfg.get("width", 64)
        num_blocks = train_cfg.get("num_blocks", 4)
        print(f"Read architecture from {config_path}: width={width}, num_blocks={num_blocks}")
    else:
        # Fallback to the defaults that match our training config
        width = 64
        num_blocks = 4
        print(f"No config.yaml found; using default architecture: width={width}, num_blocks={num_blocks}")
    
    model = load_model(model_path, device, width=width, num_blocks=num_blocks,
                       uncertainty=args.uncertainty)
    
    files = sorted(list(input_dir.glob("*.npy")))
    if not files:
        print(f"No .npy files found in {input_dir}")
        return
        
    print(f"Starting batched inference on {len(files)} files (Batch size: {args.batch_size})...")
    
    # Warmup
    if 'cuda' in str(device):
        dummy_tensor = torch.zeros(args.batch_size, 2, 64, 64, device=device).to(memory_format=torch.channels_last)
        dummy_sigma = torch.zeros(args.batch_size, 1, device=device)
        dummy_params = torch.zeros(args.batch_size, 2, device=device)
        with torch.inference_mode():
            _ = model(dummy_tensor, dummy_sigma, dummy_params)
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    for i in range(0, len(files), args.batch_size):
        batch_files = files[i:i+args.batch_size]
        process_batch(model, batch_files, output_dir, device, use_tta=args.tta, return_uncertainty=args.uncertainty)

        
    if 'cuda' in str(device):
        torch.cuda.synchronize()
        
    total_time = time.perf_counter() - start_time
    mean_time_ms = (total_time / len(files)) * 1000
    
    print(f"Inference complete.")
    print(f"Total time: {total_time:.4f} s")
    print(f"Mean time per image: {mean_time_ms:.2f} ms")
    if 'cuda' in str(device):
        print(f"Hardware: {torch.cuda.get_device_name(device)}")

if __name__ == "__main__":
    main()
