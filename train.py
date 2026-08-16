import os
import yaml
import argparse
import time
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from pathlib import Path
import math

from src.dataset import KLAImageDataset
from src.model import NAFNetSR
from src.losses import CompositeLoss
from src.noise_estimate import estimate_noise_sigma

@dataclass
class TrainConfig:
    # Model
    width: int = 64
    num_blocks: int = 4
    uncertainty: bool = False
    
    # Training
    batch_size: int = 16
    epochs: int = 200
    lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    
    # Dataset
    crop_size: int = 64
    synthetic_ratio: float = 0.5
    use_cutblur: bool = True
    augment: bool = True
    
    # Loss weights
    use_nll: bool = False
    use_msssim: bool = False
    use_fft: bool = False
    use_lpips: bool = False
    use_gradient: bool = False
    use_forward: bool = False
    w_charb: float = 1.0
    
    # Loss weights (used by CompositeLoss via config.get())
    w_msssim: float = 0.1
    w_fft: float = 0.1
    w_lpips: float = 0.1
    w_grad: float = 0.1
    w_forward: float = 0.1
    w_nll: float = 1.0

    # Paths
    output_dir: str = "checkpoints/baseline"

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def load_shadow(self, shadow_dict):
        for name, param in shadow_dict.items():
            if name in self.shadow:
                self.shadow[name] = param.clone()

def evaluate(model, dataloader, device, ema=None):
    if ema:
        ema.apply_shadow(model)
    model.eval()
    
    mse_sum = 0.0
    bicubic_mse_sum = 0.0
    count = 0
    
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            lr = batch["lr"].to(device, memory_format=torch.channels_last)
            hr = batch["hr"].to(device)
            norm_params = batch["norm_params"].to(device)
            
            B = lr.shape[0]
            sigma = torch.full((B, 1), 0.15, device=device)
            
            outputs = model(lr, sigma, norm_params)
            
            if isinstance(outputs, tuple):
                pred_hr = outputs[0]
            else:
                pred_hr = outputs
                
            pred_hr = pred_hr.float()
            
            mse_sum += F.mse_loss(pred_hr, hr, reduction='sum').item()
            
            # Bicubic baseline
            low_val = norm_params[:, 0].view(-1, 1, 1, 1)
            scale = norm_params[:, 1].view(-1, 1, 1, 1)
            lr_unnorm = lr[:, 0:1] * scale + low_val
            
            bicubic = F.interpolate(lr_unnorm, scale_factor=2, mode='bicubic', align_corners=False)
            bicubic = torch.clamp(bicubic, 0.0, 1.0)
            bicubic_mse_sum += F.mse_loss(bicubic, hr, reduction='sum').item()
            
            count += B * hr.shape[2] * hr.shape[3]
            
    rmse = math.sqrt(mse_sum / count) if count > 0 else 0
    bicubic_rmse = math.sqrt(bicubic_mse_sum / count) if count > 0 else 0
    
    model.train()
    return rmse, bicubic_rmse

def train(config: TrainConfig, resume=False):
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(config), f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets
    train_ds = KLAImageDataset(split="train", 
                               crop_size=config.crop_size,
                               synthetic_ratio=config.synthetic_ratio,
                               use_cutblur=config.use_cutblur,
                               augment=config.augment)
                               
    val_in_ds = KLAImageDataset(split="val_in_dist", crop_size=128)
    val_ood_ds = KLAImageDataset(split="val_ood", crop_size=128)
    
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_in_dl = DataLoader(val_in_ds, batch_size=1, shuffle=False)
    val_ood_dl = DataLoader(val_ood_ds, batch_size=1, shuffle=False)
    
    model = NAFNetSR(width=config.width, num_blocks=config.num_blocks, uncertainty=config.uncertainty)
    model = model.to(device, memory_format=torch.channels_last)
    
    ema = EMA(model, config.ema_decay)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_dl), eta_min=config.min_lr)
    
    loss_fn = CompositeLoss(asdict(config), device=device)
    # Mixed-precision disabled: bfloat16 autocast causes cuDNN errors on T4/V100 GPUs.
    # Model is small enough that float32 training has negligible throughput cost.
    
    start_epoch = 0
    log_file = out_dir / "training_log.csv"
    
    # Setup CSV Logging
    log_exists = log_file.exists() and resume
    csv_file = open(log_file, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not log_exists:
        csv_writer.writerow(["Epoch", "Time_s", "Train_Loss", "Val_InDist_RMSE", "Val_OOD_RMSE", "Bicubic_InDist", "Bicubic_OOD"])
        csv_file.flush()
    
    # Resume Logic
    if resume:
        latest_ckpt = out_dir / "latest.pt"
        if latest_ckpt.exists():
            print(f"Resuming from {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            ema.load_shadow(checkpoint['ema_shadow'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # scaler loading removed: full float32 training
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed at epoch {start_epoch + 1}")
        else:
            print("No checkpoint found at latest.pt. Starting from scratch.")
    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for i, batch in enumerate(train_dl):
            lr = batch["lr"].to(device, memory_format=torch.channels_last)
            hr = batch["hr"].to(device)
            norm_params = batch["norm_params"].to(device)
            
            B = lr.shape[0]
            sigma = torch.full((B, 1), 0.15, device=device)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(lr, sigma, norm_params)
            if config.uncertainty:
                pred_hr, log_var = outputs
            else:
                pred_hr, log_var = outputs, None
                
            loss, loss_dict = loss_fn(pred_hr, hr, log_var=log_var)
            
            loss.backward()
            optimizer.step()
            
            ema.update(model)
            scheduler.step()
            
            epoch_loss += loss.item()
            
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(train_dl)
        
        # Validation every epoch
        in_rmse, bic_in = evaluate(model, val_in_dl, device, ema)
        ood_rmse, bic_ood = evaluate(model, val_ood_dl, device, ema)
        
        print(f"Epoch {epoch+1}/{config.epochs} [{epoch_time:.1f}s], Loss: {avg_loss:.6f}")
        print(f"  Val In-Dist RMSE: {in_rmse:.4f} (Bicubic: {bic_in:.4f})")
        print(f"  Val OOD RMSE: {ood_rmse:.4f} (Bicubic: {bic_ood:.4f})")
        
        # Log to CSV
        csv_writer.writerow([epoch+1, f"{epoch_time:.1f}", f"{avg_loss:.6f}", f"{in_rmse:.4f}", f"{ood_rmse:.4f}", f"{bic_in:.4f}", f"{bic_ood:.4f}"])
        csv_file.flush()
        
        # Checkpointing
        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_shadow': ema.shadow,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # scaler removed: full float32 training
        }
        
        # Save epoch checkpoint and latest
        torch.save(ckpt_state, out_dir / f"checkpoint_{epoch+1}.pt")
        torch.save(ckpt_state, out_dir / "latest.pt")
        
    csv_file.close()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--width", type=int, default=None, help="Override model width")
    parser.add_argument("--num_blocks", type=int, default=None, help="Override num blocks")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()
    
    # Start with defaults
    cfg = TrainConfig()
    
    # Load YAML config if provided
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        # Get field types from the dataclass for proper casting
        field_types = {f.name: f.type for f in cfg.__dataclass_fields__.values()}
        for k, v in yaml_cfg.items():
            if hasattr(cfg, k):
                # Cast to the declared type (handles YAML parsing 2e-4 as string)
                expected_type = field_types.get(k)
                if expected_type == float:
                    v = float(v)
                elif expected_type == int:
                    v = int(v)
                elif expected_type == bool:
                    v = bool(v)
                setattr(cfg, k, v)
            else:
                print(f"Warning: unknown config key '{k}', ignoring")
    
    # CLI overrides take precedence
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.width is not None:
        cfg.width = args.width
    if args.num_blocks is not None:
        cfg.num_blocks = args.num_blocks
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.lr is not None:
        cfg.lr = args.lr
    
    print(f"Training config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    
    train(cfg, resume=args.resume)
