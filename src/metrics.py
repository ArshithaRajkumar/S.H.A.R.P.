import numpy as np
import torch

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    psnr, ssim = None, None

try:
    import lpips as lpips_lib
    _lpips_model = None
except ImportError:
    lpips_lib = None
    _lpips_model = None

def get_lpips_model(device='cpu'):
    global _lpips_model
    if _lpips_model is None and lpips_lib is not None:
        _lpips_model = lpips_lib.LPIPS(net='vgg').to(device)
        for p in _lpips_model.parameters():
            p.requires_grad = False
    return _lpips_model

def compute_metrics(pred, gt, device='cpu'):
    """
    Computes PSNR, SSIM, and LPIPS on float arrays in the GT range [0, 1].
    Args:
        pred: numpy array (float32) [0, 1]
        gt: numpy array (float32) [0, 1]
        device: device for LPIPS model
    Returns:
        dict with psnr, ssim, lpips
    """
    metrics = {}
    
    if psnr is not None:
        metrics['psnr'] = float(psnr(gt, pred, data_range=1.0))
        metrics['ssim'] = float(ssim(gt, pred, data_range=1.0))
    else:
        metrics['psnr'] = float('nan')
        metrics['ssim'] = float('nan')
        
    model_lpips = get_lpips_model(device)
    if model_lpips is not None:
        # Convert to [-1, 1]
        p_t = torch.from_numpy(pred).float() * 2.0 - 1.0
        g_t = torch.from_numpy(gt).float() * 2.0 - 1.0
        
        # Add batch and channel dims, replicate to 3 channels for VGG
        p_t = p_t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        g_t = g_t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        
        with torch.inference_mode():
            val = model_lpips(p_t, g_t).item()
        metrics['lpips'] = float(val)
    else:
        metrics['lpips'] = float('nan')
        
    return metrics
