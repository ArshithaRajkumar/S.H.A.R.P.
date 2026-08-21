import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps_sq = eps ** 2
        
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps_sq))

class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred.float())
        target_fft = torch.fft.rfft2(target.float())
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        
    def forward(self, pred, target):
        pred_gx = F.conv2d(pred, self.kernel_x, padding=1)
        pred_gy = F.conv2d(pred, self.kernel_y, padding=1)
        
        target_gx = F.conv2d(target, self.kernel_x, padding=1)
        target_gy = F.conv2d(target, self.kernel_y, padding=1)
        
        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)

class ForwardConsistencyLoss(nn.Module):
    def __init__(self, blur_sigma=0.3):
        super().__init__()
        # Measured operator A: blur + bicubic downsample
        # We approximate blur with a simple gaussian kernel if needed, or just rely on downsample if blur is small.
        # Let's use a 5x5 average pool for "local mean filtered version of input" as suggested.
        self.local_mean = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        
    def forward(self, pred_hr, lr_input):
        # A*y on prediction (Downsample prediction)
        # Using bicubic downsample to match phase 1 finding
        A_pred = F.interpolate(pred_hr, scale_factor=0.5, mode='bicubic', align_corners=False)
        
        # Local mean of x
        # lr_input has 2 channels (norm, log), we take the first channel
        lr_img = lr_input[:, 0:1, :, :]
        
        # We need to un-normalize it to match the pred_hr range (which is [0, 1])
        # Wait, the loss should probably be computed on the normalized space or both in [0, 1].
        # It's easier if the caller passes un-normalized LR or we assume lr_input is un-normalized here?
        # Actually, let's just assume the caller passes the GT-range LR image as lr_target.
        
        pass # Will refine in the combined loss module

class GaussianNLLLoss(nn.Module):
    def __init__(self, min_log_var=-10.0, max_log_var=10.0):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.charbonnier = CharbonnierLoss()
        
    def forward(self, pred, target, log_var, step=None, warmup_steps=10000):
        # Clamp log_var
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        
        if step is not None and step < warmup_steps:
            return self.charbonnier(pred, target)
            
        var = torch.exp(log_var)
        nll = 0.5 * ((pred - target) ** 2) / var + 0.5 * log_var
        return torch.mean(nll)

class CompositeLoss(nn.Module):
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.config = config
        
        self.charb = CharbonnierLoss()
        if config.get("use_msssim", False):
            from pytorch_msssim import ms_ssim
            self.ms_ssim = ms_ssim
        if config.get("use_fft", False):
            self.fft = FFTLoss()
        if config.get("use_lpips", False):
            import lpips
            self.lpips = lpips.LPIPS(net='vgg').to(device)
            # Freeze LPIPS
            for p in self.lpips.parameters():
                p.requires_grad = False
        if config.get("use_gradient", False):
            self.gradient = GradientLoss().to(device)
        if config.get("use_forward", False):
            self.forward_consistency = ForwardConsistencyLoss().to(device)
            self.local_mean_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        if config.get("use_nll", False):
            self.nll = GaussianNLLLoss().to(device)
            
        self.step = 0
            
    def forward(self, pred_hr, target_hr, lr_unnorm=None, log_var=None):
        loss_dict = {}
        total_loss = 0.0
        
        # Primary reconstruction (Charbonnier)
        if self.config.get("use_nll", False) and log_var is not None:
            nll_loss = self.nll(pred_hr, target_hr, log_var, self.step, self.config.get("nll_warmup", 10000))
            loss_dict['nll'] = nll_loss
            total_loss += self.config.get("w_nll", 1.0) * nll_loss
        else:
            c_loss = self.charb(pred_hr, target_hr)
            loss_dict['charb'] = c_loss
            total_loss += self.config.get("w_charb", 1.0) * c_loss
            
        if self.config.get("use_msssim", False):
            # ms_ssim returns a value in [0, 1], higher is better
            msssim_val = self.ms_ssim(pred_hr, target_hr, data_range=1.0, size_average=True, win_size=7)
            msssim_loss = 1.0 - msssim_val
            loss_dict['msssim'] = msssim_loss
            total_loss += self.config.get("w_msssim", 0.1) * msssim_loss
            
        if self.config.get("use_fft", False):
            f_loss = self.fft(pred_hr, target_hr)
            loss_dict['fft'] = f_loss
            total_loss += self.config.get("w_fft", 0.1) * f_loss
            
        if self.config.get("use_lpips", False):
            # Scale from [0, 1] to [-1, 1]
            p_hr_lpips = pred_hr * 2.0 - 1.0
            t_hr_lpips = target_hr * 2.0 - 1.0
            
            # Replicate to 3 channels
            if p_hr_lpips.shape[1] == 1:
                p_hr_lpips = p_hr_lpips.repeat(1, 3, 1, 1)
                t_hr_lpips = t_hr_lpips.repeat(1, 3, 1, 1)
                
            lpips_loss = self.lpips(p_hr_lpips, t_hr_lpips).mean()
            loss_dict['lpips'] = lpips_loss
            total_loss += self.config.get("w_lpips", 0.1) * lpips_loss
            
        if self.config.get("use_gradient", False):
            g_loss = self.gradient(pred_hr, target_hr)
            loss_dict['grad'] = g_loss
            total_loss += self.config.get("w_grad", 0.1) * g_loss
            
        if self.config.get("use_forward", False) and lr_unnorm is not None:
            # A_pred: bicubic downsample
            A_pred = F.interpolate(pred_hr, scale_factor=0.5, mode='bicubic', align_corners=False)
            
            # local mean of LR input
            lr_mean = self.local_mean_pool(lr_unnorm)
            
            fwd_loss = F.l1_loss(A_pred, lr_mean)
            loss_dict['forward'] = fwd_loss
            total_loss += self.config.get("w_forward", 0.1) * fwd_loss
            
        self.step += 1
        return total_loss, loss_dict
