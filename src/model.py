import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * self.conv(self.gap(x))

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c) 
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dwconv = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(c)
        self.conv2 = nn.Conv2d(c, c, 1)
        
    def forward(self, x, gamma, beta):
        res = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv2(x)
        
        # FiLM: gamma and beta have shape (B, C, 1, 1)
        x = x * gamma + beta
        return res + x

class NAFNetSR(nn.Module):
    def __init__(self, in_channels=2, width=256, num_blocks=16, uncertainty=True):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.uncertainty = uncertainty
        
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        
        self.blocks = nn.ModuleList([NAFBlock(width) for _ in range(num_blocks)])
        
        # Output head: PixelShuffle x2
        # For x2 upscale, we need 4 * out_channels
        self.out_channels = 2 if uncertainty else 1
        self.up_conv = nn.Conv2d(width, self.out_channels * 4, 3, padding=1)
        # Zero-initialize the final projection so the initial residual is exactly 0
        nn.init.zeros_(self.up_conv.weight)
        if self.up_conv.bias is not None:
            nn.init.zeros_(self.up_conv.bias)
            
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # FiLM generator (reduced hidden width to save params)
        film_hidden = 64
        self.film_mlp = nn.Sequential(
            nn.Linear(1, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, num_blocks * width * 2)
        )
        
    def forward(self, x, sigma, norm_params=None):
        """
        x: (B, 2, H, W) normalized input and log(norm + 1e-3)
        sigma: (B, 1) scalar noise estimate
        norm_params: (B, 2) tensor of (low_val, scale) used for inverse normalization
        """
        B = x.shape[0]
        sigma = sigma.view(B, 1)
        
        # Generate FiLM params
        film_params = self.film_mlp(sigma) # (B, num_blocks * width * 2)
        film_params = film_params.view(B, self.num_blocks, 2, self.width, 1, 1)
        
        feat = self.intro(x)
        
        for i, block in enumerate(self.blocks):
            gamma = film_params[:, i, 0] + 1.0
            beta = film_params[:, i, 1]
            feat = block(feat, gamma, beta)
            
        out = self.pixel_shuffle(self.up_conv(feat))
        
        pred_res = out[:, 0:1, :, :]
        
        # Global residual: add a bicubic x2 upsample of the input (channel 0)
        base_img = x[:, 0:1, :, :]
        base_up = F.interpolate(base_img, scale_factor=2, mode='bicubic', align_corners=False)
        
        pred_hr_norm = base_up + pred_res
        
        # Final: inverse-normalize, then clip to GT range [0, 1]
        if norm_params is not None:
            low_val = norm_params[:, 0].view(-1, 1, 1, 1)
            scale = norm_params[:, 1].view(-1, 1, 1, 1)
            pred_hr = pred_hr_norm * scale + low_val
            pred_hr = torch.clamp(pred_hr, 0.0, 1.0)
        else:
            pred_hr = torch.clamp(pred_hr_norm, 0.0, 1.0)
            
        if self.uncertainty:
            log_var = out[:, 1:2, :, :]
            return pred_hr, log_var
        else:
            return pred_hr

if __name__ == "__main__":
    # Test model
    model = NAFNetSR(width=256, num_blocks=16, uncertainty=True)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f} M")
    
    B, C, H, W = 4, 2, 64, 64
    x = torch.randn(B, C, H, W)
    sigma = torch.rand(B, 1)
    norm_params = torch.rand(B, 2)
    
    pred_hr, log_var = model(x, sigma, norm_params)
    print(f"pred_hr shape: {pred_hr.shape}")
    print(f"log_var shape: {log_var.shape}")
