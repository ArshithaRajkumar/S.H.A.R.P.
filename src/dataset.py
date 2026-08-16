import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Adjust paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
GT_DIR = BASE_DIR / "train" / "train" / "GT"
LR_DIR = BASE_DIR / "train" / "train" / "NoisyLR"
SPLITS_FILE = BASE_DIR / "analysis" / "dataset_splits.json"

from .normalization import robust_normalize
from .degradation import generate_synthetic_pair

class KLAImageDataset(Dataset):
    def __init__(self, 
                 split="train", 
                 crop_size=64, # LR crop size
                 synthetic_ratio=0.5, 
                 use_cutblur=True,
                 augment=True):
        """
        Args:
            split: "train", "val_in_dist", or "val_ood"
            crop_size: Size of the random crop in the LR domain. HR crop will be 2x.
            synthetic_ratio: Probability [0, 1] of generating a synthetic pair on the fly.
            use_cutblur: Whether to apply CutBlur data augmentation.
            augment: Whether to apply D4 geometric augmentations.
        """
        self.split = split
        self.crop_size = crop_size
        self.hr_crop_size = crop_size * 2
        self.synthetic_ratio = synthetic_ratio
        self.use_cutblur = use_cutblur
        self.augment = augment
        
        with open(SPLITS_FILE, "r") as f:
            splits = json.load(f)
            
        self.filenames = splits[split]
        if len(self.filenames) == 0:
            raise ValueError(f"No files found for split {split}")

    def __len__(self):
        return len(self.filenames)
        
    def _random_crop(self, lr, hr):
        h, w = lr.shape
        if h <= self.crop_size or w <= self.crop_size:
            # Pad if too small, though they are 128x128
            return lr, hr
            
        y = np.random.randint(0, h - self.crop_size)
        x = np.random.randint(0, w - self.crop_size)
        
        lr_crop = lr[y : y + self.crop_size, x : x + self.crop_size]
        hr_crop = hr[y*2 : (y + self.crop_size)*2, x*2 : (x + self.crop_size)*2]
        
        return lr_crop, hr_crop
        
    def _augment(self, lr, hr):
        # D4 augmentations (flips + rotations)
        hflip = np.random.rand() < 0.5
        vflip = np.random.rand() < 0.5
        rot90 = np.random.rand() < 0.5
        
        if hflip:
            lr = np.fliplr(lr)
            hr = np.fliplr(hr)
        if vflip:
            lr = np.flipud(lr)
            hr = np.flipud(hr)
        if rot90:
            lr = np.rot90(lr, axes=(0, 1))
            hr = np.rot90(hr, axes=(0, 1))
            
        return lr.copy(), hr.copy()
        
    def _cutblur(self, lr, hr):
        """
        CutBlur augmentation: mix HR into LR or vice versa.
        Since HR is 2x LR, we downsample the HR patch to LR resolution for mixing,
        or upsample LR patch to HR resolution.
        The brief says CutBlur, which usually replaces a patch in LR with the corresponding HR patch (downsampled),
        or replaces a patch in HR with the corresponding LR patch (upsampled).
        Here we'll do the LR-domain CutBlur: pasting HR patch (downsampled) into LR.
        """
        if np.random.rand() < 0.5:
            return lr, hr
            
        h, w = lr.shape
        
        # Random patch size between 20% and 50% of the image
        ch = int(np.random.uniform(0.2, 0.5) * h)
        cw = int(np.random.uniform(0.2, 0.5) * w)
        
        y = np.random.randint(0, h - ch)
        x = np.random.randint(0, w - cw)
        
        # Paste HR into LR (Requires downsampling HR patch)
        # Using simple average pooling for downsampling the patch
        hr_patch = hr[y*2 : (y+ch)*2, x*2 : (x+cw)*2]
        hr_patch_ds = hr_patch.reshape(ch, 2, cw, 2).mean(axis=(1, 3))
        
        if np.random.rand() < 0.5:
            # Paste HR->LR
            lr[y:y+ch, x:x+cw] = hr_patch_ds
        else:
            # Paste LR->HR
            lr_patch_us = lr[y:y+ch, x:x+cw].repeat(2, axis=0).repeat(2, axis=1)
            hr[y*2 : (y+ch)*2, x*2 : (x+cw)*2] = lr_patch_us
            
        return lr, hr

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        
        gt = np.load(GT_DIR / fn)
        
        # Decide whether to use real pair or generate synthetic
        if self.split == "train" and np.random.rand() < self.synthetic_ratio:
            lr, hr = generate_synthetic_pair(gt)
        else:
            lr = np.load(LR_DIR / fn)
            hr = gt
            
        if self.split == "train":
            lr, hr = self._random_crop(lr, hr)
            if self.augment:
                lr, hr = self._augment(lr, hr)
            if self.use_cutblur:
                lr, hr = self._cutblur(lr, hr)
        else:
            # For validation, we could center crop or take the whole image
            # The brief says random crops for train. For val, let's keep it full size (128/256)
            pass
            
        # Normalization
        # According to Phase 3 model: 
        # "Input: 2 channels -- the normalized array and log(normalized + 1e-3), clamped at zero."
        # We perform robust normalization on the LR image.
        lr_norm, params = robust_normalize(lr)
        
        # The HR image is our target. 
        # The model outputs a residual, added to bicubic upsampled input.
        # It's cleaner to return unnormalized HR and unnormalized LR here, 
        # or handle normalization inside the dataset or model.
        # The brief says: "final: inverse-normalize, then clip to GT range".
        # So we pass parameters so the training loop/model can use them.
        
        # We will return the normalized LR as base input, the params for inverse, and the raw HR target.
        
        lr_ch1 = lr_norm
        lr_ch2 = np.log(np.maximum(lr_norm + 1e-3, 1e-6))
        lr_ch2 = np.maximum(lr_ch2, 0.0) # clamped at zero
        
        # Stack channels
        lr_tensor = torch.from_numpy(np.stack([lr_ch1, lr_ch2], axis=0)).float()
        hr_tensor = torch.from_numpy(hr).unsqueeze(0).float()
        
        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "norm_params": torch.tensor(params, dtype=torch.float32), # (low_val, scale)
            "filename": fn
        }

if __name__ == "__main__":
    ds = KLAImageDataset(split="train", synthetic_ratio=0.5, use_cutblur=True)
    sample = ds[0]
    print(sample["lr"].shape)
    print(sample["hr"].shape)
    print(sample["norm_params"])
