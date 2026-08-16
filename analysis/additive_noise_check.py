import numpy as np
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter
import sys

GT_DIR  = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\GT")
LR_DIR  = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\NoisyLR")

def ds_bicubic(img):
    return zoom(img, 0.5, order=3, prefilter=True)

def main():
    gt_names = sorted(f.name for f in GT_DIR.glob("*.npy"))[:100]  # sample 100 images
    
    ds_vals = []
    res_vals = []
    
    for fn in gt_names:
        gt = np.load(GT_DIR / fn)
        lr = np.load(LR_DIR / fn)
        
        gt_b = gaussian_filter(gt, 0.3)
        ds = ds_bicubic(gt_b)
        
        res = lr - ds
        
        ds_vals.append(ds.flatten())
        res_vals.append(res.flatten())
        
    ds_all = np.concatenate(ds_vals)
    res_all = np.concatenate(res_vals)
    
    # We want to model: res = ds * n_mult + n_add
    # var(res) = ds^2 * var(n_mult) + var(n_add) (assuming independent)
    
    # Bin by ds
    bins = np.linspace(0, 1, 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    var_res = []
    valid_centers = []
    
    print(f"{'ds_bin':>10} | {'count':>10} | {'std(res)':>10} | {'var(res)':>10}")
    print("-" * 47)
    
    for i in range(len(bins)-1):
        mask = (ds_all >= bins[i]) & (ds_all < bins[i+1])
        if mask.sum() > 1000:
            v = np.var(res_all[mask])
            s = np.std(res_all[mask])
            c = mask.sum()
            var_res.append(v)
            valid_centers.append(bin_centers[i])
            print(f"{bin_centers[i]:10.3f} | {c:10d} | {s:10.5f} | {v:10.5f}")
            
    # Fit line: y = m * x + c
    # y = var(res), x = ds^2
    # m = var(n_mult), c = var(n_add)
    x = np.array(valid_centers)**2
    y = np.array(var_res)
    
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    std_mult = np.sqrt(max(0, m))
    std_add = np.sqrt(max(0, c))
    
    print("-" * 47)
    print("Fitted Model: var(res) = ds^2 * var(n_mult) + var(n_add)")
    print(f"var(n_mult) = {m:.6f}  => std(n_mult) = {std_mult:.5f}")
    print(f"var(n_add)  = {c:.6f}  => std(n_add)  = {std_add:.5f}")

if __name__ == "__main__":
    main()
