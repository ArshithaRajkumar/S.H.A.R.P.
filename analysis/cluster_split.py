import numpy as np
from pathlib import Path
from scipy.cluster.vq import kmeans2, whiten
from scipy.ndimage import sobel
import json

GT_DIR = Path(r"C:\Users\LENOVO\Downloads\KLA\train\train\GT")
OUT_DIR = Path(__file__).resolve().parent

def get_stats(img):
    """Extract simple structure statistics from an image."""
    mean = img.mean()
    std = img.std()
    
    # Gradient magnitude (edge density)
    sx = sobel(img, axis=0, mode='constant')
    sy = sobel(img, axis=1, mode='constant')
    grad_mag = np.hypot(sx, sy)
    grad_mean = grad_mag.mean()
    grad_std = grad_mag.std()
    
    return [mean, std, grad_mean, grad_std]

def main():
    print("Extracting features for clustering...")
    gt_names = sorted(f.name for f in GT_DIR.glob("*.npy"))
    
    features = []
    for i, fn in enumerate(gt_names):
        gt = np.load(GT_DIR / fn)
        features.append(get_stats(gt))
        if (i+1) % 400 == 0:
            print(f"Processed {i+1}/{len(gt_names)}")
            
    features = np.array(features)
    
    # Normalize features using scipy whiten (divides by standard deviation)
    feats_norm = whiten(features)
    
    print("Clustering into 3 categories...")
    # Use scipy kmeans2 to cluster into 3 categories
    np.random.seed(42)
    centroids, labels = kmeans2(feats_norm, 3, minit='points')
    
    counts = np.bincount(labels)
    print(f"Cluster sizes: {counts}")
    
    # Hold out the smallest cluster to ensure OOD is challenging but we keep enough training data
    held_out_cluster = int(np.argmin(counts))
    print(f"Holding out cluster {held_out_cluster} (size: {counts[held_out_cluster]}) as OOD split.")
    
    splits = {
        "train": [],
        "val_in_dist": [],
        "val_ood": []
    }
    
    # To have an in-distribution val set, we'll reserve 10% of the in-distribution clusters
    np.random.seed(42)
    
    for fn, label in zip(gt_names, labels):
        if label == held_out_cluster:
            splits["val_ood"].append(fn)
        else:
            if np.random.rand() < 0.1:
                splits["val_in_dist"].append(fn)
            else:
                splits["train"].append(fn)
                
    print(f"Final splits: Train: {len(splits['train'])}, Val-In: {len(splits['val_in_dist'])}, Val-OOD: {len(splits['val_ood'])}")
    
    split_file = OUT_DIR / "dataset_splits.json"
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Saved splits to {split_file}")

if __name__ == "__main__":
    main()
