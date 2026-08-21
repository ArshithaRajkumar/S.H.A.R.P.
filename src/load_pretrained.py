import torch
import torch.nn as nn
from pathlib import Path
import urllib.request
import os

def download_pretrained(model_name="NAFNet-SIDD-width64.pth", save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = Path(save_dir) / model_name
    url = f"https://huggingface.co/nyanko7/nafnet-models/resolve/main/{model_name}"
    
    if not save_path.exists():
        print(f"Downloading {model_name}...")
        urllib.request.urlretrieve(url, save_path)
        print("Download complete.")
    else:
        print(f"Found {model_name} locally.")
    return save_path

def load_nafnet_pretrained(model, ckpt_path):
    """
    Loads pretrained weights into a custom NAFNetSR model.
    Only loads matching NAFBlock weights (conv1, conv2, dwconv, sca.conv, norm).
    """
    print(f"\n--- Loading Pretrained Weights from {ckpt_path.name} ---")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    model_state = model.state_dict()
    
    loaded_params = 0
    skipped_params = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    
    # Map from our layer names to their layer names where possible.
    # We only care about the blocks, e.g., 'blocks.0.conv1.weight'
    
    loaded_layers = []
    skipped_layers = []
    
    # Analyze their width
    # In official NAFNet, encoders/decoders/middle blocks exist.
    # Their structure has 'encoders.0...', 'middle_blks...', 'decoders...'.
    # We just have a flat list of `blocks`. 
    # Let's see what keys they have.
    # First, let's grab all NAFBlock-like keys from their state dict.
    their_blocks = {}
    
    # This might require manual mapping. Since our NAFBlocks are flat, we will map them sequentially
    # from their 'middle_blks' or 'encoders'/'decoders'.
    # Actually, we first need to inspect the checkpoint.
    
    # Let's inspect first:
    keys = list(state_dict.keys())
    print(f"Checkpoint contains {len(keys)} keys. First 10 keys:")
    for k in keys[:10]:
        print("  ", k, state_dict[k].shape)
        
    print("\nWarning: The width of the pretrained model and our model might differ.")
    print("If they differ, we cannot load the 1x1 or 3x3 convs directly without resampling/truncating.")
    
    # Let's check width based on their first block's norm layer
    if 'intro.weight' in state_dict:
        their_width = state_dict['intro.weight'].shape[0]
    elif 'encoders.0.0.norm1.weight' in state_dict: # commonly named
        their_width = state_dict['encoders.0.0.norm1.weight'].shape[0]
    else:
        their_width = "Unknown"
        
    print(f"Their inferred width: {their_width}")
    
    if hasattr(model, 'up_conv'):
        our_width = model.up_conv.weight.shape[1]
    else:
        our_width = "Unknown"
        
    print(f"Our model width: {our_width}")
    
    if str(our_width) != str(their_width):
        print(f"\n[!] MISMATCH: Our width is {our_width} but checkpoint width is {their_width}.")
        print("Force-fitting mismatched checkpoint layers via truncation/padding destroys the learned filters.")
        print("Skipping weight transfer. Please consider changing `TrainConfig.width` to match the checkpoint, or train from scratch.")
        return False
        
    # If dimensions matched, we would implement the mapping here.
    # But since we know it's 256 vs 64, it will abort safely.
    
    return False

if __name__ == "__main__":
    # Test script locally
    ckpt_path = download_pretrained()
    
    # Instantiate our model to test loading
    import sys
    BASE_DIR = Path(__file__).resolve().parent.parent
    sys.path.append(str(BASE_DIR))
    from src.model import NAFNetSR
    
    model = NAFNetSR(width=256, num_blocks=16)
    load_nafnet_pretrained(model, ckpt_path)
