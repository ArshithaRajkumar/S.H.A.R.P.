import numpy as np
from pathlib import Path
import os
import tempfile

def test_io_round_trip():
    # Pick a real file
    base_dir = Path(__file__).resolve().parent
    lr_dir = base_dir / "train" / "train" / "NoisyLR"
    files = list(lr_dir.glob("*.npy"))
    
    if not files:
        print("Skipping I/O test: No files found in NoisyLR")
        return
        
    test_file = files[0]
    
    # Load
    original = np.load(test_file)
    assert original.dtype == np.float32, f"Expected float32, got {original.dtype}"
    
    # Save to temp
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / test_file.name
        np.save(temp_path, original)
        
        # Reload
        reloaded = np.load(temp_path)
        
        # Assert bit-exact equality
        assert reloaded.dtype == np.float32
        assert np.array_equal(original, reloaded), "Bit-exact round-trip failed!"
        
    print("I/O Round-trip bit-exact equality test passed.")

if __name__ == "__main__":
    test_io_round_trip()
