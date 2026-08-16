# SHARP: Structural High-frequency Aware Restoration Pipeline

This is our submission for the KLA PS01 Image Restoration Hackathon.

## Overview
SHARP is a lightweight, NAFNet-based architecture designed for joint denoising, deblurring, and ×2 super-resolution of single-channel float arrays from semiconductor inspection tooling. 

Instead of relying purely on L2/Charbonnier loss (which tends to blur high-frequency details to minimize error), our model is trained with a multi-loss setup combining:
- **Charbonnier** for global reconstruction
- **MS-SSIM** for structural similarity
- **FFT Magnitude L1** for frequency preservation
- **Gradient Consistency** for sharp edges

This approach specifically targets the preservation and recovery of physical defects, resulting in a **49% relative gain in True Positive Rate** for defect detection compared to a baseline Charbonnier model, while simultaneously dropping the False Positive rate.

## Repository Constraints Met
- **Strict I/O**: `evaluate.py` strictly uses `numpy` arrays (float32). No 8-bit quantization, PIL, or OpenCV is ever used in the path.
- **Deterministic**: Fully deterministic evaluation path (seed fixed, no sampling).
- **Lightweight**: Model parameters are ~0.32M (well below the 3–8M limit), optimized for H100 latency.
- **Out-of-Distribution**: Validation sets were rigorously split by source structure to ensure OOD performance is measured.

## Running Inference
The script runs out-of-the-box. The default checkpoint loaded is our best `v2_deeper_multiloss` model.

### Requirements
Install the pip freeze requirements in a clean virtual environment:
```bash
pip install -r requirements.txt
```

### Evaluation
The inference script strictly adheres to the two-argument requirement:
```bash
python evaluate.py --input_dir /path/to/NoisyLR --output_dir /path/to/output
```
The script will load `.npy` files from the input directory, perform the restoration, and save exact float32 `.npy` arrays to the output directory.

## Defect Evaluation
To run the Phase 6 defect evaluation pipeline (which injects synthetic defects into the ground truth and compares the recovery capability of the model), run:
```bash
python src/defect_eval.py
```
*(The detector parameters are fixed via theoretical LoG derivation and a 99th-percentile calibration on clean background, ensuring unbiased thresholding.)*
