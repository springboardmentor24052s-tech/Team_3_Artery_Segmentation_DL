
# Artery Segmentation and Stenosis Detection

## Overview

This module implements deep learning-based artery segmentation using the ARCADE dataset. It includes Baseline U-Net and Attention U-Net models for segmenting artery regions in angiography images.

## Project Structure

```text
segmentation/
├── src/
│   ├── dataset.py
│   ├── unet.py
│   ├── attention_unet.py
│   ├── train.py
│   ├── train_attention.py
│   ├── compare_models.py
│   ├── threshold_analysis.py
│   ├── postprocess.py
│   ├── extract_centerlines.py
│   ├── estimate_vessel_diameter.py
│   └── ...
├── requirements.txt
└── README.md
```

## Models

- **Baseline U-Net:** Performs artery image segmentation.
- **Attention U-Net:** Uses attention gates to focus on relevant vessel regions.

## Dataset

The module uses the ARCADE dataset.

Place the dataset in the appropriate location and configure the dataset paths in `src/dataset.py`.

Dataset files are not included in this repository.

## Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r segmentation/requirements.txt
```

## Training

Run commands from the repository root:

```powershell
python -m segmentation.src.train
python -m segmentation.src.train_attention
```

Check the configuration and dataset paths in the scripts before running training.

## Evaluation

The module includes scripts for:

- Comparing Baseline U-Net and Attention U-Net.
- Evaluating segmentation metrics.
- Performing threshold analysis.
- Post-processing predicted masks.
- Extracting vessel centerlines.
- Estimating vessel diameter.

## Evaluation Metrics

- Dice Score
- Precision
- Recall
- Specificity
- Pixel Accuracy
- Hausdorff Distance
- Average Surface Distance

## Outputs

Generated checkpoints, prediction masks, and visualizations are stored locally in the `outputs/` directory and are excluded from Git.

## Dependencies

See `requirements.txt` for the Python package versions used by this module.
