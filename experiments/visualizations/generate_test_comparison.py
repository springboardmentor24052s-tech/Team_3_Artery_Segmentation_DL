from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model_multiscale import MultiScaleResidualUNet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT = Path(
    "stenosis_detection/checkpoints/"
    "resunet_multiscale_frozen_finetuned_best.pt"
)

OUTPUT_DIR = Path(
    "stenosis_detection/outputs/test_comparisons"
)

THRESHOLD = 0.90
DEVICE = torch.device("cpu")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading final stenosis segmentation model...")

model = MultiScaleResidualUNet(
    in_channels=1,
    out_channels=1
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint loaded successfully.")


# --------------------------------------------------
# LOAD HELD-OUT TEST DATA
# --------------------------------------------------

print("\nLoading ARCADE test dataset...")

test_dataset = StenosisDataset(split="test")

print("Test images:", len(test_dataset))

# Select the first test image containing a
# ground-truth stenosis region.
selected_index = None

for index in range(len(test_dataset)):
    image, mask = test_dataset[index]

    if torch.count_nonzero(mask).item() > 0:
        selected_index = index
        break

if selected_index is None:
    raise RuntimeError(
        "No test image with a stenosis annotation was found."
    )

image, mask = test_dataset[selected_index]

filename = test_dataset.images[selected_index]["file_name"]

print("Selected test image:", filename)


# --------------------------------------------------
# GENERATE PREDICTION
# --------------------------------------------------

print("\nGenerating model prediction...")

with torch.no_grad():
    input_tensor = image.unsqueeze(0).to(DEVICE)

    logits = model(input_tensor)

    probabilities = torch.sigmoid(logits)[0, 0]

    prediction = (
        probabilities >= THRESHOLD
    ).cpu().numpy()

ground_truth = mask[0].numpy() > 0.5

original = (
    image[0].numpy() * 255
).clip(0, 255).astype(np.uint8)


# --------------------------------------------------
# CALCULATE IMAGE-LEVEL METRICS
# --------------------------------------------------

tp = np.logical_and(prediction, ground_truth).sum()
fp = np.logical_and(prediction, ~ground_truth).sum()
fn = np.logical_and(~prediction, ground_truth).sum()

dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
iou = tp / (tp + fp + fn + 1e-8)

print("\nIMAGE-LEVEL RESULTS")
print("------------------------------")
print(f"Dice:      {dice:.4f}")
print(f"IoU:       {iou:.4f}")
print(f"Threshold: {THRESHOLD}")
print(f"TP:        {tp}")
print(f"FP:        {fp}")
print(f"FN:        {fn}")


# --------------------------------------------------
# CREATE VISUALIZATIONS
# --------------------------------------------------

original_rgb = np.stack(
    [original, original, original],
    axis=-1
)

gt_rgb = (
    ground_truth.astype(np.uint8) * 255
)

pred_rgb = (
    prediction.astype(np.uint8) * 255
)

# Overlay:
# Green = Ground truth only
# Red = Prediction only
# Yellow = Overlap

overlay = original_rgb.copy().astype(np.float32)

gt_only = ground_truth & ~prediction
pred_only = prediction & ~ground_truth
overlap = ground_truth & prediction

alpha = 0.55

overlay[gt_only] = (
    (1 - alpha) * overlay[gt_only]
    + alpha * np.array([0, 255, 0])
)

overlay[pred_only] = (
    (1 - alpha) * overlay[pred_only]
    + alpha * np.array([255, 0, 0])
)

overlay[overlap] = (
    (1 - alpha) * overlay[overlap]
    + alpha * np.array([255, 255, 0])
)

overlay = overlay.clip(0, 255).astype(np.uint8)


# --------------------------------------------------
# SAVE INDIVIDUAL OUTPUTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

stem = Path(filename).stem

Image.fromarray(original).save(
    OUTPUT_DIR / f"{stem}_original.png"
)

Image.fromarray(gt_rgb).save(
    OUTPUT_DIR / f"{stem}_ground_truth.png"
)

Image.fromarray(pred_rgb).save(
    OUTPUT_DIR / f"{stem}_prediction.png"
)

Image.fromarray(overlay).save(
    OUTPUT_DIR / f"{stem}_overlay.png"
)


# --------------------------------------------------
# CREATE FOUR-PANEL COMPARISON
# --------------------------------------------------

panel_size = 512
title_height = 38
gap = 16

canvas_width = panel_size * 2 + gap
canvas_height = (
    (panel_size + title_height) * 2 + gap
)

canvas = Image.new(
    "RGB",
    (canvas_width, canvas_height),
    "white"
)

draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default()

panels = [
    ("Original Angiogram", Image.fromarray(original_rgb)),
    ("Ground Truth Mask", Image.fromarray(gt_rgb).convert("RGB")),
    ("Model Prediction (0.90)", Image.fromarray(pred_rgb).convert("RGB")),
    ("Prediction vs Ground Truth", Image.fromarray(overlay))
]

positions = [
    (0, 0),
    (panel_size + gap, 0),
    (0, panel_size + title_height + gap),
    (panel_size + gap, panel_size + title_height + gap)
]

for (title, panel), (x, y) in zip(panels, positions):

    draw.text(
        (x + 8, y + 12),
        title,
        fill="black",
        font=font
    )

    canvas.paste(
        panel,
        (x, y + title_height)
    )

comparison_path = (
    OUTPUT_DIR / f"{stem}_comparison.png"
)

canvas.save(comparison_path)

print("\nSUCCESS!")
print("------------------------------")
print("Saved comparison:", comparison_path)
print("Saved individual images in:", OUTPUT_DIR)
