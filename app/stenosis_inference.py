
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from stenosis_detection.model_multiscale import MultiScaleResidualUNet


BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    BASE_DIR
    / "stenosis_detection"
    / "checkpoints"
    / "resunet_multiscale_frozen_finetuned_best.pt"
)

PREDICTION_DIR = BASE_DIR / "processed" / "predictions"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stenosis_model():
    """Load the selected trained stenosis segmentation model."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = MultiScaleResidualUNet(
        in_channels=1,
        out_channels=1
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("FrozenFineTuned stenosis model loaded.")
    print(f"Device: {DEVICE}")
    print(f"Threshold: {THRESHOLD}")

    return model


stenosis_model = load_stenosis_model()


def predict_stenosis(image_id: str) -> dict:
    """Predict stenosis regions and return their locations."""

    array_path = (
        BASE_DIR / "processed" / f"{image_id}_normalized.npy"
    )

    if not array_path.exists():
        raise FileNotFoundError(
            "Preprocessed image not found. "
            "Run the preprocessing endpoint first."
        )

    image_array = np.load(array_path).astype(np.float32)

    if image_array.ndim != 2:
        raise ValueError("Expected a single-channel grayscale image.")

    if not np.isfinite(image_array).all():
        raise ValueError("Image contains invalid pixel values.")

    if image_array.shape != (512, 512):
        raise ValueError(
            "Model expects a 512 x 512 image. "
            f"Received {image_array.shape[1]} x {image_array.shape[0]}."
        )

    if image_array.min() < 0 or image_array.max() > 1:
        raise ValueError("Image values must be normalized between 0 and 1.")

    input_tensor = torch.from_numpy(
        image_array.copy()
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = stenosis_model(input_tensor)
        probabilities = torch.sigmoid(logits)

    probability_map = probabilities[0, 0].cpu().numpy()

    binary_mask = probability_map >= THRESHOLD

    labeled_mask, component_count = ndimage.label(binary_mask)

    regions = []

    for component_id, region_slice in enumerate(
        ndimage.find_objects(labeled_mask),
        start=1
    ):
        if region_slice is None:
            continue

        component_mask = labeled_mask[region_slice] == component_id
        area = int(component_mask.sum())

        if area == 0:
            continue

        y_slice, x_slice = region_slice

        local_y, local_x = np.where(component_mask)

        regions.append({
            "region_id": len(regions) + 1,
            "area_pixels": area,
            "centroid": {
                "x": round(
                    float((local_x + x_slice.start).mean()), 2
                ),
                "y": round(
                    float((local_y + y_slice.start).mean()), 2
                )
            },
            "bounding_box": {
                "x_min": int(x_slice.start),
                "y_min": int(y_slice.start),
                "x_max": int(x_slice.stop - 1),
                "y_max": int(y_slice.stop - 1)
            },
            "mean_probability": round(
                float(probability_map[region_slice][component_mask].mean()),
                4
            )
        })

    # Save binary segmentation mask.
    mask_pixels = binary_mask.astype(np.uint8) * 255

    mask_path = (
        PREDICTION_DIR / f"{image_id}_stenosis_mask.png"
    )

    Image.fromarray(mask_pixels).save(mask_path)

    # Save probability visualization.
    probability_pixels = np.round(
        np.clip(probability_map, 0, 1) * 255
    ).astype(np.uint8)

    probability_path = (
        PREDICTION_DIR / f"{image_id}_probability.png"
    )

    # Reuse an existing probability image to avoid the
    # Windows file-overwrite issue encountered previously.
    if not probability_path.exists():
        Image.fromarray(probability_pixels).save(probability_path)

    return {
        "image_id": image_id,
        "model": "FrozenFineTuned MultiScaleResidualUNet",
        "threshold": THRESHOLD,
        "image_width": int(image_array.shape[1]),
        "image_height": int(image_array.shape[0]),
        "predicted_stenosis_detected": bool(binary_mask.any()),
        "candidate_region_count": len(regions),
        "regions": regions,
        "predicted_mask_path": str(
            mask_path.relative_to(BASE_DIR)
        ),
        "probability_map_path": str(
            probability_path.relative_to(BASE_DIR)
        ),
        "note": (
            "Predicted segmentation regions are research outputs "
            "and are not clinically validated diagnoses."
        )
    }
