import math
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps, UnidentifiedImageError
from pydicom.errors import InvalidDicomError

from segmentation.src.unet import UNet


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = (
    BASE_DIR
    / "app"
    / "checkpoints"
    / "segmentation"
    / "unet_baseline_round2_best.pth"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5
EXPECTED_EPOCH = 11
EXPECTED_VALIDATION_DICE = 0.5207999770392526
EXPECTED_STATE_DICT_ENTRIES = 118
_MODELS = {}
_CHECKPOINT_METADATA = {}


def load_baseline_unet(device=None):
    target_device = torch.device(device or DEVICE)
    if str(target_device) in _MODELS:
        return _MODELS[str(target_device)]

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Baseline U-Net checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Baseline U-Net checkpoint must be a mapping")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Baseline U-Net checkpoint does not contain model_state_dict"
        )
    try:
        epoch = int(checkpoint["epoch"])
        validation_dice = float(checkpoint["val_dice"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Baseline U-Net checkpoint metadata is invalid") from exc
    if (
        epoch != EXPECTED_EPOCH
        or len(state_dict) != EXPECTED_STATE_DICT_ENTRIES
        or not math.isclose(
            validation_dice,
            EXPECTED_VALIDATION_DICE,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Checkpoint does not match the selected Baseline U-Net")

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(state_dict, strict=True)
    model.to(target_device)
    model.eval()
    _MODELS[str(target_device)] = model
    _CHECKPOINT_METADATA.update(
        epoch=epoch,
        validation_dice=validation_dice,
        state_dict_entries=len(state_dict),
    )
    return model


def _read_rgb_image(image_path: Path) -> np.ndarray:
    extension = image_path.suffix.lower()
    try:
        if extension == ".dcm":
            dataset = pydicom.dcmread(image_path)
            if "PixelData" not in dataset:
                raise ValueError("DICOM image has no pixel data")

            pixels = dataset.pixel_array.astype(np.float32)
            if pixels.ndim != 2 or pixels.size == 0:
                raise ValueError(
                    "Only single-frame grayscale DICOM images are supported"
                )

            pixels = (
                pixels * float(getattr(dataset, "RescaleSlope", 1))
                + float(getattr(dataset, "RescaleIntercept", 0))
            )
            if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixels = pixels.max() - pixels

            low, high = float(pixels.min()), float(pixels.max())
            if not np.isfinite(pixels).all():
                raise ValueError("DICOM image contains invalid pixel values")
            if high > low:
                pixels = (pixels - low) / (high - low)
            else:
                pixels = np.zeros_like(pixels, dtype=np.float32)
            grayscale = np.round(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
            return np.repeat(grayscale[:, :, None], 3, axis=2)

        with Image.open(image_path) as source:
            rgb_image = ImageOps.exif_transpose(source).convert("RGB")
            return np.asarray(rgb_image, dtype=np.uint8).copy()
    except (
        Image.DecompressionBombError,
        InvalidDicomError,
        OSError,
        RuntimeError,
        UnidentifiedImageError,
    ) as exc:
        raise ValueError("Could not decode the uploaded image") from exc


def _predict_with_model(image_array: np.ndarray, model, target_device):
    height, width = image_array.shape[:2]
    tensor = (
        torch.from_numpy(image_array.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=target_device, dtype=torch.float32)
        .div_(255.0)
    )
    pad_height = (-height) % 16
    pad_width = (-width) % 16
    if pad_height or pad_width:
        tensor = functional.pad(tensor, (0, pad_width, 0, pad_height))

    with torch.inference_mode():
        logits = model(tensor)
        if logits.ndim != 4 or logits.shape[:2] != (1, 1):
            raise RuntimeError("Baseline U-Net returned an invalid output shape")
        probabilities = torch.sigmoid(logits)[0, 0, :height, :width]

    probability_map = probabilities.cpu().numpy()
    if probability_map.shape != (height, width) or not np.isfinite(
        probability_map
    ).all():
        raise RuntimeError("Baseline U-Net returned invalid probabilities")

    return probability_map.astype(np.float32, copy=False)


def predict_artery_segmentation(image_path: Path, device=None) -> dict:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Uploaded image not found: {image_path}")

    image_array = _read_rgb_image(image_path)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Expected a three-channel RGB image")
    if image_array.shape[0] == 0 or image_array.shape[1] == 0:
        raise ValueError("Image dimensions must be non-zero")

    target_device = torch.device(device or DEVICE)
    model = load_baseline_unet(target_device)
    try:
        probability_map = _predict_with_model(
            image_array,
            model,
            target_device,
        )
    except torch.cuda.OutOfMemoryError:
        if target_device.type != "cuda":
            raise
        model.to("cpu")
        _MODELS.pop(str(target_device), None)
        torch.cuda.empty_cache()
        target_device = torch.device("cpu")
        probability_map = _predict_with_model(
            image_array,
            load_baseline_unet(target_device),
            target_device,
        )

    binary_mask = probability_map >= THRESHOLD
    return {
        "probability_map": probability_map,
        "binary_mask": binary_mask,
        "input_rgb": image_array,
        "threshold": THRESHOLD,
        "image_width": int(image_array.shape[1]),
        "image_height": int(image_array.shape[0]),
        "mask_pixel_count": int(binary_mask.sum()),
        "device": str(target_device),
        "checkpoint_epoch": _CHECKPOINT_METADATA["epoch"],
        "validation_dice": _CHECKPOINT_METADATA["validation_dice"],
    }


def save_segmentation_artifacts(prediction: dict, analysis_run_id: str) -> dict:
    prediction_dir = BASE_DIR / "processed" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{analysis_run_id}_baseline_unet"
    mask = prediction["binary_mask"]
    input_pixels = prediction["input_rgb"]
    overlay = input_pixels.astype(np.float32)
    overlay[mask] = overlay[mask] * 0.45 + np.array(
        [32, 220, 160],
        dtype=np.float32,
    ) * 0.55

    images = {
        "artery_input_preview": Image.fromarray(input_pixels, mode="RGB"),
        "artery_segmentation_mask": Image.fromarray(
            mask.astype(np.uint8) * 255,
            mode="L",
        ),
        "artery_probability_map": Image.fromarray(
            np.round(
                np.clip(prediction["probability_map"], 0, 1) * 255
            ).astype(np.uint8),
            mode="L",
        ),
        "artery_segmentation_overlay": Image.fromarray(
            np.clip(np.round(overlay), 0, 255).astype(np.uint8),
            mode="RGB",
        ),
    }

    paths = {}
    created_paths = []
    try:
        for artifact_type, image in images.items():
            path = prediction_dir / f"{stem}_{artifact_type}.png"
            image.save(path, format="PNG")
            created_paths.append(path)
            paths[artifact_type] = path.relative_to(BASE_DIR).as_posix()
    except OSError:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    return paths
