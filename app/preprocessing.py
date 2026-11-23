from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(image_id: str) -> dict:
    """Convert an uploaded angiography image to grayscale,
    normalize its pixel values, and save model-ready output.
    """
    matches = [
        path for path in UPLOAD_DIR.iterdir()
        if path.stem == image_id
        and path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".dcm"
        }
    ]

    if not matches:
        raise FileNotFoundError("Uploaded image not found")

    source_path = matches[0]

    # Load image pixels
    if source_path.suffix.lower() == ".dcm":
        dicom = pydicom.dcmread(source_path)
        pixels = dicom.pixel_array.astype(np.float32)

        if pixels.ndim != 2:
            raise ValueError(
                "Only single-frame grayscale DICOM images are supported"
            )

        pixels = (
            pixels * float(getattr(dicom, "RescaleSlope", 1))
            + float(getattr(dicom, "RescaleIntercept", 0))
        )

        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels

        # Scale DICOM intensity values to [0, 1]
        low, high = pixels.min(), pixels.max()
        if high > low:
            normalized = (pixels - low) / (high - low)
        else:
            normalized = np.zeros_like(pixels, dtype=np.float32)

    else:
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image)
            grayscale = image.convert("L")
            pixels = np.asarray(grayscale, dtype=np.float32)

        normalized = pixels / 255.0

    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    # Save normalized array for future model input
    array_path = PROCESSED_DIR / f"{image_id}_normalized.npy"
    np.save(array_path, normalized)

    # Save an 8-bit PNG for visual inspection
    output_pixels = np.round(normalized * 255).astype(np.uint8)
    processed_image = Image.fromarray(output_pixels)

    output_path = PROCESSED_DIR / f"{image_id}_processed.png"
    processed_image.save(output_path)

    return {
        "image_id": image_id,
        "original_path": str(source_path.relative_to(BASE_DIR)),
        "processed_path": str(output_path.relative_to(BASE_DIR)),
        "normalized_array_path": str(array_path.relative_to(BASE_DIR)),
        "width": processed_image.width,
        "height": processed_image.height,
        "channels": 1,
        "pixel_range": [float(normalized.min()), float(normalized.max())],
        "data_type": str(normalized.dtype),
    }