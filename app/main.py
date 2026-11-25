from dotenv import load_dotenv

load_dotenv()

from app.routers import auth, patients, cases

from app.preprocessing import preprocess_image
from app.stenosis_inference import predict_stenosis

from pathlib import Path
from uuid import uuid4
import io

import numpy as np
import pydicom
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydicom.errors import InvalidDicomError
from app.stenosis_classification import analyze_stenosis
from app.stenosis_measurement import measure_stenosis

app = FastAPI(
    title="Artery Image API",
    description="Image upload, preprocessing and stenosis analysis",
    version="0.2.0",
)

app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(cases.router)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/images")
async def upload_image(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: PNG, JPEG, DICOM",
        )

    content = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB")

    try:
        if extension == ".dcm":
            dataset = pydicom.dcmread(io.BytesIO(content))

            if "PixelData" not in dataset:
                raise ValueError("DICOM has no pixel data")

            pixels = dataset.pixel_array

            if pixels.size == 0 or not np.isfinite(pixels).all():
                raise ValueError("Invalid DICOM pixel data")

        else:
            with Image.open(io.BytesIO(content)) as image:
                actual_format = image.format
                expected_formats = {
                    ".png": "PNG",
                    ".jpg": "JPEG",
                    ".jpeg": "JPEG",
                }

                if actual_format != expected_formats[extension]:
                    raise ValueError(
                        "File extension does not match image format"
                    )

                image.verify()

    except (
        UnidentifiedImageError,
        InvalidDicomError,
        OSError,
        ValueError,
        AttributeError,
        KeyError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image file",
        ) from exc

    stored_name = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / stored_name
    destination.write_bytes(content)

    return {
        "message": "Image uploaded successfully",
        "image_id": destination.stem,
        "filename": stored_name,
        "format": extension.lstrip("."),
        "size_bytes": len(content),
        "storage_path": str(destination.relative_to(BASE_DIR)),
    }


@app.post("/images/{image_id}/preprocess")
def preprocess_uploaded_image(image_id: str):
    try:
        result = preprocess_image(image_id)

        return {
            "message": "Image preprocessed successfully",
            **result,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not preprocess image",
        ) from exc


@app.post("/images/{image_id}/predict")
def predict_uploaded_image(image_id: str):
    try:
        # 1. Detect and localize stenosis regions
        result = predict_stenosis(image_id)

        # 2. Load normalized angiogram
        image_path = Path("processed") / f"{image_id}_normalized.npy"
        image = np.load(image_path).astype(np.float32)

        # 3. Load predicted stenosis mask
        mask_path = Path(result["predicted_mask_path"])
        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        # 4. Estimate narrowing and classify severity
        measurement_result = measure_stenosis(image, mask)

        # 5. Attach measurements to corresponding regions
        measurements = {
            item["region_id"]: item
            for item in measurement_result["measurements"]
        }

        for region in result["regions"]:
            region["measurement"] = measurements.get(
                region["region_id"],
                {
                    "measurement_available": False,
                    "severity": "Unavailable"
                }
            )

        return {
            "message": "Stenosis analysis completed successfully",
            **result,
            "measurement_method": measurement_result["measurement_method"],
            "successful_measurements": measurement_result["successful_measurements"],
            "measurement_note": measurement_result["note"]
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not complete stenosis analysis",
        ) from exc