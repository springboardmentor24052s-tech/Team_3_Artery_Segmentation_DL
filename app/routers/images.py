import io
import mimetypes
import pickle
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pydicom
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydicom.errors import InvalidDicomError
from PIL import Image as PILImage, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.permissions import get_accessible_case, get_accessible_image
from app.database import get_db
from app.models import (
    AnalysisRun,
    Artifact,
    Image as ImageRecord,
    ImagingCase,
    ModelVersion,
    Report,
    Segment,
    StenosisFinding,
    User,
)
from app.baseline_segmentation import (
    CHECKPOINT_PATH as BASELINE_UNET_CHECKPOINT_PATH,
    DEVICE as BASELINE_UNET_DEVICE,
    predict_artery_segmentation,
    save_segmentation_artifacts,
    load_baseline_unet,
)
from app.preprocessing import preprocess_image
from app.routers.auth import get_current_user
from app.schemas import ImageResponse
from app.stenosis_inference import predict_stenosis
from app.stenosis_measurement import measure_stenosis

router = APIRouter(prefix="/images", tags=["Image Management"])

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
PREDICTION_DIR = PROCESSED_DIR / "predictions"
MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}


def inspect_image(extension: str, content: bytes) -> tuple[int, int]:
    try:
        if extension == ".dcm":
            dataset = pydicom.dcmread(io.BytesIO(content))

            if "PixelData" not in dataset:
                raise ValueError("DICOM has no pixel data")

            pixels = dataset.pixel_array
            if (
                pixels.ndim != 2
                or pixels.size == 0
                or not np.isfinite(pixels).all()
            ):
                raise ValueError(
                    "Only single-frame grayscale DICOM images are supported"
                )

            return int(dataset.Columns), int(dataset.Rows)

        expected_formats = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
        }

        with PILImage.open(io.BytesIO(content)) as image:
            if image.format != expected_formats[extension]:
                raise ValueError("File extension does not match image format")

            width, height = image.size
            image.verify()

        return width, height
    except (
        UnidentifiedImageError,
        InvalidDicomError,
        PILImage.DecompressionBombError,
        OSError,
        AttributeError,
        KeyError,
        RuntimeError,
    ) as exc:
        raise ValueError("Invalid or unsupported image file") from exc


def _resolve_upload_path(image: ImageRecord) -> Path:
    upload_root = UPLOAD_DIR.resolve()
    path = (BASE_DIR / image.storage_path).resolve()

    if upload_root not in path.parents:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image storage path is invalid",
        )

    return path


def _mark_analysis_failed(
    db: Session,
    run: AnalysisRun,
    error: Exception,
) -> None:
    run.status = "failed"
    run.completed_at = datetime.now(timezone.utc)
    run.error_message = f"Analysis failed ({type(error).__name__})"
    case = db.get(ImagingCase, run.case_id)
    if case is not None:
        case.analysis_status = "failed"

    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store analysis failure status",
        ) from exc


def _persist_analysis(
    db: Session,
    case: ImagingCase,
    image: ImageRecord,
    user: User,
    run: AnalysisRun,
    prediction: dict,
    measurements: dict,
) -> None:
    regions = prediction["regions"]
    measurements_by_region = {
        item["region_id"]: item for item in measurements["measurements"]
    }
    probabilities = [
        region["mean_probability"]
        for region in regions
        if region.get("mean_probability") is not None
    ]

    run.status = "completed"
    run.completed_at = datetime.now(timezone.utc)
    case.analysis_status = "completed"
    case.model_version = prediction["model"]
    case.updated_at = run.completed_at

    db.add(
        Segment(
            analysis_run_id=run.id,
            label="predicted_stenosis_mask",
            mask_path=prediction["predicted_mask_path"],
            confidence=float(np.mean(probabilities)) if probabilities else None,
        )
    )

    for region in regions:
        measurement = measurements_by_region.get(region["region_id"], {})
        narrowing = measurement.get("percentage_narrowing")
        db.add(
            StenosisFinding(
                analysis_run_id=run.id,
                segment_location=f"region-{region['region_id']}",
                stenosis_percentage=float(narrowing) if narrowing is not None else None,
                severity=measurement.get("severity", "Unavailable"),
                measurement_details={
                    "measurement_available": measurement.get(
                        "measurement_available", False
                    ),
                    "minimum_diameter_pixels": measurement.get(
                        "minimum_diameter_pixels"
                    ),
                    "reference_diameter_pixels": measurement.get(
                        "reference_diameter_pixels"
                    ),
                    "measurement_type": measurement.get("measurement_type"),
                    "area_pixels": region.get("area_pixels"),
                    "centroid": region.get("centroid"),
                    "bounding_box": region.get("bounding_box"),
                    "mean_probability": region.get("mean_probability"),
                },
            )
        )

    for artifact_type, path, mime_type in (
        (
            "stenosis_mask",
            prediction["predicted_mask_path"],
            "image/png",
        ),
        (
            "probability_map",
            prediction["probability_map_path"],
            "image/png",
        ),
        (
            "localization_overlay",
            prediction["localization_overlay_path"],
            "image/png",
        ),
    ):
        db.add(
            Artifact(
                analysis_run_id=run.id,
                image_id=image.id,
                artifact_type=artifact_type,
                storage_path=path,
                mime_type=mime_type,
            )
        )

    db.add(
        Report(
            case_id=case.id,
            analysis_run_id=run.id,
            created_by=user.id,
            summary=(
                f"Research analysis completed. {len(regions)} stenosis region(s) "
                f"detected; {measurements['successful_measurements']} "
                "measurement(s) available."
            ),
        )
    )


def _mark_segmentation_failed(
    db: Session,
    run: AnalysisRun,
    error: Exception,
) -> None:
    run.status = "failed"
    run.completed_at = datetime.now(timezone.utc)
    run.error_message = f"Segmentation failed ({type(error).__name__})"
    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store segmentation failure status",
        ) from exc


def _persist_segmentation(
    db: Session,
    image: ImageRecord,
    user: User,
    run: AnalysisRun,
    prediction: dict,
    artifact_paths: dict,
) -> None:
    completed_at = datetime.now(timezone.utc)
    run.status = "completed"
    run.completed_at = completed_at

    db.add(
        Segment(
            analysis_run_id=run.id,
            label="baseline_unet_artery_mask",
            mask_path=artifact_paths["artery_segmentation_mask"],
        )
    )
    for artifact_type, path in artifact_paths.items():
        db.add(
            Artifact(
                analysis_run_id=run.id,
                image_id=image.id,
                artifact_type=artifact_type,
                storage_path=path,
                mime_type="image/png",
            )
        )

    db.add(
        Report(
            case_id=run.case_id,
            analysis_run_id=run.id,
            created_by=user.id,
            summary=(
                "Baseline U-Net artery segmentation completed. "
                f"Checkpoint: {BASELINE_UNET_CHECKPOINT_PATH.name}; "
                f"epoch: {prediction['checkpoint_epoch']}; "
                f"validation Dice: {prediction['validation_dice']:.16f}; "
                f"threshold: {prediction['threshold']}; "
                f"image: {prediction['image_width']}x"
                f"{prediction['image_height']}; "
                f"foreground pixels: {prediction['mask_pixel_count']}."
            ),
        )
    )


@router.post(
    "",
    response_model=ImageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    extension = Path((file.filename or "").replace("\\", "/")).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supported formats: PNG, JPEG, DICOM",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 20 MB",
        )

    try:
        width, height = inspect_image(extension, content)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unsupported image file",
        ) from exc

    case = get_accessible_case(db, case_id, current_user)
    image_id = uuid4().hex
    stored_name = f"{image_id}{extension}"
    destination = UPLOAD_DIR / stored_name
    relative_path = destination.relative_to(BASE_DIR).as_posix()
    safe_filename = Path(
        (file.filename or "upload").replace("\\", "/")
    ).name
    image = ImageRecord(
        id=image_id,
        case_id=case.id,
        uploaded_by=current_user.id,
        original_filename=safe_filename[:255],
        storage_path=relative_path,
        file_format=extension.lstrip("."),
        file_size_bytes=len(content),
        width=width,
        height=height,
    )

    file_created = False
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stored_file:
            file_created = True
            stored_file.write(content)

        db.add(image)
        case.image_path = relative_path
        case.updated_at = datetime.now(timezone.utc)
        db.flush()
        db.refresh(image)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if file_created:
            destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not associate image with the case",
        ) from exc
    except (OSError, SQLAlchemyError) as exc:
        db.rollback()
        if file_created:
            destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store image",
        ) from exc

    return image


@router.post("/{image_id}/segment")
def segment_uploaded_image(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    image_record = get_accessible_image(db, image_id, current_user)
    case = get_accessible_case(db, image_record.case_id, current_user)
    image_path = _resolve_upload_path(image_record)
    if not image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Uploaded image file not found",
        )

    target_device = BASELINE_UNET_DEVICE
    try:
        load_baseline_unet(target_device)
    except torch.cuda.OutOfMemoryError as exc:
        if target_device.type != "cuda":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not load the Baseline U-Net model",
            ) from exc
        torch.cuda.empty_cache()
        target_device = torch.device("cpu")
        try:
            load_baseline_unet(target_device)
        except (
            EOFError,
            MemoryError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            ValueError,
        ) as cpu_exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not load the Baseline U-Net model",
            ) from cpu_exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Baseline U-Net checkpoint is unavailable",
        ) from exc
    except (
        EOFError,
        MemoryError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not load the Baseline U-Net model",
        ) from exc

    model_version = db.scalar(
        select(ModelVersion).where(
            ModelVersion.model_name == "Baseline U-Net artery segmentation",
            ModelVersion.version == "round2-best",
        )
    )
    if model_version is None:
        model_version = ModelVersion(
            model_name="Baseline U-Net artery segmentation",
            version="round2-best",
            architecture="Baseline U-Net",
            weights_path=BASELINE_UNET_CHECKPOINT_PATH.relative_to(
                BASE_DIR
            ).as_posix(),
            framework="PyTorch",
            metrics={
                "validation_dice": 0.5207999770392526,
                "best_epoch": 11,
                "state_dict_entries": 118,
            },
        )
        db.add(model_version)
        db.flush()

    started_at = datetime.now(timezone.utc)
    run = AnalysisRun(
        case_id=case.id,
        image_id=image_record.id,
        model_version_id=model_version.id,
        started_by=current_user.id,
        status="running",
        started_at=started_at,
    )
    try:
        db.add(run)
        db.flush()
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not start segmentation",
        ) from exc

    try:
        prediction = predict_artery_segmentation(
            image_path,
            device=target_device,
        )
    except FileNotFoundError as exc:
        _mark_segmentation_failed(db, run, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Uploaded image file not found",
        ) from exc
    except ValueError as exc:
        _mark_segmentation_failed(db, run, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not process image for segmentation",
        ) from exc
    except (OSError, RuntimeError) as exc:
        _mark_segmentation_failed(db, run, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not complete Baseline U-Net segmentation",
        ) from exc

    artifact_paths = {}
    try:
        artifact_paths = save_segmentation_artifacts(prediction, run.id)
        _persist_segmentation(
            db,
            image_record,
            current_user,
            run,
            prediction,
            artifact_paths,
        )
        db.commit()
    except (OSError, SQLAlchemyError) as exc:
        db.rollback()
        for relative_path in artifact_paths.values():
            (BASE_DIR / relative_path).unlink(missing_ok=True)
        _mark_segmentation_failed(db, run, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store segmentation results",
        ) from exc

    return {
        "analysis_run_id": run.id,
        "message": "Baseline U-Net artery segmentation completed successfully",
        "model": "Baseline U-Net artery segmentation",
        "model_version": "round2-best",
        "threshold": prediction["threshold"],
        "image_width": prediction["image_width"],
        "image_height": prediction["image_height"],
        "mask_pixel_count": prediction["mask_pixel_count"],
        "device": prediction["device"],
        "artifacts": artifact_paths,
    }


@router.get("", response_model=list[ImageResponse])
def list_case_images(
    case_id: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = get_accessible_case(db, case_id, current_user)
    return db.scalars(
        select(ImageRecord)
        .where(ImageRecord.case_id == case.id)
        .order_by(ImageRecord.uploaded_at.desc())
    ).all()


@router.get("/{image_id}", response_model=ImageResponse)
def get_image_metadata(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_accessible_image(db, image_id, current_user)


@router.get("/{image_id}/content")
def get_image_content(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from fastapi.responses import FileResponse

    image = get_accessible_image(db, image_id, current_user)
    path = _resolve_upload_path(image)

    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file not found",
        )

    media_type = mimetypes.guess_type(image.original_filename)[0]
    return FileResponse(
        path,
        media_type=media_type or "application/octet-stream",
        filename=image.original_filename,
    )


@router.post("/{image_id}/preprocess")
def preprocess_uploaded_image(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_accessible_image(db, image_id, current_user)

    try:
        return {
            "message": "Image preprocessed successfully",
            **preprocess_image(image_id),
        }
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Uploaded image file not found",
        ) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not preprocess image",
        ) from exc


@router.post("/{image_id}/predict")
def predict_uploaded_image(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    image_record = get_accessible_image(db, image_id, current_user)
    case = get_accessible_case(db, image_record.case_id, current_user)
    started_at = datetime.now(timezone.utc)
    run = AnalysisRun(
        case_id=case.id,
        image_id=image_record.id,
        started_by=current_user.id,
        status="running",
        started_at=started_at,
    )

    try:
        db.add(run)
        case.analysis_status = "running"
        db.flush()
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not start analysis",
        ) from exc

    try:
        prediction = predict_stenosis(image_id)
        normalized_path = PROCESSED_DIR / f"{image_id}_normalized.npy"
        image = np.load(normalized_path).astype(np.float32)
        mask_path = BASE_DIR / prediction["predicted_mask_path"]
        mask = np.asarray(PILImage.open(mask_path).convert("L"))
        measurement_result = measure_stenosis(image, mask)
    except FileNotFoundError as exc:
        _mark_analysis_failed(db, run, exc)
        if "Model checkpoint not found" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Stenosis model checkpoint is unavailable",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Uploaded or preprocessed image was not found",
        ) from exc
    except ValueError as exc:
        _mark_analysis_failed(db, run, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (OSError, RuntimeError) as exc:
        _mark_analysis_failed(db, run, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not complete stenosis analysis",
        ) from exc

    try:
        _persist_analysis(
            db,
            case,
            image_record,
            current_user,
            run,
            prediction,
            measurement_result,
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        _mark_analysis_failed(db, run, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store analysis results",
        ) from exc

    measurements_by_region = {
        item["region_id"]: item
        for item in measurement_result["measurements"]
    }
    for region in prediction["regions"]:
        region["measurement"] = measurements_by_region.get(
            region["region_id"],
            {
                "measurement_available": False,
                "severity": "Unavailable",
            },
        )

    return {
        "analysis_run_id": run.id,
        "message": "Stenosis analysis completed successfully",
        **prediction,
        "measurement_method": measurement_result["measurement_method"],
        "successful_measurements": measurement_result[
            "successful_measurements"
        ],
        "measurement_note": measurement_result["note"],
    }
