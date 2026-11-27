
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.permissions import (
    check_case_access,
    check_patient_access,
    get_accessible_case,
)
from app.database import get_db
from app.models import (
    AnalysisRun,
    Artifact,
    ImagingCase,
    Patient,
    Report,
    Segment,
    StenosisFinding,
    User,
)
from app.routers.auth import get_current_user
from app.schemas import CaseCreate, CaseResponse

router = APIRouter(tags=["Case Management"])


@router.post(
    "/cases",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_case(
    data: CaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = db.get(Patient, data.patient_id)

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )

    check_patient_access(patient, current_user)

    case = ImagingCase(
        patient_id=patient.id,
        created_by=current_user.id,
        modality=data.modality,
        scan_date=data.scan_date,
        anatomical_region=data.anatomical_region,
    )

    db.add(case)
    try:
        db.flush()
        db.refresh(case)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not create imaging case because a related record conflicts",
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create imaging case",
        ) from exc

    return case


@router.get("/cases", response_model=list[CaseResponse])
def list_cases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = select(ImagingCase)

    if current_user.role != "administrator":
        query = query.where(
            ImagingCase.created_by == current_user.id
        )

    query = query.order_by(ImagingCase.created_at.desc())

    return db.scalars(query).all()


@router.get(
    "/cases/{case_id}",
    response_model=CaseResponse,
)
def get_case(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(ImagingCase, case_id)

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Imaging case not found",
        )

    check_case_access(case, current_user)
    return case


@router.get(
    "/patients/{patient_id}/cases",
    response_model=list[CaseResponse],
)
def list_patient_cases(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = db.get(Patient, patient_id)

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )

    check_patient_access(patient, current_user)

    query = select(ImagingCase).where(
        ImagingCase.patient_id == patient_id
    )

    if current_user.role != "administrator":
        query = query.where(
            ImagingCase.created_by == current_user.id
        )

    query = query.order_by(ImagingCase.created_at.desc())

    return db.scalars(query).all()


def analysis_run_payload(db: Session, run: AnalysisRun) -> dict:
    segments = db.scalars(
        select(Segment).where(Segment.analysis_run_id == run.id)
    ).all()
    findings = db.scalars(
        select(StenosisFinding).where(StenosisFinding.analysis_run_id == run.id)
    ).all()
    artifacts = db.scalars(
        select(Artifact).where(Artifact.analysis_run_id == run.id)
    ).all()
    reports = db.scalars(
        select(Report).where(Report.analysis_run_id == run.id)
    ).all()

    return {
        "id": run.id,
        "case_id": run.case_id,
        "image_id": run.image_id,
        "model_version_id": run.model_version_id,
        "started_by": run.started_by,
        "status": run.status,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "error_message": run.error_message,
        "segments": [
            {
                "id": item.id,
                "label": item.label,
                "artery_name": item.artery_name,
                "mask_path": item.mask_path,
                "confidence": item.confidence,
                "created_at": item.created_at,
            }
            for item in segments
        ],
        "stenosis_findings": [
            {
                "id": item.id,
                "artery_name": item.artery_name,
                "segment_location": item.segment_location,
                "stenosis_percentage": item.stenosis_percentage,
                "severity": item.severity,
                "measurement_details": item.measurement_details,
                "created_at": item.created_at,
            }
            for item in findings
        ],
        "artifacts": [
            {
                "id": item.id,
                "image_id": item.image_id,
                "artifact_type": item.artifact_type,
                "storage_path": item.storage_path,
                "mime_type": item.mime_type,
                "created_at": item.created_at,
            }
            for item in artifacts
        ],
        "reports": [
            {
                "id": item.id,
                "report_path": item.report_path,
                "summary": item.summary,
                "generated_at": item.generated_at,
            }
            for item in reports
        ],
    }


@router.get(
    "/cases/{case_id}/analysis-runs",
)
def list_case_analysis_runs(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(ImagingCase, case_id)

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Imaging case not found",
        )

    check_case_access(case, current_user)
    runs = db.scalars(
        select(AnalysisRun)
        .where(AnalysisRun.case_id == case_id)
        .order_by(AnalysisRun.started_at.desc())
    ).all()
    return [analysis_run_payload(db, run) for run in runs]


@router.get(
    "/analysis-runs/{analysis_run_id}",
)
def get_analysis_run(
    analysis_run_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = db.get(AnalysisRun, analysis_run_id)

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis run not found",
        )

    get_accessible_case(db, run.case_id, current_user)
    return analysis_run_payload(db, run)


@router.get("/analysis-runs/{analysis_run_id}/artifacts/{artifact_type}")
def get_analysis_artifact(
    analysis_run_id: str,
    artifact_type: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = db.get(AnalysisRun, analysis_run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis run not found",
        )

    get_accessible_case(db, run.case_id, current_user)
    artifact = db.scalar(
        select(Artifact).where(
            Artifact.analysis_run_id == run.id,
            Artifact.artifact_type == artifact_type,
        )
    )
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis artifact not found",
        )

    base_dir = Path(__file__).resolve().parents[2]
    predictions_dir = (base_dir / "processed" / "predictions").resolve()
    path = (base_dir / artifact.storage_path).resolve()
    if predictions_dir not in path.parents or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis artifact file not found",
        )

    return FileResponse(
        path,
        media_type=artifact.mime_type or "application/octet-stream",
        filename=path.name,
    )


@router.get("/reports/{report_id}/download")
def download_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )

    get_accessible_case(db, report.case_id, current_user)
    if report.analysis_run_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report has no associated analysis",
        )
    run = db.get(AnalysisRun, report.analysis_run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report analysis not found",
        )

    payload = {
        "report_id": report.id,
        "case_id": report.case_id,
        "analysis_run": analysis_run_payload(db, run),
        "summary": report.summary,
        "generated_at": report.generated_at,
        "disclaimer": (
            "Research output only. Experimental image-based estimates are "
            "not clinically validated and require qualified review."
        ),
    }
    return JSONResponse(
        content=jsonable_encoder(payload),
        headers={
            "Content-Disposition": (
                f'attachment; filename="stenosis-report-{report.id}.json"'
            )
        },
    )
