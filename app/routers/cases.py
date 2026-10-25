
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ImagingCase, Patient, User
from app.routers.auth import get_current_user
from app.schemas import CaseCreate, CaseResponse

router = APIRouter(tags=["Case Management"])


def check_patient_access(patient: Patient, user: User):
    if (
        user.role != "administrator"
        and patient.created_by != user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot access this patient record",
        )


def check_case_access(case: ImagingCase, user: User):
    if (
        user.role != "administrator"
        and case.created_by != user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot access this imaging case",
        )


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
    db.commit()
    db.refresh(case)

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
