from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Patient, User
from app.routers.auth import get_current_user
from app.schemas import PatientCreate, PatientResponse

router = APIRouter(prefix="/patients", tags=["Patient Management"])


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_patient(
    data: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = Patient(
        patient_code=data.patient_code,
        age=data.age,
        gender=data.gender,
        medical_history=data.medical_history,
        risk_factors=data.risk_factors,
        clinical_notes=data.clinical_notes,
        created_by=current_user.id,
    )

    db.add(patient)

    try:
        db.commit()
        db.refresh(patient)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Patient code already exists",
        )

    return patient


@router.get("", response_model=list[PatientResponse])
def list_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = select(Patient)

    # Administrators can see all records.
    # Other users see only records they created.
    if current_user.role != "administrator":
        query = query.where(Patient.created_by == current_user.id)

    return db.scalars(query.order_by(Patient.created_at.desc())).all()


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
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

    if (
        current_user.role != "administrator"
        and patient.created_by != current_user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot access this patient record",
        )

    return patient