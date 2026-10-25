
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRole(str, Enum):
    RADIOLOGIST_CLINICIAN = "radiologist_clinician"
    MEDICAL_RESEARCHER = "medical_researcher"
    DATA_SCIENTIST_ML_ENGINEER = "data_scientist_ml_engineer"
    ADMINISTRATOR = "administrator"


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=2, max_length=150)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


from datetime import date
from pydantic import Field


class PatientCreate(BaseModel):
    patient_code: str = Field(min_length=3, max_length=50)
    age: int = Field(ge=0, le=120)
    gender: str = Field(min_length=1, max_length=30)
    medical_history: str | None = None
    risk_factors: str | None = None
    clinical_notes: str | None = None


class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    patient_code: str
    age: int
    gender: str
    medical_history: str | None
    risk_factors: str | None
    clinical_notes: str | None
    created_by: str


class CaseCreate(BaseModel):
    patient_id: str
    modality: str = Field(default="CT", max_length=30)
    scan_date: date
    anatomical_region: str = Field(
        default="Coronary arteries",
        min_length=1,
        max_length=100,
    )


class CaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    patient_id: str
    created_by: str
    modality: str
    scan_date: date
    anatomical_region: str
    analysis_status: str
    model_version: str | None

