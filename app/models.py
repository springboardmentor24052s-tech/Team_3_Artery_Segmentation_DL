
import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.database import Base


def generate_id():
    return str(uuid.uuid4())


# 1. USERS
class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_id)
    email = Column(String(254), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(150), nullable=False)
    role = Column(
        String(50),
        nullable=False,
        server_default=text("'medical_researcher'")
    )
    is_active = Column(Boolean, nullable=False, server_default=text("TRUE"))
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 2. PATIENTS
class Patient(Base):
    __tablename__ = "patients"

    id = Column(String(36), primary_key=True, default=generate_id)
    patient_code = Column(String(50), unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(30), nullable=False)
    medical_history = Column(Text)
    risk_factors = Column(Text)
    clinical_notes = Column(Text)
    created_by = Column(
        String(36),
        ForeignKey("users.id"),
        nullable=False
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 3. IMAGING CASES
class ImagingCase(Base):
    __tablename__ = "imaging_cases"

    id = Column(String(36), primary_key=True, default=generate_id)
    patient_id = Column(
        String(36),
        ForeignKey("patients.id"),
        nullable=False,
        index=True
    )
    created_by = Column(
        String(36),
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )
    modality = Column(String(30), nullable=False, server_default=text("'CT'"))
    scan_date = Column(Date, nullable=False)
    anatomical_region = Column(
        String(100),
        nullable=False,
        server_default=text("'Coronary arteries'")
    )
    image_path = Column(String(500))
    analysis_status = Column(
        String(30),
        nullable=False,
        server_default=text("'pending'")
    )
    model_version = Column(String(100))
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 4. IMAGES
class Image(Base):
    __tablename__ = "images"

    id = Column(String(36), primary_key=True, default=generate_id)
    case_id = Column(
        String(36),
        ForeignKey("imaging_cases.id"),
        nullable=False
    )
    uploaded_by = Column(
        String(36),
        ForeignKey("users.id"),
        nullable=False
    )
    original_filename = Column(String(255), nullable=False)
    storage_path = Column(String(500), nullable=False)
    file_format = Column(String(30))
    file_size_bytes = Column(BigInteger)
    width = Column(Integer)
    height = Column(Integer)
    uploaded_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 5. MODEL VERSIONS
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=generate_id)
    model_name = Column(String(150), nullable=False)
    version = Column(String(50), nullable=False)
    architecture = Column(String(150))
    weights_path = Column(String(500))
    framework = Column(String(50))
    metrics = Column(JSONB)
    is_active = Column(Boolean, nullable=False, server_default=text("TRUE"))
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 6. ANALYSIS RUNS
class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(String(36), primary_key=True, default=generate_id)
    case_id = Column(
        String(36),
        ForeignKey("imaging_cases.id"),
        nullable=False
    )
    image_id = Column(String(36), ForeignKey("images.id"))
    model_version_id = Column(
        String(36),
        ForeignKey("model_versions.id")
    )
    started_by = Column(String(36), ForeignKey("users.id"))
    status = Column(
        String(30),
        nullable=False,
        server_default=text("'pending'")
    )
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)


# 7. SEGMENTS
class Segment(Base):
    __tablename__ = "segments"

    id = Column(String(36), primary_key=True, default=generate_id)
    analysis_run_id = Column(
        String(36),
        ForeignKey("analysis_runs.id"),
        nullable=False
    )
    label = Column(String(100))
    artery_name = Column(String(150))
    mask_path = Column(String(500))
    confidence = Column(Numeric(6, 5))
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 8. STENOSIS FINDINGS
class StenosisFinding(Base):
    __tablename__ = "stenosis_findings"

    id = Column(String(36), primary_key=True, default=generate_id)
    analysis_run_id = Column(
        String(36),
        ForeignKey("analysis_runs.id"),
        nullable=False
    )
    artery_name = Column(String(150))
    segment_location = Column(String(150))
    stenosis_percentage = Column(Numeric(6, 2))
    severity = Column(String(50))
    measurement_details = Column(JSONB)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 9. ARTIFACTS
class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(String(36), primary_key=True, default=generate_id)
    analysis_run_id = Column(
        String(36),
        ForeignKey("analysis_runs.id"),
        nullable=False
    )
    image_id = Column(String(36), ForeignKey("images.id"))
    artifact_type = Column(String(100), nullable=False)
    storage_path = Column(String(500), nullable=False)
    mime_type = Column(String(100))
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 10. REPORTS
class Report(Base):
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=generate_id)
    case_id = Column(
        String(36),
        ForeignKey("imaging_cases.id"),
        nullable=False
    )
    analysis_run_id = Column(
        String(36),
        ForeignKey("analysis_runs.id")
    )
    created_by = Column(
        String(36),
        ForeignKey("users.id"),
        nullable=False
    )
    report_path = Column(String(500))
    summary = Column(Text)
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 11. ALERTS
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=generate_id)
    user_id = Column(
        String(36),
        ForeignKey("users.id"),
        nullable=False
    )
    case_id = Column(String(36), ForeignKey("imaging_cases.id"))
    alert_type = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, nullable=False, server_default=text("FALSE"))
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )


# 12. AUDIT LOGS
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_id)
    user_id = Column(String(36), ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    entity_type = Column(String(100))
    entity_id = Column(String(36))
    details = Column(JSONB)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )
