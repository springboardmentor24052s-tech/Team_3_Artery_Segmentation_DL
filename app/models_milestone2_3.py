"""
Additive database models for Milestones 2 and 3.
These models reuse the existing Base, User, and ImagingCase models.
They intentionally use distinct table names to avoid silently colliding with
legacy tables that may already exist in artery_stenosis_db.
"""
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, Boolean
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app import models


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MLModelVersion(Base):
    __tablename__ = "ml_model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)  # segmentation / stenosis_detection
    architecture: Mapped[str | None] = mapped_column(String(150))
    framework: Mapped[str | None] = mapped_column(String(80))
    model_uri: Mapped[str | None] = mapped_column(String(700))
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class MLAnalysisRun(Base):
    __tablename__ = "ml_analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("imaging_cases.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("ml_model_versions.id", ondelete="SET NULL"))
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"))
    pipeline_stage: Mapped[str] = mapped_column(String(50), nullable=False)  # segmentation / stenosis_detection
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    parameters: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    mask_uri: Mapped[str | None] = mapped_column(String(700))
    overlay_uri: Mapped[str | None] = mapped_column(String(700))
    mask_format: Mapped[str | None] = mapped_column(String(40))
    dice_score: Mapped[float | None] = mapped_column(Float)
    iou_score: Mapped[float | None] = mapped_column(Float)
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class VesselSegment(Base):
    __tablename__ = "vessel_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    segment_label: Mapped[str | None] = mapped_column(String(100))
    anatomy: Mapped[str | None] = mapped_column(String(100))
    geometry_uri: Mapped[str | None] = mapped_column(String(700))
    coordinates: Mapped[dict | list | None] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class VesselCenterline(Base):
    __tablename__ = "vessel_centerlines"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    vessel_segment_id: Mapped[str] = mapped_column(String(36), ForeignKey("vessel_segments.id", ondelete="CASCADE"), nullable=False, index=True)
    points: Mapped[list | dict | None] = mapped_column(JSON)
    coordinate_system: Mapped[str | None] = mapped_column(String(80))
    data_uri: Mapped[str | None] = mapped_column(String(700))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class VesselMeasurement(Base):
    __tablename__ = "vessel_measurements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    vessel_segment_id: Mapped[str] = mapped_column(String(36), ForeignKey("vessel_segments.id", ondelete="CASCADE"), nullable=False, index=True)
    measurement_type: Mapped[str] = mapped_column(String(60), nullable=False)  # diameter / length / area
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    position: Mapped[dict | list | None] = mapped_column(JSON)
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class StenosisCandidate(Base):
    __tablename__ = "stenosis_candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    vessel_segment_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("vessel_segments.id", ondelete="SET NULL"))
    start_position: Mapped[dict | list | None] = mapped_column(JSON)
    end_position: Mapped[dict | list | None] = mapped_column(JSON)
    region_uri: Mapped[str | None] = mapped_column(String(700))
    minimum_diameter_mm: Mapped[float | None] = mapped_column(Float)
    reference_diameter_mm: Mapped[float | None] = mapped_column(Float)
    narrowing_percent: Mapped[float | None] = mapped_column(Float)
    candidate_score: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending_review")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class StenosisLocalization(Base):
    __tablename__ = "stenosis_localizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    candidate_id: Mapped[str] = mapped_column(String(36), ForeignKey("stenosis_candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    slice_index: Mapped[int | None] = mapped_column(Integer)
    bounding_box: Mapped[dict | list | None] = mapped_column(JSON)
    contour_points: Mapped[list | dict | None] = mapped_column(JSON)
    coordinate_system: Mapped[str | None] = mapped_column(String(80))
    overlay_uri: Mapped[str | None] = mapped_column(String(700))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class StenosisClassification(Base):
    __tablename__ = "stenosis_classifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    candidate_id: Mapped[str] = mapped_column(String(36), ForeignKey("stenosis_candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    predicted_class: Mapped[str] = mapped_column(String(60), nullable=False)  # normal / stenosis / uncertain
    severity: Mapped[str | None] = mapped_column(String(40))
    probability: Mapped[float | None] = mapped_column(Float)
    narrowing_percent: Mapped[float | None] = mapped_column(Float)
    model_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("ml_model_versions.id", ondelete="SET NULL"))
    reviewed_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"))
    review_status: Mapped[str] = mapped_column(String(30), nullable=False, default="unreviewed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ModelEvaluation(Base):
    __tablename__ = "ml_model_evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_model_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_name: Mapped[str | None] = mapped_column(String(150))
    split_name: Mapped[str | None] = mapped_column(String(40))
    sample_count: Mapped[int | None] = mapped_column(Integer)
    dice_score: Mapped[float | None] = mapped_column(Float)
    iou_score: Mapped[float | None] = mapped_column(Float)
    precision_score: Mapped[float | None] = mapped_column(Float)
    recall_score: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    auc_score: Mapped[float | None] = mapped_column(Float)
    metrics_json: Mapped[dict | None] = mapped_column(JSON)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AnalysisArtifact(Base):
    __tablename__ = "ml_analysis_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    artifact_type: Mapped[str] = mapped_column(String(60), nullable=False)
    file_uri: Mapped[str] = mapped_column(String(700), nullable=False)
    file_format: Mapped[str | None] = mapped_column(String(40))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AnalysisReport(Base):
    __tablename__ = "ml_analysis_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    report_type: Mapped[str] = mapped_column(String(60), nullable=False)
    file_uri: Mapped[str | None] = mapped_column(String(700))
    report_data: Mapped[dict | None] = mapped_column(JSON)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


Index("ix_stenosis_candidates_status", StenosisCandidate.status)
Index("ix_ml_analysis_runs_stage_status", MLAnalysisRun.pipeline_stage, MLAnalysisRun.status)
