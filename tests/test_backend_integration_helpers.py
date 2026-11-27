import io
import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("JWT_SECRET_KEY", "backend-integration-test-secret")

from fastapi import HTTPException
from PIL import Image as PILImage

from app.core.permissions import get_accessible_image
from app.models import (
    AnalysisRun,
    Artifact,
    Image as ImageRecord,
    ImagingCase,
    Report,
    Segment,
    StenosisFinding,
    User,
)
from app.routers.images import inspect_image, _persist_analysis


class FakeSession:
    def __init__(self, *records):
        self.records = {
            (type(record), record.id): record
            for record in records
        }
        self.added = []

    def get(self, model, record_id):
        return self.records.get((model, record_id))

    def add(self, record):
        self.added.append(record)


class BackendIntegrationHelperTests(unittest.TestCase):
    def setUp(self):
        self.owner = User(
            id="owner-id",
            email="owner@example.test",
            password_hash="unused",
            full_name="Owner",
            role="medical_researcher",
            is_active=True,
        )
        self.other_user = User(
            id="other-id",
            email="other@example.test",
            password_hash="unused",
            full_name="Other",
            role="medical_researcher",
            is_active=True,
        )
        self.case = ImagingCase(
            id="case-id",
            patient_id="patient-id",
            created_by=self.owner.id,
            modality="CT",
        )
        self.image = ImageRecord(
            id="image-id",
            case_id=self.case.id,
            uploaded_by=self.owner.id,
            original_filename="angiogram.png",
            storage_path="uploads/image-id.png",
        )

    def test_image_validation_checks_extension_against_decoded_format(self):
        buffer = io.BytesIO()
        PILImage.new("L", (7, 5), color=128).save(buffer, format="PNG")

        self.assertEqual(inspect_image(".png", buffer.getvalue()), (7, 5))

        with self.assertRaisesRegex(ValueError, "extension"):
            inspect_image(".jpg", buffer.getvalue())

    def test_image_access_is_scoped_to_case_owner(self):
        db = FakeSession(self.image, self.case)

        self.assertIs(
            get_accessible_image(db, self.image.id, self.owner),
            self.image,
        )
        with self.assertRaises(HTTPException) as raised:
            get_accessible_image(db, self.image.id, self.other_user)
        self.assertEqual(raised.exception.status_code, 403)

    def test_admin_can_access_other_users_image(self):
        administrator = User(
            id="admin-id",
            email="admin@example.test",
            password_hash="unused",
            full_name="Admin",
            role="administrator",
            is_active=True,
        )
        db = FakeSession(self.image, self.case)

        self.assertIs(
            get_accessible_image(db, self.image.id, administrator),
            self.image,
        )

    def test_analysis_persistence_links_results_to_existing_records(self):
        db = FakeSession()
        run = AnalysisRun(
            id="run-id",
            case_id=self.case.id,
            image_id=self.image.id,
            started_by=self.owner.id,
            status="running",
        )
        prediction = {
            "model": "test-model",
            "predicted_mask_path": "processed/mask.png",
            "probability_map_path": "processed/probability.png",
            "localization_overlay_path": "processed/overlay.png",
            "regions": [
                {
                    "region_id": 1,
                    "mean_probability": 0.8,
                    "area_pixels": 12,
                    "centroid": {"x": 4, "y": 2},
                    "bounding_box": {"x_min": 1, "x_max": 6},
                }
            ],
        }
        measurements = {
            "measurements": [
                {
                    "region_id": 1,
                    "measurement_available": True,
                    "percentage_narrowing": 35.5,
                    "severity": "Mild",
                    "minimum_diameter_pixels": 2.0,
                    "reference_diameter_pixels": 3.1,
                    "measurement_type": "pixel estimate",
                }
            ],
            "successful_measurements": 1,
        }

        _persist_analysis(
            db,
            self.case,
            self.image,
            self.owner,
            run,
            prediction,
            measurements,
        )

        self.assertEqual(run.status, "completed")
        self.assertEqual(self.case.analysis_status, "completed")
        self.assertEqual(len(db.added), 6)
        self.assertTrue(all(record.analysis_run_id == run.id for record in db.added))
        finding = next(
            record for record in db.added
            if isinstance(record, StenosisFinding)
        )
        self.assertEqual(float(finding.stenosis_percentage), 35.5)
        self.assertTrue(
            any(isinstance(record, Segment) for record in db.added)
        )
        self.assertEqual(sum(isinstance(record, Artifact) for record in db.added), 3)
        self.assertTrue(any(isinstance(record, Report) for record in db.added))


if __name__ == "__main__":
    unittest.main()
