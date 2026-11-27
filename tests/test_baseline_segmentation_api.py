import io
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import Artifact, Image as ImageRecord


BASE_DIR = Path(__file__).resolve().parents[1]


@compiles(JSONB, "sqlite")
def compile_jsonb_for_sqlite(type_, compiler, **kwargs):
    return "JSON"


class BaselineSegmentationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=cls.engine)
        cls.session_factory = sessionmaker(
            bind=cls.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=cls.engine)
        cls.engine.dispose()

    def setUp(self):
        def override_get_db():
            with self.session_factory() as db:
                yield db

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)
        self.created_paths = []

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()
        for path in self.created_paths:
            path.unlink(missing_ok=True)

    def _register_and_login(self, email):
        registration = self.client.post(
            "/auth/register",
            json={
                "email": email,
                "password": "valid-test-password",
                "full_name": "Segmentation Test",
            },
        )
        self.assertEqual(registration.status_code, 201, registration.text)
        login = self.client.post(
            "/auth/login",
            json={"email": email, "password": "valid-test-password"},
        )
        self.assertEqual(login.status_code, 200, login.text)
        return {"Authorization": f"Bearer {login.json()['access_token']}"}

    def test_authenticated_segmentation_upload_persistence_and_authorization(self):
        unique_id = uuid.uuid4().hex
        owner_headers = self._register_and_login(
            f"owner-{unique_id}@example.com"
        )
        self.assertEqual(
            self.client.get("/auth/me", headers=owner_headers).status_code,
            200,
        )

        patient_response = self.client.post(
            "/patients",
            headers=owner_headers,
            json={
                "patient_code": f"PAT-{unique_id[:12]}",
                "age": 56,
                "gender": "unknown",
            },
        )
        self.assertEqual(patient_response.status_code, 201, patient_response.text)
        case_response = self.client.post(
            "/cases",
            headers=owner_headers,
            json={
                "patient_id": patient_response.json()["id"],
                "modality": "Angiography",
                "scan_date": "2026-11-27",
                "anatomical_region": "Coronary arteries",
            },
        )
        self.assertEqual(case_response.status_code, 201, case_response.text)
        case_id = case_response.json()["id"]

        sample_path = BASE_DIR / "backend" / "sample_angiogram.png"
        upload_response = self.client.post(
            "/images",
            headers=owner_headers,
            data={"case_id": case_id},
            files={
                "file": (
                    "sample_angiogram.png",
                    sample_path.read_bytes(),
                    "image/png",
                )
            },
        )
        self.assertEqual(upload_response.status_code, 201, upload_response.text)
        image_id = upload_response.json()["id"]
        with self.session_factory() as db:
            stored_image = db.get(ImageRecord, image_id)
            self.created_paths.append(BASE_DIR / stored_image.storage_path)

        segmentation_response = self.client.post(
            f"/images/{image_id}/segment",
            headers=owner_headers,
            json={},
        )
        self.assertEqual(
            segmentation_response.status_code,
            200,
            segmentation_response.text,
        )
        prediction = segmentation_response.json()
        self.assertEqual(prediction["model_version"], "round2-best")
        self.assertEqual(prediction["image_width"], 512)
        self.assertEqual(prediction["image_height"], 512)
        self.assertGreater(prediction["mask_pixel_count"], 0)
        run_id = prediction["analysis_run_id"]

        runs_response = self.client.get(
            f"/cases/{case_id}/analysis-runs",
            headers=owner_headers,
        )
        self.assertEqual(runs_response.status_code, 200)
        run = next(item for item in runs_response.json() if item["id"] == run_id)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(
            [segment["label"] for segment in run["segments"]],
            ["baseline_unet_artery_mask"],
        )
        self.assertEqual(
            {artifact["artifact_type"] for artifact in run["artifacts"]},
            {
                "artery_input_preview",
                "artery_segmentation_mask",
                "artery_probability_map",
                "artery_segmentation_overlay",
            },
        )
        self.created_paths.extend(
            BASE_DIR / artifact["storage_path"] for artifact in run["artifacts"]
        )

        retrieved = self.client.get(
            f"/analysis-runs/{run_id}",
            headers=owner_headers,
        )
        self.assertEqual(retrieved.status_code, 200)
        mask_response = self.client.get(
            f"/analysis-runs/{run_id}/artifacts/artery_segmentation_mask",
            headers=owner_headers,
        )
        self.assertEqual(mask_response.status_code, 200)
        with PILImage.open(io.BytesIO(mask_response.content)) as mask:
            self.assertEqual(mask.size, (512, 512))
            self.assertGreater(np.asarray(mask).sum(), 0)

        preprocessing_response = self.client.post(
            f"/images/{image_id}/preprocess",
            headers=owner_headers,
        )
        self.assertEqual(
            preprocessing_response.status_code,
            200,
            preprocessing_response.text,
        )
        self.created_paths.extend(
            BASE_DIR / preprocessing_response.json()[key]
            for key in ("processed_path", "normalized_array_path")
        )
        stenosis_response = self.client.post(
            f"/images/{image_id}/predict",
            headers=owner_headers,
        )
        self.assertEqual(stenosis_response.status_code, 200, stenosis_response.text)
        stenosis_run_id = stenosis_response.json()["analysis_run_id"]
        stenosis_run_response = self.client.get(
            f"/analysis-runs/{stenosis_run_id}",
            headers=owner_headers,
        )
        self.assertEqual(stenosis_run_response.status_code, 200)
        stenosis_run = stenosis_run_response.json()
        self.assertEqual(stenosis_run["status"], "completed")
        stenosis_artifacts = {
            artifact["artifact_type"]: artifact
            for artifact in stenosis_run["artifacts"]
        }
        self.assertIn("stenosis_mask", stenosis_artifacts)
        self.created_paths.extend(
            BASE_DIR / artifact["storage_path"]
            for artifact in stenosis_artifacts.values()
        )
        report_id = stenosis_run["reports"][0]["id"]
        report_response = self.client.get(
            f"/reports/{report_id}/download",
            headers=owner_headers,
        )
        self.assertEqual(report_response.status_code, 200)

        other_headers = self._register_and_login(
            f"other-{unique_id}@example.com"
        )
        forbidden = self.client.post(
            f"/images/{image_id}/segment",
            headers=other_headers,
        )
        self.assertEqual(forbidden.status_code, 403)

        invalid_upload = self.client.post(
            "/images",
            headers=owner_headers,
            data={"case_id": case_id},
            files={"file": ("invalid.png", b"not an image", "image/png")},
        )
        self.assertEqual(invalid_upload.status_code, 400)

        with patch(
            "app.routers.images.predict_artery_segmentation",
            side_effect=ValueError("test inference failure"),
        ):
            failed_response = self.client.post(
                f"/images/{image_id}/segment",
                headers=owner_headers,
            )
        self.assertEqual(failed_response.status_code, 400)
        updated_runs = self.client.get(
            f"/cases/{case_id}/analysis-runs",
            headers=owner_headers,
        ).json()
        failed_run = next(
            item for item in updated_runs if item["status"] == "failed"
        )
        self.assertEqual(failed_run["segments"], [])
        self.assertEqual(failed_run["artifacts"], [])
        self.assertEqual(
            self.client.get(
                f"/cases/{case_id}",
                headers=owner_headers,
            ).json()["analysis_status"],
            "completed",
        )


if __name__ == "__main__":
    unittest.main()
