import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from app.baseline_segmentation import (
    CHECKPOINT_PATH,
    DEVICE,
    THRESHOLD,
    load_baseline_unet,
    predict_artery_segmentation,
    save_segmentation_artifacts,
)
from app.routers.images import _persist_segmentation
from app.models import AnalysisRun, Image as ImageRecord, ImagingCase, User


BASE_DIR = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self):
        self.added = []

    def add(self, record):
        self.added.append(record)


class BaselineSegmentationTests(unittest.TestCase):
    def test_exact_checkpoint_loads_strictly_on_cpu(self):
        self.assertTrue(CHECKPOINT_PATH.is_file())
        model = load_baseline_unet(device="cpu")
        self.assertFalse(model.training)
        self.assertEqual(len(model.state_dict()), 118)
        self.assertTrue(
            all(parameter.device.type == "cpu" for parameter in model.parameters())
        )

    def test_real_sample_inference_preserves_dimensions_and_artifact_format(self):
        sample_path = BASE_DIR / "backend" / "sample_angiogram.png"
        image_id = uuid.uuid4().hex
        created_paths = []
        try:
            prediction = predict_artery_segmentation(sample_path, device="cpu")
            self.assertEqual(prediction["probability_map"].shape, (512, 512))
            self.assertEqual(prediction["binary_mask"].shape, (512, 512))
            self.assertTrue(np.isfinite(prediction["probability_map"]).all())
            self.assertTrue(
                np.array_equal(
                    prediction["binary_mask"],
                    prediction["probability_map"] >= THRESHOLD,
                )
            )
            self.assertGreater(prediction["mask_pixel_count"], 0)

            paths = save_segmentation_artifacts(prediction, image_id)
            created_paths.extend(BASE_DIR / path for path in paths.values())
            self.assertEqual(len(paths), 4)
            for path in created_paths:
                with Image.open(path) as output:
                    self.assertEqual(output.size, (512, 512))
                    self.assertEqual(output.format, "PNG")
        finally:
            for path in created_paths:
                path.unlink(missing_ok=True)

    def test_invalid_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "decode"):
            predict_artery_segmentation(BASE_DIR / "README_DATABASE.md", device="cpu")

    def test_inference_crops_padding_to_original_odd_dimensions(self):
        with TemporaryDirectory() as directory:
            sample_path = Path(directory) / "odd-dimensions.png"
            Image.new("RGB", (37, 29), color=(96, 128, 160)).save(sample_path)

            prediction = predict_artery_segmentation(sample_path, device="cpu")

        self.assertEqual(prediction["probability_map"].shape, (29, 37))
        self.assertEqual(prediction["binary_mask"].shape, (29, 37))

    def test_segmentation_result_persistence_uses_existing_analysis_tables(self):
        user = User(id="user-id", email="user@example.test", password_hash="x", full_name="User")
        case = ImagingCase(id="case-id", patient_id="patient-id", created_by=user.id)
        image = ImageRecord(
            id="image-id",
            case_id=case.id,
            uploaded_by=user.id,
            original_filename="angiogram.png",
            storage_path="uploads/image-id.png",
        )
        run = AnalysisRun(
            id="run-id",
            case_id=case.id,
            image_id=image.id,
            started_by=user.id,
            status="running",
        )
        prediction = {
            "checkpoint_epoch": 11,
            "validation_dice": 0.5207999770392526,
            "threshold": THRESHOLD,
            "image_width": 32,
            "image_height": 24,
            "mask_pixel_count": 17,
        }
        artifact_paths = {
            "artery_input_preview": "processed/predictions/input.png",
            "artery_segmentation_mask": "processed/predictions/mask.png",
            "artery_probability_map": "processed/predictions/probability.png",
            "artery_segmentation_overlay": "processed/predictions/overlay.png",
        }
        db = FakeSession()

        _persist_segmentation(db, image, user, run, prediction, artifact_paths)

        self.assertEqual(run.status, "completed")
        self.assertEqual(
            sum(record.__class__.__name__ == "Segment" for record in db.added),
            1,
        )
        self.assertEqual(
            sum(record.__class__.__name__ == "Artifact" for record in db.added),
            4,
        )
        report = next(record for record in db.added if record.__class__.__name__ == "Report")
        self.assertIn("Baseline U-Net artery segmentation", report.summary)
        self.assertIn("foreground pixels: 17", report.summary)


if __name__ == "__main__":
    unittest.main()
