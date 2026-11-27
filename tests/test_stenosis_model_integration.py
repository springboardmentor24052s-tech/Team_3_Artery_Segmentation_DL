import uuid
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.stenosis_inference import BASE_DIR, PREDICTION_DIR, load_stenosis_model, predict_stenosis
from app.stenosis_measurement import measure_stenosis


class StenosisModelIntegrationTests(unittest.TestCase):
    def test_checkpoint_loads_strictly_on_cpu(self):
        model = load_stenosis_model(device="cpu")

        self.assertFalse(model.training)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))

    def test_actual_sample_inference_and_measurements_run_on_cpu(self):
        sample_path = (
            BASE_DIR
            / "processed"
            / "211657f025944712b7934d8988be26be_normalized.npy"
        )
        self.assertTrue(sample_path.is_file())
        sample = np.load(sample_path)
        image_id = f"cpu_test_{uuid.uuid4().hex}"
        normalized_path = BASE_DIR / "processed" / f"{image_id}_normalized.npy"
        mask_path = PREDICTION_DIR / f"{image_id}_stenosis_mask.png"
        probability_path = PREDICTION_DIR / f"{image_id}_probability.png"
        overlay_path = PREDICTION_DIR / f"{image_id}_stenosis_overlay.png"

        try:
            np.save(normalized_path, sample)
            prediction = predict_stenosis(image_id, device="cpu")
            mask = np.asarray(Image.open(mask_path).convert("L"))
            with Image.open(overlay_path) as overlay:
                self.assertEqual(overlay.size, (512, 512))
                self.assertEqual(overlay.mode, "RGB")
            measurements = measure_stenosis(sample, mask)

            self.assertEqual((prediction["image_height"], prediction["image_width"]), (512, 512))
            self.assertEqual(
                prediction["predicted_stenosis_detected"],
                prediction["candidate_region_count"] > 0,
            )
            self.assertEqual(
                measurements["successful_measurements"]
                + sum(not item["measurement_available"] for item in measurements["measurements"]),
                measurements["region_count"],
            )
            for measurement in measurements["measurements"]:
                if measurement["measurement_available"]:
                    self.assertGreaterEqual(measurement["percentage_narrowing"], 0)
                    self.assertLessEqual(measurement["percentage_narrowing"], 100)
        finally:
            for path in (normalized_path, mask_path, probability_path, overlay_path):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
