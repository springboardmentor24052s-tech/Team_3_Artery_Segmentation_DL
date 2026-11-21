
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATASET = Path.home() / "Downloads" / "arcade" / "arcade"
OUTPUT = Path("outputs") / "masks_preview"
SAMPLES_PER_SPLIT = 3


def create_mask(width, height, annotations):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for annotation in annotations:
        for polygon in annotation.get("segmentation", []):
            if len(polygon) < 6:
                continue

            points = [
                (polygon[i], polygon[i + 1])
                for i in range(0, len(polygon), 2)
            ]
            draw.polygon(points, fill=255)

    return mask


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        folder = DATASET / "stenosis" / split
        json_path = folder / "annotations" / f"{split}.json"
        image_dir = folder / "images"

        with json_path.open(encoding="utf-8") as file:
            data = json.load(file)

        annotations_by_image = {}
        for annotation in data["annotations"]:
            annotations_by_image.setdefault(
                annotation["image_id"], []
            ).append(annotation)

        saved = 0

        for image_info in data["images"]:
            if saved >= SAMPLES_PER_SPLIT:
                break

            image_path = image_dir / image_info["file_name"]

            with Image.open(image_path) as original:
                original = original.convert("RGB")

            mask = create_mask(
                original.width,
                original.height,
                annotations_by_image.get(image_info["id"], [])
            )

            original_array = np.array(original).copy()
            mask_array = np.array(mask) > 0

            # Highlight annotated regions in red.
            original_array[mask_array] = (
                0.6 * original_array[mask_array]
                + 0.4 * np.array([255, 0, 0])
            ).astype(np.uint8)

            overlay = Image.fromarray(original_array)

            filename = Path(image_info["file_name"]).stem
            output_path = OUTPUT / f"{split}_{filename}_overlay.png"
            overlay.save(output_path)

            print(f"Saved: {output_path}")
            saved += 1

    print(f"\nOverlays saved in: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
