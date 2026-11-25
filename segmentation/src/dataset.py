
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


DATASET_ROOT = Path(r"C:\Users\Yamuna\Downloads\arcade\arcade")


class StenosisDataset(Dataset):
    def __init__(self, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        split_dir = DATASET_ROOT / "stenosis" / split
        self.image_dir = split_dir / "images"
        annotation_file = split_dir / "annotations" / f"{split}.json"

        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file}"
            )

        with annotation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.images = data["images"]
        self.annotations_by_image = {}

        for annotation in data["annotations"]:
            image_id = annotation["image_id"]
            self.annotations_by_image.setdefault(image_id, []).append(
                annotation
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_id = image_info["id"]
        image_path = self.image_dir / image_info["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        mask_image = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask_image)

        for annotation in self.annotations_by_image.get(image_id, []):
            for polygon in annotation.get("segmentation", []):
                if len(polygon) >= 6 and len(polygon) % 2 == 0:
                    points = [
                        (polygon[i], polygon[i + 1])
                        for i in range(0, len(polygon), 2)
                    ]
                    drawer.polygon(points, outline=1, fill=1)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask_image, dtype=np.float32)

        # Convert image to [channels, height, width].
        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )

        # Convert mask to [1, height, width].
        mask_tensor = torch.from_numpy(
            mask_array[None, :, :].copy()
        )

        return image_tensor, mask_tensor
