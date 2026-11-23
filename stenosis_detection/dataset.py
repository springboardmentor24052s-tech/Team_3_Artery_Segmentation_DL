
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


DATA_ROOT = Path("datasets/arcade/arcade/stenosis")

# Exclude images duplicated between train and validation.
EXCLUDED_TRAIN_IMAGES = {
    "218.png",
    "799.png",
    "864.png",
    "916.png"
}


class StenosisDataset(Dataset):

    def __init__(self, split="train"):

        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid dataset split")

        self.split = split
        self.image_dir = DATA_ROOT / split / "images"

        annotation_path = (
            DATA_ROOT / split / "annotations" / f"{split}.json"
        )

        with open(annotation_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]

        # Exclude duplicated training images.
        if split == "train":
            self.images = [
                image_info
                for image_info in self.images
                if Path(image_info["file_name"]).name
                not in EXCLUDED_TRAIN_IMAGES
            ]

        self.annotations_by_image = defaultdict(list)

        for ann in data["annotations"]:
            if ann["category_id"] == 26:
                self.annotations_by_image[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_info = self.images[index]

        image_id = image_info["id"]
        filename = image_info["file_name"]

        # Load original angiography image.
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("L")

        # Preserve original image preprocessing (0-1 scaling).
        image_array = (
            np.asarray(image, dtype=np.float32).copy() / 255.0
        )

        # Create an empty binary mask.
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Convert stenosis polygons into mask regions.
        annotations = self.annotations_by_image[image_id]

        for ann in annotations:
            for polygon in ann["segmentation"]:

                points = [
                    (polygon[i], polygon[i + 1])
                    for i in range(0, len(polygon), 2)
                ]

                draw.polygon(points, fill=255)

        # Convert mask into a NumPy array.
        mask_array = (
            np.asarray(mask, dtype=np.float32).copy() / 255.0
        )

        # Convert arrays into PyTorch tensors.
        image_tensor = torch.from_numpy(
            image_array
        ).unsqueeze(0)

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0)

        return image_tensor, mask_tensor
