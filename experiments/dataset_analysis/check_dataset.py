
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

root = Path("datasets/arcade/arcade/stenosis")

for split in ["train", "val", "test"]:

    folder = root / split
    image_dir = folder / "images"
    json_path = folder / "annotations" / f"{split}.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    annotations_by_image = defaultdict(list)

    for ann in annotations:
        annotations_by_image[ann["image_id"]].append(ann)

    empty_images = []
    dimension_counts = Counter()
    invalid_polygons = 0
    out_of_bounds = 0
    invalid_boxes = 0

    for img in images:

        image_id = img["id"]
        filename = img["file_name"]
        image_path = image_dir / filename

        with Image.open(image_path) as im:
            actual_width, actual_height = im.size

        dimension_counts[(actual_width, actual_height)] += 1

        if (actual_width != img["width"] or
                actual_height != img["height"]):
            print("DIMENSION MISMATCH:", filename)

        anns = annotations_by_image[image_id]

        if not anns:
            empty_images.append(filename)

        for ann in anns:

            polygons = ann.get("segmentation", [])

            if not isinstance(polygons, list) or not polygons:
                invalid_polygons += 1
                continue

            for polygon in polygons:

                if not isinstance(polygon, list) or len(polygon) < 6:
                    invalid_polygons += 1
                    continue

                if len(polygon) % 2 != 0:
                    invalid_polygons += 1
                    continue

                if not all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    for v in polygon
                ):
                    invalid_polygons += 1
                    continue

                xs = polygon[0::2]
                ys = polygon[1::2]

                if (
                    min(xs) < 0 or max(xs) > img["width"] or
                    min(ys) < 0 or max(ys) > img["height"]
                ):
                    out_of_bounds += 1

            bbox = ann.get("bbox", [])

            if (
                len(bbox) != 4 or
                not all(isinstance(v, (int, float)) and math.isfinite(v)
                        for v in bbox) or
                bbox[2] <= 0 or bbox[3] <= 0
            ):
                invalid_boxes += 1

    annotation_counts = Counter(
        len(annotations_by_image[img["id"]])
        for img in images
    )

    print("\n" + "=" * 45)
    print("SPLIT:", split.upper())
    print("=" * 45)
    print("Image dimensions:", dict(dimension_counts))
    print("Images without annotations:", len(empty_images))
    print("Annotations per image:", dict(annotation_counts))
    print("Invalid polygons:", invalid_polygons)
    print("Out-of-bounds polygons:", out_of_bounds)
    print("Invalid bounding boxes:", invalid_boxes)

    if empty_images:
        print("Unannotated examples:", empty_images[:5])

    if annotations:
        print("\nSample annotation:")
        print(annotations[0])

print("\nDEEP INSPECTION COMPLETED!")
