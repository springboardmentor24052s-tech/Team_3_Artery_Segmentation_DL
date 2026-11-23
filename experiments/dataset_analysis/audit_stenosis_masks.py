
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw


DATA_ROOT = Path("datasets/arcade/arcade/stenosis")

SPLITS = ["train", "val", "test"]

STENOSIS_CATEGORY_ID = 26


def polygon_is_valid(polygon):

    if not isinstance(polygon, list):
        return False

    if len(polygon) < 6:
        return False

    if len(polygon) % 2 != 0:
        return False

    try:
        coordinates = np.asarray(
            polygon,
            dtype=np.float64
        )

        return np.isfinite(coordinates).all()

    except (TypeError, ValueError):
        return False


def audit_split(split):

    print("\n" + "=" * 55)
    print(f"DATASET SPLIT: {split.upper()}")
    print("=" * 55)

    split_root = DATA_ROOT / split

    image_dir = split_root / "images"

    annotation_path = (
        split_root
        / "annotations"
        / f"{split}.json"
    )

    with open(annotation_path, "r") as file:
        data = json.load(file)

    images = data["images"]

    annotations = [
        ann for ann in data["annotations"]
        if ann.get("category_id") == STENOSIS_CATEGORY_ID
    ]

    image_ids = {
        image_info["id"]
        for image_info in images
    }

    annotations_by_image = defaultdict(list)

    for ann in annotations:
        annotations_by_image[ann["image_id"]].append(ann)

    # Counters
    invalid_polygons = 0
    out_of_bounds_polygons = 0
    malformed_annotations = 0
    missing_images = 0
    duplicate_filenames = 0
    duplicate_annotations = 0

    empty_masks = 0
    nonempty_masks = 0

    total_mask_pixels = 0
    total_image_pixels = 0

    image_annotation_counts = Counter()

    filenames = [
        image_info["file_name"]
        for image_info in images
    ]

    duplicate_filenames = (
        len(filenames) - len(set(filenames))
    )

    seen_annotations = set()

    for ann in annotations:

        image_id = ann.get("image_id")

        if image_id not in image_ids:
            malformed_annotations += 1
            continue

        annotation_key = (
            image_id,
            str(ann.get("segmentation"))
        )

        if annotation_key in seen_annotations:
            duplicate_annotations += 1

        seen_annotations.add(annotation_key)

        segmentations = ann.get("segmentation")

        if not isinstance(segmentations, list):
            malformed_annotations += 1
            continue

        for polygon in segmentations:

            if not polygon_is_valid(polygon):
                invalid_polygons += 1
                continue

            coordinates = np.asarray(
                polygon,
                dtype=np.float64
            ).reshape(-1, 2)

            image_info = next(
                (
                    item for item in images
                    if item["id"] == image_id
                ),
                None
            )

            if image_info is None:
                continue

            width = image_info["width"]
            height = image_info["height"]

            x = coordinates[:, 0]
            y = coordinates[:, 1]

            if (
                np.any(x < 0)
                or np.any(y < 0)
                or np.any(x > width)
                or np.any(y > height)
            ):
                out_of_bounds_polygons += 1

    # Generate and inspect masks
    for index, image_info in enumerate(images):

        filename = image_info["file_name"]

        image_path = image_dir / filename

        if not image_path.exists():
            missing_images += 1
            continue

        image = Image.open(image_path).convert("L")

        mask = Image.new("L", image.size, 0)

        draw = ImageDraw.Draw(mask)

        image_id = image_info["id"]

        image_annotations = annotations_by_image[image_id]

        image_annotation_counts[len(image_annotations)] += 1

        for ann in image_annotations:

            segmentations = ann.get("segmentation", [])

            if not isinstance(segmentations, list):
                continue

            for polygon in segmentations:

                if not polygon_is_valid(polygon):
                    continue

                coordinates = np.asarray(
                    polygon,
                    dtype=np.float64
                ).reshape(-1, 2)

                width, height = image.size

                if (
                    np.any(coordinates[:, 0] < 0)
                    or np.any(coordinates[:, 1] < 0)
                    or np.any(coordinates[:, 0] > width)
                    or np.any(coordinates[:, 1] > height)
                ):
                    continue

                points = [
                    (float(x), float(y))
                    for x, y in coordinates
                ]

                draw.polygon(points, fill=255)

        mask_array = np.asarray(mask)

        mask_pixels = int(
            np.count_nonzero(mask_array)
        )

        total_mask_pixels += mask_pixels

        total_image_pixels += (
            image.size[0] * image.size[1]
        )

        if mask_pixels == 0:
            empty_masks += 1
        else:
            nonempty_masks += 1

        if index % 200 == 0 and index > 0:
            print(
                f"Inspected {index}/{len(images)} images"
            )

    coverage = (
        100 * total_mask_pixels / total_image_pixels
        if total_image_pixels else 0
    )

    print("\nANNOTATION STATISTICS")
    print("-" * 40)

    print("Images:", len(images))
    print("Stenosis annotations:", len(annotations))
    print("Invalid polygons:", invalid_polygons)
    print("Out-of-bounds polygons:", out_of_bounds_polygons)
    print("Malformed annotations:", malformed_annotations)
    print("Missing image files:", missing_images)
    print("Duplicate filenames:", duplicate_filenames)
    print("Duplicate annotations:", duplicate_annotations)

    print("\nMASK STATISTICS")
    print("-" * 40)

    print("Images with empty masks:", empty_masks)
    print("Images with nonempty masks:", nonempty_masks)
    print("Total annotated pixels:", total_mask_pixels)
    print("Total image pixels:", total_image_pixels)
    print("Annotated pixel coverage: {:.5f}%".format(coverage))

    print("\nANNOTATIONS PER IMAGE")
    print("-" * 40)

    for count in sorted(image_annotation_counts):
        print(
            f"{count} annotations: "
            f"{image_annotation_counts[count]} images"
        )


def main():

    for split in SPLITS:
        audit_split(split)

    print("\n" + "=" * 55)
    print("MASK AUDIT COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
