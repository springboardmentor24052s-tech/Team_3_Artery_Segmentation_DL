
import json
import hashlib
from pathlib import Path

ROOT = Path("datasets/arcade/arcade/stenosis")


def inspect_image(split, filename):

    json_path = ROOT / split / "annotations" / f"{split}.json"
    image_path = ROOT / split / "images" / filename

    with open(json_path, "r") as f:
        data = json.load(f)

    image_info = next(
        img for img in data["images"]
        if Path(img["file_name"]).name == filename
    )

    annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] == image_info["id"]
    ]

    with open(image_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"\n{split.upper()} — {filename}")
    print("-" * 35)
    print("Image ID:", image_info["id"])
    print("Image dimensions:",
          image_info["width"], "x", image_info["height"])
    print("File size:", image_path.stat().st_size, "bytes")
    print("Raw file SHA-256:", file_hash)
    print("Stenosis annotations:", len(annotations))

    for i, ann in enumerate(annotations, start=1):
        print(f"\nAnnotation {i}")
        print("Category ID:", ann["category_id"])
        print("Bounding box:", ann["bbox"])
        print("Area:", ann.get("area"))


inspect_image("train", "218.png")
inspect_image("val", "58.png")

print("\nDUPLICATE INSPECTION COMPLETED!")
