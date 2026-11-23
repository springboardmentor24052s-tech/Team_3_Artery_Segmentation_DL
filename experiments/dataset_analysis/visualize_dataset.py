import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

root = Path("datasets/arcade/arcade/stenosis/train")
image_dir = root / "images"
json_path = root / "annotations" / "train.json"

with open(json_path, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

# Group annotations by image ID
annotations_by_image = {}

for ann in annotations:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# Select three annotated images
samples = [
    img for img in images
    if annotations_by_image.get(img["id"])
][:3]

fig, axes = plt.subplots(len(samples), 3, figsize=(12, 12))

for i, img_data in enumerate(samples):
    image_path = image_dir / img_data["file_name"]

    original = Image.open(image_path).convert("RGB")
    mask = Image.new("L", original.size, 0)

    draw = ImageDraw.Draw(mask)

    for ann in annotations_by_image[img_data["id"]]:
        for polygon in ann["segmentation"]:
            points = [
                (polygon[j], polygon[j + 1])
                for j in range(0, len(polygon), 2)
            ]

            draw.polygon(points, fill=255)

    # Create red overlay
    overlay = original.copy()
    red_layer = Image.new("RGB", original.size, (255, 0, 0))
    overlay.paste(red_layer, mask=mask)

    blended = Image.blend(original, overlay, 0.4)

    axes[i, 0].imshow(original)
    axes[i, 0].set_title(f"Original: {img_data['file_name']}")

    axes[i, 1].imshow(blended)
    axes[i, 1].set_title("Stenosis Overlay")

    axes[i, 2].imshow(mask, cmap="gray")
    axes[i, 2].set_title("Binary Mask")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.show()
