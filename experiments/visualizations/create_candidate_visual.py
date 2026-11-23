from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

image_id = "ee91867d66a8472ebcf621cb65b552c7"

base = Path("processed")
mask_path = base / "predictions" / f"{image_id}_stenosis_mask.png"
image_path = base / f"{image_id}_processed.png"
output_path = base / "predictions" / f"{image_id}_candidate_regions.png"

# Load the existing prediction mask
mask = np.array(Image.open(mask_path).convert("L")) > 0

# Identify connected components, matching the inference approach
labels, count = ndimage.label(mask)

# Load the preprocessed angiogram as the background
image = Image.open(image_path).convert("RGB")
if image.size != (mask.shape[1], mask.shape[0]):
    image = image.resize((mask.shape[1], mask.shape[0]))

draw = ImageDraw.Draw(image)

for region_id in range(1, count + 1):
    region = labels == region_id
    y, x = np.where(region)

    if len(x) == 0:
        continue

    x_min, x_max = int(x.min()), int(x.max())
    y_min, y_max = int(y.min()), int(y.max())

    centroid_x = int(x.mean())
    centroid_y = int(y.mean())

    draw.rectangle(
        (x_min, y_min, x_max, y_max),
        outline="red",
        width=2
    )

    draw.ellipse(
        (centroid_x - 3, centroid_y - 3,
         centroid_x + 3, centroid_y + 3),
        fill="yellow"
    )

    draw.text(
        (x_min, max(0, y_min - 14)),
        f"R{region_id}",
        fill="red"
    )

image.save(output_path)

print("Candidate regions detected:", count)
print("Image saved to:", output_path)
