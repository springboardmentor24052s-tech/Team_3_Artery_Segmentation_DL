
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_DIR = Path(
    "datasets/arcade/arcade/stenosis/train/images"
)

hash_groups = defaultdict(list)

print("Checking for duplicate images within TRAIN...\n")

for image_path in sorted(IMAGE_DIR.glob("*.png")):

    with Image.open(image_path) as img:
        pixels = img.convert("L").tobytes()

    image_hash = hashlib.sha256(pixels).hexdigest()

    hash_groups[image_hash].append(image_path.name)

duplicates = [
    filenames
    for filenames in hash_groups.values()
    if len(filenames) > 1
]

print("Total image files:", sum(len(v) for v in hash_groups.values()))
print("Unique image contents:", len(hash_groups))
print("Duplicate groups:", len(duplicates))

print("\nDUPLICATE GROUPS")
print("----------------")

if not duplicates:
    print("No duplicate images found within TRAIN.")

else:
    for i, filenames in enumerate(duplicates, start=1):
        print(f"Group {i}: {filenames}")

print("\nTRAIN DUPLICATE CHECK COMPLETED!")
