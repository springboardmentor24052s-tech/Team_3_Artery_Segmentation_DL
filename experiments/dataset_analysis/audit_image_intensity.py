
from pathlib import Path
import numpy as np
from PIL import Image

DATA_ROOT = Path("datasets/arcade/arcade/stenosis")

for split in ["train", "val", "test"]:

    image_dir = DATA_ROOT / split / "images"
    image_paths = sorted(image_dir.glob("*"))

    means = []
    stds = []
    p1_values = []
    p99_values = []
    dark_pixels = []
    bright_pixels = []

    print("\n" + "=" * 55)
    print(f"IMAGE INTENSITY AUDIT: {split.upper()}")
    print("=" * 55)

    for index, path in enumerate(image_paths):

        try:
            image = Image.open(path).convert("L")
            array = np.asarray(image, dtype=np.float32)

            means.append(array.mean())
            stds.append(array.std())
            p1_values.append(np.percentile(array, 1))
            p99_values.append(np.percentile(array, 99))

            dark_pixels.append(np.mean(array <= 5) * 100)
            bright_pixels.append(np.mean(array >= 250) * 100)

        except Exception as error:
            print(f"Could not read {path.name}: {error}")

    def report(label, values):
        print(
            f"{label:<25}"
            f"Median: {np.median(values):8.2f}  "
            f"Min: {np.min(values):8.2f}  "
            f"Max: {np.max(values):8.2f}"
        )

    print(f"Images inspected: {len(means)}")

    report("Mean brightness:", means)
    report("Contrast (std):", stds)
    report("1st percentile:", p1_values)
    report("99th percentile:", p99_values)
    report("Very dark pixels %:", dark_pixels)
    report("Very bright pixels %:", bright_pixels)

print("\nIMAGE INTENSITY AUDIT COMPLETED")
