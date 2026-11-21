
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

INPUT_DIR = Path("outputs/baseline_threshold_070_masks")
OUTPUT_DIR = Path("outputs/extracted_vessel_structures")

MIN_COMPONENT_SIZE = 6

# 8-connectivity: diagonal neighboring pixels are connected.
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def extract_vessels(mask):
    binary_mask = mask.astype(bool)

    labeled_mask, component_count = ndimage.label(
        binary_mask,
        structure=STRUCTURE,
    )

    component_sizes = np.bincount(labeled_mask.ravel())

    # Keep components containing at least 6 pixels.
    keep_components = component_sizes >= MIN_COMPONENT_SIZE
    keep_components[0] = False

    cleaned_mask = keep_components[labeled_mask]

    removed_pixels = int(binary_mask.sum() - cleaned_mask.sum())
    kept_components = int(keep_components.sum())

    return cleaned_mask.astype(np.uint8), component_count, kept_components, removed_pixels


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(INPUT_DIR.glob("*.npy"))

    if not mask_files:
        raise FileNotFoundError(f"No NumPy masks found in {INPUT_DIR}")

    total_removed = 0

    print("Input masks:", len(mask_files))
    print("Minimum component size:", MIN_COMPONENT_SIZE)

    for index, mask_path in enumerate(mask_files, start=1):
        mask = np.load(mask_path)

        cleaned_mask, original_count, kept_count, removed_pixels = (
            extract_vessels(mask)
        )

        np.save(
            OUTPUT_DIR / mask_path.name,
            cleaned_mask,
        )

        png_path = OUTPUT_DIR / f"{mask_path.stem}.png"

        Image.fromarray(cleaned_mask * 255).save(png_path)

        total_removed += removed_pixels

        if index <= 5 or index % 20 == 0:
            print(
                f"{mask_path.name}: "
                f"components={original_count}, "
                f"kept={kept_count}, "
                f"removed_pixels={removed_pixels}"
            )

    print("\nVessel structure extraction completed.")
    print("Processed masks:", len(mask_files))
    print("Total isolated pixels removed:", total_removed)
    print("Output folder:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
