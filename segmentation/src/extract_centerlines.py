
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

INPUT_DIR = Path("outputs/extracted_vessel_structures")
OUTPUT_DIR = Path("outputs/vessel_centerlines")


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    mask_files = sorted(INPUT_DIR.glob("*.npy"))

    if not mask_files:
        raise FileNotFoundError("No NumPy masks found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Input masks:", len(mask_files))

    total_centerline_pixels = 0

    for index, mask_path in enumerate(mask_files, start=1):
        mask = np.load(mask_path)

        # Convert the binary mask into a Boolean array.
        binary_mask = mask > 0

        # Extract a one-pixel-wide centerline.
        centerline = skeletonize(binary_mask)

        centerline = centerline.astype(np.uint8)

        # Save NumPy centerline.
        np.save(
            OUTPUT_DIR / mask_path.name,
            centerline,
        )

        # Save PNG visualization.
        png_path = OUTPUT_DIR / f"{mask_path.stem}.png"

        Image.fromarray(centerline * 255).save(png_path)

        pixel_count = int(centerline.sum())
        total_centerline_pixels += pixel_count

        if index <= 5 or index % 20 == 0:
            print(
                f"{index}/{len(mask_files)}: "
                f"{mask_path.name}, "
                f"centerline pixels={pixel_count}"
            )

    print("\nCenterline extraction completed.")
    print("Processed masks:", len(mask_files))
    print("Total centerline pixels:", total_centerline_pixels)
    print("Output folder:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
