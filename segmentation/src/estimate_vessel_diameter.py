
from pathlib import Path
import csv

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# Input folders
MASK_DIR = Path("outputs/extracted_vessel_structures")
CENTERLINE_DIR = Path("outputs/vessel_centerlines")

# Output folders
OUTPUT_DIR = Path("outputs/vessel_diameter")
DIAMETER_MAP_DIR = OUTPUT_DIR / "diameter_maps"
VISUAL_DIR = OUTPUT_DIR / "visualizations"

def main():
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask folder not found: {MASK_DIR}")

    if not CENTERLINE_DIR.exists():
        raise FileNotFoundError(
            f"Centerline folder not found: {CENTERLINE_DIR}"
        )

    mask_files = sorted(MASK_DIR.glob("*.npy"))

    if not mask_files:
        raise FileNotFoundError("No vessel masks found.")

    DIAMETER_MAP_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "diameter_measurements.csv"

    total_points = 0
    all_diameters = []

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "image",
            "centerline_pixels",
            "mean_diameter_px",
            "median_diameter_px",
            "minimum_diameter_px",
            "maximum_diameter_px"
        ])

        for index, mask_path in enumerate(mask_files, start=1):

            centerline_path = CENTERLINE_DIR / mask_path.name

            if not centerline_path.exists():
                print(f"Skipping {mask_path.name}: centerline missing")
                continue

            # Load binary vessel mask and centerline
            vessel_mask = np.load(mask_path) > 0
            centerline = np.load(centerline_path) > 0

            if vessel_mask.shape != centerline.shape:
                print(f"Skipping {mask_path.name}: shape mismatch")
                continue

            # Keep only centerline pixels inside the vessel mask
            centerline = centerline & vessel_mask

            if not np.any(vessel_mask) or not np.any(centerline):
                print(f"Skipping {mask_path.name}: empty mask or centerline")
                continue

            # Distance from each vessel pixel to the nearest background pixel
            distance_map = distance_transform_edt(vessel_mask)

            # Local diameter = 2 * distance to nearest boundary
            diameter_values = 2.0 * distance_map[centerline]

            # Store diameter only at centerline locations
            diameter_map = np.zeros(vessel_mask.shape, dtype=np.float32)
            diameter_map[centerline] = diameter_values

            # Save numerical diameter map
            np.save(
                DIAMETER_MAP_DIR / mask_path.name,
                diameter_map
            )

            # Create normalized visualization
            max_diameter = float(diameter_values.max())

            if max_diameter > 0:
                visual = np.zeros(
                    diameter_map.shape,
                    dtype=np.uint8
                )
                visual[centerline] = np.clip(
                    diameter_values / max_diameter * 255,
                    0,
                    255
                ).astype(np.uint8)

                Image.fromarray(visual).save(
                    VISUAL_DIR / f"{mask_path.stem}.png"
                )

            # Statistics
            mean_d = float(np.mean(diameter_values))
            median_d = float(np.median(diameter_values))
            min_d = float(np.min(diameter_values))
            max_d = float(np.max(diameter_values))

            writer.writerow([
                mask_path.name,
                len(diameter_values),
                round(mean_d, 4),
                round(median_d, 4),
                round(min_d, 4),
                round(max_d, 4)
            ])

            total_points += len(diameter_values)
            all_diameters.extend(diameter_values.tolist())

            if index % 20 == 0 or index == len(mask_files):
                print(
                    f"Processed {index}/{len(mask_files)} images"
                )

    print("\nVessel diameter estimation completed.")

    print(f"Total centerline points: {total_points}")
    print(f"CSV measurements: {csv_path}")
    print(f"Diameter maps: {DIAMETER_MAP_DIR}")
    print(f"Visualizations: {VISUAL_DIR}")

    if all_diameters:
        values = np.asarray(all_diameters)

        print("\nOverall diameter statistics (pixels):")
        print(f"Mean:   {values.mean():.3f}")
        print(f"Median: {np.median(values):.3f}")
        print(f"Minimum: {values.min():.3f}")
        print(f"Maximum: {values.max():.3f}")


if __name__ == "__main__":
    main()
