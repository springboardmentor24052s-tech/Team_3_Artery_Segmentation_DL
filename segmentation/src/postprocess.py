
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


INPUT_FOLDERS = {
    "baseline": Path("outputs/baseline_binary_masks"),
    "attention": Path("outputs/attention_binary_masks"),
}

OUTPUT_FOLDERS = {
    "baseline": Path("outputs/baseline_processed_masks"),
    "attention": Path("outputs/attention_processed_masks"),
}

# Remove only components containing 5 pixels or fewer.
MIN_COMPONENT_SIZE = 6

# 8-connectivity preserves diagonal connections.
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def remove_small_components(mask):
    mask = mask.astype(bool)

    labels, count = ndimage.label(
        mask,
        structure=STRUCTURE
    )

    component_sizes = np.bincount(labels.ravel())

    # Component 0 represents the background.
    keep = component_sizes >= MIN_COMPONENT_SIZE
    keep[0] = False

    cleaned_mask = keep[labels]

    removed_pixels = int(mask.sum() - cleaned_mask.sum())

    return cleaned_mask.astype(np.uint8), count, removed_pixels


def save_comparison(original, processed, output_path):
    original_image = Image.fromarray(
        original.astype(np.uint8) * 255
    )

    processed_image = Image.fromarray(
        processed.astype(np.uint8) * 255
    )

    canvas = Image.new("RGB", (1024, 550), "white")

    canvas.paste(original_image.convert("RGB"), (0, 38))
    canvas.paste(processed_image.convert("RGB"), (512, 38))

    draw = ImageDraw.Draw(canvas)

    draw.text((10, 10), "Original mask", fill="black")
    draw.text((522, 10), "Processed mask", fill="black")

    canvas.save(output_path)


def process_folder(model_name, input_folder, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(input_folder.glob("*.npy"))

    print(f"\nProcessing: {model_name}")
    print("-" * 50)

    for mask_path in mask_files:

        original = np.load(mask_path)

        processed, component_count, removed_pixels = (
            remove_small_components(original)
        )

        # Save processed NumPy mask.
        npy_path = output_folder / mask_path.name
        np.save(npy_path, processed)

        # Save processed PNG mask.
        png_path = output_folder / (
            mask_path.stem + ".png"
        )

        Image.fromarray(
            processed * 255
        ).convert("L").save(png_path)

        # Save original-versus-processed comparison.
        comparison_path = output_folder / (
            mask_path.stem + "_comparison.png"
        )

        save_comparison(
            original,
            processed,
            comparison_path
        )

        print(f"\nImage: {mask_path.name}")
        print(f"Original pixels: {int(original.sum())}")
        print(f"Processed pixels: {int(processed.sum())}")
        print(f"Original components: {component_count}")
        print(f"Removed pixels: {removed_pixels}")
        print(f"Saved: {png_path}")


def main():
    for model_name, input_folder in INPUT_FOLDERS.items():

        output_folder = OUTPUT_FOLDERS[model_name]

        process_folder(
            model_name,
            input_folder,
            output_folder
        )

    print("\nPost-processing completed.")
    print("Original masks remain unchanged.")


if __name__ == "__main__":
    main()
