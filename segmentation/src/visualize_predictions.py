
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.dataset import StenosisDataset
from src.unet import UNet


THRESHOLD = 0.40
NUM_IMAGES = 10


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.65):
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    mask = mask.astype(bool)
    color_array = np.array(color, dtype=np.float32)

    base[mask] = (1 - alpha) * base[mask] + alpha * color_array

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def create_error_map(prediction, ground_truth):
    error_map = np.zeros((*prediction.shape, 3), dtype=np.uint8)

    tp = prediction & ground_truth
    fp = prediction & ~ground_truth
    fn = ~prediction & ground_truth

    error_map[tp] = (0, 255, 0)
    error_map[fp] = (255, 0, 0)
    error_map[fn] = (0, 0, 255)

    return Image.fromarray(error_map)


def main():
    device = torch.device("cpu")

    output_dir = Path("outputs/round2_baseline_visualizations")
    mask_dir = Path("outputs/round2_baseline_masks")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path("outputs/unet_baseline_round2_best.pth")

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint:", checkpoint_path)
    print("Training epoch:", checkpoint["epoch"])
    print("Saved validation Dice:", checkpoint["val_dice"])
    print("Threshold:", THRESHOLD)

    dataset = StenosisDataset("val")

    with torch.no_grad():
        for index in range(min(NUM_IMAGES, len(dataset))):
            image_tensor, mask_tensor = dataset[index]

            logits = model(image_tensor.unsqueeze(0).to(device))
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

            image_info = dataset.images[index]
            image_path = dataset.image_dir / image_info["file_name"]

            image = Image.open(image_path).convert("RGB").resize((512, 512))

            ground_truth = mask_tensor[0].numpy() >= 0.5
            prediction = probabilities >= THRESHOLD

            dice = (
                2 * np.logical_and(prediction, ground_truth).sum() + 1
            ) / (prediction.sum() + ground_truth.sum() + 1)

            binary_mask = prediction.astype(np.uint8) * 255

            Image.fromarray(binary_mask).save(
                mask_dir / f"baseline_mask_{index + 1}.png"
            )

            np.save(
                mask_dir / f"baseline_mask_{index + 1}.npy",
                prediction.astype(np.uint8),
            )

            target_overlay = overlay_mask(image, ground_truth)
            prediction_overlay = overlay_mask(image, prediction)
            error_map = create_error_map(prediction, ground_truth)

            probability_image = Image.fromarray(
                (probabilities * 255).clip(0, 255).astype(np.uint8)
            ).convert("RGB")

            canvas = Image.new("RGB", (1024, 1050), "white")
            draw = ImageDraw.Draw(canvas)

            panels = [
                ("Original", image),
                ("Ground truth", target_overlay),
                (f"Prediction (Dice: {dice:.3f})", prediction_overlay),
                ("Probability map", probability_image),
                ("Error map: Green=TP, Red=FP, Blue=FN", error_map),
            ]

            for panel_index, (label, panel) in enumerate(panels):
                x = (panel_index % 2) * 512
                y = (panel_index // 2) * 350

                canvas.paste(panel, (x, y + 25))
                draw.text((x + 8, y + 5), label, fill="black")

            filename = output_dir / f"baseline_validation_{index + 1}.png"
            canvas.save(filename)

            print(
                f"Image {index + 1}: Dice={dice:.4f}, "
                f"TP={np.logical_and(prediction, ground_truth).sum()}, "
                f"FP={np.logical_and(prediction, ~ground_truth).sum()}, "
                f"FN={np.logical_and(~prediction, ground_truth).sum()}"
            )

            print("Saved:", filename.resolve())

    print("\nBaseline visualization completed.")
    print("Output folder:", output_dir.resolve())


if __name__ == "__main__":
    main()
