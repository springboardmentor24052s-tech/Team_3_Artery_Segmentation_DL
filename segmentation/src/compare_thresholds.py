
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.dataset import StenosisDataset
from src.unet import UNet

THRESHOLDS = [0.60, 0.70]


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.65):
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    mask = mask.astype(bool)
    color_array = np.array(color, dtype=np.float32)

    base[mask] = (1 - alpha) * base[mask] + alpha * color_array

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def main():
    device = torch.device("cpu")

    checkpoint_path = "outputs/unet_baseline_continued_best.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = StenosisDataset("val")

    for threshold in THRESHOLDS:
        output_dir = Path(f"outputs/threshold_{int(threshold * 100):03d}")
        mask_dir = output_dir / "binary_masks"

        output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating predictions at threshold {threshold}")

        with torch.no_grad():
            for index in range(min(5, len(dataset))):
                image_tensor, mask_tensor = dataset[index]

                logits = model(image_tensor.unsqueeze(0).to(device))
                probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

                image_info = dataset.images[index]
                image_path = dataset.image_dir / image_info["file_name"]

                image = Image.open(image_path).convert("RGB").resize((512, 512))

                ground_truth = mask_tensor[0].numpy() >= 0.5
                prediction = probabilities >= threshold

                binary_mask = prediction.astype(np.uint8) * 255

                Image.fromarray(binary_mask).save(
                    mask_dir / f"mask_{index + 1}.png"
                )

                np.save(
                    mask_dir / f"mask_{index + 1}.npy",
                    prediction.astype(np.uint8),
                )

                target_overlay = overlay_mask(image, ground_truth)
                prediction_overlay = overlay_mask(image, prediction)

                probability_image = Image.fromarray(
                    (probabilities * 255).clip(0, 255).astype(np.uint8)
                ).convert("RGB")

                canvas = Image.new("RGB", (1024, 1050), "white")

                panels = [
                    ("Original", image),
                    ("Ground truth", target_overlay),
                    (f"Baseline threshold {threshold}", prediction_overlay),
                    ("Probability map", probability_image),
                ]

                draw = ImageDraw.Draw(canvas)

                for panel_index, (label, panel) in enumerate(panels):
                    x = (panel_index % 2) * 512
                    y = (panel_index // 2) * 525

                    canvas.paste(panel, (x, y + 25))
                    draw.text((x + 8, y + 5), label, fill="black")

                filename = output_dir / f"validation_{index + 1}.png"
                canvas.save(filename)

                print("Saved:", filename.resolve())

    print("\nThreshold comparison images generated successfully.")


if __name__ == "__main__":
    main()
