
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import StenosisDataset
from src.unet import UNet

THRESHOLD = 0.70


def main():
    device = torch.device("cpu")

    checkpoint_path = "outputs/unet_baseline_continued_best.pth"
    output_dir = Path("outputs/baseline_threshold_070_masks")

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = StenosisDataset("val")

    print("Model: Baseline U-Net")
    print("Checkpoint:", checkpoint_path)
    print("Threshold:", THRESHOLD)
    print("Validation images:", len(dataset))

    with torch.no_grad():
        for index in range(len(dataset)):
            image_tensor, _ = dataset[index]

            logits = model(image_tensor.unsqueeze(0).to(device))
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prediction = (probabilities >= THRESHOLD).astype(np.uint8)

            image_info = dataset.images[index]
            original_name = Path(image_info["file_name"]).stem

            filename = f"{index + 1:04d}_{original_name}"

            np.save(
                output_dir / f"{filename}.npy",
                prediction,
            )

            mask_image = Image.fromarray(prediction * 255)
            mask_image.save(output_dir / f"{filename}.png")

            if (index + 1) % 20 == 0:
                print(f"Generated masks: {index + 1}/{len(dataset)}")

    print("\nMask generation completed.")
    print("Total masks:", len(dataset))
    print("Saved in:", output_dir.resolve())


if __name__ == "__main__":
    main()
