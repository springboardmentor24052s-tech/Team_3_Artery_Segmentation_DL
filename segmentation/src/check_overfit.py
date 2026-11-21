
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from src.dataset import StenosisDataset
from src.unet import UNet


def main():
    device = torch.device("cpu")

    dataset = StenosisDataset("train")
    small_dataset = Subset(dataset, list(range(4)))
    loader = DataLoader(small_dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(
        torch.load(
            "outputs/overfit_test.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    output_dir = Path("outputs/overfit_visuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_dice = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader, start=1):
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            intersection = (predictions * masks).sum().item()
            denominator = predictions.sum().item() + masks.sum().item()
            dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
            total_dice += dice

            image = images[0].permute(1, 2, 0).numpy()
            image = image.clip(0, 1)
            ground_truth = masks[0, 0].numpy()
            prediction = predictions[0, 0].cpu().numpy()
            probability = probabilities[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Original")

            axes[1].imshow(image, cmap="gray")
            axes[1].imshow(ground_truth, cmap="Reds", alpha=0.7)
            axes[1].set_title("Ground Truth")

            axes[2].imshow(image, cmap="gray")
            axes[2].imshow(prediction, cmap="Reds", alpha=0.7)
            axes[2].set_title(f"Prediction (Dice={dice:.3f})")

            axes[3].imshow(probability, cmap="gray", vmin=0, vmax=1)
            axes[3].set_title("Probability Map")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            save_path = output_dir / f"training_{i}.png"
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            print(f"Image {i}: Dice={dice:.4f}")
            print(f"Saved: {save_path}")

    print(f"Mean training Dice: {total_dice / len(loader):.4f}")
    print(f"Visualizations saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
