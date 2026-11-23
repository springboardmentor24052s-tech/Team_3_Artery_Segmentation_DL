
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    "stenosis_detection/checkpoints/resunet_stenosis_best.pt"
)

THRESHOLD = 0.8


def main():

    print("Loading validation dataset...")

    dataset = StenosisDataset("val")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Loading best checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = ResidualUNet().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Generating predictions...")

    # Select up to 6 validation images
    number_of_images = min(6, len(dataset))

    fig, axes = plt.subplots(
        number_of_images,
        3,
        figsize=(12, 4 * number_of_images)
    )

    if number_of_images == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():

        for index in range(number_of_images):

            image, mask = dataset[index]

            input_image = image.unsqueeze(0).to(DEVICE)

            logits = model(input_image)

            probabilities = torch.sigmoid(logits)

            prediction = (
                probabilities >= THRESHOLD
            ).float()

            image_np = image[0].numpy()
            mask_np = mask[0].numpy()
            prediction_np = prediction[0, 0].numpy()

            axes[index, 0].imshow(
                image_np,
                cmap="gray"
            )
            axes[index, 0].set_title("Original Angiography")

            axes[index, 1].imshow(
                image_np,
                cmap="gray"
            )
            axes[index, 1].imshow(
                mask_np,
                cmap="Reds",
                alpha=0.5
            )
            axes[index, 1].set_title("Ground Truth")

            axes[index, 2].imshow(
                image_np,
                cmap="gray"
            )
            axes[index, 2].imshow(
                prediction_np,
                cmap="Blues",
                alpha=0.5
            )
            axes[index, 2].set_title("Model Prediction")

            for column in range(3):
                axes[index, column].axis("off")

    plt.tight_layout()

    output_path = "stenosis_prediction_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("Visualization saved:", output_path)

    plt.show()


if __name__ == "__main__":
    main()
