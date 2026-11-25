
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import StenosisDataset
from src.unet import UNet


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    dataset = StenosisDataset("train")
    small_dataset = Subset(dataset, list(range(4)))

    loader = DataLoader(
        small_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([10.0], device=device)
    )

    print("Testing on 4 training images")

    for step in range(1, 51):
        model.train()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        print(
            f"Step {step}/50 | "
            f"Average loss: {total_loss / len(loader):.4f}"
        )
    torch.save(model.state_dict(), "outputs/overfit_test.pth")
    print("Saved overfit-test model to outputs/overfit_test.pth")


if __name__ == "__main__":
    main()
