
import torch
from torch.utils.data import DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet
from stenosis_detection.train import validate, DEVICE


CHECKPOINT_PATH = (
    "stenosis_detection/checkpoints/resunet_stenosis_best.pt"
)


def main():

    print("Loading validation dataset...")

    val_dataset = StenosisDataset("val")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Loading best model checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = ResidualUNet().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\nCheckpoint information")
    print("----------------------")
    print("Epoch:", checkpoint["epoch"])
    print("Saved Soft Dice:", checkpoint["best_val_soft_dice"])
    print("Saved Hard Dice:", checkpoint["val_hard_dice"])
    print("Saved IoU:", checkpoint["val_iou"])

    print("\nRe-evaluating on all validation images...")

    soft_dice, hard_dice, iou = validate(
        model,
        val_loader
    )

    print("\nINDEPENDENT VALIDATION RESULTS")
    print("------------------------------")
    print("Validation images:", len(val_dataset))
    print(f"Soft Dice: {soft_dice:.6f}")
    print(f"Hard Dice: {hard_dice:.6f}")
    print(f"IoU:       {iou:.6f}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
