
import torch
import numpy as np

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet

# -----------------------------------------
# SETTINGS
# -----------------------------------------

CHECKPOINT = (
    "stenosis_detection/checkpoints/"
    "resunet_hardnegative_best.pt"
)

THRESHOLD = 0.80

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model = ResidualUNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])

# -----------------------------------------
# LOAD VALIDATION DATA
# -----------------------------------------

dataset = StenosisDataset("val")

false_positive_counts = []
false_negative_counts = []
true_positive_counts = []

# -----------------------------------------
# EVALUATION
# -----------------------------------------

print("\nAnalysing false positives...")

with torch.no_grad():

    for index in range(len(dataset)):

        image, mask = dataset[index]

        image = image.unsqueeze(0).to(DEVICE)

        probabilities = torch.sigmoid(model(image))

        prediction = (
            probabilities[0, 0].cpu().numpy()
            >= THRESHOLD
        )

        ground_truth = mask[0].numpy() > 0.5

        tp = int(np.logical_and(
            prediction, ground_truth
        ).sum())

        fp = int(np.logical_and(
            prediction, ~ground_truth
        ).sum())

        fn = int(np.logical_and(
            ~prediction, ground_truth
        ).sum())

        true_positive_counts.append(tp)
        false_positive_counts.append(fp)
        false_negative_counts.append(fn)

        if (index + 1) % 50 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(dataset)} images"
            )

# -----------------------------------------
# DISTRIBUTION ANALYSIS
# -----------------------------------------

fp = np.array(false_positive_counts)
fn = np.array(false_negative_counts)
tp = np.array(true_positive_counts)

total_fp = fp.sum()
total_fn = fn.sum()
total_tp = tp.sum()

print("\n" + "=" * 65)
print("FALSE-POSITIVE DISTRIBUTION")
print("=" * 65)

print("Total false positives:", total_fp)
print("Total false negatives:", total_fn)
print("Total true positives:", total_tp)

print("\nIMAGE DISTRIBUTION")
print("-" * 50)

print("Images with zero false positives:", np.sum(fp == 0))
print("Images with false positives:", np.sum(fp > 0))

print("Images with more than 500 FP pixels:", np.sum(fp > 500))
print("Images with more than 1000 FP pixels:", np.sum(fp > 1000))
print("Images with more than 2000 FP pixels:", np.sum(fp > 2000))

print("\nFALSE POSITIVE STATISTICS")
print("-" * 50)

print("Mean FP pixels per image:", round(float(fp.mean()), 2))
print("Median FP pixels per image:", round(float(np.median(fp)), 2))
print("Maximum FP pixels in one image:", int(fp.max()))

print("\nTOP 10 FALSE-POSITIVE IMAGES")
print("-" * 50)

top_indices = np.argsort(fp)[::-1][:10]

for rank, index in enumerate(top_indices, start=1):

    print(
        f"Rank {rank}: Image index {index}, "
        f"FP={fp[index]}, "
        f"FN={fn[index]}, "
        f"TP={tp[index]}"
    )

# -----------------------------------------
# CONCENTRATION
# -----------------------------------------

sorted_fp = np.sort(fp)[::-1]

if total_fp > 0:

    for count in [10, 20, 40, 50]:

        contribution = (
            sorted_fp[:count].sum() / total_fp
        ) * 100

        print(
            f"Top {count} images account for "
            f"{contribution:.2f}% of all FP pixels"
        )

print("\nAnalysis completed.")
