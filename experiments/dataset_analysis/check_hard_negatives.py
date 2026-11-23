
import json
from collections import Counter
from pathlib import Path

path = Path(
    "stenosis_detection/hard_negative_candidates.json"
)

with open(path, "r", encoding="utf-8") as file:
    candidates = json.load(file)

print("=" * 55)
print("HARD-NEGATIVE CANDIDATE AUDIT")
print("=" * 55)

print("Total candidates:", len(candidates))

image_counts = Counter(
    item["image_index"] for item in candidates
)

print("Unique training images:", len(image_counts))

predicted_pixels = [
    item["predicted_pixels"] for item in candidates
]

probabilities = [
    item["max_probability"] for item in candidates
]

print("\nPREDICTED PIXELS")
print("-------------------------")
print("Minimum:", min(predicted_pixels))
print("Maximum:", max(predicted_pixels))
print("Average:", round(sum(predicted_pixels) / len(predicted_pixels), 2))

print("\nMAXIMUM PROBABILITY")
print("-------------------------")
print("Minimum:", min(probabilities))
print("Maximum:", max(probabilities))
print("Average:", round(sum(probabilities) / len(probabilities), 4))

print("\nTOP 10 CANDIDATES")
print("-------------------------")

for item in sorted(
    candidates,
    key=lambda x: x["predicted_pixels"],
    reverse=True
)[:10]:
    print(item)

print("\nCOORDINATE VALIDATION")
print("-------------------------")

invalid = [
    item for item in candidates
    if item["x"] < 0
    or item["y"] < 0
    or item["x"] + 256 > 512
    or item["y"] + 256 > 512
]

print("Invalid coordinates:", len(invalid))

print("\nAUDIT COMPLETED")
