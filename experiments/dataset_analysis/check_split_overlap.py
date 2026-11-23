
import hashlib
from pathlib import Path
from PIL import Image

from stenosis_detection.dataset import StenosisDataset


def get_hashes(dataset):

    hashes = {}

    for image_info in dataset.images:

        filename = Path(image_info["file_name"]).name
        image_path = dataset.image_dir / filename

        with Image.open(image_path) as img:
            pixels = img.convert("L").tobytes()

        digest = hashlib.sha256(pixels).hexdigest()
        hashes[digest] = filename

    return hashes


print("Checking updated dataset splits...\n")

train = get_hashes(StenosisDataset("train"))
val = get_hashes(StenosisDataset("val"))
test = get_hashes(StenosisDataset("test"))

train_val = set(train) & set(val)
train_test = set(train) & set(test)
val_test = set(val) & set(test)

print("TRAIN unique images:", len(train))
print("VALIDATION unique images:", len(val))
print("TEST unique images:", len(test))

print("\nCROSS-SPLIT OVERLAP")
print("-------------------")
print("Train-Val:", len(train_val))
print("Train-Test:", len(train_test))
print("Val-Test:", len(val_test))

if not (train_val or train_test or val_test):
    print("\nSUCCESS: No identical images across splits!")
else:
    print("\nWARNING: Duplicate images still exist!")

    for digest in train_val:
        print("Train:", train[digest], "| Val:", val[digest])

    for digest in train_test:
        print("Train:", train[digest], "| Test:", test[digest])

    for digest in val_test:
        print("Val:", val[digest], "| Test:", test[digest])

print("\nFINAL SPLIT CHECK COMPLETED!")
