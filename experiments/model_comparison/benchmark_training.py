
import time
import torch
from torch.utils.data import DataLoader, Subset

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.train import PatchDataset, CombinedLoss
from stenosis_detection.model import ResidualUNet


torch.set_num_threads(4)

DEVICE = torch.device("cpu")
BATCH_SIZE = 2
WARMUP_BATCHES = 1
TIMED_BATCHES = 5

print("Loading training dataset...")

base_dataset = StenosisDataset("train")
train_dataset = PatchDataset(
    base_dataset,
    patches_per_image=2
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Total batches per epoch:", len(train_loader))

print("\nInitializing model...")

model = ResidualUNet().to(DEVICE)
criterion = CombinedLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

model.train()
iterator = iter(train_loader)

print("\nRunning warm-up batch...")

for _ in range(WARMUP_BATCHES):
    images, masks = next(iterator)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()

print("Warm-up completed.")

print("\nBenchmarking training batches...")

start_time = time.perf_counter()

for step in range(TIMED_BATCHES):

    images, masks = next(iterator)

    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, masks)

    loss.backward()
    optimizer.step()

    print(
        f"Batch {step + 1}/{TIMED_BATCHES} "
        f"| Loss: {loss.item():.4f}"
    )

elapsed = time.perf_counter() - start_time

seconds_per_batch = elapsed / TIMED_BATCHES
estimated_epoch = seconds_per_batch * len(train_loader)

print("\nBENCHMARK RESULTS")
print("-----------------")
print(f"Time for {TIMED_BATCHES} batches: {elapsed:.2f} seconds")
print(f"Average time per batch: {seconds_per_batch:.2f} seconds")
print(f"Estimated epoch time: {estimated_epoch / 60:.2f} minutes")
print(f"Estimated 30 epochs: {estimated_epoch * 30 / 3600:.2f} hours")

print("\nCPU BENCHMARK COMPLETED!")
