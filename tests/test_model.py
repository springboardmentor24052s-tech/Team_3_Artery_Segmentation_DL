
import torch
from stenosis_detection.model import ResidualUNet

model = ResidualUNet()

# Create a sample 512 x 512 input image
sample = torch.randn(1, 1, 512, 512)

# Test the model
with torch.no_grad():
    output = model(sample)

print("Input shape:", sample.shape)
print("Output shape:", output.shape)

print("Total parameters:",
      sum(p.numel() for p in model.parameters()))

assert output.shape == (1, 1, 512, 512)

print("\nRESUNET TEST PASSED!")
