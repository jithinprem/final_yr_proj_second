import torch

padded = []
label = [13,33,22,55,67,53]
# label = torch.tensor(label)
for lab in label:
    padded.extend([lab])

print(padded)
