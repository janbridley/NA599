import torch

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from tqdm import tqdm

from SCNN import SteerableCNN, IM_SIZE
from lattice import *
from data import SimulationData

## Device:
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Building kernels...")

# Define lattice parameters
MONOCLINIC_CELL_PARAMS = {"L2": 1.5, "theta": np.pi / 3, "centered": False}
ORTHORHOMBIC_CELL_PARAMS = {"L2": 1.2, "theta": np.pi / 2, "centered": False}
HEXAGONAL_CELL_PARAMS = {"L2": 1.0, "theta": 2 * np.pi / 3, "centered": False}
SQUARE_CELL_PARAMS = {"L2": 1.0, "theta": np.pi / 2, "centered": False}
CENTERED_CELL_PARAMS = {"L2": 1.2, "theta": np.pi / 2, "centered": True}
SIG_MATCH = 0.5  # Example blur value

# Generate kernels using the imported function
kernels = {
    "monoclinic": bravais_kernel(**MONOCLINIC_CELL_PARAMS, blur_sigma=SIG_MATCH),
    "orthorhombic": bravais_kernel(**ORTHORHOMBIC_CELL_PARAMS, blur_sigma=SIG_MATCH),
    "hexagonal": bravais_kernel(**HEXAGONAL_CELL_PARAMS, blur_sigma=SIG_MATCH),
    "square": bravais_kernel(**SQUARE_CELL_PARAMS, blur_sigma=SIG_MATCH),
    "centered": bravais_kernel(**CENTERED_CELL_PARAMS, blur_sigma=SIG_MATCH),
}


print("Build the dataset...")
dat = SimulationData(mode="test")
im, label = dat[1]
print(f"Label: {label}")
plt.imshow(im)
plt.show()

im_array = np.array(im)
best_kernel, results = test_kernels_on_image(im_array, kernels)

print(f"Best matching kernel for label {label}: {best_kernel}")
print("Similarity scores for each kernel:", results)


'''
class CrystalDataset(Dataset):
    def __init__(self, images, labels, kernels, transform=None):
        self.images = images
        self.labels = labels
        self.kernels = kernels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        kernel = self.kernels[label]

        if self.transform:
            image = self.transform(image)
            kernel = self.transform(kernel)

        return image, kernel, label




# Process both the Kernel and CNN
class KernelMatchingCNN(torch.nn.Module):
    def __init__(self):
        super(KernelMatchingCNN, self).__init__()
        self.image_branch = SteerableCNN()
        self.kernel_branch = SteerableCNN()
        self.fc = torch.nn.Linear(512, 5)

    def forward(self, image, kernel):
        image_features = self.image_branch(image)
        kernel_features = self.kernel_branch(kernel)
        combined = torch.cat((image_features, kernel_features), dim=1)
        return self.fc(combined)

# images are padded to have shape IM_SIZExIM_SIZE.
pad = Pad((0, 0, 1, 1), fill=0)
'''
