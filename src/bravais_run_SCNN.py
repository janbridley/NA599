import torch

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from tqdm import tqdm

from SCNN import SteerableCNN, IM_SIZE
from lattice import *
from data import SimulationData


## Define MSE function
def test_kernels_on_im(image, kernels):
    """
    Test different kernels on a single image.
    Uses Mean Square Error as the methof to determine loss.
    """
    results = {}
    image_tensor = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)

    for name, kernel in kernels.items():
        kernel_tensor = torch.tensor(kernel).float().unsqueeze(0).unsqueeze(0)
        convolved = F.conv2d(image_tensor, kernel_tensor, padding='same')
        mse = F.mse_loss(convolved, image_tensor)
        results[name] = mse.item()

    best_kernel = min(results, key=results.get)
    return best_kernel, results

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

#/anvil/scratch/x-kjensen/NA599/workspace/32c4dff28f4650a0d72cdb966cffc409

class SimulationData2(Dataset):
    def __init__(self, mode, transform=None):
        import signac

        self.transform = transform

        project = signac.get_project()
        jobs = project.find_jobs()

        self.labels = []
        self.images = []

        for job in jobs:
            data = np.load("/anvil/scratch/x-kjensen/NA599/workspace/32c4dff28f4650a0d72cdb966cffc409/data.npz")

            # Labels are stored in the data args.
            # For now, this is just timestep, but could contain an OP
            self.labels.extend([f"{job.id[:8]}_{arg}" for arg in data["args"]])
            self.images.extend(data["kwds"].astype(np.uint8))

        assert len(self.labels) == len(self.images), "Data labels do not match images!"
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        assert image.shape == FINAL_DATA_SHAPE
        image = Image.fromarray(image.astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

print("Build the dataset...")
dat = SimulationData2(mode="test")
im, label = dat[1]
print(f"Label: {label}")
plt.imshow(im)
plt.show()

im_array = np.array(im)
best_kernel, results = test_kernels_on_im(im_array, kernels)

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
