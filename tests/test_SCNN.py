import torch

from escnn import gspaces
from escnn import nn

from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode

import numpy as np

from PIL import Image

# Import model
from ../SCNN import SteerableCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Download the dataset:
# download the dataset
!wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
# uncompress the zip file
!unzip -n mnist_rotation_new.zip -d mnist_rotation_new


