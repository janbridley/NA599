import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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

from SCNN import SteerableCNN

'''
ABOUT test_SCNN: 

    The SCNN model is randomly initialized. Therefore, we do not expect it to produce the right class probabilities.However, the model should still produce the same output for rotated versions of the same image. This is true for rotations by multiples of pi/2, but is only approximate for rotations by pi/4. Here, we feed eight rotated versions of the first image in the test set and print the output logits of the model for each of them.

    The output of the model is already almost invariant. However, we still observe small fluctuations in the outputs.This is because the model contains some operations which might break equivariance. For instance, every convolution includes a padding of 
 pixels per side. This is adds information about the actual orientation of the grid where the image/feature map is sampled because the padding is not rotated with the image.

'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Building dataset...")
## Build the dataset
class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "../data/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "../data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image, mode='F')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

# images are padded to have shape 29x29.
# this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
pad = Pad((0, 0, 1, 1), fill=0)

# to reduce interpolation artifacts (e.g. when testing the model on rotated images),
# we upsample an image by a factor of 3, rotate it and finally downsample it again
resize1 = Resize(87)
resize2 = Resize(29)

totensor = ToTensor()

print("Building model...")
## Build the model
model = SteerableCNN().to(device)

## Test it on a random image rotated 8 times. 
# should procude same output for rotated version of same image.

def test_model(model: torch.nn.Module, x: Image):
    np.set_printoptions(linewidth=10000)

    # evaluate the `model` on 8 rotated versions of the input image `x`
    model.eval()

    x = resize1(pad(x))

    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(8):
            x_transformed = totensor(resize2(x.rotate(r*45., Image.BILINEAR))).reshape(1, 1, 29, 29)
            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()

            angle = r * 45
            print("{:5d} : {}".format(angle, y))
    print('##########################################################################################')
    print()


print("Building test set...")
# build the test set
raw_mnist_test = MnistRotDataset(mode='test')

# retrieve the first image from the test set
x, y = next(iter(raw_mnist_test))

print("Evaluating the model")
# evaluate the model
test_model(model, x)
