import torch


from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode


from SCNN import SteerableCNN, IM_SIZE


## Device:
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Build the dataset...")
from data import MnistRotDataset

# images are padded to have shape IM_SIZExIM_SIZE.
# this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
pad = Pad((0, 0, 1, 1), fill=0)

# to reduce interpolation artifacts (e.g. when testing the model on rotated images),
# we upsample an image by a factor of 3, rotate it and finally downsample it again
resize1 = Resize(IM_SIZE * 3)
resize2 = Resize(IM_SIZE)

totensor = ToTensor()

## Build the model:
print("Building Model...")
model = SteerableCNN().to(device)

# Now randomly initialized. we do not expect it to produce the right class probabilities
# BUT! Should still produce the same output for rotated versions of the same image.

## Train the model:
print("Set up model training...")
train_transform = Compose(
    [
        pad,
        resize1,
        RandomRotation(180.0, interpolation=InterpolationMode.BILINEAR, expand=False),
        resize2,
        totensor,
    ]
)

mnist_train = MnistRotDataset(mode="train", transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)


test_transform = Compose(
    [
        pad,
        totensor,
    ]
)
mnist_test = MnistRotDataset(mode="test", transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

print("Beginning training loop...")
for epoch in range(31):
    model.train()
    for i, (x, t) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(device)
        t = t.to(device)

        y = model(x)

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):
                x = x.to(device)
                t = t.to(device)

                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
        print(f"epoch {epoch} | test accuracy: {correct/total*100.}")

raw_mnist_test = MnistRotDataset(mode="test")

# retrieve the first image from the test set
x, y = next(iter(raw_mnist_test))

print("Evaluating the model...")
# evaluate the model
test_model(model, x)

print("Completed.")
