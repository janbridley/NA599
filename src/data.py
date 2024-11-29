from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from lattice import FINAL_DATA_SHAPE


class MnistRotDataset(Dataset):
    def __init__(self, mode, transform=None):
        assert mode in ["train", "test"]

        if mode == "train":
            file = "../data/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "../data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

        print("Data files successfully found.")

        self.transform = transform

        data = np.loadtxt(file, delimiter=" ")

        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image, mode="F")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class SimulationData(Dataset):
    def __init__(self, mode, transform=None):
        import signac

        self.transform = transform

        project = signac.get_project()
        jobs = project.find_jobs()

        self.labels = []
        self.images = []

        for job in jobs:
            print('Transforming job ', job, ' out of ', len(jobs), '.')
            data = np.load(job.fn("data.npz"))

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dat = SimulationData(mode="test")
    im, label = dat[1]
    print(f"Label: {label}")
    plt.imshow(im)
    plt.show()
