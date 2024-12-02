# ruff: noqa: E701
import scipy as sp
import numpy as np
import freud
from freud.data import UnitCell

# from matplotlib.transforms import Affine2D


SEED = 1253
FIGSIZE = (14, 7)
FINAL_DATA_SHAPE = (193, 193) # Square images with odd numbers of pixels


def make_bravais2d(n, L2=1.0, theta=np.pi / 2, centered=False, sigma_noise=0.0):
    """Generate a simple bravais lattice in two dimensions.

    For simple square, set L2=1 and theta=np.pi/2
    For simple orthorhombic, set L2!=1 and theta=np.pi/2
    For simple monoclinic, set L2>0 and theta != np.pi/2
    For simple hexagonal, set L2=1 and theta = 2*np.pi/3

    For centered orthorhombic lattices, set Ly!=1 and centered=True.

    """
    if centered:
        assert theta == np.pi / 2, "Nonrectangular centered lattices are not valid!"

    L1, α, β = 1, np.pi / 2, np.pi / 2
    box = freud.box.Box.from_box_lengths_and_angles(L1, L2, 0, α, β, theta)
    #box = freud.box.Box(Lx=L1, Ly=L2, is2D=True, tilt_factors=(0, theta))
    basis_positions = [[0.5, 0.5, 0], [0.0, 0.0, 0]] if centered else [[0.5, 0.5, 0]]

    uc = UnitCell(box, basis_positions=basis_positions)
    box, pos = uc.generate_system(n, sigma_noise=sigma_noise, seed=SEED)
    return box, pos - pos.mean(axis=0)


def lattice2image(box, pos, px=64, blur_sigma=0.0, pad_width=0):
    """Convert a freud system to a 2D image with an optional Gaussian blur."""

    # Calculate rectangular image with aspect ratio matching the input box
    assert np.isclose(box.xy, 0.0), "Box should be rectangular to bin correctly!"
    px = (
        px,
        int(box.Ly / box.Lx * px),
    )  # if box.Ly > box.Lx else (int(box.Lx / box.Ly * px), px)

    image, x, y = np.histogram2d(pos[:, 0], pos[:, 1], bins=px)
    image = image.T
    x = x[:-1] + np.diff(x)
    y = y[:-1] + np.diff(y)

    # Remove points from outside the box. Should never trigger for orthogonal boxes
    BOX_PADDING_SCALE = 0.9  # Required to properly wrap points near the box edges
    coords = np.asarray(np.meshgrid(x, y)).reshape(2, -1).T
    coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    is_inside = ~box.contains(BOX_PADDING_SCALE * coords).reshape(px).T
    image[is_inside] = np.nan

    image = np.pad(
        image,
        pad_width=pad_width,
        constant_values=np.nan if np.isnan(image).any() else 0,
    )

    # TODO: image does not handle boundaries properly: real image is not square
    # image = sp.ndimage.gaussian_filter(image, sigma=blur_sigma, mode="wrap")
    image = sp.ndimage.gaussian_filter(image, sigma=blur_sigma, mode="constant")

    return image.astype(np.float32)


def slice_to_orthogonal(box, pos):
    """Convert an orthogonal (rectangular) system from a nonorthogonal one."""
    ny, theta = np.array(box.to_box_lengths_and_angles())[[1, -1]]
    x = ny * np.cos(theta)
    y = ny * np.sin(theta)

    orthogonal_box = freud.box.Box(box.Lx, box.Ly, 0, 0, 0, 0)
    return orthogonal_box, orthogonal_box.wrap(pos), (x, y)


def gkern(l=5, sigma=1.0):  # noqa: E741
    """Create a gaussian kernel with side length `l` and a sigma of `sigma`.

    Source: https://stackoverflow.com/a/43346070/21897583
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


def bravais_kernel(L2=1.0, theta=np.pi / 2, centered=False, blur_sigma=0.0, px=9):
    """Create a one unit-cell kernel from bravais lattice parameters."""
    box, pos, _ = slice_to_orthogonal(
        *make_bravais2d(n=(1, 2, 1), L2=1.0, theta=theta, centered=centered)
    )
    return lattice2image(box, pos, px=px, blur_sigma=blur_sigma, pad_width=1).T


def frame2image(
    frame: "gsd.hoomd.Frame", px=64, cut_size: tuple[int, int] | None = None
):
    """Render a GSD frame to a rectangular image."""
    box, pos = freud.box.Box(*frame.configuration.box), frame.particles.position

    im = lattice2image(*slice_to_orthogonal(box, pos)[:2], px=px)

    if cut_size is not None:
        im = im[: cut_size[0], : cut_size[1]]
    return im


# Actual MNIST data set is a raw data array - no need to save to image file! Can do npz

if __name__ == "__main__":
    """"""
    import gsd.hoomd
    import signac
    from tqdm import tqdm

    project = signac.get_project()
    job = project.open_job(id="32c4dff28f4650a0d72cdb966cffc409")

    def save_trajectory_as_images(job):
        """Save a gsd trajectory stored in a signac job to a .npz file."""

        data, args = [], []
        with gsd.hoomd.open(job.fn("trajectory.gsd"), "r") as traj:
            print(f"N: {traj[0].particles.N}")
            if traj[0].particles.N != 1024:
                raise ValueError(
                    "Images sizes won't be correct for different system sizes."
                )

            for f in tqdm(traj[::10]):
                data.append(frame2image(f, px=256))
                args.append(f"ts{f.configuration.step}")

        min_size = np.min([im.shape for im in data], axis=0)

        rect_data = [im[: FINAL_DATA_SHAPE[0], : FINAL_DATA_SHAPE[1]] for im in data]

        # Perform some basic verification
        assert (
            len(np.array(rect_data)) > 0
        ), "Verification failed! Arrays are not orthorhombic."
        assert (min_size[0] >= FINAL_DATA_SHAPE[0]) and (
            min_size[1] >= FINAL_DATA_SHAPE[1]
        )

        np.savez_compressed(job.fn("data.npz"), args=args, kwds=rect_data)
        return rect_data, args

    for job in project.find_jobs():
        save_trajectory_as_images(job)
    # plt.imshow(im)
    # plt.show()
