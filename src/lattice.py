# ruff: noqa: E701
import scipy as sp
import numpy as np
import freud
from freud.data import UnitCell
from freud.locality import Voronoi
import matplotlib.pyplot as plt
# from matplotlib.transforms import Affine2D


SEED = 1253
np.random.seed(SEED)

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

    L1, alpha, beta = 1, np.pi / 2, np.pi / 2
    box = freud.box.Box.from_box_lengths_and_angles(L1, L2, 0, alpha, beta, theta)
    #box = freud.box.Box(Lx=L1, Ly=L2, is2D=True, tilt_factors=(0, theta))
    basis_positions = [[0.5, 0.5, 0], [0.0, 0.0, 0]] if centered else [[0.5, 0.5, 0]]

    uc = UnitCell(box, basis_positions=basis_positions)
    box, pos = uc.generate_system(n, sigma_noise=sigma_noise, seed=SEED)
    return box, pos - pos.mean(axis=0)

def lat2im_KT(box, pos, px=64, blur_sigma=0.0, pad_width=0):
    assert np.isclose(box.xy, 0.0), "Box should be rectangular to bin correctly!"
    bin_width = box.Lx / px
    x_max = max(pos[:,0])
    x_min = min(pos[:,0])
    x_bins = bin_width * np.arange(np.floor(x_min / bin_width), np.ceil(x_max / bin_width) + 1)
    print('px: ',px)
    print('len: ',len(x_bins))
    #print(x_bins)
    y_max = max(pos[:,1])
    y_min = min(pos[:,1])
    y_bins = bin_width * np.arange(np.floor(y_min / bin_width), np.ceil(y_max / bin_width) + 1)
    
    plt.figure(figsize=(20,20))
    plt.hlines(y_bins, min(pos[:,0]), max(pos[:,0]), color='k', alpha=.5)
    plt.vlines(x_bins, min(pos[:,1]), max(pos[:,1]), color='k', alpha=.5)
    plt.scatter(pos[:, 0], -pos[:, 1])
    #plt.scatter(x,np.zeros_like(x))
    plt.axis('equal')
    plt.savefig(f"../figs/pixelspace_NEW_{pos[3,0]:.2f}.png")
    plt.close()

    image, x, y = np.histogram2d(pos[:, 0], pos[:, 1], bins=(x_bins,y_bins))
    image = image.T
    
    plt.figure()
    plt.set_cmap('cmc.acton')
    plt.imshow(image, aspect="equal")
    plt.axis('equal')
    plt.savefig(f"../figs/KT_hist_{pos[3,0]:.2f}.png",transparent=True, bbox_inches="tight")
    plt.close()
    
    # Remove points from outside the box. Should never trigger for orthogonal boxes
    if not np.isclose(box.xy,0.0):
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


#def bravais_kernel(L2=1.0, theta=np.pi / 2, centered=False, blur_sigma=0.0, px=9, n=(3,3)):
def bravais_kernel(L2=1.0, theta=np.pi / 2, centered=False, blur_sigma=0.0, n=(3,3), rot_angle = 0, px_width=1):
    """Create a one unit-cell kernel from bravais lattice parameters."""
    print(f"L2: {L2}")
    print(f"theta: {theta/np.pi/np.pi:.4f}π")
    #print(f"px: {px}")
    box, pos = make_bravais2d(n=(n[0]+2, n[1]+2, 1), L2=L2, theta=theta, centered=centered)
    print('len pos 0: ', len(pos))
    #box, pos, _ = slice_to_orthogonal(
    #    *make_bravais2d(n=(*n, 1), L2=L2, theta=theta, centered=centered))
    
    box, unrot_pos, _ = slice_to_orthogonal(box, pos)
    
    #pos[:,:-1] = unrot_pos[:,:-1] @ rotation_matrix(rot_angle).T
    pos[:,:-1] = unrot_pos[:,:-1] @ rotation_matrix(rot_angle)
    
    print('rot_angle: ', rot_angle)
    
    '''
    plt.scatter(unrot_pos[:,0],-unrot_pos[:,1], s=1)
    plt.scatter(pos[:,0],-pos[:,1], s=1)
    plt.axis('equal')
    plt.savefig(f"../figs/kpos_{rot_angle:.2f}.png",transparent=True, dpi=300, bbox_inches="tight")
    plt.close()
    '''

    pos = np.array(pos)
    rad = pos[:,0]**2 + pos[:,1]**2
    sort_idx = np.argsort(rad)
    close_idx = sort_idx[:(n[0] * n[1])]
    pos = pos[close_idx]
    print(pos) 
    print(box)
    print('len pos 1: ', len(pos))
   
    '''
    plt.scatter(pos[:,0], -pos[:,1])
    plt.axis('equal')
    plt.savefig(f'../figs/example2_pts_final_{rot_angle:.2f}.png',transparent=True, bbox_inches="tight")
    plt.close()
    '''

    x_min = min(pos[:,0])
    x_max = max(pos[:,0])

    px = np.ceil((box.Lx/2)/px_width - (-box.Lx/2)/px_width) # number of pixels depends on rotation of point before image

    #return lattice2image(box, pos, px=px, blur_sigma=blur_sigma, pad_width=4)
    return lat2im_KT(box, pos, px=px, blur_sigma=blur_sigma, pad_width=4)


def frame2image(
    frame: "gsd.hoomd.Frame", px=64, cut_size: tuple[int, int] | None = None
):
    """Render a GSD frame to a rectangular image."""
    box, pos = freud.box.Box(*frame.configuration.box), frame.particles.position

    im = lattice2image(*slice_to_orthogonal(box, pos)[:2], px=px)

    if cut_size is not None:
        im = im[: cut_size[0], : cut_size[1]]
    return im

def rotation_matrix(angle):
    """Create the 2D rotation matrix for the provided angle."""
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

def make_polycrystalline(midpoints, theta, n=100, L2=1.0, sigma_noise=0.0):
    """
    Generate a polycrystalline lattice using Voronoi region
    """
    midpoints = np.asarray(midpoints)
    box = freud.box.Box.from_box_lengths_and_angles(n, n, 0, np.pi/2, np.pi/2, theta)
    box, _, _ = slice_to_orthogonal(box, midpoints)
    vor = Voronoi()
    vor.compute((box, midpoints))
    # plt.figure(figsize = (10,10))
    # ax = plt.gca()
    # vor.plot(ax=ax)
    #ax.scatter(midpoints[:, 0], midpoints[:, 1], s=10, c="k")

    all_positions = []
    rotations = []
    # Generate lattice for each Voronoi region
    for i, midpoint in enumerate(midpoints):
        rotation_angle = np.random.uniform(0, 2*np.pi)
        rotations.append(rotation_angle)
        rot_matrix = rotation_matrix(rotation_angle)
        if i==0:
            rot_matrix = rotation_matrix(0)
        _, region_pos = make_bravais2d(
            10*n, # Jank -- fix this if you ever use this in the future.
            L2=L2,
            theta=theta,
            sigma_noise=sigma_noise
        )

        
        #region_pos[:,:-1] = region_pos[:,:-1] @ rot_matrix.T
        region_pos[:,:-1] = region_pos[:,:-1] @ rot_matrix
        
        translated_pos = region_pos #+ midpoint

        '''
        plt.scatter(translated_pos[:,0],-translated_pos[:,1], s=.3)
        plt.axis('equal')
        plt.xlim(0,4)
        plt.ylim(-2,2)
        plt.savefig("../figs/pos_before_cut.png",transparent=True, dpi=500, bbox_inches="tight")
        plt.close()
        '''
        
        distances = box.compute_all_distances(translated_pos, midpoints)
        
        # Filter points within Voronoi region
        mask_is_closest = np.argmin(distances, axis = 1) == i
        mask_in_box = box.contains(translated_pos)
        filtered_pos = translated_pos[mask_is_closest & mask_in_box]
        wrapped_pos = box.wrap(filtered_pos)
        all_positions.append(wrapped_pos)
    
    rotations[0] = 0
    print('rotations end of makepoly:\n', rotations)

    # Combine all positions
    final_positions = np.vstack(all_positions)
    '''
    plt.scatter(final_positions[:,0],-final_positions[:,1], s=.4)
    plt.axis('equal')
    plt.savefig("../figs/pos_after_cut.png",transparent=True, dpi=300, bbox_inches="tight")
    plt.close()
    '''
    
    # Create a box that encompasses all points
    
    final_box, final_positions, _ = slice_to_orthogonal(box, final_positions)
    # ax.scatter(final_positions[:, 0], final_positions[:, 1], s=4, c="b")
    return final_box, final_positions, rotations


def lattice2image(box, pos, px=64, blur_sigma=0.0, pad_width=0):
    """Convert a freud system to a 2D image with an optional Gaussian blur."""

    # Calculate rectangular image with aspect ratio matching the input box
    assert np.isclose(box.xy, 0.0), "Box should be rectangular to bin correctly!"
    px = (px, int(box.Ly / box.Lx * px),)  
    # if box.Ly > box.Lx else (int(box.Lx / box.Ly * px), px)
    '''
    plt.scatter(pos[:, 0], -pos[:, 1]) 
    plt.axis('equal')
    plt.savefig(f"../figs/Porfavormantegasusmanos{pos[3,0]:.2f}.png")
    plt.close()
    '''
    image, x, y = np.histogram2d(pos[:, 0], pos[:, 1], bins=px)
    image = image.T
    x = x[:-1] + np.diff(x)
    y = y[:-1] + np.diff(y)
    print(np.diff(x))


    # Remove points from outside the box. Should never trigger for orthogonal boxes
    BOX_PADDING_SCALE = 0.9  # Required to properly wrap points near the box edges
    ''' 
    plt.figure(figsize=(20,20))
    plt.hlines(y,min(pos[:,1]),max(pos[:,1]), color='k', alpha=.5) 
    plt.vlines(x,min(pos[:,0]),max(pos[:,0]), color='k', alpha=.5)
    plt.scatter(pos[:, 0], -pos[:, 1]) 
    #plt.scatter(x,np.zeros_like(x))
    plt.axis('equal')
    plt.savefig(f"../figs/pixelspace_{pos[3,0]:.2f}.png")
    plt.close()
    
    plt.figure()
    plt.set_cmap('cmc.acton')
    plt.imshow(image, aspect="equal")
    plt.axis('equal')
    plt.savefig(f"../figs/example2_image_b4pad_{pos[3,0]:.2f}.png",transparent=True, bbox_inches="tight")
    plt.close()
    '''
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
#midpoints = np.array([[-0.45,-0.25,0],[-0.4,0.25,0],[0.35,0.35,0]])
#make_polycrystalline(midpoints, theta =2*np.pi/3, n=30, L2=1.0, sigma_noise=0.0)

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
