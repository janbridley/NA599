import numpy as np
import freud
import coxeter
from freud.data import UnitCell
from freud.locality import Voronoi
import matplotlib.pyplot as plt
from lattice import make_bravais2d, slice_to_orthogonal
from scipy.spatial.distance import cdist

SEED = 126
def rotation_matrix(angle):
    """Create the 2D rotation matrix for the provided angle."""
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

def make_polycrystalline(midpoints, theta, n=100, L2=1.0, sigma_noise=0.0):
    """
    Generate a polycrystalline lattice using Voronoi regions.

    Parameters:
    -----------
    midpoints : numpy.ndarray
        Midpoints defining grain centers (2D array of coordinates)
    n : int, optional
        Number of points to generate
    L2 : float, optional
        Length parameter for lattice generation (default 1.0)
    sigma_noise : float, optional
        Noise parameter for lattice point generation (default 0.0)

    Returns:
    --------
    tuple : (freud.box.Box, numpy.ndarray)
        A tuple containing the box and positions of the polycrystalline lattice
    """
    midpoints = np.asarray(midpoints)
    box = freud.box.Box.from_box_lengths_and_angles(n, n, 0, np.pi/2, np.pi/2, theta)
    box, _, _ = slice_to_orthogonal(box, midpoints)
    vor = Voronoi()
    vor.compute((box, midpoints))
    plt.figure(figsize = (10,10))
    ax = plt.gca()
    vor.plot(ax=ax)
    #ax.scatter(midpoints[:, 0], midpoints[:, 1], s=10, c="k")

    all_positions = []
    # Generate lattice for each Voronoi region
    for i, midpoint in enumerate(midpoints):
        rotation_angle = np.random.uniform(0, 2*np.pi)
        rot_matrix = rotation_matrix(rotation_angle)
        if i==0:
            rot_matrix = rotation_matrix(0)
        _, region_pos = make_bravais2d(
            10*n,
            L2=L2,
            theta=theta,
            sigma_noise=sigma_noise
        )
        region_pos[:,:-1] = region_pos[:,:-1] @ rot_matrix.T
        translated_pos = region_pos #+ midpoint
        
        distances = box.compute_all_distances(translated_pos, midpoints)

        # Filter points within Voronoi region
        mask_is_closest = np.argmin(distances, axis = 1) == i
        mask_in_box = box.contains(translated_pos)
        filtered_pos = translated_pos[mask_is_closest & mask_in_box]
        wrapped_pos = box.wrap(filtered_pos)
        all_positions.append(wrapped_pos)
    
    # Combine all positions
    final_positions = np.vstack(all_positions)
    
    # Create a box that encompasses all points
    
    final_box, final_positions, _ = slice_to_orthogonal(box, final_positions)
    print(final_positions)
    ax.scatter(final_positions[:, 0], final_positions[:, 1], s=4, c="b")
    plt.savefig("voronoi_regions.png")
    return final_box, final_positions


midpoints = np.array([[-0.45,-0.25,0],[-0.4,0.25,0],[0.35,0.35,0]])
make_polycrystalline(midpoints, theta =2*np.pi/3, n=30, L2=1.0, sigma_noise=0.0)
