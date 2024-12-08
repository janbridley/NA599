# Now we will plot
# ruff: noqa: F403, F405
from lattice import *

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from freud.plot import system_plot

if __name__ == "__main__":
    FIGSIZE = 14,3.5
    # Define unit cell params for testing
    MONOCLINIC_CELL_PARAMS = {"L2": 1.17, "theta": np.pi / 5.1}
    HEXAGONAL_CELL_PARAMS = {"L2": 1, "theta": np.pi / 3}
    RECTANGULAR_CELL_PARAMS = {"L2": 2}
    RECTANGULAR_CENTERED_CELL_PARAMS = {"L2": 2, "centered": True}

    CONVOLVE_MODE = "same"#"full" # "same"

    N_KERN = (3,3)
    N_LATT = 8

    PX = 256

    SIG_MATCH = PX / 128
    box, pos = make_bravais2d(N_LATT, **MONOCLINIC_CELL_PARAMS, sigma_noise=0.0)
    
    orthogonal_box, wrapped, (x, y) = slice_to_orthogonal(box, pos)

    midpoints = np.array([[-0.45,-0.25,0],[-0.4,0.25,0],[0.35,0.35,0]])
    box, pos, rotations = make_polycrystalline(midpoints, **MONOCLINIC_CELL_PARAMS, n=8, sigma_noise=0.0)

    print('rotations: ', rotations)

    im = lattice2image(box, pos, blur_sigma=0, px=PX)
    kernel = np.pad(gkern(16, sigma=SIG_MATCH),4)
    im_gauss = sp.signal.convolve(im, kernel, mode=CONVOLVE_MODE)
    kernel = np.pad(gkern(16, sigma=SIG_MATCH),4)

    real_space_pixel_width = box.Lx / PX

    im_gauss = sp.signal.convolve(im, kernel, mode=CONVOLVE_MODE)
    bkern0 = bravais_kernel(**MONOCLINIC_CELL_PARAMS, 
            blur_sigma=SIG_MATCH, px_width=real_space_pixel_width, 
            n=N_KERN, rot_angle = rotations[0])
    im_brav0 = sp.signal.convolve(im, bkern0, mode=CONVOLVE_MODE)

    bkern1 = bravais_kernel(**MONOCLINIC_CELL_PARAMS, 
            blur_sigma=SIG_MATCH, px_width=real_space_pixel_width, 
            n=N_KERN, rot_angle = rotations[1])
    im_brav1 = sp.signal.convolve(im, bkern1, mode=CONVOLVE_MODE)

    bkern2 = bravais_kernel(**MONOCLINIC_CELL_PARAMS, 
            blur_sigma=SIG_MATCH,px_width=real_space_pixel_width, 
            n=N_KERN, rot_angle = rotations[2])
    im_brav2 = sp.signal.convolve(im, bkern2, mode=CONVOLVE_MODE)

    ## PLOTTING
    fig, ax = plt.subplots(2, 4, figsize=FIGSIZE, gridspec_kw={"height_ratios": [1, 1], "width_ratios": [1.0, 1, 1, 1]}, dpi=300)
    plt.set_cmap('cmc.acton')
    
    # Combine the first column subplots (ax[0, 0] and ax[1, 0])
    ax_main = fig.add_subplot(2, 4, (1, 5))  # Combines the first column (spans rows 1 and 2)
    ax_main.imshow(im_gauss, aspect="equal")
    ax_main.set(xticks=[], yticks=[], title="Discretized system")

    ax[0, 0].set(xticks=[], yticks=[])
    ax[1, 0].set(xticks=[], yticks=[])
    ax[0, 0].axis('off')
    ax[1, 0].axis('off')

    # Plot gkern at [0, 1]
    ax[0, 1].imshow(bkern0, aspect="equal")
    ax[0, 1].set(xticks=[], yticks=[], title=f"Kernel A")
    # Plot blurred at [1, 1]
    ax[1, 1].imshow(im_brav0)
    ax[1, 1].set(xticks=[], yticks=[], title=f"Convolution Layer A")

    # Matching unit-cell kernel at [0, 2]
    ax[0, 2].imshow(bkern1)
    ax[0, 2].set(xticks=[], yticks=[], title="Kernel B")
    '''   
    dict_text = "\n".join(
        [f"{key if key != 'theta' else 'θ'}: {(value if key != 'theta' else str(round(value/np.pi, 4))+'π')}" for key, value in MONOCLINIC_CELL_PARAMS.items()]
    )
    ax[0, 2].text(
        0.5,
        0.7,
        dict_text,
        color="white",
        ha="center",
        va="center",
        transform=ax[0, 2].transAxes,
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "#430153", "alpha": 0.8},
    )
        '''

    # Convolved with matching unit-cell kernel at [1, 2]
    ax[1, 2].imshow(im_brav1, aspect="equal")
    ax[1, 2].set(xticks=[], yticks=[], title="Convolution Layer B")

    # Non-matching unit-cell kernel at [0, 3]
    ax[0, 3].imshow(bkern2)
    ax[0, 3].set(xticks=[], yticks=[], title="Kernel C")
    '''
    dict_text = "\n".join(
        [f"{key if key != 'theta' else 'θ'}: {(value if key != 'theta' else str(round(value/np.pi, 4))+'π')}" for key, value in HEXAGONAL_CELL_PARAMS.items()]
    )
    ax[0, 3].text(
        0.5,

        dict_text,
        color="white",
        ha="center",
        va="center",
        transform=ax[0, 3].transAxes,
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "#430153", "alpha": 0.8},
    )
    '''

    # Convolved with non-matching unit-cell kernel at [1, 3]
    ax[1, 3].imshow(im_brav2, aspect="equal")
    ax[1, 3].set(xticks=[], yticks=[], title="Convolution Layer C")

    plt.savefig("../figs/paper_plot.png",transparent=True, bbox_inches="tight")
    # plt.show()

    plt.close()


