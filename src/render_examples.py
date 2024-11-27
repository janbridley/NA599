from lattice import *

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from freud.plot import system_plot



if __name__ == "__main__":
    SIG_MATCH = 0.8

    # Define unit cell params for testing
    MONOCLINIC_CELL_PARAMS = {"L2": 1.4, "theta": np.pi / 5}
    HEXAGONAL_CELL_PARAMS = {"L2": 1, "theta": np.pi / 3}
    RECTANGULAR_CELL_PARAMS = {"L2": 2}
    RECTANGULAR_CENTERED_CELL_PARAMS = {"L2": 2, "centered": True}

    # box, pos = make_bravais2d((4,4, 1), L2=1, theta=2 * np.pi / 3)  # hexagonal
    # box, pos = make_bravais2d(4, **RECTANGULAR_CENTERED_CELL_PARAMS) # Centered rectangular

    box, pos = make_bravais2d(8, **MONOCLINIC_CELL_PARAMS)

    ax, _ = system_plot((box, pos))

    orthogonal_box, wrapped, (x, y) = slice_to_orthogonal(box, pos)

    ax.scatter(*wrapped.T[:2], color="cyan")
    orthogonal_box.plot(ax=ax)
    # plt.show()
    plt.close()

    # Now our data is in an orthogonal box and we can convolve it!
    fig, ax = plt.subplots(2, 3, figsize=FIGSIZE)  # , sharex=True, sharey=True)
    plt.set_cmap('cmc.acton')
    im = lattice2image(orthogonal_box, wrapped, blur_sigma=0)
    ax[0, 0].imshow(im, aspect="equal")
    # ax[0].set_title("Discretized Data (Perfect)")
    ax[0, 0].set(xticks=[], yticks=[], title="Discretized Data (Perfect)")

    # First, convolve with gaussian to verify correctness
    kernel = gkern(5)
    im_gauss = sp.signal.convolve(im, kernel, mode="same")
    ax[1, 0].imshow(im_gauss, aspect="equal")
    ax[1, 0].set(xticks=[], yticks=[], title="Convolved with Gaussian (σ=1.0)")

    # matching
    bkern = bravais_kernel(**MONOCLINIC_CELL_PARAMS, blur_sigma=SIG_MATCH)
    ax[0, 1].imshow(bkern)
    ax[0, 1].set(xticks=[], yticks=[], title="Matching unit-cell kernel")
    dict_text = "\n".join(
        [f"{key}: {value:.4f}" for key, value in MONOCLINIC_CELL_PARAMS.items()]
    )
    print(dict_text)
    ax[0, 1].text(
        0.5,
        0.5,
        dict_text,
        color="white",
        ha="center",
        va="center",
        transform=ax[0, 1].transAxes,
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "#430153", "alpha": 0.8},
    )

    im_brav = sp.signal.convolve(im, bkern, mode="same")
    ax[1, 1].imshow(im_brav, aspect="equal")
    ax[1, 1].set(xticks=[], yticks=[], title="Convolved with Matching Unit Cell")

    # non-matchin
    non_matching_bkern = bravais_kernel(**HEXAGONAL_CELL_PARAMS, blur_sigma=SIG_MATCH)
    ax[0, 2].imshow(non_matching_bkern)
    ax[0, 2].set(xticks=[], yticks=[], title="Non-matching unit-cell kernel")
    dict_text = "\n".join(
        [f"{key}: {value:.4f}" for key, value in HEXAGONAL_CELL_PARAMS.items()]
    )
    ax[0, 2].text(
        0.5,
        0.5,
        dict_text,
        color="white",
        ha="center",
        va="center",
        transform=ax[0, 2].transAxes,
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "#430153", "alpha": 0.8},
    )

    im_nm_brav = sp.signal.convolve(im, non_matching_bkern, mode="same")
    ax[1, 2].imshow(im_nm_brav, aspect="equal")
    ax[1, 2].set(xticks=[], yticks=[], title="Convolved with Incorrect Unit Cell")

    plt.savefig("figs/example_image.svg", transparent=True, bbox_inches="tight")
    plt.show()
