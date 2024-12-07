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
    RECTANGULAR_CENTERED_CELL_PARAMS = {"L2": 2,"centered": True}

    CONVOLVE_MODE = "same"#"full" # "same"

    N_KERN = (3,3)
    N_LATT = 8
    
    PX = 256
    #OUT_PX = PX * N_KERN[0]*N_KERN[1]// N_LATT**2 * 4

    SIG_MATCH = PX / 128


    # box, pos = make_bravais2d((4,4, 1), L2=1, theta=2 * np.pi / 3)  # hexagonal
    # box, pos = make_bravais2d(4, **RECTANGULAR_CENTERED_CELL_PARAMS) # Centered rectangular

    box, pos = make_bravais2d(N_LATT, **MONOCLINIC_CELL_PARAMS, sigma_noise=0.0)
    print('box.Lx: ', box.Lx) 
    print(f"N: {len(pos)}")
    # ax, _ = system_plot((box, np.zeros((0,3))))
    ax = box.plot(linestyle="--", label="Natural Basis")
    ax.scatter(*pos.T[:2], color="#70618C")

    orthogonal_box, wrapped, (x, y) = slice_to_orthogonal(box, pos)

    ax.scatter(*wrapped.T[:2], color="#C76E95")
    orthogonal_box.plot(ax=ax, label="Rectilinear Basis")
    ax.set(xticks=[], yticks=[], xlabel="", ylabel="", title="Original System")
    plt.legend()
    # plt.savefig("figs/wrapped.png", transparent=True, bbox_inches="tight", dpi=300)
    # plt.show()
    plt.close()

    # Now our data is in an orthogonal box and we can convolve it!
    fig, ax = plt.subplots(2, 4, figsize=FIGSIZE, gridspec_kw={"height_ratios": [1, 1], "width_ratios": [1.0, 1, 1, 1]}, dpi=300)
    plt.set_cmap('cmc.acton')

    midpoints = np.array([[-0.45,-0.25,0],[-0.4,0.25,0],[0.35,0.35,0]])
    # box, pos = make_polycrystalline(midpoints, theta =2*np.pi/3, n=30, L2=1.0, sigma_noise=0.0)
    box, pos, rotations = make_polycrystalline(midpoints, **MONOCLINIC_CELL_PARAMS, n=8, sigma_noise=0.0)
    
    #im = lattice2image(box, pos, blur_sigma=SIG_MATCH, px=PX)
    im = lat2im_KT(box, pos, blur_sigma=SIG_MATCH, px=PX)
    real_space_pixel_width = box.Lx / PX
    kernel = np.pad(gkern(16, sigma=SIG_MATCH),4)
    im_gauss = sp.signal.convolve(im, kernel, mode=CONVOLVE_MODE)
    bkern = bravais_kernel(**MONOCLINIC_CELL_PARAMS, blur_sigma=SIG_MATCH, 
            n=N_KERN, rot_angle = rotations[0], px_width=real_space_pixel_width)
    print('shape bkern: ',bkern.shape)
    im_brav = sp.signal.convolve(im, bkern, mode=CONVOLVE_MODE)
    non_matching_bkern = bravais_kernel(**HEXAGONAL_CELL_PARAMS, blur_sigma=SIG_MATCH, px_width=real_space_pixel_width)
    im_nm_brav = sp.signal.convolve(im, non_matching_bkern, mode=CONVOLVE_MODE)


    import torch.nn.functional as F
    import torch

    im_tensor = torch.tensor(im_gauss).float()#.unsqueeze(0).unsqueeze(0)
    im_brav_tensor = torch.tensor(im_brav).float()#.unsqueeze(0).unsqueeze(0)
    im_nm_brav_tensor = torch.tensor(im_nm_brav).float()#.unsqueeze(0).unsqueeze(0)
    mse = F.mse_loss(im_tensor, im_brav_tensor)
    print(f"RMSD matching: {mse.item()}")
    mse_nm = F.mse_loss(im_tensor, im_nm_brav_tensor)
    print(f"RMSD nonmatching: {mse_nm.item()}")

    # Combine the first column subplots (ax[0, 0] and ax[1, 0])
    ax_main = fig.add_subplot(2, 4, (1, 5))  # Combines the first column (spans rows 1 and 2)
    ax_main.imshow(im, aspect="equal")
    ax_main.set(xticks=[], yticks=[], title="Discretized system")

    
    ax[0, 0].set(xticks=[], yticks=[])
    ax[1, 0].set(xticks=[], yticks=[])
    ax[0, 0].axis('off')
    ax[1, 0].axis('off')

    # Plot gkern at [0, 1]
    ax[0, 1].imshow(kernel, aspect="equal")
    ax[0, 1].set(xticks=[], yticks=[], title=f"Gaussian Kernel (σ={SIG_MATCH})")
    # Plot blurred at [1, 1]
    ax[1, 1].imshow(im_gauss)
    ax[1, 1].set(xticks=[], yticks=[], title=f"Convolved with Gaussian (σ={SIG_MATCH})")

    # Matching unit-cell kernel at [0, 2]
    ax[0, 2].imshow(bkern)
    ax[0, 2].set(xticks=[], yticks=[], title="Matching unit-cell kernel")
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

    # Convolved with matching unit-cell kernel at [1, 2]
    ax[1, 2].imshow(im_brav, aspect="equal")
    ax[1, 2].set(xticks=[], yticks=[], title="Convolved with Matching Unit Cell")

    # Non-matching unit-cell kernel at [0, 3]
    ax[0, 3].imshow(non_matching_bkern)
    ax[0, 3].set(xticks=[], yticks=[], title="Non-matching unit-cell kernel")
    dict_text = "\n".join(
        [f"{key if key != 'theta' else 'θ'}: {(value if key != 'theta' else str(round(value/np.pi, 4))+'π')}" for key, value in HEXAGONAL_CELL_PARAMS.items()]
    )
    
    ax[0, 3].text(
        0.5,
        0.7,
        dict_text,
        color="white",
        ha="center",
        va="center",
        transform=ax[0, 3].transAxes,
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "#430153", "alpha": 0.8},
    )
    
    # Convolved with non-matching unit-cell kernel at [1, 3]
    ax[1, 3].imshow(im_nm_brav, aspect="equal")
    ax[1, 3].set(xticks=[], yticks=[], title="Convolved with Incorrect Unit Cell")
    
    plt.savefig("../figs/example2.png",transparent=True, bbox_inches="tight")
    # plt.show()

    plt.close()

    # Plotting kernels to debug
    
    plt.figure()
    fig, ax = plt.subplots(1, 2, dpi=300)
    plt.set_cmap('cmc.acton')
    ax[0].imshow(bkern, aspect="equal")
    ax[1].imshow(im_gauss[:bkern.shape[0], im_gauss.shape[1]-bkern.shape[1]:], aspect="equal")
    plt.savefig("../figs/example2_kernels.png",transparent=True, bbox_inches="tight")
    

