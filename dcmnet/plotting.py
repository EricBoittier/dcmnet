from dcmnet.loss import (
    esp_mono_loss_pots,
    esp_loss_pots,
    esp_loss_eval,
    get_predictions,
)
from dcmnet.utils import clip_colors, apply_model
from dcmnet.utils import reshape_dipole
from dcmnet.multimodel import get_atoms_dcmol
from dcmnet.multipoles import plot_3d
from dcmnet.rdkit_utils import get_mol_from_id
import numpy as np
import matplotlib.pyplot as plt
import optax
from jax import numpy as jnp
from scipy.spatial.distance import cdist
import ase
from ase.visualize.plot import plot_atoms
from rdkit.Chem import Draw

# set the default color map to RWB
plt.set_cmap("bwr")


def evaluate_dc(batch, dipo, mono, batch_size, nDCM, plot=False, rcut=100):
    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"], batch["mono"], batch_size, nDCM
    )

    mono_pred = esp_loss_pots(
        batch["positions"],
        batch["mono"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )

    non_zero = np.nonzero(batch["mono"])

    esp_errors = []
    xyzs = batch["positions"].reshape(batch_size, 60, 3)
    elems = batch["atomic_numbers"].reshape(batch_size, 60)
    monos_gt = batch["mono"].reshape(batch_size, 60)
    monos_pred = mono.reshape(batch_size, 60, nDCM)

    mols = get_mol_from_id(batch)
    images = [Draw.MolToImage(_) for _ in mols]

    for mbID in range(batch_size):

        xyz = xyzs[mbID]
        elem = elems[mbID]
        mono_gt = monos_gt[mbID]
        mono_pred_ = monos_pred[mbID]
        non_zero = np.nonzero(mono_gt)

        vdws = batch["vdw_surface"][mbID][: batch["ngrid"][mbID]]
        diff = xyzs[mbID][:, None, :] - vdws[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        min_d = np.min(r, axis=-2)
        wheremind = np.where(min_d < rcut, min_d, 0)
        idx_cut = np.nonzero(wheremind)[0]
        loss1 = (
            esp_loss_eval(
                esp_dc_pred[mbID][: batch["ngrid"][mbID]][idx_cut],
                batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut],
                batch["ngrid"][mbID],
            )
            * 627.509
        )
        loss2 = (
            esp_loss_eval(
                mono_pred[mbID][: batch["ngrid"][mbID]][idx_cut],
                batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut],
                batch["ngrid"][mbID],
            )
            * 627.509
        )
        esp_errors.append([loss1, loss2])

        if plot:

            fig = plt.figure(figsize=(12, 12))

            ax_scatter = fig.add_subplot(331)

            ax_scatter.scatter(mono_gt, mono_pred_.sum(axis=-1).squeeze())

            loss = jnp.mean(
                abs(batch["mono"][non_zero] - mono.sum(axis=-1).squeeze()[non_zero])
            )

            plt.title(f"MAE: {loss:.3f}")

            ax_scatter.plot([-1, 1], [-1, 1], c="k", alpha=0.5)
            ax_scatter.set_xlim(-1, 1)
            ax_scatter.set_ylim(-1, 1)
            ax_scatter.set_aspect("equal")

            ax_scatter2 = fig.add_subplot(332)

            ax_scatter2.scatter(
                esp_dc_pred[mbID][: batch["ngrid"][mbID]],
                batch["esp"][mbID][: batch["ngrid"][mbID]],
                alpha=0.1,
            )

            ax_scatter2.set_xlim(-0.1, 0.1)
            ax_scatter2.set_ylim(-0.1, 0.1)
            ax_scatter2.plot([-1, 1], [-1, 1], c="k", alpha=0.5)
            ax_scatter2.set_aspect("equal")

            ax_rdkit = fig.add_subplot(333, frameon=False)
            ax_rdkit.imshow(images[mbID])
            ax_rdkit.axis("off")

            ax1 = fig.add_subplot(334, projection="3d")
            s = ax1.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            ax1.set_title(f"GT {batch['id'][mbID]}")

            ax2 = fig.add_subplot(335, projection="3d")
            s = ax2.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(esp_dc_pred[mbID][: batch["ngrid"][mbID]][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            ax2.set_title(f"dcmnet: {loss1:.3f} (kcal/mol)/$e$")

            ax4 = fig.add_subplot(336, projection="3d")
            s = ax4.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(
                    esp_dc_pred[mbID][: batch["ngrid"][mbID]][idx_cut]
                    - batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]
                ),
                vmin=-0.015,
                vmax=0.015,
            )

            axmol = fig.add_subplot(337, frameon=False)
            atoms = ase.Atoms(
                numbers=elem,
                positions=xyz,
            )
            plot_atoms(atoms, axmol, rotation=("-45x,-45y,0z"), scale=1)
            axmol.axis("off")

            ax3 = fig.add_subplot(338, projection="3d")
            s = ax3.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(mono_pred[mbID][: batch["ngrid"][mbID]][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            ax3.set_title(f"mono: {loss2:.3f} (kcal/mol)/$e$")

            ax5 = fig.add_subplot(339, projection="3d")
            s = ax5.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(
                    mono_pred[mbID][: batch["ngrid"][mbID]][idx_cut]
                    - batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]
                ),
                vmin=-0.015,
                vmax=0.015,
            )

            for _ in [ax1, ax2, ax3, ax4, ax5]:
                _.set_xlim(-10, 10)
                _.set_ylim(-10, 10)
                _.set_zlim(-10, 10)

            # adjust white space
            plt.subplots_adjust(wspace=0.5, hspace=0.5)

            plt.show()

    return esp_errors, mono_pred


def plot_3d_combined(combined, batch, batch_size=1):
    xyz2 = combined[:, :3]
    q2 = combined[:, 3]
    i = 0
    nonzero = np.nonzero(batch["atomic_numbers"].reshape(batch_size, 60)[i])
    xyz = batch["positions"].reshape(batch_size, 60, 3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size, 60)[i][nonzero]

    from ase import Atoms
    from ase.visualize import view

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in q2], xyz2)
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3


def plot_model(DCM2, params, batch, batch_size, nDCM):
    mono_dc2, dipo_dc2 = apply_model(DCM2, params, batch, batch_size)

    esp_errors, mono_pred = evaluate_dc(
        batch, dipo_dc2, mono_dc2, batch_size, nDCM, plot=True
    )

    atoms, dcmol, grid, esp, esp_dc_pred = create_plots2(
        mono_dc2, dipo_dc2, batch, batch_size, nDCM
    )
    outDict = {
        "mono": mono_dc2,
        "dipo": dipo_dc2,
        "esp_errors": esp_errors,
        "atoms": atoms,
        "dcmol": dcmol,
        "grid": grid,
        "esp": esp,
        "esp_dc_pred": esp_dc_pred,
        "esp_mono_pred": mono_pred,
    }
    return outDict


def plot_esp(esp, batch, batch_size, rcut=4.0):
    mbID = 0
    xyzs = batch["positions"].reshape(batch_size, 60, 3)
    vdws = batch["vdw_surface"][mbID][: batch["ngrid"][mbID]]
    diff = xyzs[mbID][:, None, :] - vdws[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    min_d = np.min(r, axis=-2)
    wheremind = np.where(min_d < rcut, min_d, 0)
    idx_cut = np.nonzero(wheremind)[0]

    mono_pred = esp_loss_pots(
        batch["positions"],
        batch["mono"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )

    print(len(mono_pred[0][idx_cut]))

    loss_mono = optax.l2_loss(
        mono_pred[0][idx_cut] * 627.509, batch["esp"][0][idx_cut] * 627.509
    )
    loss_mono = np.mean(loss_mono * 2) ** 0.5
    loss_dc = optax.l2_loss(esp[idx_cut] * 627.509, batch["esp"][0][idx_cut] * 627.509)
    loss_dc = np.mean(loss_dc * 2) ** 0.5

    fig = plt.figure(figsize=(12, 6))

    # set white background
    fig.patch.set_facecolor("white")
    # whitebackground in 3d
    fig.patch.set_alpha(0.0)

    ax1 = fig.add_subplot(151, projection="3d")
    s = ax1.scatter(
        *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
        c=clip_colors(batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]),
        vmin=-0.015,
        vmax=0.015,
    )
    ax1.set_title(f"GT {mbID}")

    ax2 = fig.add_subplot(152, projection="3d")
    s = ax2.scatter(
        *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
        c=clip_colors(esp[idx_cut]),
        vmin=-0.015,
        vmax=0.015,
    )
    ax2.set_title(loss_dc)

    ax4 = fig.add_subplot(153, projection="3d")
    s = ax4.scatter(
        *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
        c=clip_colors(
            esp[idx_cut] - batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]
        ),
        vmin=-0.015,
        vmax=0.015,
    )

    ax3 = fig.add_subplot(154, projection="3d")
    s = ax3.scatter(
        *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
        c=clip_colors(mono_pred[mbID][: batch["ngrid"][mbID]][idx_cut]),
        vmin=-0.015,
        vmax=0.015,
    )
    ax3.set_title(loss_mono)

    ax5 = fig.add_subplot(155, projection="3d")
    s = ax5.scatter(
        *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
        c=clip_colors(
            mono_pred[mbID][: batch["ngrid"][mbID]][idx_cut]
            - batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]
        ),
        vmin=-0.015,
        vmax=0.015,
    )

    for _ in [ax1, ax2, ax3]:
        _.set_xlim(-10, 10)
        _.set_ylim(-10, 10)
        _.set_zlim(-10, 10)
    plt.show()
    return loss_dc, loss_mono


def plot_3d_combined(combined, batch, batch_size):
    xyz2 = combined[:, :3]
    q2 = combined[:, 3]
    i = 0
    nonzero = np.nonzero(batch["atomic_numbers"].reshape(batch_size, 60)[i])
    xyz = batch["positions"].reshape(batch_size, 60, 3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size, 60)[i][nonzero]

    from ase import Atoms
    from ase.visualize import view

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in q2], xyz2)
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3


def create_plots2(mono_dc2, dipo_dc2, batch, batch_size, nDCM):
    esp_dc_pred, mono_pred = get_predictions(
        mono_dc2, dipo_dc2, batch, batch_size, nDCM
    )
    dipo_dc2 = reshape_dipole(dipo_dc2, nDCM)
    atoms, dcmol, end = get_atoms_dcmol(batch, mono_dc2, dipo_dc2, nDCM)

    grid = batch["vdw_surface"][0]
    # esp = esp_dc_pred[0]
    # esp = batch["esp"][0]
    esp = esp_dc_pred[0] - batch["esp"][0]

    print(
        "rmse:",
        jnp.mean(2 * optax.l2_loss(esp_dc_pred[0] * 627.503, batch["esp"][0] * 627.503))
        ** 0.5,
    )

    xyz = batch["positions"][:end]

    cull_min = 2.5
    cull_max = 4.0
    grid_idx = np.where(np.all(cdist(grid, xyz) >= (cull_min - 1e-10), axis=-1))[0]
    print(grid_idx)
    grid_idx = jnp.asarray(grid_idx)
    grid = grid[grid_idx]
    esp = esp[grid_idx]
    grid_idx = np.where(np.all(cdist(grid, xyz) >= (cull_max - 1e-10), axis=-1))[0]
    grid_idx = [_ for _ in range(grid.shape[0]) if _ not in grid_idx]
    grid_idx = jnp.asarray(grid_idx)
    grid = grid[grid_idx]
    esp = esp[grid_idx]
    # try:
    #     display(get_rdkit(batch))
    # except:
    #     pass
    plot_3d(grid, esp, atoms=atoms + dcmol)
    return atoms, dcmol, grid, esp, esp_dc_pred[0]
