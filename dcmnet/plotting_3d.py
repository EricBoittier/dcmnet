import numpy as np
from ase import Atoms
from ase.visualize import view


def plot_3d_molecule(batch, batch_size):
    i = 0
    b1_ = batch["atomic_numbers"].reshape(batch_size, 60)[i]
    c1_ = batch["mono"].reshape(batch_size, 60)[i]
    nonzero = np.nonzero(c1_)
    i = 0
    xyz = batch["positions"].reshape(batch_size, 60, 3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size, 60)[i][nonzero]

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    return V1


def plot_3d_models(mono, dc, dcq, batch, batch_size):
    n_dcm = mono.shape[1]
    i = 0
    b1_ = batch["atomic_numbers"].reshape(batch_size, 60)[i]
    c1_ = batch["mono"].reshape(batch_size, 60)[i]
    print(b1_)
    nonzero = np.nonzero(c1_)
    i = 0
    xyz = batch["positions"].reshape(batch_size, 60, 3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size, 60)[i][nonzero]

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    idx = len(nonzero[0]) * n_dcm
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in dcq[i][:idx]], dc[i][:idx])
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3
