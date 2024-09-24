import e3x
import jax
import jax.numpy as jnp
import numpy as np
import ase.data
from scipy.spatial.distance import cdist
import pandas as pd

def cut_vdw(grid, xyz, elements, vdw_scale=2.0):
    """ """
    if type(elements[0]) == str:
        elements = [ase.data.atomic_numbers[s] for s in elements]
    vdw_radii = [ase.data.vdw_radii[s] for s in elements]
    vdw_radii = np.array(vdw_radii) * vdw_scale
    distances = cdist(grid, xyz)
    mask = distances < vdw_radii
    closest_atom = np.argmin(distances, axis=1)
    closest_atom_type = elements[closest_atom]
    mask = ~mask.any(axis=1)
    return mask, closest_atom_type

def prepare_multiple_datasets(key, num_train, num_valid, filename=["esp2000.npz"], clean=False):
    """
    Prepare multiple datasets for training and validation.

    Args:
        key: Random key for dataset shuffling.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.
        filename (list): List of filenames to load datasets from.

    Returns:
        tuple: A tuple containing the prepared data and keys.
    """
    # Load the datasets
    datasets = [np.load(f) for f in filename]

    for dataset in datasets:
        for k, v in dataset.items():
            print(k, v.shape)

    dataid = np.concatenate([dataset["id"] for dataset in datasets])
    if clean:
        failed = pd.read_csv("/pchem-data/meuwly/boittier/home/jaxeq/data/qm9-fails.csv")
        failed = list(failed["0"])
        not_failed = [i for i in range(len(dataid)) if str(dataid[i]) not in failed ]
        n_failed =  len(dataid) - len(not_failed)
        print("n_failed:", n_failed)
        num_train = max([0, num_train - n_failed])
        if num_train == 0:
            num_valid = max([0, num_valid - n_failed]) 
        print(num_train, num_valid)
        
    dataid = dataid[not_failed]
    dataR = np.concatenate([dataset["R"] for dataset in datasets])[not_failed]
    dataZ = np.concatenate([dataset["Z"] for dataset in datasets])[not_failed]
    dataN = np.concatenate([dataset["N"] for dataset in datasets])[not_failed]
    dataMono = np.concatenate([dataset["mono"] for dataset in datasets])[not_failed]
    dataEsp = np.concatenate([dataset["esp"] for dataset in datasets])[not_failed]
    dataVDW = np.concatenate([dataset["vdw_surface"] for dataset in datasets])[not_failed]
    dataNgrid = np.concatenate([dataset["n_grid"] for dataset in datasets])[not_failed]
    dataD = np.concatenate([dataset["D"] for dataset in datasets])[not_failed]
    dataDxyz = np.concatenate([dataset["Dxyz"] for dataset in datasets])[not_failed]
    dataCOM = np.concatenate([dataset["com"] for dataset in datasets])[not_failed]
    print("creating_mask")
    dataESPmask = np.array([cut_vdw(dataVDW[i], dataR[i], dataZ[i])[0] for i in range(len(dataZ))])

    data = [
        dataR,
        dataZ,
        dataN,
        dataMono,
        dataEsp,
        dataVDW,
        dataNgrid,
        dataD,
        dataDxyz,
        dataCOM,
        dataESPmask,
        dataid,
    ]
    keys = [
        "R",
        "Z",
        "N",
        "mono",
        "esp",
        "vdw_surface",
        "n_grid",
        "D",
        "Dxyz",
        "com",
        "espMask",
        "id",
    ]
    assert_dataset_size(dataR, num_train, num_valid)
    
    return data, keys, num_train, num_valid


def prepare_datasets(key, num_train, num_valid, filename, clean=False):
    """
    Prepare datasets for training and validation.

    Args:
        key: Random key for dataset shuffling.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.
        filename (str or list): Filename(s) to load datasets from.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    # Load the datasets
    if isinstance(filename, str):
        filename = [filename]

    data, keys, num_train, num_valid = prepare_multiple_datasets(key, num_train, num_valid, filename, clean=clean)

    train_choice, valid_choice = get_choices(key, len(data[0]), num_train, num_valid)

    train_data, valid_data = make_dicts(data, keys, train_choice, valid_choice)

    return train_data, valid_data


def assert_dataset_size(dataR, num_train, num_valid):
    """
    Assert that the dataset contains enough entries for training and validation.

    Args:
        dataR: The dataset to check.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.

    Raises:
        RuntimeError: If the dataset doesn't contain enough entries.
    """
    assert num_train >= 0
    assert num_valid >= 0
    # Make sure that the dataset contains enough entries.
    num_data = len(dataR)
    print(num_data)
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f"datasets only contains {num_data} points, "
            f"requested num_train={num_train}, num_valid={num_valid}"
        )


def get_choices(key, num_data, num_train, num_valid):
    """
    Randomly draw train and validation sets from the dataset.

    Args:
        key: Random key for shuffling.
        num_data (int): Total number of data points.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.

    Returns:
        tuple: A tuple containing train_choice and valid_choice arrays.
    """
    # Randomly draw train and validation sets from dataset.
    choice = np.asarray(
        jax.random.choice(key, num_data, shape=(num_data,), replace=False)
    )
    train_choice = choice[:num_train]
    valid_choice = choice[num_train : num_train + num_valid]
    return train_choice, valid_choice


def make_dicts(data, keys, train_choice, valid_choice):
    """
    Create dictionaries for train and validation data.

    Args:
        data (list): List of data arrays.
        keys (list): List of keys for the data arrays.
        train_choice (array): Indices for training data.
        valid_choice (array): Indices for validation data.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    train_data, valid_data = dict(), dict()

    for i, k in enumerate(keys):
        train_data[k] = data[i][train_choice]
        valid_data[k] = data[i][valid_choice]

    return train_data, valid_data


def print_shapes(train_data, valid_data):
    """
    Print the shapes of train and validation data.

    Args:
        train_data (dict): Dictionary containing training data.
        valid_data (dict): Dictionary containing validation data.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    print("...")
    print("...")
    for k, v in train_data.items():
        print(k, v.shape)
    print("...")
    for k, v in valid_data.items():
        print(k, v.shape)

    return train_data, valid_data


def prepare_batches(key, data, batch_size, include_id=False, data_keys=None) -> list:
    """
    Prepare batches for training.

    Args:
        key: Random key for shuffling.
        data (dict): Dictionary containing the dataset.
        batch_size (int): Size of each batch.
        include_id (bool): Whether to include ID in the output.
        data_keys (list): List of keys to include in the output.

    Returns:
        list: A list of dictionaries, each representing a batch.
    """
    # Determine the number of training steps per epoch.
    data_size = len(data["mono"])
    steps_per_epoch = data_size // batch_size

    # Draw random permutations for fetching batches from the train data.
    perms = jax.random.permutation(key, data_size)
    perms = perms[
        : steps_per_epoch * batch_size
    ]  # Skip the last batch (if incomplete).
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Prepare entries that are identical for each batch.
    num_atoms = 60
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)

    output = []
    data_keys = [
        "R",
        "Z",
        "N",
        "mono",
        "esp",
        "vdw_surface",
        "n_grid",
        "D",
        "Dxyz",
        "espMask",
        "com",
    ]
    if include_id:
        data_keys.append("id")

    for perm in perms:
        dict_ = dict()
        for k, v in data.items():
            if k in data_keys:
                if k == "R":
                    dict_[k] = v[perm].reshape(-1, 3)
                elif k == "Z":
                    dict_[k] = v[perm].reshape(-1)
                elif k == "mono":
                    dict_[k] = v[perm].reshape(-1)

                else:
                    dict_[k] = v[perm]

        dict_["dst_idx"] = dst_idx
        dict_["src_idx"] = src_idx
        dict_["batch_segments"] = batch_segments
        output.append(dict_)

    return output
