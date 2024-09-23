import e3x
import jax
import jax.numpy as jnp
import numpy as np


def prepare_multiple_datasets(key, num_train, num_valid, filename=["esp2000.npz"]):
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

    dataR = np.concatenate([dataset["R"] for dataset in datasets])
    dataZ = np.concatenate([dataset["Z"] for dataset in datasets])
    dataN = np.concatenate([dataset["N"] for dataset in datasets])
    dataMono = np.concatenate([dataset["mono"] for dataset in datasets])
    dataEsp = np.concatenate([dataset["esp"] for dataset in datasets])
    dataVDW = np.concatenate([dataset["vdw_surface"] for dataset in datasets])
    dataNgrid = np.concatenate([dataset["n_grid"] for dataset in datasets])
    dataid = np.concatenate([dataset["id"] for dataset in datasets])
    dataD = np.concatenate([dataset["D"] for dataset in datasets])
    dataDxyz = np.concatenate([dataset["Dxyz"] for dataset in datasets])
    dataCOM = np.concatenate([dataset["com"] for dataset in datasets])

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
        "id",
    ]
    assert_dataset_size(dataR, num_train, num_valid)
    return data, keys


def prepare_datasets(key, num_train, num_valid, filename):
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

    data, keys = prepare_multiple_datasets(key, num_train, num_valid, filename)

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
