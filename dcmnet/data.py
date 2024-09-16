import e3x
import jax
import jax.numpy as jnp
import numpy as np


def prepare_multiple_datasets(key, num_train, num_valid, filename=["esp2000.npz"]):
    # Load the datasets
    datasets = [np.load(f) for f in filename]

    for dataset in datasets:
        for k, v in dataset.items():
            print(k, v.shape)

    dataR = np.concatenate([dataset["R"] for dataset in datasets])
    dataZ = np.concatenate([dataset["Z"] for dataset in datasets])
    dataMono = np.concatenate([dataset["mono"] for dataset in datasets])
    dataEsp = np.concatenate([dataset["esp"] for dataset in datasets])
    dataVDW = np.concatenate([dataset["vdw_surface"] for dataset in datasets])
    dataNgrid = np.concatenate([dataset["n_grid"] for dataset in datasets])
    dataid = np.concatenate([dataset["id"] for dataset in datasets])

    data = [dataR, dataZ, dataMono, dataEsp, dataVDW, dataNgrid, dataid]
    keys = ["R", "Z", "mono", "esp", "vdw_surface", "n_grid", "id"]
    assert_dataset_size(dataR, num_train, num_valid)
    return data, keys


def prepare_datasets(key, num_train, num_valid, filename=["esp2000.npz"]):
    # Load the datasets
    if isinstance(filename, str):
        filename = [filename]

    data, keys = prepare_multiple_datasets(key, num_train, num_valid, filename)

    train_choice, valid_choice = get_choices(key, len(data[0]), num_train, num_valid)

    train_data, valid_data = make_dicts(data, keys, train_choice, valid_choice)

    return train_data, valid_data


def assert_dataset_size(dataR, num_train, num_valid):

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
    # Randomly draw train and validation sets from dataset.
    choice = np.asarray(
        jax.random.choice(key, num_data, shape=(num_data,), replace=False)
    )
    train_choice = choice[:num_train]
    valid_choice = choice[num_train : num_train + num_valid]
    return train_choice, valid_choice


def make_dicts(data, keys, train_choice, valid_choice):
    train_data, valid_data = dict(), dict()

    for k in keys:
        train_data[k] = data[k][train_choice]
        valid_data[k] = data[k][valid_choice]

    return train_data, valid_data


def print_shapes(train_data, valid_data):
    print("...")
    print("...")
    for k, v in train_data.items():
        print(k, v.shape)
    print("...")
    for k, v in valid_data.items():
        print(k, v.shape)

    return train_data, valid_data


def prepare_batches(key, data, batch_size, include_id=False):
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
    num_atoms = len(data["atomic_numbers"][0])
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)
    # Assemble and return batches.
    if include_id:
        return [
            dict(
                mono=data["mono"][perm].reshape(-1),
                ngrid=data["ngrid"][perm].reshape(-1),
                esp=data["esp"][perm],  # .reshape(-1),
                vdw_surface=data["vdw_surface"][perm],  # .reshape(-1, 3),
                atomic_numbers=data["atomic_numbers"][perm].reshape(-1),
                positions=data["positions"][perm].reshape(-1, 3),
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=batch_segments,
                id=data["id"][perm],
            )
            for perm in perms
        ]
    else:
        return [
            dict(
                mono=data["mono"][perm].reshape(-1),
                ngrid=data["ngrid"][perm].reshape(-1),
                esp=data["esp"][perm],  # .reshape(-1),
                vdw_surface=data["vdw_surface"][perm],  # .reshape(-1, 3),
                atomic_numbers=data["atomic_numbers"][perm].reshape(-1),
                positions=data["positions"][perm].reshape(-1, 3),
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=batch_segments,
            )
            for perm in perms
        ]
