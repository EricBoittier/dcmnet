import numpy as np
import jax
import jax.numpy as jnp
import e3x


def prepare_datasets(key, num_train, num_valid, filename="esp2000.npz"):
    # Load the dataset.
    dataset = np.load(filename)

    for k, v in dataset.items():
        print(k, v.shape)

    dataR = dataset["R"]
    dataZ = dataset["Z"]
    dataMono = dataset["mono"]
    dataEsp = dataset["esp"]
    dataVDW = dataset["vdw_surface"]
    dataNgrid = dataset["n_grid"]
    dataid = dataset["id"]

    # Make sure that the dataset contains enough entries.
    num_data = len(dataR)
    print(num_data)
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f"datasets only contains {num_data} points, "
            f"requested num_train={num_train}, num_valid={num_valid}"
        )

    # Randomly draw train and validation sets from dataset.
    choice = np.asarray(
        jax.random.choice(key, num_data, shape=(num_draw,), replace=False)
    )
    train_choice = choice[:num_train]
    valid_choice = choice[num_train:]

    atomic_numbers = dataZ

    # Collect and return train and validation sets.
    train_data = dict(
        atomic_numbers=jnp.asarray(atomic_numbers[train_choice]),
        ngrid=jnp.array(dataNgrid[train_choice]),
        positions=jnp.asarray(dataR[train_choice]),
        mono=jnp.asarray(dataMono[train_choice]),
        esp=jnp.asarray(dataEsp[train_choice]),
        vdw_surface=jnp.asarray(dataVDW[train_choice]),
        id=dataid[train_choice],
    )
    valid_data = dict(
        atomic_numbers=jnp.asarray(atomic_numbers[valid_choice]),
        positions=jnp.asarray(dataR[valid_choice]),
        mono=jnp.asarray(dataMono[valid_choice]),
        ngrid=jnp.array(dataNgrid[valid_choice]),
        esp=jnp.asarray(dataEsp[valid_choice]),
        vdw_surface=jnp.asarray(dataVDW[valid_choice]),
        id=dataid[valid_choice],
    )
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
