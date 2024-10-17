
if __name__ == "__main__":
    import argparse
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument(
        "--data_dir", type=str, default="/pchem-data/meuwly/boittier/home/jaxeq/"
    )
    args.add_argument("--model_dir", type=str, default="model")
    args.add_argument("--num_epochs", type=int, default=5_000)
    args.add_argument("--learning_rate", type=float, default=0.001)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--esp_w", type=float, default=10000.0)
    args.add_argument("--num_epics", type=int, default=1)
    args.add_argument("--n_feat", type=int, default=16)
    args.add_argument("--n_basis", type=int, default=16)
    args.add_argument("--max_degree", type=int, default=2)
    args.add_argument("--n_mp", type=int, default=2)
    args.add_argument("--restart", type=str, default=None)
    args.add_argument("--random_seed", type=int, default=0)
    args.add_argument("--n_dcm", type=int, default=1)
    args.add_argument("--n_gpu", type=str, default="0")
    args.add_argument("--data", type=str, default="qm9-esp40000-0.npz")
    args.add_argument("--n_train", type=int, default=80_000)
    args.add_argument("--n_valid", type=int, default=2_000)
    args.add_argument("--type", type=str, default="default")
    args.add_argument(
        "--include_pseudotensors", default=False, action=argparse.BooleanOptionalAction
    )
    args = args.parse_args()
    print("args:")
    for k, v in vars(args).items():
        print(f"{k} = {v}")

    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_gpu
    import jax
    devices = jax.local_devices()
    print(devices)
    print(jax.default_backend())
    print(jax.devices())

    import sys


    import pandas as pd

    import dcmnet

    print(sys.path)
    import pickle
    import time
    from pathlib import Path

    from tensorboardX import SummaryWriter
    from utils import safe_mkdir

    from dcmnet.data import prepare_datasets
    from dcmnet.modules import MessagePassingModel
    from dcmnet.training import train_model
    from dcmnet.training_dipole import train_model_dipo

    training = train_model if args.type == "default" else train_model_dipo

    NATOMS = 60

    # Model hyperparameters.
    features = args.n_feat
    max_degree = args.max_degree
    num_iterations = args.n_mp
    num_basis_functions = args.n_basis
    include_pseudotensors = args.include_pseudotensors
    cutoff = 4.0
    n_dcm = args.n_dcm
    # Training hyperparameters.
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    esp_w = args.esp_w
    restart_params = None
    num_epochs = args.num_epochs
    isRestart = args.restart is not None


    if isRestart:
        from dcmnet.analysis import create_model_and_params

        message_passing_model, restart_params, job_parms = create_model_and_params(args.restart)
        args.random_seed = int(job_parms["random_seed"])

    else:
        # Create model.
        message_passing_model = MessagePassingModel(
            features=features,
            max_degree=max_degree,
            num_iterations=num_iterations,
            num_basis_functions=num_basis_functions,
            cutoff=cutoff,
            n_dcm=n_dcm,
            include_pseudotensors=include_pseudotensors,
        )



    # data_key, train_key = jax.random.split(jax.random.PRNGKey(args.random_seed), 2)

    data_key, train_key = jax.random.split(jax.random.PRNGKey(1), 2)

    # load data
    data_file = Path(args.data_dir) / args.data
    data = [
        Path(args.data_dir) / "data/qm9-esp-dip-40000-0.npz",
        Path(args.data_dir) / "data/qm9-esp-dip-40000-1.npz",
        Path(args.data_dir) / "data/qm9-esp-dip-40000-2.npz",
        # Path(args.data_dir) / "data/spice2-esp-dip-1977-0.npz", 
    ]
    train_data, valid_data = prepare_datasets(
        data_key, args.n_train, args.n_valid, data, clean=True
        # data_key, 1877, 100, data, clean=True
    )
    n_dcm = message_passing_model.n_dcm
    args.n_dcm = n_dcm
    args.n_train = len(train_data["Z"])
    args.n_valid = len(valid_data["Z"])
    args.data = "_".join([str(_) for _ in data])
    
    # make checkpoint directory
    safe_mkdir(
        f"/pchem-data/meuwly/boittier/home/jaxeq/checkpoints2/dcm{n_dcm}-{esp_w}"
    )
    # Set up TensorBoard writer
    log_dir = (
        "/pchem-data/meuwly/boittier/home/jaxeq/all_runs/diponore/"
        + time.strftime("%Y%m%d-%H%M%S")
        + f"dcm-{n_dcm}-w-{esp_w}-re-{isRestart}-pt{message_passing_model.include_pseudotensors}"
    )
    safe_mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    print(log_dir)
    # make a note of the settings
    with open(log_dir + "/manifest.txt", "w") as f:
        f.write(str(message_passing_model))
        for k, v in vars(args).items():
            f.write(f"\n{k} = {v}")

    for epic in range(1, args.num_epics + 1):
        print(f"epic {epic}")
        params, val = training(
            key=train_key,
            model=message_passing_model,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            writer=writer,
            # restart_params=restart_params,
            restart_params=None,
            esp_w=esp_w,
            ndcm=n_dcm,
        )
