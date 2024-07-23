import pandas as pd
import dcmnet
import sys

# sys.path.append("/home/boittier/jaxeq/dcmnet")
print(sys.path)
from dcmnet.modules import MessagePassingModel
from dcmnet.data import prepare_datasets
import os
import jax
import pickle
from tensorboardX import SummaryWriter
import time
from utils import safe_mkdir
from dcmnet.training import train_model
from pathlib import Path

if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument("--data_dir", type=str, default="/home/boittier/jaxeq/data")
    args.add_argument("--model_dir", type=str, default="model")
    args.add_argument("--num_epochs", type=int, default=10000)
    args.add_argument("--learning_rate", type=float, default=0.0001)
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--esp_w", type=float, default=10.0)
    args.add_argument("--num_epics", type=int, default=100)
    args.add_argument("--restart", type=str, default=None)
    args.add_argument("--random_seed", type=int, default=0)
    args.add_argument("--n_dcm", type=int, default=1)
    args.add_argument("--n_gpu", type=str, default="0")
    args.add_argument("--data", type=str, default="qm9-esp40000-0.npz")
    args.add_argument("--ntrain", type=str, default=32000)
    args.add_argument("--nvalid", type=str, default=8000)
    args = args.parse_args()
    print("args:")
    for k, v in vars(args).items():
        print(f"{k} = {v}")

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_gpu
    devices = jax.local_devices()
    print(devices)
    print(jax.default_backend())
    print(jax.devices())

    NATOMS = 60
    data_key, train_key = jax.random.split(jax.random.PRNGKey(args.random_seed), 2)

    # Model hyperparameters.
    features = 16
    max_degree = 2
    num_iterations = 3
    num_basis_functions = 16
    cutoff = 4.0

    n_dcm = args.n_dcm
    # Training hyperparameters.
    learning_rate = args.learning_rate
    batch_size = 8
    esp_w = args.esp_w
    restart_params = args.restart
    if restart_params is not None:
        restart_params = pd.read_pickle(restart_params)
    params = restart_params
    num_epochs = args.num_epochs
    data_file = Path(args.data_dir) / args.data
    train_data, valid_data = prepare_datasets(
        data_key, args.ntrain, args.nvalid, filename=data_file
    )

    # Create and train model.
    message_passing_model = MessagePassingModel(
        features=features,
        max_degree=max_degree,
        num_iterations=num_iterations,
        num_basis_functions=num_basis_functions,
        cutoff=cutoff,
        n_dcm=n_dcm,
    )

    # make checkpoint directory
    safe_mkdir(f"/home/boittier/jaxeq/checkpoints2/dcm{n_dcm}-{esp_w}")
    isRestart = args.restart is not None
    # Set up TensorBoard writer
    log_dir = (
        "/home/boittier/jaxeq/runs2/"
        + time.strftime("%Y%m%d-%H%M%S")
        + f"dcm-{n_dcm}-espw-{esp_w}-restart-{isRestart}"
    )
    writer = SummaryWriter(log_dir)
    print(log_dir)
    with open(log_dir + "/manifest.txt", "w") as f:
        f.write(str(message_passing_model))
        for k, v in vars(args).items():
            f.write(f"\n{k} = {v}")

    for epic in range(1, args.num_epics + 1):
        print(f"epic {epic}")
        params, val = train_model(
            key=train_key,
            model=message_passing_model,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            writer=writer,
            restart_params=params,
            esp_w=esp_w * epic,
            ndcm=n_dcm,
        )

        # open a file, where you want to store the data
        with open(
            f"/home/boittier/jaxeq/checkpoints2/dcm{n_dcm}-{esp_w}/{epic}-{val}-esp_params.pkl",
            "wb",
        ) as file:
            pickle.dump(params, file)
