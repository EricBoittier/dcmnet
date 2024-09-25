import functools
import pickle

import e3x
import jax
import jax.numpy as jnp
import optax
from dcmnet.loss import esp_mono_loss

from dcmnet.data import prepare_batches, prepare_datasets


@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "optimizer_update", "batch_size", "esp_w", "ndcm"),
)
def train_step(
    model_apply, optimizer_update, batch, batch_size, opt_state, params, esp_w, ndcm
):
    def loss_fn(params):
        mono, dipo = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
        )
        loss = esp_mono_loss(
            dipo_prediction=dipo,
            mono_prediction=mono,
            vdw_surface=batch["vdw_surface"],
            esp_target=batch["esp"],
            mono=batch["mono"],
            ngrid=batch["n_grid"],
            n_atoms=batch["N"],
            batch_size=batch_size,
            esp_w=esp_w,
            n_dcm=ndcm,
        )
        return loss, (mono, dipo)

    (loss, (mono, dipo)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@functools.partial(
    jax.jit, static_argnames=("model_apply", "batch_size", "esp_w", "ndcm")
)
def eval_step(model_apply, batch, batch_size, params, esp_w, ndcm):
    mono, dipo = model_apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    loss = esp_mono_loss(
        dipo_prediction=dipo,
        mono_prediction=mono,
        vdw_surface=batch["vdw_surface"],
        esp_target=batch["esp"],
        mono=batch["mono"],
        ngrid=batch["n_grid"],
        n_atoms=batch["N"],
        batch_size=batch_size,
        esp_w=esp_w,
        n_dcm=ndcm,
    )
    return loss


def train_model(
    key,
    model,
    train_data,
    valid_data,
    num_epochs,
    learning_rate,
    batch_size,
    writer,
    ndcm,
    esp_w=1.0,
    restart_params=None,
    ema_decay=0.999  
):
    best = 10**7
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data["Z"][0]))
    params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    if restart_params is not None:
        params = restart_params

    opt_state = optimizer.init(params)
    # Initialize EMA parameters (a copy of the initial parameters)
    ema_params = initialize_ema_params(params)
    
    print("Preparing batches")
    print("..................")
    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    print("Training")
    print("..................")
    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)
        # Loop over train batches.
        train_loss = 0.0
        for i, batch in enumerate(train_batches):
            params, opt_state, loss = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                esp_w=esp_w,
                ndcm=ndcm,
            )

            ema_params = update_ema_params(ema_params, params, ema_decay)
            
            train_loss += (loss - train_loss) / (i + 1)

        # Evaluate on validation set.
        valid_loss = 0.0
        for i, batch in enumerate(valid_batches):
            loss = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=ema_params,
                esp_w=esp_w,
                ndcm=ndcm,
            )
            valid_loss += (loss - valid_loss) / (i + 1)

        # Print progress.
        print(f"epoch: {epoch: 3d}      train:   valid:")
        print(f"    loss [a.u.]             {train_loss : 8.3e} {valid_loss : 8.3e}")
        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        # writer.add_scalar("RMSE/train", jnp.sqrt(2 * train_loss), epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        # writer.add_scalar("RMSE/valid", jnp.sqrt(2 * valid_loss), epoch)

        if valid_loss < best:
            best = valid_loss
            # open a file, where you want to store the data
            with open(f"{writer.logdir}/best_{esp_w}_params.pkl", "wb") as file:
                pickle.dump(ema_params, file)

    # Return final model parameters.
    return params, valid_loss
