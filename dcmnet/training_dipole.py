import functools
import pickle

import e3x
import jax
import jax.numpy as jnp
import optax
from dcmnet.loss import dipo_esp_mono_loss
from dcmnet.data import prepare_batches, prepare_datasets

import jax
import jax.numpy as jnp
import optax
from collections import deque
from typing import Dict, Optional, Literal




def clip_grads_by_global_norm(grads, max_norm):
    """
    Clips gradients by their global norm.
    
    Args:
    - grads: The gradients to clip.
    - max_norm: The maximum allowed global norm.
    
    Returns:
    - clipped_grads: The gradients after global norm clipping.
    """
    # Compute the global L2 norm of the gradients
    global_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
    
    # Compute the clipping factor (ratio of max_norm to global norm)
    clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))  # Add a small value for numerical stability
    
    # Scale all gradients by the clip_factor if needed
    clipped_grads = jax.tree_map(lambda g: g * clip_factor, grads)
    
    return clipped_grads


import optax
import jax.numpy as jnp

def create_adam_optimizer_with_exponential_decay(
    initial_lr, final_lr, transition_steps, total_steps
):
    """
    Create an Adam optimizer with an exponentially decaying learning rate.
    
    Args:
    - initial_lr: Initial learning rate (e.g., 5e-4).
    - final_lr: Final learning rate (e.g., 1e-5).
    - transition_steps: How many steps before the learning rate starts decaying.
    - total_steps: The total number of training steps.
    
    Returns:
    - An Adam optimizer with exponential decay.
    """
    # Calculate the decay rate needed to go from initial_lr to final_lr over total_steps
    # decay_rate = (final_lr / initial_lr) ** (1 / total_steps)
    
    # # Learning rate schedule with exponential decay
    # lr_schedule = optax.exponential_decay(
    #     init_value=initial_lr,
    #     transition_steps=transition_steps,
    #     decay_rate=decay_rate,
    #     end_value=final_lr,  # Set the final value to explicitly stop at final_lr
    #     staircase=False  # Smooth decay, set True if you want step-wise decay
    # )

    num_cycles = 10
    lr_schedule = optax.join_schedules(schedules=
                                        [optax.cosine_onecycle_schedule(
        peak_value=0.0005 - 0.00005*i,
        transition_steps=500,
        div_factor=1.1,
        final_div_factor=2
    ) for i in range(num_cycles)], 
                                       boundaries=jnp.cumsum(jnp.array([500] * num_cycles)))


    # lr_schedule = optax.cosine_onecycle_schedule(
    #     peak_value=0.001,
    #     transition_steps=3000,
    #     div_factor=2,
    #     final_div_factor=2
    # )

    # Adam optimizer with the learning rate schedule
    optimizer = optax.adamw(learning_rate=lr_schedule)
    # optimizer = optax.adam(learning_rate=0.00001)
    # optimizer = optax.amsgrad(learning_rate=lr_schedule)
    return optimizer


def initialize_ema_params(params):
    """
    Initialize EMA parameters. Typically initialized to the same values as the initial model parameters.
    
    Args:
    - params: Initial model parameters.
    
    Returns:
    - A copy of the parameters for EMA tracking.
    """
    return jax.tree_util.tree_map(lambda p: p, params)  # Creates a copy of the initial parameters

def update_ema_params(ema_params, new_params, decay):
    """
    Update EMA parameters using exponential moving average.
    
    Args:
    - ema_params: Current EMA parameters.
    - new_params: Updated parameters from the current training step.
    - decay: Decay factor for EMA (e.g., 0.999).
    
    Returns:
    - Updated EMA parameters.
    """
    return jax.tree_util.tree_map(
        lambda ema, new: decay * ema + (1.0 - decay) * new, ema_params, new_params
    )



@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "optimizer_update", "batch_size", "esp_w", "ndcm"),
)
def train_step_dipo(
    model_apply, optimizer_update, batch, batch_size, opt_state, params, esp_w, ndcm,
     clip_norm=2.0
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
        esp_l, mono_l, dipo_l = dipo_esp_mono_loss(
            dipo_prediction=dipo,
            mono_prediction=mono,
            vdw_surface=batch["vdw_surface"],
            esp_target=batch["esp"],
            mono=batch["mono"],
            Dxyz=batch["Dxyz"],
            com=batch["com"],
            espMask=batch["espMask"],
            n_atoms=batch["N"],
            batch_size=batch_size,
            esp_w=esp_w,
            n_dcm=ndcm,
        )
        loss = esp_l + mono_l + dipo_l
        return loss, (mono, dipo, esp_l, mono_l, dipo_l)

    (loss, (mono, dipo, esp_l, mono_l, dipo_l)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Clip gradients by their global norm
    clipped_grads = clip_grads_by_global_norm(grad, clip_norm)
    updates, opt_state = optimizer_update(clipped_grads, opt_state, params)
    # updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, esp_l, mono_l, dipo_l


@functools.partial(
    jax.jit, static_argnames=("model_apply", "batch_size", "esp_w", "ndcm")
)
def eval_step_dipo(model_apply, batch, batch_size, params, esp_w, ndcm):
    mono, dipo = model_apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    esp_l, mono_l, dipo_l = dipo_esp_mono_loss(
        dipo_prediction=dipo,
        mono_prediction=mono,
        vdw_surface=batch["vdw_surface"],
        esp_target=batch["esp"],
        mono=batch["mono"],
        Dxyz=batch["Dxyz"],
        com=batch["com"],
        espMask=batch["espMask"],
        n_atoms=batch["N"],
        batch_size=batch_size,
        esp_w=esp_w,
        n_dcm=ndcm,
    )
    loss = esp_l + mono_l + dipo_l
    return loss, esp_l, mono_l, dipo_l


def train_model_dipo(
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
    # ema_decay=0.999,
):
    best = 10**7
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)

    # optimizer = optax.adam(learning_rate)
    # Example parameters for the optimizer
    initial_lr = learning_rate         
    final_lr = 1e-6           # 1 * 10^(-5)
    transition_steps = 10   # Number of steps before decaying

    # Create the Adam optimizer with the exponential decay schedule
    optimizer = create_adam_optimizer_with_exponential_decay(
        initial_lr=initial_lr,
        final_lr=final_lr,
        transition_steps=transition_steps,
        total_steps=num_epochs
    )
    
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
    # ema_params = initialize_ema_params(params)
    print("Preparing batches")
    print("..................")
    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)
        # Loop over train batches.
        train_loss = 0.0
        train_esp_l = 0.0
        train_mono_l = 0.0
        train_dipo_l = 0.0
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, esp_l, mono_l, dipo_l = train_step_dipo(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                esp_w=esp_w,
                ndcm=ndcm,
            )
            train_loss += (loss - train_loss) / (i + 1)
            train_esp_l += (esp_l - train_esp_l) / (i + 1)
            train_mono_l += (mono_l - train_mono_l) / (i + 1)
            train_dipo_l += (dipo_l - train_dipo_l) / (i + 1)
            
        del train_batches
        
        # Evaluate on validation set.
        valid_loss = 0.0
        valid_esp_l = 0.0
        valid_mono_l = 0.0
        valid_dipo_l = 0.0
        for i, batch in enumerate(valid_batches):
            loss, esp_l, mono_l, dipo_l = eval_step_dipo(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=params,
                esp_w=esp_w,
                ndcm=ndcm,
            )
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_esp_l += (esp_l - valid_esp_l) / (i + 1)
            valid_mono_l += (mono_l - valid_mono_l) / (i + 1)
            valid_dipo_l += (dipo_l - valid_dipo_l) / (i + 1)

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("esp_l/train", train_esp_l, epoch)
        writer.add_scalar("esp_l/valid", valid_esp_l, epoch)
        writer.add_scalar("mono_l/train", train_mono_l, epoch)
        writer.add_scalar("mono_l/valid", valid_mono_l, epoch)
        writer.add_scalar("dipo_l/train", train_dipo_l, epoch)
        writer.add_scalar("dipo_l/valid", valid_dipo_l, epoch)

        if valid_loss < best:
            best = valid_loss
            # open a file, where you want to store the data
            with open(f"{writer.logdir}/best_{esp_w}_params.pkl", "wb") as file:
                pickle.dump(params, file)
                
        writer.add_scalar("Loss/bestValid", best, epoch)

    # Return final model parameters.
    return params, valid_loss
