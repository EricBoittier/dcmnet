import functools

import jax
from dcmnet.electrostatics import batched_electrostatic_potential
import optax
import jax.numpy as jnp
import numpy as np
from dcmnet.modules import NATOMS
from dcmnet.utils import reshape_dipole


@functools.partial(jax.jit, static_argnames=("batch_size", "esp_w", "n_dcm"))
def esp_mono_loss(
    dipo_prediction,
    mono_prediction,
    esp_target,
    vdw_surface,
    mono,
    batch_size,
    esp_w,
    n_dcm,
):
    """ """
    l2_loss_mono = optax.l2_loss(mono_prediction.sum(axis=-1), mono)
    mono_loss = jnp.mean(l2_loss_mono)
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, NATOMS * n_dcm, 3)
    # mono = jnp.repeat(mono.reshape(batch_size, NATOMS), n_dcm, axis=-1)
    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    l2_loss = optax.l2_loss(batched_pred, esp_target)
    esp_loss = jnp.mean(l2_loss)
    return esp_loss * esp_w + mono_loss


def esp_mono_loss_pots(
    dipo_prediction, mono_prediction, vdw_surface, mono, batch_size, n_dcm
):
    """ """
    d = dipo_prediction.reshape(batch_size, NATOMS, 3, n_dcm)
    d = jnp.moveaxis(d, -1, -2)
    d = d.reshape(batch_size, NATOMS * n_dcm, 3)
    mono = jnp.repeat(mono.reshape(batch_size, NATOMS), n_dcm, axis=-1)
    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)

    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)

    return batched_pred


def esp_loss_pots(dipo_prediction, mono_prediction, vdw_surface, mono, batch_size):
    d = dipo_prediction.reshape(batch_size, NATOMS, 3)
    mono = mono.reshape(batch_size, NATOMS)
    m = mono_prediction.reshape(batch_size, NATOMS)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)

    return batched_pred


def mean_absolute_error(prediction, target, batch_size):
    nonzero = jnp.nonzero(target, size=batch_size * NATOMS)
    return jnp.mean(jnp.abs(prediction[nonzero] - target[nonzero]))


def esp_loss_eval(pred, target, ngrid):
    target = target.flatten()
    esp_non_zero = np.nonzero(target)
    l2_loss = optax.l2_loss(pred[esp_non_zero], target[esp_non_zero]) * 2
    esp_loss = np.mean(l2_loss) ** 0.5
    return esp_loss


def get_predictions(mono_dc2, dipo_dc2, batch, batch_size, n_dcm):
    mono = mono_dc2
    dipo = dipo_dc2

    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"], batch["mono"], batch_size, n_dcm
    )

    mono_pred = esp_loss_pots(
        batch["positions"],
        batch["mono"],
        # batch["esp"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )
    return esp_dc_pred, mono_pred
