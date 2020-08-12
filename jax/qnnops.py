import itertools
from collections import OrderedDict
from datetime import datetime

import jax
import jax.numpy as jnp
import qutip
import wandb
from jax.experimental import optimizers

import expmgr
import gate_jax as gates

from jax.config import config
config.update("jax_enable_x64", True)


def create_target_states(n_qubits, n_samples, seed=None):
    """ Create multiple target states with given qubit number.

    Args:
        n_qubits: int, number of qubits
        n_samples: int, number of samples
        seed: int, random seed
    Returns:
        jnp.ndarray, state vectors of shape (n_samples, 2^n_qubits)
    """

    dim = 2 ** n_qubits
    haar_random_states = [
        qutip.rand_ket_haar(N=dim, seed=seed).get_data().toarray().T
        for _ in range(n_samples)]
    return jnp.vstack(haar_random_states)


def initialize_circuit_params(rng, n_qubits, n_layers):
    """ Initialize a state

    Args:
        rng: PRNGKey, random generation key
        n_qubits: int, number of qubits
        n_layers: int, number of layers

    Returns:
        PRNGKey, random generation key
        jnp.ndarray: initial random values in [0, 2pi)
    """

    rng, sub_rng = jax.random.split(rng)
    # TODO(jdk): Do we need to change this as a list of params rather than
    #  flatten vector? Like, [random(n_qubits) for _ in range(n_layers)]
    params = jax.random.uniform(sub_rng, (n_qubits * n_layers,)) * 2 * jnp.pi
    return rng, params


def state_norm(state):
    """ Compute the norm of a state """
    return jnp.real(jnp.sum(state * state.conj()))  # norm must be real.


def block(params, qubits, state, n_qubit, rot_axis='Y'):
    rot_axis = rot_axis.upper()
    if rot_axis == 'X':
        rotation_gate = gates.rx
    elif rot_axis == 'Y':
        rotation_gate = gates.ry
    elif rot_axis == 'Z':
        rotation_gate = gates.rz
    else:
        raise ValueError("rot_axis should be either 'X', 'Y', or 'Z'.")

    # Rotation layer
    for qubit, param in zip(qubits, params):
        state = rotation_gate(param, n_qubit, qubit) @ state

    # CZ layer
    entangler_pairs = sorted(
        itertools.combinations(range(len(qubits)), 2),
        key=lambda x: abs(x[0] - x[1]), reverse=False)

    for control, target in entangler_pairs:
        state = gates.cz_gate(n_qubit, control, target) @ state

    return state


def alternating_layer_ansatz(params, n_qubits, block_size, n_layers, rot_axis='Y'):
    # TODO(jdk): Check this function later whether we need to revise for scalability.
    rot_axis = rot_axis.upper()
    assert rot_axis in ('X', 'Y', 'Z')
    assert n_qubits % block_size == 0
    assert len(params) == n_qubits * n_layers

    # Initial state
    state = jnp.array([0] * (2 ** n_qubits - 1) + [1], dtype=jnp.complex64)

    for d in range(n_layers):
        block_idx = jnp.arange(n_qubits)
        if d % 2:
            block_idx = jnp.roll(block_idx, -(block_size // 2))
        block_idx = jnp.reshape(block_idx, (-1, block_size))

        for i in range(block_idx.shape[0]):
            state = block(params=params[block_idx[i] + d * n_qubits],
                          qubits=block_idx[i], state=state, n_qubit=n_qubits,
                          rot_axis=rot_axis)

    return state


def train_loop(loss_fn, init_params, train_steps=int(1e4), lr=0.01,
               loss_args=None, early_stopping=False, monitor=None,
               log_every=1):
    """ Training loop.

    Args:
        loss_fn: callable, loss function whose first argument must be params.
        init_params: jnp.array, initial trainable parameter values
        train_steps: int, total number of training steps
        lr: float, initial learning rate
        loss_args: dict, additional loss arguments if needed.
        early_stopping: bool, whether to early stop if the train loss value
            doesn't decrease further. (Not implemented yet)
        monitor: callable -> dict, monitoring function on training.
        log_every: int, logging every N steps.
    Returns:
        params: jnp.array, optimized parameters
        history: dict, training history.
    """

    assert monitor is None or callable(monitor), 'the monitoring function must be callable.'

    loss_args = loss_args or {}
    train_steps = int(train_steps)  # to guarantee an integer type value.
    scheduler = optimizers.inverse_time_decay(lr, train_steps, decay_rate=0.5)
    init_fun, update_fun, get_params = optimizers.adam(scheduler)
    optimizer_state = init_fun(init_params)
    history = {'loss': [], 'grad': []}
    min_loss = float('inf')
    for step in range(train_steps):
        params = get_params(optimizer_state)
        loss, grad = jax.value_and_grad(loss_fn)(params, **loss_args)
        optimizer_state = update_fun(step, grad, optimizer_state)
        updated_params = get_params(optimizer_state)

        history['loss'].append(loss)
        history['grad'].append(grad)
        if loss < min_loss:
            jnp.save(expmgr.get_result_path('checkpoint_best.npy'), updated_params)
            min_loss = loss

        if step % log_every == 0:
            grad_norm = jnp.linalg.norm(grad).item()
            logging_output = OrderedDict(loss=loss.item(), lr=scheduler(step), grad_norm=grad_norm)
            if monitor is not None:
                logging_output.update(monitor(params=params))
            logging_output['min_loss'] = min_loss.item()
            logging_str = ' | '.join('='.join([k, str(v)]) for k, v in logging_output.items())
            expmgr.log(f'Step[{step:d}]: {logging_str}')
            wandb.log(logging_output, step=step)
            wandb.run.summary['min_loss'] = min_loss.item()
            jnp.save(expmgr.get_result_path('checkpoint_last.npy'), updated_params)

        if early_stopping:
            # TODO(jdk): implement early stopping feature.
            pass
    jnp.savez(expmgr.get_result_path('history.npz'), **history)
    return get_params(optimizer_state), history


PauliBasis = jnp.array([[[1., 0., ], [0., 1., ]],
                        [[0., 1., ], [1., 0., ]],
                        [[0., -1j, ], [1j, 0., ]],
                        [[1., 0., ], [0., -1., ]]], dtype=jnp.complex128)


def energy(hamiltonian, state):
    """ Compute the energy level of a state under given hamiltonian.

    E = <s| H |s>

    Args:
        hamiltonian: jnp.array, of shape (2 ** qubit, 2 ** qubit),
            hamiltonian matrix
        state: jnp.array, of shape (2 ** qubit,) a state vector
    Returns:
        jnp.scalar, energy
    """
    return jnp.real(state.T.conj() @ hamiltonian @ state)


def fidelity(state, target_state):
    """ Compute the fidelity between two states. """
    return jnp.abs(state.T.conj() @ target_state) ** 2
