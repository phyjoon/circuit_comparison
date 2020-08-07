import jax 
import jax.numpy as jnp
from itertools import combinations
from gate_jax import *

def _block(params, qubits, state, n_qubit, rot_axis='Y'):
    if rot_axis == 'Y' or rot_axis == 'y':
        rgate = ry
    elif rot_axis == 'X' or rot_axis == 'x':
        rgate = rx
    elif rot_axis == 'Z' or rot_axis == 'z':
        rgate = rz
    else:
        raise ValueError("rot_axis should be either 'X', 'Y', or 'Z'.")

    # Rotation layer
    for qubit, param in zip(qubits, params):
        state = rgate(param, n_qubit, qubit) @ state
    
    # CZ layer
    entangler_pairs = sorted(combinations(range(len(qubits)), 2), 
                             key=lambda x: abs(x[0] - x[1]), reverse=False)
    
    for control, target in entangler_pairs:
        state = cz_gate(n_qubit, control, target) @ state

    return state


def alternating_layer_ansatz(params, n_qubit, s_block, n_layer, rot_axis='Y'):
    
    assert n_qubit % s_block == 0
    assert len(params) == n_qubit * n_layer
    
    # Initial state
    state = jnp.array([0] * (2 ** n_qubit - 1) + [1], dtype=jnp.complex64)
    
    for d in range(n_layer):
        block_idx = jnp.arange(n_qubit)
        if d % 2: 
            block_idx = jnp.roll(block_idx, -(s_block // 2))
        block_idx = jnp.reshape(block_idx, (-1, s_block))
        
        for i in range(block_idx.shape[0]):
            state = _block(params=params[block_idx[i] + d * n_qubit], 
                           qubits=block_idx[i], state=state, n_qubit=n_qubit, rot_axis=rot_axis)
        
    return state
                   