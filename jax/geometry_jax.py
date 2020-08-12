import jax
import jax.numpy as jnp
import numpy as onp

import matplotlib.pyplot as plt
from circuit_ansatz_jax import alternating_layer_ansatz
from jax.config import config
config.update("jax_enable_x64", True)

def fidelity(params, n_qubit, s_block, n_layer, rot_axis, target_state):
    ansatz_state = alternating_layer_ansatz(params, n_qubit, s_block, n_layer, rot_axis)
    return jnp.float32(jnp.abs(ansatz_state.T.conj() @ target_state) ** 2)


def energy(hamiltonian, state):
    return jnp.real(state.T.conj() @ hamiltonian @ state)

def fidelity_local_plot(ham_matrix, state_0, state_1, origin, direction_0, direction_1, 
                        n_qubit, s_block, n_layer, rot_axis='Y'):

    @jax.vmap
    def state_along_flat_0(alpha, beta):
        state = origin + alpha * direction_0 + beta * direction_1
        return fidelity(state, n_qubit, s_block, n_layer, rot_axis, target_state=state_0)

    @jax.vmap
    def state_along_flat_1(alpha, beta):
        state = origin + alpha * direction_0 + beta * direction_1
        return fidelity(state, n_qubit, s_block, n_layer, rot_axis, target_state=state_1)

    @jax.vmap
    def energy_along_flat(alpha, beta):
        state = alternating_layer_ansatz(origin + alpha * direction_0 + beta * direction_1, 
                                         n_qubit, s_block, n_layer, rot_axis)
        return energy(ham_matrix, state)


    alpha = onp.linspace(-2, 2, 50)
    beta = onp.linspace(-2, 2, 50)

    A, B = onp.meshgrid(alpha, beta)
    Z0 = state_along_flat_0(A.flatten(), B.flatten()).reshape(50, 50)
    Z1 = state_along_flat_1(A.flatten(), B.flatten()).reshape(50, 50)
    E0 = energy_along_flat(A.flatten(), B.flatten()).reshape(50, 50)

    vv = onp.linspace(0, 1.0, 50, endpoint=True)
    fig, ax = plt.subplots(2, 2, figsize=(12,12))
    f1 = ax[0,0].contour(A, B, Z0, vv, cmap='magma')
    ax[0,0].set_title('Fidelity with the ground state')
    plt.clabel(f1, inline=True, fontsize=8)
    # plt.colorbar(f1, ax=ax[0,0], ticks=vv)
    f2 = ax[0,1].contour(A, B, Z1, vv, cmap='magma')
    ax[0,1].set_title('Fidelity with the next-to-ground state')
    # plt.colorbar(f2, ax=ax[0,1])
    plt.clabel(f2, inline=True, fontsize=8)
    f3 = ax[1,0].contour(A, B, Z0 + Z1, vv, cmap='magma')
    ax[1,0].set_title('Fidelity with the two lowest states')
    # plt.colorbar(f3, ax=ax[1,0])
    plt.clabel(f3, inline=True, fontsize=8)
    ax[1,1].set_title('Energy (loss) landscape')
    f4 = ax[1,1].contour(A, B, E0, 30, cmap='magma')
    # plt.colorbar(f4)
    plt.clabel(f4, inline=True, fontsize=8)

    fig.colorbar(f1, ax=ax.flatten().ravel(), ticks=vv)
    plt.show()
    