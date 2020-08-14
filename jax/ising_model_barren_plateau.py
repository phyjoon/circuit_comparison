import argparse

import jax
import jax.numpy as jnp
import numpy as onp
import wandb

import expmgr
import qnnops


from jax.config import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('Barren Plateau Test for Ising Model')
parser.add_argument('--n-qubits', type=int, metavar='N', required=True,
                    help='Number of qubits')
parser.add_argument('--n-layers', type=int, metavar='N', required=True,
                    help='Number of alternating layers')
parser.add_argument('--rot-axis', type=str, metavar='R', required=True,
                    choices=['x', 'y', 'z'],
                    help='Direction of rotation gates.')
parser.add_argument('--sample-size', type=int, metavar='N', required=True,
                    help='Size of sample set of gradients.')
parser.add_argument('--block-size', type=int, metavar='N', required=True,
                    help='Size of a block to entangle multiple qubits.')
parser.add_argument('--g', type=float, metavar='M', required=True,
                    help='Transverse magnetic field')
parser.add_argument('--h', type=float, metavar='M', required=True,
                    help='Longitudinal magnetic field')
parser.add_argument('--seed', type=int, metavar='M', required=True,
                    help='Random seed')
parser.add_argument('--jax-enable-x64', action='store_true',
                    help='Enable jax x64 option.')
parser.add_argument('--exp-name', type=str, metavar='NAME', default=None,
                    help='Experiment name. If None, the default format will be used.')
args = parser.parse_args()


seed = args.seed
n_qubits, n_layers, rot_axis = args.n_qubits, args.n_layers, args.rot_axis
block_size = args.block_size
sample_size = args.sample_size
g, h = args.g, args.h

if not args.exp_name:
    args.exp_name = f'Ising - Q{n_qubits}L{n_layers}R{rot_axis}BS{block_size} - g{g}h{h} - S{seed} - SN{sample_size}'
expmgr.init(project='VanishingGrad', name=args.exp_name, config=args)


# Construct the hamiltonian matrix of Ising model.
ham_matrix = 0

# Nearest-neighbor interaction
spin_coupling = jnp.kron(qnnops.PauliBasis[3], qnnops.PauliBasis[3])

for i in range(n_qubits - 1):
    ham_matrix -= jnp.kron(jnp.kron(jnp.eye(2 ** i), spin_coupling),
                           jnp.eye(2 ** (n_qubits - 2 - i)))
    ham_matrix -= jnp.kron(jnp.kron(qnnops.PauliBasis[3], jnp.eye(2 ** (n_qubits - 2))),
                           qnnops.PauliBasis[3])  # Periodic B.C

# Transverse magnetic field
for i in range(n_qubits):
    ham_matrix -= g * jnp.kron(jnp.kron(jnp.eye(2 ** i), qnnops.PauliBasis[1]),
                               jnp.eye(2 ** (n_qubits - 1 - i)))

# Longitudinal magnetic field
for i in range(n_qubits):
    ham_matrix -= h * jnp.kron(jnp.kron(jnp.eye(2 ** i), qnnops.PauliBasis[3]),
                               jnp.eye(2 ** (n_qubits - 1 - i)))

# Bandwidth
eigval, _ = onp.linalg.eigh(ham_matrix)
bandwidth = jnp.real(jnp.max(eigval) - jnp.min(eigval))

print(f'Bandwidth={bandwidth}')
wandb.config.bandwidth = str(bandwidth)


def circuit(params):
    return qnnops.alternating_layer_ansatz(params, n_qubits, block_size, n_layers, rot_axis)


def loss(params):
    ansatz_state = circuit(params)
    return qnnops.energy(ham_matrix, ansatz_state) / bandwidth

# Set of random seeds for parameter sampling
rng = jax.random.PRNGKey(seed)

# Collect the norms of gradients
grads = []


for step in range(sample_size):
    rng, param_rng = jax.random.split(rng)
    _, init_params = qnnops.initialize_circuit_params(param_rng, n_qubits, n_layers)
    grads.append(jax.grad(loss)(init_params))

grads = jnp.vstack(grads)

grads_mean, grads_var, grads_norm = jnp.mean(grads, axis=0), jnp.var(grads, axis=0), jnp.linalg.norm(grads, axis=1)
expmgr.save_array(expmgr.get_result_path('grads_mean.npy'), grads_mean)
expmgr.save_array(expmgr.get_result_path('grads_var.npy'), grads_var)
expmgr.save_array(expmgr.get_result_path('grads_norm.npy'), grads_norm)

wandb.config.grads_mean = str(grads_mean)
wandb.config.grads_var = str(grads_var)
wandb.config.grads_norm = str(grads_norm)


wandb.log({'means_mean': jnp.mean(grads_mean).item(),
           'means_var': jnp.var(grads_mean).item(),
           'vars_mean': jnp.mean(grads_var).item(),
           'vars_var': jnp.var(grads_var).item(),  
           'norms_mean': jnp.mean(grads_norm).item(),
           'norms_var': jnp.var(grads_norm).item()})