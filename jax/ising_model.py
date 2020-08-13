import argparse

import jax
import jax.numpy as jnp
import wandb

import expmgr
import qnnops


from jax.config import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('Expressibility Test')
parser.add_argument('--n-qubits', type=int, metavar='N', required=True,
                    help='Number of qubits')
parser.add_argument('--n-layers', type=int, metavar='N', required=True,
                    help='Number of alternating layers')
parser.add_argument('--rot-axis', type=str, metavar='R', required=True,
                    choices=['x', 'y', 'z'],
                    help='Direction of rotation gates.')
parser.add_argument('--block-size', type=int, metavar='N', required=True,
                    help='Size of a block to entangle multiple qubits.')
parser.add_argument('--g', type=float, metavar='M', required=True,
                    help='Transverse magnetic field')
parser.add_argument('--h', type=float, metavar='M', required=True,
                    help='Longitudinal magnetic field')
parser.add_argument('--train-steps', type=int, metavar='N', default=int(1e3),
                    help='Number of training steps. (Default: 1000)')
parser.add_argument('--lr', type=float, metavar='LR', default=0.01,
                    help='Initial value of learning rate. (Default: 0.01)')
parser.add_argument('--log-every', type=int, metavar='N', default=1,
                    help='Logging every N steps. (Default: 1)')
parser.add_argument('--seed', type=int, metavar='N', required=True,
                    help='Random seed. For reproducibility, the value is set explicitly.')
parser.add_argument('--exp-name', type=str, metavar='NAME', default=None,
                    help='Experiment name. If None, the following format will be used as '
                         'the experiment name: Q{n_qubits}L{n_layers}_R{rot_axis}BS{block_size}')
parser.add_argument('--jax-enable-x64', action='store_true',
                    help='Enable jax x64 option.')
parser.add_argument('--quiet', action='store_true',
                    help='Quite mode (No training logs)')
args = parser.parse_args()


seed = args.seed
n_qubits, n_layers, rot_axis = args.n_qubits, args.n_layers, args.rot_axis
block_size = args.block_size
g, h = args.g, args.h
if not args.exp_name:
    args.exp_name = f'Q{n_qubits}L{n_layers}g{g}h{h}_R{rot_axis}BS{block_size}_S{seed}_LR{args.lr}'
expmgr.init(project='IsingModel', name=args.exp_name, config=args)


# Construct the hamiltonian matrix of Ising model.
ham_matrix = jnp.zeros((2 ** n_qubits, 2 ** n_qubits))

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

jnp.save(expmgr.get_result_path('hamiltonian_matrix.npy'), ham_matrix)

eigval, eigvec = jnp.linalg.eigh(ham_matrix)
eigvec = eigvec.T  # Transpose such that eigvec[i] is an eigenvector, rather than eigenftn[:, i]
ground_state = eigvec[0]
next_to_ground_state = eigvec[1]

print("The lowest eigenvalues (energy) and corresponding eigenvectors (state)")
print(f'Ground state energy={eigval[0]}, state={ground_state}')
print(f'The next-to-ground state energy={eigval[1]}, state={next_to_ground_state}')
for i in range(2, min(5, len(eigval))):
    print(f'| {i}-th state energy={eigval[i]:.4f}')
    print(f'| {i}-th state vector={eigvec[i]}')
wandb.config.eigenvalues = str(eigval)
wandb.config.ground_state = str(ground_state)
wandb.config.next_to_ground_state = str(next_to_ground_state)


def circuit(params):
    return qnnops.alternating_layer_ansatz(params, n_qubits, block_size, n_layers, rot_axis)


def loss(params):
    ansatz_state = circuit(params)
    return qnnops.energy(ham_matrix, ansatz_state)


def monitor(params, **kwargs):  # use kwargs for the flexibility.
    ansatz_state = circuit(params)
    fidelity_with_ground_state = qnnops.fidelity(ansatz_state, ground_state)
    fidelity_with_next_to_ground = qnnops.fidelity(ansatz_state, next_to_ground_state)
    return {
        'fidelity/ground': fidelity_with_ground_state.item(),
        'fidelity/next_to_ground': fidelity_with_next_to_ground.item(),
    }


rng = jax.random.PRNGKey(seed)
_, init_params = qnnops.initialize_circuit_params(rng, n_qubits, n_layers)
trained_params, _ = qnnops.train_loop(
    loss, init_params, args.train_steps, args.lr, monitor=monitor)

optimized_state = circuit(trained_params)
print('Optimized State:', optimized_state)
wandb.config.optimized_state = str(optimized_state)
jnp.save(expmgr.get_result_path('optimized_state.npy'), optimized_state)

expmgr.save_config(args)
