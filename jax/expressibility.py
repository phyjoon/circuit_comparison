import argparse

import jax

import expmgr
import qnnops

args = expmgr.load_config('alt_example.yml')
args = argparse.Namespace(**args)

seed = args.seed
n_qubits, n_layers, rot_axis = args.n_qubits, args.n_layers, args.rot_axis
block_size = args.block_size
exp_name = getattr(args, 'exp_name', f'Q{n_qubits}L{n_layers}_R{rot_axis}BS{block_size}')

target_state = qnnops.create_target_states(n_qubits, 1, seed=seed)
expmgr.init(
    project='expressibility',
    name=exp_name,
    config=args
)


def loss_fn(params):
    ansatz_state = qnnops.alternating_layer_ansatz(
        params, n_qubits=n_qubits, block_size=block_size, n_layers=n_layers, rot_axis=rot_axis)
    return qnnops.state_norm(ansatz_state - target_state) / (2 ** n_qubits)


rng = jax.random.PRNGKey(seed)
_, init_params = qnnops.initialize_circuit_params(rng, n_qubits, n_layers)
loss = qnnops.train_loop(loss_fn, init_params, args.train_steps, args.lr)
expmgr.save_config(args)
