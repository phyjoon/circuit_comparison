""" Operations to see the mode connectivity in a quantum circuit.

Reference: https://arxiv.org/pdf/1802.10026.pdf

"""
import argparse
import gc
from collections import OrderedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import wandb
from scipy.special import binom

import expmgr
import qnnops


def get_beizer_curve(n_bends=3):
    """ Get a function of Beizer curve having n bends.

    Args:
        n_bends: int, number of bends including endpoints.
    Returns:
        callable, a function of time t in [0, 1] maps to coefficients.
    """

    n = jnp.arange(n_bends)
    n_rev = n_bends - 1 - n
    coef = binom(n_bends - 1, n)  # binomial coefficients

    def parametric_beizer_curve(t, bends):  # bends: (num_bends, nL)
        if not jnp.isscalar(t):
            t = t.reshape(-1, 1)  # t should be column vector
        c = coef * jnp.power(t, n) * jnp.power(1. - t, n_rev)
        return jnp.dot(c, bends)

    return parametric_beizer_curve


def save_checkpoints(tag, step, w1, w2, params, history, optimizer_state):
    bends = jnp.vstack([w1, params, w2])
    expmgr.save_array(f'bends_{tag}.npy', bends)
    expmgr.save_array(f'bends_{tag}.npy', bends)
    qnnops.save_checkpoint(
        f'checkpoint_{tag}.pkl', step, optimizer_state, history)


def find_connected_curve(
        w1, w2, loss_fn,
        n_bends=3,
        train_steps=100,
        lr=0.05,
        scheduler_name='constant',
        batch_size=4,
        log_every=1,
        seed=43,
        use_jit=False):
    """ Find a curve connecting two local minimas

    Args:
        w1: jnp.array, the first local minima
        w2: jnp.array, the second local minima
        loss_fn: callable,
        n_bends: int, number of bends between endpoints
        train_steps: int, number of training steps
        lr: float, initial learning rate
        scheduler_name: str, scheduler name
        batch_size: int, number time step samples
        log_every: int,
        seed: int, random seed
        use_jit: bool, whether to use jit-compilation of gradient function.

    Returns:
        curve function
    """

    print('Find a curve connecting two local minimas')
    curve = get_beizer_curve(n_bends + 2)  # includes the endpoints

    def curve_loss(t_, params_):  # t_ should be (m,) vector.
        params_ = jnp.vstack([w1, params_, w2])
        c = curve(t_, params_)
        loss_sum = 0
        for c_ in c:
            loss_sum += loss_fn(c_)
        return loss_sum / float(len(t_))

    # differentiate the 2nd argument
    print('Get the gradient of the loss function')
    grad_loss_fn = jax.value_and_grad(curve_loss, argnums=1)
    if use_jit:
        print('Use jit compilation')
        grad_loss_fn = jax.jit(grad_loss_fn)

    start_step = 0
    history = {'loss': [], 'grad': [], 'params': []}
    min_loss = float('inf')

    scheduler = qnnops.get_scheduler(lr, scheduler_name)
    init_fun, update_fun, get_params = qnnops.get_optimizer('adam', None, scheduler)
    # Pick evenly divided points from the line segment between w1 and w2.
    alpha = jnp.linspace(0, 1, n_bends + 2)

    init_bends = jnp.vstack([
        alpha[i] * w1 + (1 - alpha[i]) * w2
        for i in range(1, n_bends + 1)
    ])

    print('State initialization')
    optimizer_state = init_fun(init_bends)
    params = get_params(optimizer_state)
    rng = jax.random.PRNGKey(seed)

    for step in range(start_step, train_steps):
        rng, key = jax.random.split(rng)

        # Loss = E[ L(\phi_\theta(t))]  where t ~ Unif(0, 1)
        t = jax.random.uniform(key, (batch_size,))
        loss, grad = grad_loss_fn(t, params)
        optimizer_state = update_fun(step, grad, optimizer_state)
        params = get_params(optimizer_state)

        history['loss'].append(loss.item())  # scalar?
        history['grad'].append(onp.array(grad))
        history['params'].append(onp.array(params))

        if loss < min_loss:
            min_loss = loss
            save_checkpoints('best', step, w1, w2, params, history, optimizer_state)

        if step % log_every == 0:
            grad_norm = jnp.linalg.norm(grad).item()
            logging_output = OrderedDict(
                loss=loss.item(), lr=scheduler(step), grad_norm=grad_norm)
            logging_output['min_loss'] = min_loss.item()
            expmgr.log(step, logging_output)
            save_checkpoints('last', step, w1, w2, params, history, optimizer_state)

        del loss, grad
        gc.collect()


def download_checkpoints(resdir, **kwargs):
    resdir = Path(resdir)
    ising_project = 'IsingModel'
    api = wandb.Api()
    filters = {f'config.{k}': v for k, v in kwargs.items()}
    runs = api.runs(path=ising_project, filters=filters)
    if len(runs) > 2:
        print('Use only the first two runs having different seeds.')

    seeds = set()
    params_paths = []
    for run in runs:
        seed = run.config['seed']
        if seed in seeds:
            continue  # skip the seed that has seen before.

        params_file = run.file('params_best.npy')
        print(f'| Download {run.name} / {params_file.name}')
        params_file.download(resdir / run.name, replace=True)
        params_paths.append(resdir / run.name / params_file.name)
        if len(params_paths) == 2:
            break

    params_start = params_paths[0]
    params_end = params_paths[1]
    return params_start, params_end


def main():
    parser = argparse.ArgumentParser('Mode Connectivity')
    parser.add_argument('--n-qubits', type=int, metavar='N', required=True,
                        help='Number of qubits')
    parser.add_argument('--n-layers', type=int, metavar='N', required=True,
                        help='Number of alternating layers')
    parser.add_argument('--rot-axis', type=str, metavar='R', required=True,
                        choices=['x', 'y', 'z'],
                        help='Direction of rotation gates.')
    parser.add_argument('--g', type=float, metavar='M', required=True,
                        help='Transverse magnetic field')
    parser.add_argument('--h', type=float, metavar='M', required=True,
                        help='Longitudinal magnetic field')
    parser.add_argument('--n-bends', type=int, metavar='N', default=3,
                        help='Number of bends between endpoints')
    parser.add_argument('--train-steps', type=int, metavar='N', default=int(1e3),
                        help='Number of training steps. (Default: 1000)')
    parser.add_argument('--batch-size', type=int, metavar='N', default=8,
                        help='Batch size. (Default: 8)')
    parser.add_argument('--lr', type=float, metavar='LR', default=0.05,
                        help='Initial value of learning rate. (Default: 0.05)')
    parser.add_argument('--log-every', type=int, metavar='N', default=1,
                        help='Logging every N steps. (Default: 1)')
    parser.add_argument('--seed', type=int, metavar='N', required=True,
                        help='Random seed. For reproducibility, the value is set explicitly.')
    parser.add_argument('--model-seeds', type=int, metavar='N', nargs=2, required=True,
                        help='Random seed used for model training.')
    parser.add_argument('--exp-name', type=str, metavar='NAME', default=None,
                        help='Experiment name.')
    parser.add_argument('--scheduler-name', type=str, metavar='NAME',
                        default='exponential_decay',
                        help=f'Scheduler name. Supports: {qnnops.supported_schedulers()} '
                             f'(Default: constant)')
    parser.add_argument('--params-start', type=str, metavar='PATH',
                        help='A file path of a checkpoint where the curve starts from')
    parser.add_argument('--params-end', type=str, metavar='PATH',
                        help='A file path of a checkpoint where the curve ends')
    parser.add_argument('--no-jit', dest='use_jit', action='store_false',
                        help='Disable jit option to loss function.')
    parser.add_argument('--no-time-tag', dest='time_tag', action='store_false',
                        help='Omit the time tag from experiment name.')
    parser.add_argument('--quiet', action='store_true',
                        help='Quite mode (No training logs)')
    args = parser.parse_args()

    n_qubits, n_layers = args.n_qubits, args.n_layers
    if args.exp_name is None:
        args.exp_name = f'Q{n_qubits}L{n_layers}_nB{args.n_bends}'

    if args.model_seeds is None:
        params_start, params_end = args.params_start, args.params_end
    else:
        params_start, params_end = download_checkpoints(
            'checkpoints',
            n_qubits=n_qubits, n_layers=n_layers,
            lr=0.05,  # fixed lr that we used in training
            seed={'$in': args.model_seeds})

    print('Initializing project')
    expmgr.init('ModeConnectivity', args.exp_name, args)

    print('Loading pretrained models')
    w1 = jnp.load(params_start)
    w2 = jnp.load(params_end)
    expmgr.save_array('endpoint_begin.npy', w1)
    expmgr.save_array('endpoint_end.npy', w2)

    print('Constructing Hamiltonian matrix')
    ham_matrix = qnnops.ising_hamiltonian(n_qubits=n_qubits, g=args.g, h=args.h)

    print('Define the loss function')

    def loss_fn(params):
        ansatz_state = qnnops.alternating_layer_ansatz(
            params, n_qubits, n_qubits, n_layers, args.rot_axis)
        return qnnops.energy(ham_matrix, ansatz_state)

    find_connected_curve(
        w1, w2, loss_fn,
        n_bends=args.n_bends,
        train_steps=args.train_steps,
        lr=args.lr,
        scheduler_name=args.scheduler_name,
        batch_size=args.batch_size,
        log_every=1,
        seed=args.seed,
        use_jit=args.use_jit
    )


if __name__ == '__main__':
    main()
