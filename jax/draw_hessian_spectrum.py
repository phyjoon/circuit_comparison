""" Drawing Hessian spectrum plots

Usage examples
 $ python draw_hessian_spectrum.py --n-qubits 6 --n-layers-list 6 12 --lr 0.05

"""
import argparse
import shutil
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

import qnnops


def iterate_artifacts(project, filters=None):
    filters = filters or {}
    filters = {f'config.{k}': v for k, v in filters.items()}
    api = wandb.Api()
    runs = api.runs(project, filters=filters)
    for run in runs:
        for file in run.files():
            yield run, file


def get_normalized_name(config):
    return f'Q{config["n_qubits"]}L{config["n_layers"]}' \
           f'_LR{config["lr"]}_S{config["seed"]}'


def download_circuits(project, filters=None, resdir='results_hessian'):
    resdir = Path(resdir)
    if not resdir.exists():
        resdir.mkdir(parents=True)
    pretrained_fname = 'params_best.npy'
    print(f'Download pretrained models in {project}')
    print(f'| target circuits: {filters}')
    print(f'| pretrained name: {pretrained_fname}')
    opt_circuits = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for run, file in iterate_artifacts(project, filters):
            if file.name == pretrained_fname:  # use the best one.
                f = file.download(tmpdir)
                src = f.name
                dst = resdir / f'{get_normalized_name(run.config)}.npy'
                shutil.move(src, dst)
                print(f'| ...downloading from [{run.name}:{file.name}] to [{dst}]')
                opt_circuits.append((run.config, dst))
    return opt_circuits


def get_loss_fn(ham_matrix, n_qubits, n_layers, rot_axis):

    def loss(params):
        ansatz_state = qnnops.alternating_layer_ansatz(
            params, n_qubits, n_qubits, n_layers, rot_axis)
        return qnnops.energy(ham_matrix, ansatz_state)

    return loss


def compute_hessian_eigenvalues(hessian_fn, x):
    """ Compute eigenvalues of given cost function.

    Args:
        hessian_fn: Callable, the hessian function of the cost function.
        x: array, vector where to compute Hessian.

    Returns:
        DeviceArray, eigenvalues.
    """
    hessian_mat = hessian_fn(x)
    eigvals = jnp.linalg.eigvals(hessian_mat)
    # All eigenvalues of a hessian matrix must be real
    # because of the symmetry of the Hessian matrix.
    eigvals = jnp.real(eigvals)
    eigvals = -jnp.sort(-eigvals)  # trick to sort in descending order.
    return hessian_mat, eigvals


def plot_histogram(eigenvalues, save_path=None):
    """ Plot and save a histogram of eigenvalues."""
    plt.figure()
    plt.hist(eigenvalues)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.close()
    return eigenvalues


def plot_spectrum(eigenvalues, save_path=None):
    """ Plot and save eigenvalue spectrum."""
    plt.figure()
    plt.bar(jnp.arange(len(eigenvalues)), eigenvalues)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.close()
    return eigenvalues


def get_arguments():
    parser = argparse.ArgumentParser('Hessian spectrum')
    parser.add_argument('--project', type=str, default='IsingModel',
                        help='Project name where the trained model')
    parser.add_argument('--n-qubits', type=int, metavar='N', required=True,
                        help='Number of qubits')
    parser.add_argument('--n-layers-list', type=int, nargs='+',
                        metavar='N',
                        help='List of target number of alternating layers. '
                             'If None, do not filter on n_layers'
                             'Ex) --n-layers-list 6 12 18 24')
    parser.add_argument('--lr', type=float, metavar='LR', default=None,
                        help='Constraint: Learning rate value. Do not filter '
                             'on the learning rate if None (Default: None)')
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    project = args.project
    n_qubits, g, h = args.n_qubits, 2, 0
    filters = dict(n_qubits=n_qubits, g=g, h=h)
    if args.n_layers_list is not None:
        print(f'Add [n_layers] filter: {args.n_layers_list}')
        filters['n_layers'] = {'$in': args.n_layers_list}
    if args.lr is not None:
        print(f'Add [lr] filter: {args.lr}')
        filters['lr'] = args.lr
    resdir = Path('results_hessian')
    opt_circuits = download_circuits(project, filters, resdir)

    ham_matrix = qnnops.ising_hamiltonian(n_qubits, g, h)

    print('Hessian spectrum')
    hess_fns = {}
    for cfg, fpath in opt_circuits:
        print(f'| computing hessian of {fpath}')
        params = jnp.load(fpath)
        circuit_name = f'Q{n_qubits}-L{cfg["n_layers"]}-R{cfg["rot_axis"]}'
        if circuit_name not in hess_fns:
            loss_fn = get_loss_fn(
                ham_matrix, n_qubits, cfg['n_layers'], cfg['rot_axis'])
            hess_fns[circuit_name] = jax.hessian(loss_fn)
        _, hess_eigvals = compute_hessian_eigenvalues(hess_fns[circuit_name], params)

        name = get_normalized_name(cfg)
        jnp.savez(
            resdir / f'{name}_all.npz',
            params=params,
            ham_matrix=ham_matrix,
            hess_spectrum=hess_eigvals,
        )
        print(f'| ...plotting hessian spectrum and histogram')
        plot_spectrum(hess_eigvals, resdir / f'{name}_spectrum.pdf')
        plot_histogram(hess_eigvals, resdir / f'{name}_histogram.pdf')


if __name__ == '__main__':
    main()
