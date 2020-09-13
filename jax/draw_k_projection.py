import argparse
import ast
import pickle
import re
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import wandb

import qnnops


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, required=True,
                        help='Run id which can be found at url'
                             'app.wandb.ai/vqc-quantum/{project_name}/runs/{run_id}')
    parser.add_argument('--step-interval', type=int, default=1,
                        help='Interval between two consecutive training steps '
                             'to draw their landscapes.')
    parser.add_argument('--surface-dims', type=int, nargs='+',
                        help='The dimension of hyper-surface to project onto.')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum number of training steps to draw.')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='Transparency (Default: 0.9)')
    parser.add_argument('--color', type=str, default='hsv',
                        help='The colormap of trajectories. (Default: hsv)')
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help='Whether to show figure.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging')
    args = parser.parse_args()
    return args


def download(run_id):
    api = wandb.Api()
    run = api.run(f'IsingModel/{run_id}')
    config = argparse.Namespace(**run.config)
    resdir = Path(f'results_mri/{run.name}_{run_id}')
    ckpt_file = run.file('checkpoint_last.pkl').download(resdir, replace=True)
    with open(ckpt_file.name, 'rb') as f:
        checkpoint = pickle.load(f)
    history = checkpoint['history']

    best_file = run.file('params_best.npy').download(resdir, replace=True)
    opt_params = jnp.load(best_file.name)

    return resdir, config, history, opt_params


def hessian_spectrum(loss_fn, params):
    hessian_fn = qnnops.memory_efficient_hessian(loss_fn)
    print('Constructing hessian matrix...', end='')
    hess_mat = hessian_fn(params)
    print(f'Done: shape={hess_mat.shape}')

    print('Computing eigenvalue and eigenvectors of hessian matrix...', end='')
    hess_eigvals, hess_eigvecs = onp.linalg.eigh(hess_mat)
    indices = onp.argsort(-hess_eigvals)
    hess_eigvecs = hess_eigvecs[indices]
    hess_eigvecs = hess_eigvecs.T
    hess_eigvals = hess_eigvals[indices]
    print('Hessian spectrum:', hess_eigvals[:5])
    return hess_eigvecs, hess_eigvals


def reconstruct_model(config):
    ham_matrix = qnnops.ising_hamiltonian(
        n_qubits=config.n_qubits, g=config.g, h=config.h)

    ev_str = re.sub(r' +', r' ', config.eigenvalues.replace('\n', '')).replace(' ', ',')
    try:
        ev = ast.literal_eval(ev_str)
        ground_energy = ev[0]
    except ValueError:
        print(f'Malformed string: {ev_str}')
        print(f'Parse the ground state energy manually.')
        ground_energy = float(ev_str.split(',')[0].lstrip('['))

    def loss_fn(params):
        ansatz_state = qnnops.alternating_layer_ansatz(
            params, config.n_qubits, config.n_qubits, config.n_layers, config.rot_axis)
        return qnnops.energy(ham_matrix, ansatz_state) - ground_energy

    return loss_fn


def draw_landscape(alpha, beta, energy_fn):
    A, B = onp.meshgrid(alpha, beta)
    grid_shape = A.shape
    E0 = energy_fn(A.flatten(), B.flatten()).reshape(*grid_shape)
    return E0


class HessProjector:

    def __init__(self, loss_fn, origin, resdir):
        self.loss = loss_fn
        self.origin = origin
        self.resdir = Path(resdir)
        self.hess_eigvecs = None
        self.hess_eigvals = None
        self.dim = None

    @property
    def hess_eigvals_cache_path(self):
        return self.resdir / 'hess_eigvals.pkl'

    @property
    def hess_eigvecs_cache_path(self):
        return self.resdir / 'hess_eigvecs.pkl'

    def setup(self, dim):
        if self.hess_eigvals is None and self.hess_eigvecs is None:
            # Lazy loading hessian spectrum.
            try:
                print('Use precomputed eigenvectors of Hessian matrix')
                with self.hess_eigvals_cache_path.open('rb') as file:
                    self.hess_eigvals = pickle.load(file)
                with self.hess_eigvecs_cache_path.open('rb') as file:
                    self.hess_eigvecs = pickle.load(file)
            except FileNotFoundError:
                self.hess_eigvecs, self.hess_eigvals = hessian_spectrum(
                    self.loss, self.origin)
                print('Caching hessian spectrum')
                with self.hess_eigvals_cache_path.open('wb') as file:
                    pickle.dump(self.hess_eigvals, file)
                with self.hess_eigvecs_cache_path.open('wb') as file:
                    pickle.dump(self.hess_eigvecs, file)
        if dim > len(self.hess_eigvals):
            raise ValueError(
                f'The dimension of Hessian is less than the surface dim. '
                f'{len(self.hess_eigvals)} < {dim}.')
        self.dim = dim
        print(f'Top-{self.dim} Spectrum: {self.hess_eigvals[:self.dim]}')

    def basin_volume(self):
        return onp.sum(onp.log(self.hess_eigvals[:self.dim]))

    def project(self, x):
        x_proj = onp.zeros((self.dim,), dtype=self.origin.dtype)
        for i in range(self.dim):
            params = x - self.origin
            direction = self.hess_eigvecs[i]
            x_proj[i] = onp.dot(params, direction) / onp.linalg.norm(direction)
        return x_proj

    def distance(self, x):
        x_proj = self.project(x)
        return onp.linalg.norm(x_proj)


def main():
    args = get_arguments()
    resdir, config, history, opt_params = download(args.run_id)
    loss_fn = reconstruct_model(config)
    projector = HessProjector(loss_fn, opt_params, resdir)

    loss_history = history['loss']
    params_history = history['params']
    if args.max_steps is not None:
        loss_history = loss_history[:args.max_steps]
        params_history = params_history[:args.max_steps]

    for surface_dim in args.surface_dims:
        print(f'Surface dimension: {surface_dim}')
        projector.setup(surface_dim)
        print(f'The volume of the basin of an attractor: {projector.basin_volume()}')
        losses = []
        distances = []
        for step, (loss, params) in enumerate(zip(loss_history, params_history)):
            if step % args.step_interval == 0:
                distance = projector.distance(params)
                if args.verbose:
                    print(f'[Step {step}] project distance/loss =  {distance:.4f} / {loss:.4f}')
                losses.append(loss)
                distances.append(distance)
        losses = onp.hstack(losses)
        distances = onp.hstack(distances)
        onp.savez(resdir / f'loss_and_projected_distance_k{surface_dim}.npz',
                  losses=losses, distances=distances)
        plt.figure()
        plt.plot(distances, losses, 'k:', alpha=0.5, zorder=1)
        plt.scatter(distances, losses, c=onp.arange(len(distances)),
                    facecolor='none',
                    alpha=args.alpha, cmap=args.color, zorder=2)
        plt.colorbar()
        plt.xlabel(r'$S_k(\mathrm{\theta}^*)$-projected distance', fontsize=13)
        plt.ylabel(r'$E(\mathrm{\theta}) - E_0$', fontsize=13)
        plt.grid(True, c='0.5', ls=':', lw=0.5)
        plt.tight_layout()
        plt.savefig(resdir / f'projected_distance_k{surface_dim}.pdf')
        if args.show:
            plt.show()
        plt.close()


if __name__ == '__main__':
    main()
