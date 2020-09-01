import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import yaml

import qnnops

# TODO(jdk): The data will be taken from wandb via API later.

# resdir = Path('results/20200831_174252_IsingModel_Q8L80RyBS8 - g2.0h0.0 - S99 - LR0.05 - trajectory')
resdir = Path('results/20200831_205142_IsingModel_Q6L27RyBS6 - g2.0h0.0 - S3 - LR0.05 - trajectory')
with (resdir / 'hparams.yaml').open() as f:
    cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg)

print('Loading hamiltonian matrix')
ham_matrix = onp.load(resdir / 'hamiltonian_matrix.npy')
eigval, _ = onp.linalg.eigh(ham_matrix)


def circuit(params):
    return qnnops.alternating_layer_ansatz(
        params,
        n_qubits=cfg.n_qubits,
        block_size=cfg.n_qubits,
        n_layers=cfg.n_layers,
        rot_axis=cfg.rot_axis)


def loss(params):
    ansatz_state = circuit(params)
    return qnnops.energy(ham_matrix, ansatz_state)


history = onp.load(resdir / 'history.npz')
loss_history = history['loss']
min_index = loss_history.argmin()

params_history = history['params']
n_steps, dim = params_history.shape
trained_params = params_history[min_index]
print(f'History: n_steps={n_steps} dim={dim}, trained_model shape={trained_params.shape}')

load_hess = True
if load_hess:
    print('Use precomputed eigenvectors of Hessian matrix')
    with (resdir / 'hess_eigvals.pkl').open('rb') as file:
        hess_eigvals = pickle.load(file)
    with (resdir / 'hess_eigvecs.pkl').open('rb') as file:
        hess_eigvecs = pickle.load(file)
else:
    hessian_fn = jax.hessian(loss)
    print('Constructing hessian matrix...', end='')
    hess_mat = hessian_fn(trained_params)
    print(f'Done: shape={hess_mat.shape}')

    print('Computing eigenvalue and eigenvectors of hessian matrix...', end='')
    hess_eigvals, hess_eigvecs = jnp.linalg.eigh(hess_mat)
    print('Done')
    hess_eigvecs = hess_eigvecs.T
    with (resdir / 'hess_eigvals.pkl').open('wb') as file:
        pickle.dump(hess_eigvals, file)
    with (resdir / 'hess_eigvecs.pkl').open('wb') as file:
        pickle.dump(hess_eigvecs, file)

print('Hessian spectrum:', hess_eigvals[:5])

dx, dy = hess_eigvecs[-1], hess_eigvecs[-2]
origin = trained_params


def project_1d(p, d):
    p = p - origin
    return onp.dot(p, d) / onp.linalg.norm(d)


def project_trajectory(params_history_, dx_, dy_):
    tx_, ty_ = [], []
    for p in params_history_:
        x = project_1d(p, dx_)
        y = project_1d(p, dy_)
        tx_.append(x)
        ty_.append(y)
    return tx_, ty_


@jax.vmap
def energy_along_flat(a, b):
    ansatz_state = qnnops.alternating_layer_ansatz(
        origin + a * dx + b * dy,
        n_qubits=cfg.n_qubits, block_size=cfg.n_qubits,
        n_layers=cfg.n_layers, rot_axis=cfg.rot_axis)
    return qnnops.energy(ham_matrix, ansatz_state) - eigval[0]


scale = 2.0
alpha = onp.linspace(-scale, scale, 50)
beta = onp.linspace(-scale, scale, 50)

print('Drawing energy landscape')
A, B = onp.meshgrid(alpha, beta)
E0 = energy_along_flat(A.flatten(), B.flatten()).reshape(50, 50)
contours = plt.contour(A, B, E0,
                       cmap='RdBu_r',
                       alpha=0.8,
                       levels=20)
plt.clabel(contours, inline=True, fontsize=7)
plt.colorbar()

print('Drawing optimization trajectory')
tx, ty = project_trajectory(params_history, dx, dy)
plt.scatter(
    tx, ty, s=45,
    c='darkslategrey',
    marker='+',
    zorder=2
)
plt.tight_layout()
plt.savefig('fig/trajectory.pdf')
plt.show()
