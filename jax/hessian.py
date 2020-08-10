import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


def compute_hessian_eigenvalues(func, x):
    """ Compute eigenvalues of given cost function.

    Args:
        func: Callable, a cost function whose Hessian is to be computed.
        x: array, vector where to compute Hessian.

    Returns:
        DeviceArray, eigenvalues.
    """

    hessian_fn = jax.hessian(func)
    hessian_mat = hessian_fn(x)
    eigvals = jnp.linalg.eigvals(hessian_mat)
    # All eigenvalues of a hessian matrix must be real
    # because of the symmetry of the Hessian matrix.
    eigvals = jnp.real(eigvals)
    return eigvals


def plot_histogram(eigenvalues, save_path=None):
    """ Plot and save a histogram of eigenvalues."""
    fig = plt.figure()
    fig.hist(eigenvalues)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path)
    return eigenvalues
