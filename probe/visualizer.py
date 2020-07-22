import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_surface(x, y, z,
                 xlabel='x', ylabel='y', zlabel='loss',
                 draw_contour=True, draw_colorbar=True,
                 show=True, pause_interval=0., save_path=None):
    """ Surface plot of 2D data values.

    Example,
    ```
        x_axis = np.linspace(-1., 1., 50)
        y_axis = np.linspace(-1., 1., 50)
        x, y = np.mesh(x_axis, y_axis)
        z = np.random.rand(50, 50)  # data values.
        plot_surface(x, y, z)
    ```

    # Args.
        x: numpy.ndarray, 2D array x values
        y: numpy.ndarray, 2D array y values
        z: numpy.ndarray, 2D array data values
        xlabel: str, label of x-axis
        ylabel: str, label of y-axis
        zlabel: str, label of z-axis
        draw_contour: bool, whether to draw contour on the bottom of plot.
        draw_colorbar: bool, whether to draw color bar as well.
        show: bool, whether to show figure window
        pause_interval: float, pausing time if positive.
        save_path: str, a file path where to save the figure.
    :return:
    """

    zlim = z.min() - 0.5, z.max() + 0.5  # margin = 0.5

    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    alpha = 0.8 if draw_contour else 1.
    surf = ax.plot_surface(
        x, y, z,
        cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=alpha,
        rstride=1, cstride=1)
    if draw_contour:
        ax.contour(
            x, y, z, 10, cmap="autumn_r", linestyles="solid", offset=zlim[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_zlim(*zlim)

    # draw z-axis on the left side
    zaxis = ax.zaxis
    tmp_planes = zaxis._PLANES
    zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                     tmp_planes[0], tmp_planes[1],
                     tmp_planes[4], tmp_planes[5])

    if draw_colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=7)

    if save_path:
        path = Path(save_path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)

    if show:
        if pause_interval > 0:
            plt.pause(pause_interval)
        else:
            plt.show()
    return fig


def test_surface_plot():
    x = np.linspace(-4, 3, 50)
    y = np.linspace(-4, 3, 50)
    x, y = np.meshgrid(x, y)

    z = np.zeros((50, 50))
    for i, j in itertools.product(range(50), range(50)):
        z[i, j] = np.sin(np.sqrt(x[i, j] ** 2 + y[i, j] ** 2))

    plot_surface(x, y, z)


if __name__ == '__main__':
    test_surface_plot()
