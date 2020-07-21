import itertools
import math

from pennylane import numpy as np  # use pennylane's one due to autograd.
import torch
import torch.nn as nn

from probe import cache, visualizer


class BaseObserver:

    def __init__(self):
        self.cache = None
        # This attribute might cause memory consumption issue when using
        # a large model architecture. But, we are considering a model of
        # relative small. Thus, for simplicity we here set baseline weights
        # as an attribute.
        self.center = None
        self.x_axis = None
        self.y_axis = None
        self.scale = 1.
        self.position = None
        self.grid_size = None

    def setup(self, **kwargs):
        """ To be refactored """
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError

    def set_weights(self, i, j):
        self.position = i, j
        self.set_weights_impl(i, j)

    def set_weights_impl(self, i, j):
        raise NotImplementedError

    def clear(self):
        self.center = None
        self.x_axis = None
        self.y_axis = None

    def run(self):
        for i, j in itertools.product(
                range(self.grid_size[0]), range(self.grid_size[1])):
            if not self.cache.is_done(i, j):
                self.set_weights(i, j)
                self.evaluate()
        self.export_cache()

    def evaluate(self):
        if self.position is None:
            raise RuntimeError('Model weights has not been set yet. '
                               'Please call `set_weights()` first.')
        loss = self.evaluate_impl()
        self.cache.save_result(loss, *self.position)

    def evaluate_impl(self):
        raise NotImplementedError

    def export_cache(self, path=None):
        self.cache.export(path)

    def check_axis_validity(self, direction, dim):
        assert dim == self.dim, \
            f'The length of given {direction}-axis must be matched ' \
            f'with the dimension of the model, {self.dim} != {dim}.'

    def retrieve_landscape(self):
        x = np.linspace(-1, 1, self.grid_size[0])
        y = np.linspace(-1, 1, self.grid_size[1])
        x, y = np.meshgrid(x, y)
        landscape = self.cache.get_values()
        return x, y, landscape

    def plot(self, draw_contour=True, draw_colorbar=True, show=True, save_path=None):
        x, y, landscape = self.retrieve_landscape()
        visualizer.plot_surface(
            x, y, landscape,
            draw_contour=draw_contour, draw_colorbar=draw_colorbar,
            show=show, save_path=save_path)


class PyTorchModelObserver(BaseObserver):

    def __init__(self, model):
        assert isinstance(model, nn.Module)
        super().__init__()
        self.model = model
        self._dim = None

    @property
    def dim(self):
        if self._dim is None:
            return sum(p.numel() for p in self.model.parameters())
        return self._dim

    def clear(self):
        super().clear()
        self.model = None
        self._dim = None

    def setup(self, center=None, x_axis=None, y_axis=None,
              scale=math.pi, grid_size=10, cache_type='numpy'):
        """ Setting up observing landscape

        # Args.
            center: iterable of Parameters, weight parameter
            x_axis: iterable of float tensors, a list of weights that indicates
                the first direction (x-axis) of the observation plane,
                where the lengths or the weight dimensions of each element must
                be matched with the model parameters.
            y_axis: iterable of float tensors, a list of weights that indicates
                the second direction (y-axis) of the observation plane,
                where the lengths or the weight dimensions of each element must
                be matched with the model parameters.
            scale: float, scaling factor of direction vectors.
            grid_size: int or tuple of length 2, the size of observation grid.
                When a single int value is given, it will be interpreted as
                (grid_size, grid_size)
            cache: cache type, caching intermediate results during traversing
                the whole grid. Recommend to use NumpyCache or SQLCache rather than
                FileCache to save storage.
        """

        # sanity check here.
        if center is None:
            center = self.model.parameters()
        self.center = center

        if x_axis is None:
            x_axis = torch.rand(self.dim, )
        self.check_axis_validity('x', sum(p.numel() for p in x_axis))
        self.x_axis = x_axis

        if y_axis is None:
            y_axis = torch.rand(self.dim, )
        self.check_axis_validity('y', sum(p.numel() for p in y_axis))
        self.y_axis = y_axis

        self.scale = scale

        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size
        self.cache = cache.init(cache_type, grid_size=grid_size)

    def set_weights_impl(self, i, j):
        i = (i - self.grid_size[0] / 2.) * 2
        j = (j - self.grid_size[1] / 2.) * 2
        for p, w, dx, dy in zip(
                self.model.parameters(), self.center,
                self.x_axis, self.y_axis):
            delta = dx / self.grid_size[0] * i * self.scale \
                    + dy / self.grid_size[1] * j * self.scale
            p.data = w + delta

    def evaluate_impl(self):
        return 0.


class PennylaneModelObserver(BaseObserver):

    def __init__(self, loss_op, grad_fn, params, param_sizes):
        """

        # Args.
            model: callable, a Python function or QNode that contains
                a combination of quantum and classical nodes
            grad_fn: callable, the function that returns the gradient of the input
            params: numpy.ndarray, the trainable parameters of the model
            param_sizes: list of int, number of parameters of each layers.
        """

        super().__init__()
        self.loss_op = loss_op
        self.grad_fn = grad_fn
        self.params = params
        self.param_sizes = param_sizes

    @property
    def dim(self):
        return sum(self.param_sizes)

    def setup(self, center=None, x_axis=None, y_axis=None,
              scale=math.pi, grid_size=10, cache_type='numpy'):
        """ Setting up observing landscape

        # Args.
            center: iterable of Parameters, weight parameter
            x_axis: numpy.ndarray, weights that indicates the x-axis of the observation plane.
            y_axis: numpy.ndarray, weights that indicates the y-axis of the observation plane.
            scale: float, scaling factor of direction vectors.
            grid_size: int or tuple of length 2, the size of observation grid.
                When a single int value is given, it will be interpreted as
                (grid_size, grid_size)
            cache_type: str , cache type, caching intermediate results during traversing
                the whole grid. Only NumpyCache is available now. SQLCache will be supported
                later.
        """

        self.center = center if center is not None else self.params
        if x_axis is None:
            x_axis = np.random.rand(self.dim, )
        self.check_axis_validity('x', x_axis.size)
        self.x_axis = x_axis / np.linalg.norm(x_axis)

        if y_axis is None:
            y_axis = np.random.rand(self.dim, )
        self.check_axis_validity('y', y_axis.size)
        self.y_axis = y_axis / np.linalg.norm(y_axis)

        self.scale = scale

        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size
        self.cache = cache.init(cache_type, grid_size=grid_size)

    def clear(self):
        super().clear()
        self.loss_op = None
        self.grad_fn = None
        self.params = None
        self.param_sizes = None

    def set_weights_impl(self, i, j):
        # To make (grid_size[0] / 2, grid_size[1] / 2) be the center.
        i = (i - self.grid_size[0] / 2.) * 2
        j = (j - self.grid_size[1] / 2.) * 2
        dx = self.x_axis / self.grid_size[0] * i * self.scale
        dy = self.y_axis / self.grid_size[1] * j * self.scale
        self.params = self.center + dx + dy

    def evaluate_impl(self):
        return self.loss_op(self.params)
