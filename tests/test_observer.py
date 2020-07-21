import itertools
import os.path
import tempfile
import unittest

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from probe import models
from probe.observer import PennylaneModelObserver
from probe.observer import PyTorchModelObserver


class DummyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 4, bias=False)
        self.fc2 = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        return x


@qml.template
def circuit(params, wires, size=None, layers=None):
    """Define the quantum circuit as a stack of VQC layers

    Args:
        params (Array): Variational parameters in the form of NumPy array
        wires (List[int]): List of qubit indices
        size (int): Number of qubits
        layers (List[int]): List of VQCs stacked up
    """
    param_size = [models.param_shape(i, size) for i in layers]
    param_idx_end = list(itertools.accumulate(param_size))
    param_idx_start = [0] + param_idx_end[:-1]

    for idx_start, idx_end, layer in zip(param_idx_start, param_idx_end, layers):
        models.__dict__[f'circuit{layer:0>2d}'](params[idx_start:idx_end], wires)


def build_dummy_model(layers, size=2):
    # Define the quantum device
    device, wires = qml.device('default.qubit', wires=size), list(range(size))
    param_size = [models.param_shape(i, size) for i in layers]

    # Define the variational parameters
    params = 2 * np.pi * (np.random.rand(sum(param_size)) - 0.5)

    # Define the Hamiltonian operators
    operators = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(size - 1)] + \
                [qml.PauliX(i) for i in range(size)]
    coeffs = np.array([0.5 for i in range(size - 1)] + [1 for i in range(size)])

    # Define the QNodeCollection
    qnodes = qml.map(circuit, operators, device, measure="expval")

    # Evaluate the QNodeCollection
    def HNode(_params):
        return np.dot(coeffs, qnodes(_params, size=size, layers=layers))

    Op, OpG = HNode, qml.grad(HNode)
    return Op, OpG, params, param_size


class ObserverTest(unittest.TestCase):

    def test_observer_initialization(self):
        dummy_net = DummyNet()
        obs = PyTorchModelObserver(dummy_net)
        obs.setup(torch.zeros(obs.dim))
        self.assertEqual(8, obs.dim)

    def test_grid_traversing(self):
        dummy_net = DummyNet()
        obs = PyTorchModelObserver(dummy_net)
        center = [torch.zeros_like(p) for p in obs.model.parameters()]
        x_axis = [torch.zeros_like(p) for p in obs.model.parameters()]
        y_axis = [torch.zeros_like(p) for p in obs.model.parameters()]
        x_axis[0][0], y_axis[0][1] = 1., 1.

        grid_size = 10
        obs.setup(center=center, x_axis=x_axis, y_axis=y_axis, grid_size=grid_size)

        for i, j in itertools.product(range(grid_size), range(grid_size)):
            obs.set_weights(i, j)
            expected = [
                0.1 * torch.tensor([2 * i - grid_size, 2 * j - grid_size, 0., 0.]).view(-1, 1),
                torch.zeros(4, ).view(1, -1)]
            for _expected, _actual in zip(expected, obs.model.parameters()):
                torch.testing.assert_allclose(_expected, _actual)

    def test_predefined_models(self):
        grid_size = 10
        layer_configs = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, layer_config in enumerate(layer_configs):
                op, grad, params, param_sizes = build_dummy_model(layer_config, 2)
                observer = PennylaneModelObserver(op, grad, params, param_sizes)
                observer.setup(scale=5., grid_size=grid_size, cache_type='numpy')
                observer.run()

                cache_path = os.path.join(tmpdir, f'config{i}.npy')
                observer.export_cache(cache_path)
                self.assertTrue(os.path.exists(cache_path))

                fig_path = os.path.join(tmpdir, f'plot{i}.png')
                observer.plot(show=False, save_path=fig_path)
                self.assertTrue(os.path.exists(fig_path))


if __name__ == '__main__':
    unittest.main()
