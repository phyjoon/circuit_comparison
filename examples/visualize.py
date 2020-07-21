import argparse
import itertools
from pathlib import Path

import pennylane as qml
import pennylane.numpy as np
from probe import models, PennylaneModelObserver

#
# Set the qubit size, random seed, and number of circuit layers
#

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=2,
                    help="Input the size of a qubit system")
parser.add_argument('--coord', type=int, nargs=2, default=[0, 1],
                    help="Input the optimizer")
parser.add_argument('--type', type=int, nargs="+", default=[1],
                    help="Input the circuit type")
args = parser.parse_args()

SIZE, LAYERS, DEPTH = args.size, args.type, len(args.type)
EXP_STRING = f"Q{SIZE:0>2d}" + "L" + '+'.join(map(str, LAYERS)) + "D" + ''.join(map(str, args.coord))
FILE_PATH = Path(".") / "data" / (EXP_STRING + ".npz")

print(f"Number of qubits: {SIZE}, Circuit type: {LAYERS}, Number of circuit layers: {DEPTH}, Experiment: {EXP_STRING}")


#
# Define the operator to be computed (e.g., Hamiltonian)
#

def Hamiltonian():
    operators = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(SIZE - 1)] + \
                [qml.PauliX(i) for i in range(SIZE)]
    coeffs = np.array([0.5 for i in range(SIZE - 1)] + [1 for i in range(SIZE)])
    return operators, coeffs


#
# Define the variational circuit, consisting of component layers specified in model.py
#

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


#
# Main block
#

# Define the quantum device
dev, wires = qml.device('default.qubit', wires=SIZE), list(range(SIZE))
param_sizes = [models.param_shape(i, SIZE) for i in LAYERS]

# Define the variational parameters
params = 2 * np.pi * (np.random.rand(sum(param_sizes)) - 0.5)

# Define the Hamiltonian operators
operators, coeffs = Hamiltonian()

# Define the QNodeCollection
qnodes = qml.map(circuit, operators, dev, measure="expval")


# Evaluate the QNodeCollection
def HNode(params):
    return np.dot(coeffs, qnodes(params, size=SIZE, layers=LAYERS))


loss_op, grad_op = HNode, qml.grad(HNode)

observer = PennylaneModelObserver(loss_op, grad_op, params, param_sizes)
observer.setup(scale=5., grid_size=20, cache_type='numpy')
observer.run()
observer.plot(
    draw_contour=True,
    draw_colorbar=True,
    show=True,
    pause_interval=0.,
    save_path=None
)
