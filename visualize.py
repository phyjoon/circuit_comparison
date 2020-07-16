import os
from pathlib import Path
import pennylane as qml
import pennylane.numpy as np

import argparse, math, itertools
import model
from model import param_shape

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorboardX import SummaryWriter

#
# Set the qubit size, random seed, and number of circuit layers
#

parser = argparse.ArgumentParser()
parser.add_argument('-size', action='store', type=int, default=2,
                    help="Input the size of a qubit system")
parser.add_argument('-coord', action='store', type=int, nargs=2, default=[0, 1], 
                    help="Input the optimizer")
parser.add_argument('-type', action='store', type=int, nargs="+", default=[1], 
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
    param_size = [model.param_shape(i, size) for i in layers]
    param_idx_end = list(itertools.accumulate(param_size))
    param_idx_start = [0] + param_idx_end[:-1]
    
    for idx_start, idx_end, layer in zip(param_idx_start, param_idx_end, layers):
        eval(f"model.circuit{layer:0>2d}(params[idx_start:idx_end], wires)")


#
# Main block
#

if __name__ == '__main__' and not os.path.isfile(FILE_PATH):

    # Define the quantum device
    dev, wires = qml.device('default.qubit', wires=SIZE), list(range(SIZE))
    param_size = [model.param_shape(i, SIZE) for i in LAYERS]

    # Define the variational parameters
    params = 2 * np.pi * (np.random.rand(sum(param_size)) - 0.5)
    
    # Define the Hamiltonian operators
    operators, coeffs = Hamiltonian()
    
    # Define the QNodeCollection
    qnodes = qml.map(circuit, operators, dev, measure="expval")

    # Evaluate the QNodeCollection
    def HNode(params):
        return np.dot(coeffs, qnodes(params, size=SIZE, layers=LAYERS))

    # Define the lattice
    X = np.linspace(-np.pi, np.pi, 100)
    Y = np.linspace(-np.pi, np.pi, 100)
    X, Y = np.meshgrid(X, Y)
    
    Op, OpG = HNode, qml.grad(HNode)
    Op_mat, OpG_mat = np.zeros(shape=(100, 100)), np.zeros(shape=(sum(param_size), 100, 100))

    for i in range(100):
        for j in range(100):
            params[args.coord[0]] = X[i, j]
            params[args.coord[1]] = Y[i, j]
            
            Op_mat[i, j] = Op(params)
            OpG_mat[:, i, j] = OpG(params)[0]

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X=X, Y=Y, Z=Op_mat)
    # plt.show()
    
    with open(FILE_PATH, "wb") as file:
        np.savez(file, H=Op_mat, G=OpG_mat)
    


    
