import pennylane as qml
import pennylane.numpy as np

import argparse, math, itertools
import model
from model import param_shape

from tensorboardX import SummaryWriter

#
# Set the qubit size, random seed, and number of circuit layers
#

parser = argparse.ArgumentParser()
parser.add_argument('-size', action='store', type=int, default=2,
                    help="Input the size of a qubit system")
parser.add_argument('-type', action='store', type=int, nargs="+", default=[1], 
                    help="Input the circuit type")
args = parser.parse_args()

SIZE, LAYERS, DEPTH = args.size, args.type, len(args.type)

print(f"Number of qubits: {SIZE}, Circuit type: {LAYERS}, Number of circuit layers: {DEPTH}")


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

if __name__ == '__main__':

    # Open the tensorboard logger
    # writer = SummaryWriter(f"runs/{SIZE}-{LAYER}")
    
    # Define the quantum device
    dev, wires = qml.device('default.qubit', wires=SIZE), list(range(SIZE))
    param_size = [model.param_shape(i, SIZE) for i in LAYERS]

    # Define the variational parameters
    params = np.arange(sum(param_size))
    
    # Define the Hamiltonian operators
    operators, coeffs = Hamiltonian()
    
    # Define the QNodeCollection
    qnodes = qml.map(circuit, operators, dev, measure="expval")

    # Evaluate the QNodeCollection
    H = np.dot(coeffs, qnodes(params, size=SIZE, layers=LAYERS))
    print(f"Hamiltonian VEV: {H}")
    
    
    ################### 
    # TODO Add a simple option for optimizers
    # TODO Randomize the params
    # TODO Manage experimental results using comet.ml or tensorboard
    ###################     
    
    
    
    # model.quantum_circuits(torch.zeros(6), [1], 3)

    
    # node = qml.map(model.quantum_circuits, [qml.PauliZ(0) @ qml.PauliZ(1)], dev,
    #                measure="expval", interface="torch")
    # qml.map()
    
    # print(node(torch.zeros(6)))
    
    # # Store all gradient expectation values
    # # grad_vals = deque([])
    
    # for 
    # model.circuit01()
    # # for i in range(50000):
    #     # RNG = np.random.default_rng()
    #     # random_gates = RNG.choice([qml.RX, qml.RY, qml.RZ], size=(LAYER, SIZE))
        
    #     qnode = qml.map(circuit, [qml.PauliZ(0) @ qml.PauliZ(1)], dev,
    #                     measure="expval", interface="torch")
        
    #     # Define the variational circuit parameters
    #     params = torch.empty(LAYER, SIZE).uniform_(0, 2 * PI).requires_grad_()
    #     optim = torch.optim.Adam([params], lr=1e-4)
        
    #     # Run the qnode
    #     target = qnode(params, random_gates=random_gates)

    #     optim.zero_grad()
    #     target.backward()
    #     grad_vals.append(params.grad[0, 0].item())
    #     writer.add_scalar(f"mean_grad", np.mean(grad_vals), i)
    #     writer.add_scalar(f"var_grad", np.var(grad_vals), i)
    #     writer.add_scalar(f"grad", grad_vals[-1], i)
    #     writer.add_scalar(f"target", target, i)
    #     # target_vals.append(target)
    #     # if i % 100 == 0:
    #     #     print(f"Iteration {i:d}: target value {target.item()} and gradient {params.grad.numpy()[0, 0]}")
    #         # print(qnode[0].draw())
    #     # optim.step()
        
    
    # writer.close()
    

    
    
    
