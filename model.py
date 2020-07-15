import pennylane as qml
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.wires import Wires
from pennylane.templates.utils import check_shape
from math import floor

def param_shape(circuit, size):
    param_dic = {1: 2 * size, 2: 2 * size}
    return param_dic[circuit]

@qml.template
def circuit01(params, wires):
    """Circuit template #01 in 1905.10876

    Args:
        params (array): Input array of size (2N, ) that corresponds to rotation parameters.
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)
    
    check_shape(params, target_shape=(2 * len(wires), ),
                msg=f"params must be of shape {(2 * len(wires), )}.")

    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:], wires, rotation='Z')
    
@qml.template
def circuit02(params, wires):
    """Circuit template #02 in 1905.10876

    Args:
        params (array): Input array of size (2N, 1).
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)
    
    check_shape(params, target_shape=(2 * len(wires), ),
                msg=f"params must be of shape {(2 * len(wires), )}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)


@qml.template
def circuit03(params, wires):
    """Circuit template #03 in 1905.10876

    Args:
        params (array): An array of shapes (3, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        the unused component, params[2][N-1], has been inevitably added.
    """
    # Input Checks
    wires = Wires(wires)
    
    check_shape(params, target_shape=(3, len(wires)),
                msg=f"params must be of shape {(3, len(wires))}.")

    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2][:-1])
    



@qml.template
def circuit04(params, wires):
    """Circuit template #04 in 1905.10876

    Args:
        params (array): An array of shapes (3, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        the unused component, params[2][N-1], has been inevitably added.
    """
    # Input Checks
    wires = Wires(wires)
    
    check_shape(params, target_shape=(3, len(wires)),
                msg=f"params must be of shape {(3, len(wires))}.")

    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2][:-1])
    




@qml.template
def circuit05(params, wires):
    """Circuit template #05 in 1905.10876

    Args:
        params (array): An array of shapes (8, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        unused components, params[a][N-1] for 2 <= a <= 5, have been added.
    """
    # Input Checks
    wires = Wires(wires)
    
    check_shape(params, target_shape=(8, len(wires)),
            msg=f"params must be of shape {(8, len(wires))}.")
    
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')
    
    for cnt, controlled in enumerate(reversed(range(len(wires)))):
        pattern = [[controlled, j] for j in reversed(range(len(wires))) if j != controlled]
        qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[cnt + 2][:-1])
    
    AngleEmbedding(params[6], wires, rotation='X')
    AngleEmbedding(params[7], wires, rotation='Z')


@qml.template
def circuit06(params, wires):
    """Circuit template #06 in 1905.10876

    Args:
        params (array): An array of shapes (8, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        unused components, params[a][N-1] for 2 <= a <= 5, have been added.
    """
    # Input Checks
    wires = Wires(wires)
    
    check_shape(params, target_shape=(8, len(wires)),
            msg=f"params must be of shape {(8, len(wires))}.")
    
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')
    
    for cnt, controlled in enumerate(reversed(range(len(wires)))):
        pattern = [[controlled, j] for j in reversed(range(len(wires))) if j != controlled]
        qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[cnt + 2][:-1])
    
    AngleEmbedding(params[6], wires, rotation='X')
    AngleEmbedding(params[7], wires, rotation='Z')

    
    
@qml.template
def circuit07(params, wires):
    """Circuit template #07 in 1905.10876

    Args:
        params (array): An array of shapes (6, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        the unused components, params[2][a] for floor(N/2) <= a and
        params[5][a] for floor((N-1)/2) <= a have been inevitably added.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(6, len(wires)),
            msg=f"params must be of shape {(6, len(wires))}.")
      
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2][:floor(len(wires)/2)])
    
    AngleEmbedding(params[3], wires, rotation='X')
    AngleEmbedding(params[4], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    print(pattern)
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[5][:floor((len(wires)-1)/2)])

@qml.template
def circuit08(params, wires):
    """Circuit template #08 in 1905.10876

    Args:
        params (array): An array of shapes (6, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        the unused components, params[2][a] for floor(N/2) <= a and
        params[5][a] for floor((N-1)/2) <= a have been inevitably added.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(6, len(wires)),
            msg=f"params must be of shape {(6, len(wires))}.")
      
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2][:floor(len(wires)/2)])
    
    AngleEmbedding(params[3], wires, rotation='X')
    AngleEmbedding(params[4], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[5][:floor((len(wires)-1)/2)])





@qml.template
def circuit09(params, wires):
    """Circuit template #09 in 1905.10876

    Args:
        params (array): An array of shapes (1, N). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(1, len(wires)),
            msg=f"params must be of shape {(1, len(wires))}.")
    
    # Define the circuit    
    qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=wires)

    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))]
    qml.broadcast(unitary=qml.CZ, pattern=pattern, wires=wires)

    AngleEmbedding(params[0], wires, rotation='X')


@qml.template
def circuit10(params, wires):
    """Circuit template #10 in 1905.10876

    Args:
        params (array): An array of shapes (2, N). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(2, len(wires)),
            msg=f"params must be of shape {(2, len(wires))}.")
    
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='Y')

    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))] + [[0, len(wires) - 1]]
    qml.broadcast(unitary=qml.CZ, pattern=pattern, wires=wires)

    AngleEmbedding(params[1], wires, rotation='Y')



@qml.template
def circuit11(params, wires):
    """Circuit template #11 in 1905.10876

    Args:
        params (array): An array of shapes (4, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        unused tensor components, params[3][b] and params[4][b] 
        for b >= N-2, have been inevitably added.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(4, len(wires)),
            msg=f"params must be of shape {(4, len(wires))}.")
    
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='Y')
    AngleEmbedding(params[1], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)
    
    AngleEmbedding(params[2][:-2], wires=wires[1:-1], rotation='Y')
    AngleEmbedding(params[3][:-2], wires=wires[1:-1], rotation='Z')

    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)

@qml.template
def circuit12(params, wires):
    """Circuit template #12 in 1905.10876

    Args:
        params (array): An array of shapes (4, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        unused tensor components, params[3][b] and params[4][b] 
        for b >= N-2, have been inevitably added.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(4, len(wires)),
            msg=f"params must be of shape {(4, len(wires))}.")
    
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='Y')
    AngleEmbedding(params[1], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CZ, pattern=pattern, wires=wires)
    
    AngleEmbedding(params[2][:-2], wires=wires[1:-1], rotation='Y')
    AngleEmbedding(params[3][:-2], wires=wires[1:-1], rotation='Z')

    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CZ, pattern=pattern, wires=wires)


@qml.template
def circuit13(params, wires):
    """Circuit template #13 in 1905.10876

    Args:
        params (array): An array of shapes (4, N). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(4, len(wires)),
            msg=f"params must be of shape {(4, len(wires))}.")
    
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='Y')
    
    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[1])
    
    AngleEmbedding(params[2], wires, rotation='Y')
    
    pattern = [[(i - 1) % len(wires), (i - 2) % len(wires)] for i in range(len(wires))]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[3])



@qml.template
def circuit14(params, wires):
    """Circuit template #14 in 1905.10876

    Args:
        params (array): An array of shapes (4, N). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(4, len(wires)),
            msg=f"params must be of shape {(4, len(wires))}.")
    
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='Y')
    
    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[1])
    
    AngleEmbedding(params[2], wires, rotation='Y')
    
    pattern = [[(i - 1) % len(wires), (i - 2) % len(wires)] for i in range(len(wires))]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[3])


@qml.template
def circuit15(params, wires):
    """Circuit template #15 in 1905.10876

    Args:
        params (array): An array of shapes (2, N). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(2, len(wires)),
            msg=f"params must be of shape {(2, len(wires))}.")
    
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='Y')
    
    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)
    
    AngleEmbedding(params[1], wires, rotation='Y')
    
    pattern = [[(i - 1) % len(wires), (i - 2) % len(wires)] for i in range(len(wires))]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)


@qml.template
def circuit16(params, wires):
    """Circuit template #16 in 1905.10876

    Args:
        params (array): An array of shapes (4, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        the unused components, params[2][a] for floor(N/2) <= a and
        params[3][a] for floor((N-1)/2) <= a have been inevitably added.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(4, len(wires)),
            msg=f"params must be of shape {(4, len(wires))}.")
      
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2][:floor(len(wires)/2)])
    
    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[3][:floor((len(wires)-1)/2)])



@qml.template
def circuit17(params, wires):
    """Circuit template #17 in 1905.10876

    Args:
        params (array): An array of shapes (4, N). 
        wires (Iterable): Wires that the template acts on.
    Comment:
        PyTorch does not support a ragged tensor. As a consequence,
        in order to put the parameter tensor in a rectangular shape, 
        the unused components, params[2][a] for floor(N/2) <= a and
        params[3][a] for floor((N-1)/2) <= a have been inevitably added.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(4, len(wires)),
            msg=f"params must be of shape {(4, len(wires))}.")
      
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2][:floor(len(wires)/2)])
    
    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[3][:floor((len(wires)-1)/2)])



@qml.template
def circuit18(params, wires):
    """Circuit template #18 in 1905.10876

    Args:
        params (array): An array of shapes (3, N). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(3, len(wires)),
            msg=f"params must be of shape {(3, len(wires))}.")
      
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')

    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2])

@qml.template
def circuit19(params, wires):
    """Circuit template #19 in 1905.10876

    Args:
        params (array): An array of shapes (3, N). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires = Wires(wires)

    check_shape(params, target_shape=(3, len(wires)),
            msg=f"params must be of shape {(3, len(wires))}.")
      
    # Define the circuit    
    AngleEmbedding(params[0], wires, rotation='X')
    AngleEmbedding(params[1], wires, rotation='Z')

    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2])
