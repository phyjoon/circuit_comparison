import pennylane as qml
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.wires import Wires
from pennylane.templates.utils import check_shape
from math import floor

def param_shape(circuit, size):
    param_dic = {1: 2 * size, 2: 2 * size, 3: 3 * size - 1, 4: 3 * size - 1, 5: size * (size + 3), 6: size * (size + 3), 
                 7: 4 * size + floor(size / 2) + floor((size - 1) / 2), 8: 4 * size + floor(size / 2) + floor((size - 1) / 2),
                 9: size, 10: 2 * size, 11: 4 * size - 4, 12: 4 * size - 4, 13: 4*size, 14: 4*size, 15: 2*size, 
                 16: 2 * size + floor(size / 2) + floor((size - 1) / 2), 17: 2 * size + floor(size / 2) + floor((size - 1) / 2),
                 18: 3*size, 19: 3*size}
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
    
    check_shape(params, target_shape=(2 * size, ),
                msg=f"params must be of shape {(2 * size, )}.")

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
    
    check_shape(params, target_shape=(2 * size, ),
                msg=f"params must be of shape {(2 * size, )}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)


@qml.template
def circuit03(params, wires):
    """Circuit template #03 in 1905.10876

    Args:
        params (array): An array of shapes (3N-1, 1). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)
    
    check_shape(params, target_shape=(3 * size - 1, ),
                msg=f"params must be of shape {(3 * size - 1, )}.")

    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2*size:])
    



@qml.template
def circuit04(params, wires):
    """Circuit template #04 in 1905.10876

    Args:
        params (array): An array of shapes (3N-1, 1). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)
    
    check_shape(params, target_shape=(3 * size - 1, ),
                msg=f"params must be of shape {(3 * size - 1, )}.")

    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2*size:])
    




@qml.template
def circuit05(params, wires):
    """Circuit template #05 in 1905.10876

    Args:
        params (array): An array of shapes (N^2 + 3N, 1). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)
    
    check_shape(params, target_shape=(size * (size + 3), ),
            msg=f"params must be of shape {(size * (size + 3), )}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')
    
    idx_start = 2 * size 
    for cnt, controlled in enumerate(reversed(range(len(wires)))):
        pattern = [[controlled, j] for j in reversed(range(len(wires))) if j != controlled]
        qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[idx_start:idx_start + (size - 1)])
        idx_start += size - 1
    
    AngleEmbedding(params[idx_start:idx_start + size], wires, rotation='X')
    AngleEmbedding(params[idx_start + size: idx_start + 2 * size], wires, rotation='Z')




@qml.template
def circuit06(params, wires):
    """Circuit template #06 in 1905.10876

    Args:
        params (array): An array of shapes (N^2 + 3N, 1). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)
    
    check_shape(params, target_shape=(size * (size + 3), ),
            msg=f"params must be of shape {(size * (size + 3), )}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')
    
    idx_start = 2 * size 
    for cnt, controlled in enumerate(reversed(range(len(wires)))):
        pattern = [[controlled, j] for j in reversed(range(len(wires))) if j != controlled]
        qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[idx_start:idx_start + (size - 1)])
        idx_start += size - 1
    
    AngleEmbedding(params[idx_start:idx_start + size], wires, rotation='X')
    AngleEmbedding(params[idx_start + size: idx_start + 2 * size], wires, rotation='Z')
    
    

    
@qml.template
def circuit07(params, wires):
    """Circuit template #07 in 1905.10876

    Args:
        params (array): An array of shapes (4N + floor(N/2) + floor((N-1)/2), ). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(4 * size + floor(size / 2) + floor((size - 1) / 2), ),
            msg=f"params must be of shape {(4 * size + floor(size / 2) + floor((size - 1) / 2), )}.")
      
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2*size:2*size+floor(size/2)])
    
    AngleEmbedding(params[2*size+floor(size/2):3*size+floor(size/2)], wires, rotation='X')
    AngleEmbedding(params[3*size+floor(size/2):4*size+floor(size/2)], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[4*size+floor(size/2):])


@qml.template
def circuit08(params, wires):
    """Circuit template #08 in 1905.10876

    Args:
        params (array): An array of shapes (4N + floor(N/2) + floor((N-1)/2), ). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(4 * size + floor(size / 2) + floor((size - 1) / 2), ),
            msg=f"params must be of shape {(4 * size + floor(size / 2) + floor((size - 1) / 2), )}.")
      
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2*size:2*size+floor(size/2)])
    
    AngleEmbedding(params[2*size+floor(size/2):3*size+floor(size/2)], wires, rotation='X')
    AngleEmbedding(params[3*size+floor(size/2):4*size+floor(size/2)], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[4*size+floor(size/2):])



@qml.template
def circuit09(params, wires):
    """Circuit template #09 in 1905.10876

    Args:
        params (array): An array of shapes (N,). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(size, ),
            msg=f"params must be of shape {(size, )}.")
    
    # Define the circuit    
    qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=wires)

    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))]
    qml.broadcast(unitary=qml.CZ, pattern=pattern, wires=wires)

    AngleEmbedding(params, wires, rotation='X')


@qml.template
def circuit10(params, wires):
    """Circuit template #10 in 1905.10876

    Args:
        params (array): An array of shapes (2N, ). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(2*size,),
            msg=f"params must be of shape {(2*size,)}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='Y')

    pattern = [[i + 1, i] for i in reversed(range(len(wires) - 1))] + [[0, len(wires) - 1]]
    qml.broadcast(unitary=qml.CZ, pattern=pattern, wires=wires)

    AngleEmbedding(params[size:], wires, rotation='Y')



@qml.template
def circuit11(params, wires):
    """Circuit template #11 in 1905.10876

    Args:
        params (array): An array of shapes (4N-4,). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(4 * size - 4, ),
            msg=f"params must be of shape {(4 * size - 4, )}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='Y')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)
    
    AngleEmbedding(params[2*size:3*size-2], wires=wires[1:-1], rotation='Y')
    AngleEmbedding(params[3*size-2:4*size-4], wires=wires[1:-1], rotation='Z')

    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)

@qml.template
def circuit12(params, wires):
    """Circuit template #12 in 1905.10876

    Args:
        params (array): An array of shapes (4N-4,). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(4 * size - 4, ),
            msg=f"params must be of shape {(4 * size - 4, )}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='Y')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')
    
    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CZ, pattern=pattern, wires=wires)
    
    AngleEmbedding(params[2*size:3*size-2], wires=wires[1:-1], rotation='Y')
    AngleEmbedding(params[3*size-2:4*size-4], wires=wires[1:-1], rotation='Z')

    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CZ, pattern=pattern, wires=wires)



@qml.template
def circuit13(params, wires):
    """Circuit template #13 in 1905.10876

    Args:
        params (array): An array of shapes (4N, 1). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(4*size,),
            msg=f"params must be of shape {(4*size,)}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='Y')
    
    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[size:2*size])
    
    AngleEmbedding(params[2*size:3*size], wires, rotation='Y')
    
    pattern = [[(i - 1) % len(wires), (i - 2) % len(wires)] for i in range(len(wires))]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[3*size:])



@qml.template
def circuit14(params, wires):
    """Circuit template #14 in 1905.10876

    Args:
        params (array): An array of shapes (4N, 1). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(4*size,),
            msg=f"params must be of shape {(4*size,)}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='Y')
    
    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[size:2*size])
    
    AngleEmbedding(params[2*size:3*size], wires, rotation='Y')
    
    pattern = [[(i - 1) % len(wires), (i - 2) % len(wires)] for i in range(len(wires))]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[3*size:])


@qml.template
def circuit15(params, wires):
    """Circuit template #15 in 1905.10876

    Args:
        params (array): An array of shapes (2N, ). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(2*size,),
            msg=f"params must be of shape {(2*size,)}.")
    
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='Y')
    
    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)
    
    AngleEmbedding(params[size:], wires, rotation='Y')
    
    pattern = [[(i - 1) % len(wires), (i - 2) % len(wires)] for i in range(len(wires))]
    qml.broadcast(unitary=qml.CNOT, pattern=pattern, wires=wires)


@qml.template
def circuit16(params, wires):
    """Circuit template #16 in 1905.10876

    Args:
        params (array): An array of shapes (2N + floor(N/2) + floor((N-1)/2), ). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(2*size + floor(size/2) + floor((size-1)/2), ),
            msg=f"params must be of shape {(2*size + floor(size/2) + floor((size-1)/2),)}.")
      
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2*size:2*size+floor(size/2)])
    
    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2*size+floor(size/2):])



@qml.template
def circuit17(params, wires):
    """Circuit template #17 in 1905.10876

    Args:
        params (array): An array of shapes (2N + floor(N/2) + floor((N-1)/2), ). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(2*size + floor(size/2) + floor((size-1)/2), ),
            msg=f"params must be of shape {(2*size + floor(size/2) + floor((size-1)/2),)}.")
      
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')

    pattern = [[i + 1, i] for i in range(0, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2*size:2*size+floor(size/2)])
    
    pattern = [[i + 1, i] for i in range(1, len(wires) - 1, 2)]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2*size+floor(size/2):])



@qml.template
def circuit18(params, wires):
    """Circuit template #18 in 1905.10876

    Args:
        params (array): An array of shapes (3N, ). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(3*size,),
            msg=f"params must be of shape {(3*size,)}.")
      
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')

    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CRZ, pattern=pattern, wires=wires, parameters=params[2*size:])

@qml.template
def circuit19(params, wires):
    """Circuit template #19 in 1905.10876

    Args:
        params (array): An array of shapes (3N, ). 
        wires (Iterable): Wires that the template acts on.
    """
    # Input Checks
    wires, size = Wires(wires), len(wires)

    check_shape(params, target_shape=(3*size,),
            msg=f"params must be of shape {(3*size,)}.")
      
    # Define the circuit    
    AngleEmbedding(params[:size], wires, rotation='X')
    AngleEmbedding(params[size:2*size], wires, rotation='Z')

    pattern = [[i, (i + 1) % len(wires)] for i in reversed(range(len(wires)))]
    qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=params[2*size:])
