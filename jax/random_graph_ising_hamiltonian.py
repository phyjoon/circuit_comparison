import numpy as np


def ising_random_hamiltonian(n_qubits, magnetic_field_transverse, magnetic_field_longitudinal, probability, seed_var=None):
    sigma_1 = np.array([[0., 1.], [1., 0.]])
    sigma_3 = np.array([[1., 0.], [0., -1.]])
    identity = np.array([[1., 0.], [0., 1.]])
    sigma_x = list()
    sigma_z = list()
    for k in range(n_qubits):
        temp_x = np.array([1])
        temp_z = np.array([1])
        for j in range(k):
            temp_x = np.kron(temp_x, identity)
            temp_z = np.kron(temp_z, identity)
        temp_x = np.kron(temp_x, sigma_1)
        temp_z = np.kron(temp_z, sigma_3)
        for i in range(int(n_qubits) - k - 1):
            temp_x = np.kron(temp_x, identity)
            temp_z = np.kron(temp_z, identity)
        sigma_x.append(temp_x)
        sigma_z.append(temp_z)
    np.random.seed(seed_var)
    indices_first = list()
    indices_second = list(range(n_qubits))
    for i in range(n_qubits):
        indices_first.append((i, (i+1) % n_qubits))
    for i in range(n_qubits - 1):
        x = np.random.rand()
        if x < probability:
            indices_first.append((i, np.random.randint(i + 1, n_qubits)))
    hamiltonian = 0
    for x, y in indices_first:
        hamiltonian -= np.linalg.multi_dot([sigma_z[x], sigma_z[y]])
    for x in indices_second:
        hamiltonian -= magnetic_field_transverse * sigma_x[x]
        hamiltonian -= magnetic_field_longitudinal * sigma_z[x]
    return hamiltonian
