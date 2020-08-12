import numpy as np
import math
from itertools import combinations


def SYK_hamiltonian(n_gamma, seed_var=None):
    sigma_1 = np.array([[0, 1], [1, 0]])
    sigma_2 = np.array([[0, 1j], [-1j, 0]])
    sigma_3 = np.array([[1, 0], [0, -1]])
    identity = np.array([[1, 0], [0, 1]])
    gamma_matrices = list()
    for k in range(n_gamma):
        temp = np.array([1])
        for j in range(k//2):
            temp = np.kron(temp, sigma_3)
        if k % 2 == 0:
            temp = np.kron(temp, sigma_1)
        else:
            temp = np.kron(temp, sigma_2)
        for i in range(int(n_gamma/2) - (k//2) - 1):
            temp = np.kron(temp, identity)
        gamma_matrices.append(temp)
    np.random.seed(seed_var)
    indices = list(combinations(range(n_gamma), 4))
    hamiltonian = 0
    couplings = list()
    for x, y, w, z in indices:
        couplings.append(np.random.normal(loc=0, scale=math.sqrt(6/math.pow(n_gamma, 3))))
        hamiltonian += (couplings[-1]/4) * np.linalg.multi_dot([gamma_matrices[x], gamma_matrices[y], gamma_matrices[w], gamma_matrices[z]])
    return hamiltonian
