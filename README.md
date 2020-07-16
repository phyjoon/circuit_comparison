# circuit_comparison

This project aims to compare various VQC architectures, as studied in https://arxiv.org/abs/1905.10876, in terms of their actual performance in finding the ground state wavefunction of a many-body system. Reachability to the true ground state does not only depend on [expressibility and entangling capability](https://arxiv.org/abs/1905.10876) but also on choice of optimization and initialization of parameters. 

## Usage
Run the program as follows, in order to construct a variational circuit supported on 3 qubits, made by stacking 1st, 18th, 16th layers (https://arxiv.org/abs/1905.10876).
```
python3 ./run.py -size 3 -type 1 18 16
```