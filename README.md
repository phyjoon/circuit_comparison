# circuit_comparison

This project aims to compare various VQC architectures, as studied in https://arxiv.org/abs/1905.10876, in terms of their actual performance in finding the ground state wavefunction of a many-body system. Reachability to the true ground state does not only depend on [expressibility and entangling capability](https://arxiv.org/abs/1905.10876) but also on choice of optimization and initialization of parameters. 

## Usage
Run the program as follows, in order to construct a variational circuit supported on 3 qubits, made by stacking 1st, 18th, 16th layers (https://arxiv.org/abs/1905.10876).
```
python3 ./run.py -size 3 -type 1 18 16
```


## Probe: Loss Landscape VisualizatiAon Tool

This project contains a visualization tool for probing loss landscapes.


### Getting Started

There are two ways to configure the package `probe`. 
Both ways allow you to use this package everywhere.

**1. Editable installation (Recommended)**

```bash
  $ git clone https://github.com/phyjoon/circuit_comparison
  $ cd circuit_comparison/probe
  # editable install the package under the current directory
  $ pip install -e .  
```


2. Edit `PYTHONPATH` environment variable

Adding the repository path to the environment allows your python 
to find `probe`.   
  
```bash
  $ git clone https://github.com/phyjoon/circuit_comparison
  $ cd circuit_comparison/probe
  $ export PYTHONPTH=$(pwd):${PYTHONPATH}  
```


### How to use

Refer [this example](examples/visualize.py).
The module `probe.observer` is the main feature to probe loss landscapes.
```python
    # After defining a quantum model in terms of loss_op, grad_op, params, param_sizes
    # Create an observer instance with a quantum model.
    observer = PennylaneModelObserver(loss_op, grad_op, params, param_sizes)
    # Setting up observation area or resolution or something else.
    observer.setup(scale=math.pi, grid_size=20, cache_type='numpy')
    # Running the observtion procedure.
    observer.run()
    # Show or save the observed landscape.
    observer.plot()
```

At now it can show only the landscape around given center, 
but we will extend to observe the trajectory of gradient procedures in training phase.
