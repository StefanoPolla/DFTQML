# DFTQML
Machine-learning Hubbard Density Functionals from Noisy Quantum-Generated Data

## usage
The code consists of: 
- a package in `./stc/dftqml`, containing the core functions
- a series of scripts in `./main` using these functions to generate data, learn and benchmark the functionals (written for perfect parallel execution on a cluster)
- the training data in hdf5 format in `./main/data-h5`
- the trained models in `./main/models`

To reproduce the results:
1. install the dftqml package by running `pip install .`
2. run the desired script in `./main`, in order (optional, the provided data/models can be used instead)
3. run the notebooks to reproduce the plots