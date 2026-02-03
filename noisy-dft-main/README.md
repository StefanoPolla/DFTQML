# Noisy DFT
> **Research Paper:** [arXiv:2409.02921](https://arxiv.org/abs/2409.02921)

[← Back to Main Repository](../README.md)

This directory contains the implementation for learning Hubbard Density Functionals from noisy, quantum-generated data.

---

##  Directory Layout

* **`../src/dftqml`**: Core package containing the ML and physics logic.
* **`data-h5/`**: Raw training data in HDF5 format.
* **`models/`**: Pre-trained model weights (serialized).
* **`scripts/`**: Parallel-optimized scripts for data generation and training.

---

##  Usage & Reproduction

To reproduce the results from the paper, follow these steps:

### 1. Environment Setup
Install the core `dftqml` package:
```
pip install -e . 

```



### 2. Training (Optional)

If you wish to re-train the models rather than using the provided ones in `models`, run the scripts in ascending enumerated order.

### 3. Visualization

Open the notebooks in this directory to regenerate the plots:

```bash
jupyter notebook notebooks/plotting_results.ipynb

