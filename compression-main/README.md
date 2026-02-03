# Learning Minimal Representations of Fermionic Ground States
> **Research Paper:** [arXiv:2512.11767](https://arxiv.org/abs/2512.11767)

[← Back to Main Repository](../README.md)

This directory contains the code to discover minimal degrees of freedom in quantum many-body ground states using an unsupervised autoencoder framework. We specifically target the $L$-site Fermi-Hubbard model to identify the $L-1$ latent dimensions.



##  Directory Layout

* **`../src/dftqml`**: Core package shared across projects.
* **`scripts/`**: Parallelized scripts for data generation and autoencoder training.
* **`notebooks/`**: Jupyter notebooks for visualizing the reconstruction thresholds and Jacobian analysis.
* **`models/`**: Saved weights for the trained encoder and decoder.

## Models & Data

To run this project, download the pre-trained weights and dataset from the [v2.0. Release](https://github.com/StefanoPolla/DFTQML/releases/tag/v2.0).
The data should be unpacked in the folder `compression-main/data/` and the models should be unpacked in the folder `compression-main/models/bs256/ae/ham-terms/`

| Asset | Description | Download |
| :--- | :--- | :--- |
| **Processed Data** | Ham-Terms data | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/data_reduced.tar.gz) |
| **Model Weights L4** | Autoencoder trained on L=4 systems | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/L4-N4-U4.0.tar.gz) |
| **Model Weights L5** | Autoencoder trained on L=5 systems | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/L5-N4-U4.0.tar.gz) |
| **Model Weights L6** | Autoencoder trained on L=6 systems | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/L6-N6-U4.0.tar.gz) |
| **Model Weights L7** | Autoencoder trained on L=7 systems | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/L7-N8-U4.0.tar.gz) |
| **Model Weights L8** | Autoencoder trained on L=8 systems | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/L8-N8-U4.0.tar.gz) |
| **Model Weights L10** | Autoencoder trained on L=10 systems | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/L10-N10-U4.0.tar.gz) |
| **Model Weights L12** | Autoencoder trained on L=12 systems | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/L12-N12-U4.0.tar.gz) |
| **Model Weights L14** | Autoencoder trained on L=14 systems | [Download (.tar.gz)](https://github.com/StefanoPolla/DFTQML/releases/download/v2.0/L14-N14-U4.0.tar.gz) |




##  Usage & Reproduction

### 1. Installation
Ensure the core `dftqml` package is installed from the root directory:
```bash
pip install -e .

```


### 2. Training (Optional)

If you wish to re-train the models rather than using the provided ones in `models`, run the scripts in ascending enumerated order.

### 3. Analysis & Plotting

To reproduce the figures in the paper (e.g., the reconstruction error as a function of latent dimension), use the provided notebooks:

```bash
jupyter notebook notebooks/plot_reconstruction_results.ipynb

```



