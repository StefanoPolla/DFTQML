import numpy as np
import matplotlib.pyplot as plt
import os.path
import h5py
from typing import Tuple
import xgboost.sklearn  
import matplotlib

DTYPE = np.float32


# ------------- Data processing -------------


def load_dft_data_h5(path: str, n_idcs: int, first_idx=0) -> Tuple:
    """
    loads DFT data from a .h5 file.

    Args:
        path: h5 file (including or excluding the .h5 extesion)
        n_idcs: number of datapoints to be loaded
        first_idx: index of the first datapoint to be loaded. Default is 0.

    Returns:
        Tuple: densities, energies
            densities is an array of shape=(n_idcs, L)
            energies is an array of shape=(n_idcs)
    """
    if not path.endswith(".h5"):
        path += ".h5"
    with h5py.File(path, "r") as f:
        indices = f["indices"]
        j = np.searchsorted(indices, first_idx)
        if indices[j] != first_idx:
            raise ValueError("first_idx not found in indices. Check dataset.")

        densities = f["densities"][j : j + n_idcs]
        dft_energies = f["dft_energies"][j : j + n_idcs]

    return densities, dft_energies


def load_dft_data_txt(directory: str, n_idcs: int) -> Tuple:
    """
    loads DFT data from a directory with .dat (text) files.

    Args:
        directory (str): directory with text files
        n_idcs (int): number of datapoints to be loaded

    Returns:
        Tuple: densities, energies
            densities is an array of shape=(n_idcs, L)
            energies is an array of shape=(n_idcs)
    """
    data = [np.loadtxt(os.path.join(directory, f"{idx}.dat")) for idx in range(n_idcs)]
    data = np.array(data)
    densities = data[:, :-1]
    dft_energies = data[:, -1]
    return densities, dft_energies


def load_dft_data(path: str, n_idcs: int, first_idx: int = 0) -> Tuple:
    """
    Load DFT data from either an h5 file or a directory containing text files.

    Args:
        path (str): The path to the h5 file or directory.
        n_idcs (int): The number of indices.

    Returns:
        Tuple: A tuple containing the loaded DFT data.
            densities is an array of shape=(n_idcs, L)
            energies is an array of shape=(n_idcs)

    Raises:
        FileNotFoundError: If the path is not found.
    """
    if path.endswith(".h5"):
        return load_dft_data_h5(path, n_idcs, first_idx)
    elif os.path.exists(path + ".h5"):
        return load_dft_data_h5(path + ".h5", n_idcs, first_idx)
    elif os.path.isdir(path):
        if first_idx != 0:
            raise NotImplementedError("first_idx is not supported for txt files")
        return load_dft_data_txt(path, n_idcs)
    else:
        raise FileNotFoundError("Path not found: " + path)


def load_potentials(path: str, n_idcs: int, first_idx: int = 0) -> np.ndarray:
    """
    Load the potentials from a .h5 file.
    """
    if not path.endswith(".h5"):
        raise NotImplementedError("only h5 files are supported at the moment")
    with h5py.File(path, "r") as f:
        indices = f["indices"]
        j = np.searchsorted(indices, first_idx)
        if indices[j] != first_idx:
            raise ValueError("first_idx not found in indices. Check dataset.")

        potentials = f["potentials"][j : j + n_idcs]
    return potentials


def load_harmonic_potentials_and_strengths(path: str, n_idcs: int, first_idx: int = 0) -> np.ndarray:
    """
    Load the harmonic potentials and strengths from a .h5 file.
    The strength W defines the potential: V_j = W * (2j/(L-1) - 1) ^ 2
    """
    if not path.endswith(".h5"):
        raise NotImplementedError("only h5 files are supported at the moment")
    with h5py.File(path, "r") as f:
        indices = f["indices"]
        j = np.searchsorted(indices, first_idx)
        if indices[j] != first_idx:
            raise ValueError("first_idx not found in indices. Check dataset.")

        potentials = f["potentials"][j : j + n_idcs]
        strengths = f["strengths"][j : j + n_idcs]
    return potentials, strengths


def augment_data(local_inputs: np.ndarray, invariant_outputs: np.ndarray) -> Tuple:
    """
    Augment the dataset according to translational and mirror symmetries.

    For each datapoint, 2 * L datapoints are created by taking shifted and
    mirrored versions of each density, each with the same energy.

    Args:
        local_inputs (np.ndarray): list of densities or RDMs of shape (n_idcs, ..., L), 
            where n_idcs is the number of datapoints, and the last index represents locality, 
            i.e. a shift in the last index corresponds to a real-space shift.
        invariant_outputs (np.ndarray): corresponding list of energies, invariant under shift.

    Returns:
        Tuple: (augmented_inputs, augmented_outputs), where:
            - augmented_inputs has shape (n_idcs * 2 * L, ..., L)
            - augmented_outputs has shape (n_idcs * 2 * L,)
    """
    shape = local_inputs.shape
    L = shape[-1]

    # Generate index matrix for translational and mirror symmetries
    symmetry_index_matrix = np.array(
        [[(s * (i - j)) % L for i in range(L)] for s in [+1, -1] for j in range(L)]
    )

    # Apply symmetry transformations
    augmented_inputs = np.take(local_inputs, symmetry_index_matrix, axis=-1)

    # flatten the selection dimension
    augmented_inputs = np.moveaxis(augmented_inputs, -2, 1) # Move the selection dim to the front
    new_shape = (-1,) + shape[1:] # Combine first two axes
    augmented_inputs = augmented_inputs.reshape(new_shape)

    # Repeat invariant outputs for each transformation
    augmented_outputs = np.repeat(invariant_outputs, 2 * L)
    return augmented_inputs, augmented_outputs


# ------------- Visualization -------------


def history_plot(history):
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], ".-", label="val_loss")
    plt.yscale("log")
    plt.xlabel("epochs")
    plt.ylabel("loss (MSE)")
    plt.legend
    plt.grid()


def performance_plot(model, x_test, y_test, **kwargs):
    """square plot representing the performance of a model"""
    if isinstance(model, xgboost.sklearn.XGBRegressor):
        prediction = np.ravel(model.predict(x_test))
    else:
        prediction = np.ravel(model(x_test).numpy())
    exact = np.ravel(y_test)
    plt.scatter(exact, prediction, **{"s": 1, **kwargs})
    plt.plot(*[[np.min([exact, prediction]), np.max([exact, prediction])]] * 2, "k-")
    plt.xlabel("y_test true value")
    plt.ylabel("model predictions")


def save_fig(
        fig: matplotlib.figure.Figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: Tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 300,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e)) 
