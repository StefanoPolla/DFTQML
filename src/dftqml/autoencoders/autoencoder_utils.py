import numpy as np
import torch.nn as nn
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler
from typing import List
import h5py
from dftqml.autoencoders.lipschitz import LipschitzLinear
from dftqml.hubbard_model.fhchain import hamiltonian_terms_from_two_rdm
import os


class DataPreprocessor:
    """
    A class to preprocess data for autoencoders, including normalization and flattening.

    Constructors:
        DataPreprocessor(flatten=True): Initializes the preprocessor with optional flattening.
        DataPreprocessor.load(checkpoint_dir): Loads the preprocessor state from an HDF5 file.

    Methods:
        save(checkpoint_dir): Saves the preprocessor state as an HDF5 file.
        fit_transform(data): Fits the preprocessor to the data and transforms it.
        transform(data): Transforms new data using the fitted preprocessor.
        inverse_transform(data): Inverse transforms the preprocessed data.
    """

    def __init__(self, flatten=True):
        """
        Initialize data preprocessor

        Args:
            normalization (str): Type of normalization to use ('minmax', 'standard', or None)
            flatten (bool): Whether to flatten the data
        """
        self.flatten = flatten
        self.scaler = StandardScaler()

    def save(self, filepath):
        """
        Save the preprocessor state as an HDF5 file

        Args:
            filepath (str): Directory to save the preprocessor state
        """
        with h5py.File(filepath, "w") as f:
            f.create_dataset("mean", data=self.scaler.mean_)
            f.create_dataset("scale", data=self.scaler.scale_)
            f.create_dataset("var", data=self.scaler.var_)
            f.attrs["n_samples_seen"] = self.scaler.n_samples_seen_
            f.attrs["flatten"] = self.flatten
            if self.flatten:
                # Save original_shape as a tuple of ints
                f.attrs["original_shape"] = self.original_shape

    @classmethod
    def load(cls, filepath):
        """
        Load the preprocessor state from an HDF5 file

        Args:
            filepath (str): Directory to load the preprocessor state from

        Returns:
            DataPreprocessor: An instance of DataPreprocessor with loaded state
        """
        with h5py.File(filepath, "r") as f:
            mean = f["mean"][()]
            scale = f["scale"][()]
            var = f["var"][()]
            flatten = bool(f.attrs["flatten"])
            n_samples_seen = f.attrs.get("n_samples_seen", None)
            original_shape = f.attrs.get("original_shape", None)
            if original_shape is not None:
                # h5py may store as np.ndarray, convert to tuple
                original_shape = tuple(original_shape)

        preprocessor = cls(flatten=flatten)
        preprocessor.scaler.mean_ = mean
        preprocessor.scaler.scale_ = scale
        preprocessor.scaler.var_ = var
        preprocessor.scaler.n_samples_seen_ = n_samples_seen
        preprocessor.original_shape = original_shape

        return preprocessor

    def fit_transform(self, data):
        """
        Fit the preprocessor to the data and transform it

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Preprocessed data
        """
        # Make a copy to avoid modifying the original data
        processed_data = data.copy()

        # Flatten the data if required
        if self.flatten:
            self.original_shape = processed_data.shape
            processed_data = processed_data.reshape(processed_data.shape[0], -1)

        # Apply normalization
        processed_data = self.scaler.fit_transform(processed_data)

        return processed_data

    def transform(self, data):
        """
        Transform new data using the fitted preprocessor

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Preprocessed data
        """
        # Make a copy to avoid modifying the original data
        processed_data = data.copy()

        # Flatten the data if required
        if self.flatten:
            processed_data = processed_data.reshape(processed_data.shape[0], -1)

        # Apply normalization
        processed_data = self.scaler.transform(processed_data)

        return processed_data

    def inverse_transform(self, data):
        """
        Inverse transform the preprocessed data

        Args:
            data (np.ndarray): Preprocessed data

        Returns:
            np.ndarray: Data in original scale
        """
        # Apply inverse normalization if specified
        if self.scaler is not None:
            data = self.scaler.inverse_transform(data)

        # Reshape the data if it was flattened
        if self.flatten:
            data = data.reshape(-1, *self.original_shape[1:])

        return data


def geometric_hidden_dims(
    input_dim: int, min_dim: int, hidden_depth: int, scaling_factor: float = 0.0
) -> List[int]:
    """Generate hidden layer dimensions for an autoencoder with optional scaled first layer.

    Args:
        input_dim: Input dimension size
        min_dim: Minimum dimension size (bottleneck)
        hidden_depth: Number of hidden layers
        scaling_factor: Factor to scale up the first layer (default=0.)
                        When set to 0., reverts to original geometric behavior

    Returns:
        List of hidden layer dimensions
    """
    # If scaling_factor is None, use original geometric progression
    if scaling_factor == 0:
        log_input = np.log(input_dim)
        log_latent = np.log(min_dim)
        logs = np.linspace(log_input, log_latent, hidden_depth + 2)[1:-1]
        dims = [max(int(np.exp(log)), min_dim) for log in logs]
        return dims

    else:
        # Handle the case when hidden_depth is 1
        if hidden_depth == 1:
            return [max(int(input_dim * scaling_factor), min_dim)]

        # For the first hidden layer, scale up from input dimension
        first_hidden = max(int(input_dim * scaling_factor), min_dim)

        # If hidden_depth is 2, we just need first_hidden and another layer
        if hidden_depth == 2:
            return [first_hidden, max(int(np.sqrt(first_hidden * min_dim)), min_dim)]

        # For more layers, create geometric progression from first_hidden to min_dim
        log_first = np.log(first_hidden)
        log_latent = np.log(min_dim)
        logs = np.linspace(log_first, log_latent, hidden_depth)[1:]
        remaining_dims = [max(int(np.exp(log)), min_dim) for log in logs]

        return [first_hidden] + remaining_dims


def linear_hidden_dims(
    input_dim: int, min_dim: int, hidden_depth: int, scaling_factor: float = 0.0
) -> List[int]:
    """Generate hidden layer dimensions for an autoencoder with optional scaled first layer.

    Args:
        input_dim: Input dimension size
        min_dim: Minimum dimension size (bottleneck)
        hidden_depth: Number of hidden layers
        scaling_factor: Factor to scale up the first layer (default=0.)
                        When set to 0., reverts to original linear behavior
    """
    if scaling_factor == 0:
        # Original linear behavior
        return [
            max(input_dim - i * (input_dim - min_dim) // (hidden_depth - 1), min_dim)
            for i in range(hidden_depth)
        ]

    else:
        # Scale the first layer
        first_hidden = max(int(input_dim * scaling_factor), min_dim)

        # If hidden_depth is 1, return only the first hidden layer
        if hidden_depth == 1:
            return [first_hidden]

        # For more layers, create linear progression from first_hidden to min_dim
        remaining_dims = [
            max(
                first_hidden - i * (first_hidden - min_dim) // (hidden_depth - 1),
                min_dim,
            )
            for i in range(1, hidden_depth)
        ]

        return [first_hidden] + remaining_dims


def init_weights(m, type):
    if isinstance(m, nn.Linear):
        if type == "kaiming":
            # Kaiming initialization for layers with ReLU-like activations
            init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        elif type == "xavier":
            init.xavier_uniform_(m.weight)
        else:
            raise ValueError(f"Unknown weight initialization type: {type}")
        if m.bias is not None:
            init.constant_(m.bias, 0)

    if isinstance(m, LipschitzLinear):
        m.reset_c()


def load_data(input_file, n_data, input_type, N, use_reduced=False):
    # ********* Load and prepare data ************
    with h5py.File(input_file, "r") as f:
        # Check if using reduced format with pre-computed Hamiltonian terms
        if use_reduced:
            if "hamiltonian_terms" not in f:
                raise ValueError(
                    f"Reduced format requested but 'hamiltonian_terms' dataset not found in {input_file}. "
                    "Available datasets: " + ", ".join(f.keys())
                )

            # Load pre-computed Hamiltonian terms directly
            ham_terms_arr = f["hamiltonian_terms"][:]

            # Enforce that input_type must be "ham_terms" when using reduced format
            if input_type != "ham_terms":
                raise ValueError(
                    "When using reduced data file, input_type must be 'ham_terms'. "
                    f"Got input_type='{input_type}'."
                )

            if ham_terms_arr.shape[0] < n_data:
                raise ValueError(
                    f"Not enough data points in the input file. Found {ham_terms_arr.shape[0]}, "
                    f"but requested {n_data}."
                )

            training_data = ham_terms_arr[:n_data]
        else:
            # Original behavior: load two_rdms and optionally convert
            if "two_rdms" not in f:
                raise ValueError(
                    f"'two_rdms' dataset not found in {input_file}. "
                    "Available datasets: " + ", ".join(f.keys()) + ". "
                )

            two_rdms = f["two_rdms"][:]

            # Select the first `n_data` two-RDMs for training.
            # Some data should be left for testing, but we do not enforce that here.
            if two_rdms.shape[0] < n_data:
                raise ValueError(
                    f"Not enough data points in the input file. Found {two_rdms.shape[0]}, "
                    f"but requested {n_data}."
                )
            two_rdms = two_rdms[:n_data]

            # Convert data to requested type. Note: the preprocessor will handle the flattening.
            if input_type == "ham_terms":
                ham_terms_arr = np.stack(
                    [hamiltonian_terms_from_two_rdm(rdm, N) for rdm in two_rdms]
                )
                training_data = ham_terms_arr
            elif input_type == "two_rdms":
                training_data = two_rdms
            else:
                raise NotImplementedError(
                    "Unsupported input type. Choose 'ham_terms' or 'two_rdms'."
                )

    return training_data


def checkpoint_dir_path(
    models_dir,
    input_type,
    L,
    N,
    U,
    interpolation,
    enable_batchnorm,
    hidden_depth,
    latent_dim,
    min_dim,
    scaling_factor,
    weight_decay,
    lambda_latent_norm,
    lambda_latent_repulsion,
    lambda_lipschitz_encoder,
    lambda_lipschitz_decoder,
    iteration,
    *,
    n_data=90000,
    linear_encoder=False,
    augment_by_symmetry=False,
    num_sources=1,
):
    encoder_type = "ae-linear" if linear_encoder else "ae"
    iteration = iteration or 0
    # Directory structure (updated):
    # models/<ae|ae-linear>/<input_type>/L{L}-N{N}-U{U}/nn-{interp}[-batchnorm]/
    #   hidden{...}_latent{...}_min{...}_scaling{...}/lbd-n{...}_r{...}_le{...}_ld{...}[_wd{wd}]/
    #     ndata{n_data}/iter{iteration}
    # Note: The optional _wd{wd} suffix is only added when weight_decay > 0 for backwards compatibility.
    path_parts = [
        models_dir,
        encoder_type,
        input_type,
        f"L{L}-N{N}-U{U}",
        "nn-" + interpolation + ("-batchnorm" if enable_batchnorm else ""),
        f"hidden{hidden_depth}_latent{latent_dim}_min{min_dim}_scaling{scaling_factor}",
        f"lbd-n{lambda_latent_norm}_r{lambda_latent_repulsion}_le{lambda_lipschitz_encoder}_ld{lambda_lipschitz_decoder}"
        + (f"_wd{weight_decay}" if weight_decay and float(weight_decay) > 0.0 else ""),
        f"ndata{n_data}"
        + ("+sa" if augment_by_symmetry else "")
        + (f"+ms{num_sources}" if num_sources and num_sources > 1 else ""),
        f"iter{iteration}",
    ]
    checkpoint_dir = os.path.join(*path_parts)
    return checkpoint_dir


def prepare_checkpoint_dir(args):
    checkpoint_dir = checkpoint_dir_path(
        args.models_dir,
        args.input_type,
        args.L,
        args.N,
        args.U,
        args.interpolation,
        args.enable_batchnorm,
        args.hidden_depth,
        args.latent_dim,
        args.min_dim,
        args.scaling_factor,
        args.weight_decay,
        args.lambda_latent_norm,
        args.lambda_latent_repulsion,
        args.lambda_lipschitz_encoder,
        args.lambda_lipschitz_decoder,
        args.iteration,
        n_data=args.n_data,
        linear_encoder=getattr(args, "linear_encoder", False),
        augment_by_symmetry=getattr(args, "augment_by_symmetry", False),
        num_sources=getattr(args, "num_sources", 1),
    )
    if os.path.exists(checkpoint_dir):
        if "training_info.json" in os.listdir(checkpoint_dir):
            print(
                f"Checkpoint directory {checkpoint_dir} already exists. "
                "Skipping training for this model."
            )
            exit(0)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint dir: {checkpoint_dir}")

    return checkpoint_dir
