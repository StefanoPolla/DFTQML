import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Tuple, Union
from dftqml.autoencoders import autoencoder_utils

from dftqml.autoencoders.loss_functions import loss_function
from dftqml.autoencoders.lipschitz import LipschitzLinear


def check_gradients(model, epoch, batch_idx):
    """Monitor gradients for exploding/vanishing issues."""
    grad_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.mean().item()
            grad_std = param.grad.data.std().item()

            grad_stats[name] = {
                "norm": grad_norm,
                "mean": grad_mean,
                "std": grad_std,
                "max": param.grad.data.max().item(),
                "min": param.grad.data.min().item(),
            }

            # Alert for potential issues
            if grad_norm > 100:
                print(f"WARNING: Large gradient in {name}: {grad_norm:.2e}")
            # elif grad_norm < 1e-7:
            #     print(f"WARNING: Vanishing gradient in {name}: {grad_norm:.2e}")

    return grad_stats


class AE(nn.Module):
    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        hidden_dims: list,
        latent_dim: int,
        activation: nn.Module = nn.Softplus(),
        use_lipschitz_layers=True,
        use_batchnorm=False,
        linear_encoder=False,
    ):
        super(AE, self).__init__()

        # model parameters
        self.activation = activation
        self.use_lipschitz_layers = use_lipschitz_layers
        self.use_batchnorm = use_batchnorm
        self.linear_encoder = linear_encoder

        # Store original input shape for unflattening
        if isinstance(input_dim, (tuple, list)):
            self.input_shape = tuple(input_dim)
            self.flattened_dim = int(torch.prod(torch.tensor(input_dim)))
            self.needs_reshaping = True
        else:
            self.input_shape = None
            self.flattened_dim = input_dim
            self.needs_reshaping = False

        # Normalization parameters (will be set during training)
        self.register_buffer("input_mean", torch.zeros(self.flattened_dim))
        self.register_buffer("input_std", torch.ones(self.flattened_dim))
        self.normalization_initialized = False

        linear_layer = LipschitzLinear if self.use_lipschitz_layers else nn.Linear

        # Encoder
        if self.linear_encoder:
            # Single dense layer encoder, never using Lipschitz layers nor batchnorm
            self.encoder = nn.Linear(self.flattened_dim, latent_dim)
        else:
            encoder_layers = []
            prev_dim = self.flattened_dim
            for h_dim in hidden_dims:
                encoder_layers.append(linear_layer(prev_dim, h_dim))
                if self.use_batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(h_dim))
                encoder_layers.append(self.activation)
                prev_dim = h_dim
            # Final encoder layer to latent dimension
            encoder_layers.append(linear_layer(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []

        # First decoder layer
        decoder_layers.append(linear_layer(latent_dim, hidden_dims[-1]))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        decoder_layers.append(self.activation)
        prev_dim = hidden_dims[-1]
        # Hidden decoder layers
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(linear_layer(prev_dim, h_dim))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(self.activation)
            prev_dim = h_dim
        # Final decoder layer to output dimension
        decoder_layers.append(linear_layer(hidden_dims[0], self.flattened_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self._initialize_weights()
        if self.use_lipschitz_layers:
            print(self.log_lipschitz_bounds)

    def encode(self, x):
        """Encode input to latent space."""
        x_flat = self._flatten_input(x)
        x_norm = self._normalize(x_flat)
        return self.encoder(x_norm)

    def decode(self, z):
        """Decode from latent space to output."""
        x_norm = self.decoder(z)
        x_denorm = self._denormalize(x_norm)
        return self._unflatten_output(x_denorm)

    def forward(self, x):
        """Full forward pass through autoencoder."""
        z = self.encode(x)
        return self.decode(z)

    def initialize_normalization(self, data):
        """Initialize normalization parameters from training data."""
        with torch.no_grad():
            data_flat = self._flatten_input(data)
            self.input_mean.copy_(data_flat.mean(dim=0))
            self.input_std.copy_(data_flat.std(dim=0))
            # Prevent division by zero
            self.input_std.clamp_(min=1e-6)
            self.normalization_initialized = True

    @property
    def log_lipschitz_bounds(self):
        if not self.use_lipschitz_layers:
            raise RuntimeError("Model does not contain any LipschitzLinear layers.")
        # Collect Lipschitz bounds while preserving gradients
        if self.linear_encoder:
            log_bound_encoder = torch.tensor(0.0, device=next(self.parameters()).device)
        else:
            encoder_bounds = [
                layer.lipschitz_bound
                for layer in self.encoder
                if isinstance(layer, LipschitzLinear)
            ]
            log_bound_encoder = torch.sum(torch.log10(torch.stack(encoder_bounds)))
        decoder_bounds = [
            layer.lipschitz_bound
            for layer in self.decoder
            if isinstance(layer, LipschitzLinear)
        ]
        log_bound_decoder = torch.sum(torch.log10(torch.stack(decoder_bounds)))
        return log_bound_encoder, log_bound_decoder

    def _initialize_weights(self):
        self.apply(lambda m: autoencoder_utils.init_weights(m, type="xavier"))

    def _flatten_input(self, x):
        """Flatten input if needed, preserving batch dimension."""
        if self.needs_reshaping and x.dim() > 2:
            return x.flatten(start_dim=1)
        return x

    def _unflatten_output(self, x):
        """Unflatten output if needed, preserving batch dimension."""
        if self.needs_reshaping and self.input_shape is not None:
            batch_size = x.shape[0]
            return x.view(batch_size, *self.input_shape)
        return x

    def _normalize(self, x):
        """Normalize flattened input data."""
        if not self.normalization_initialized:  # Check buffer value
            raise RuntimeError(
                "Normalization not initialized. Call initialize_normalization() first."
            )
        return (x - self.input_mean) / self.input_std

    def _denormalize(self, x):
        """Denormalize flattened output data."""
        if not self.normalization_initialized:  # Check buffer value
            raise RuntimeError(
                "Normalization not initialized. Call initialize_normalization() first."
            )
        return x * self.input_std + self.input_mean


def train_ae(
    data,
    *,
    # Model parameters
    hidden_depth,
    latent_dim,
    min_dim,
    scaling_factor=0.0,
    interpolation="geometric",
    enable_batchnorm=False,
    linear_encoder=False,
    # Training parameters
    batch_size=64,
    epochs=100,
    learning_rate=1e-3,
    val_split=0.1,
    early_stopping_patience=30,
    early_stopping_threshold=1e-10,
    scheduler_patience=10,
    scheduler_factor=0.5,
    scheduler_threshold=1e-8,
    # Optimizer regularization
    weight_decay=0.0,
    # Regularization parameters
    lambda_latent_norm=0.0,
    lambda_latent_repulsion=0.0,
    lambda_lipschitz_encoder=0.0,
    lambda_lipschitz_decoder=0.0,
    latent_epsilon=1e-8,
    use_statevector=False,
    reduction_type="mean",
    verbose=10,
    device="cuda",
    checkpoint_dir="checkpoints",
    augment_by_symmetry=False,
):
    """
    Train an Autoencoder on the given dataset with early stopping.
    Also saves training curves and relevant training information, including model configuration details.

    Args:
        data (np.ndarray): Input data for training.

        hidden_depth (int): Number of hidden layers in the encoder/decoder.
        latent_dim (int): Dimensionality of the latent space.
        min_dim (int): Minimum dimension for the last hidden layer.
        scaling_factor (float): Scaling factor for the first hidden layer.
        interpolation (str): Interpolation method for hidden layer dimensions ('linear' or 'geometric').
        enable_batchnorm (bool): Whether to use batch normalization between layers.
        linear_encoder (bool): Whether to use a linear encoder (single dense layer) instead of a neural network encoder.

        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        val_split (float): Fraction of data to use for validation.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
        early_stopping_threshold (float): Threshold for measuring improvement.
        scheduler_patience (int): Number of epochs with no improvement before reducing the learning rate.
        scheduler_factor (float): Factor by which the learning rate is reduced.
        scheduler_threshold (float): Threshold for measuring improvement.

        weight_decay (float): L2 weight regularization strength passed to the optimizer
            (Adam's weight_decay). Defaults to 0.0 (disabled).

        lambda_norm (float): Regularization coefficient for the latent space.
        lambda_repel (float): Coefficient for the repulsion term in the latent space.
        lambda_lipschitz (float): Coefficient for Lipschitz regularization.
            If set to zero, an unregularized model will be used.
        epsilon (float): Small constant to avoid division by zero in the repulsion term.

        device (str): Device to use for training ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
        augment_by_symmetry (bool): If True, apply translational + mirror symmetry augmentation
            to the training portion of the data (after train/val split). Only meaningful for
            inputs whose last axis encodes locality (e.g. Hamiltonian terms shaped (n, 3, L)).
            Validation data is kept unaugmented to provide an unbiased estimate.

    Returns:
        nn.Module: Trained Autoencoder model.
        dict: Training information including best validation loss and epoch.
    """

    # Create directory for checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Compute flattened input dimension. Data will be flattened by preprocessor later
    input_shape = data.shape[1:]
    flattened_dim = int(np.prod(input_shape))

    # *** Model Initialization ***

    # Generate hidden layer dimensions
    if interpolation == "linear":
        hidden_dims = autoencoder_utils.linear_hidden_dims(
            input_dim=flattened_dim,
            min_dim=min_dim,
            hidden_depth=hidden_depth,
            scaling_factor=scaling_factor,
        )
    elif interpolation == "geometric":
        hidden_dims = autoencoder_utils.geometric_hidden_dims(
            input_dim=flattened_dim,
            min_dim=min_dim,
            hidden_depth=hidden_depth,
            scaling_factor=scaling_factor,
        )
    else:
        raise ValueError(
            f"Invalid interpolation method: {interpolation}. Choose 'linear' or 'geometric'."
        )

    # Initialize model

    model = AE(
        input_shape,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        use_batchnorm=enable_batchnorm,
        linear_encoder=linear_encoder,
    ).to(device)

    if verbose:
        print(model)
        print(
            f"Lipschitz bounds: {model.log_lipschitz_bounds[0].item():.2e} (encoder) and {model.log_lipschitz_bounds[1].item():.2e} (decoder)"
        )

    # *** Optimizer, Scheduler and Early-stopping ***

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        threshold_mode="rel",
        min_lr=1e-5,
    )

    # Early stopping variables
    best_val_loss = float("inf")
    no_improvement = 0
    best_model_path = os.path.join(checkpoint_dir, "best_ae_model.pt")

    # *** Data Preparation ***

    # Split into training and validation sets using sklearn
    train_data, val_data = train_test_split(data, test_size=val_split, random_state=42)

    # --- Symmetry augmentation (training set only) ---
    if augment_by_symmetry:
        if len(train_data.shape) != 3:
            raise NotImplementedError(
                "symmetry shift is only implemented for ham_terms"
            )
        print(train_data.shape)

        L = train_data.shape[-1]
        n_data = train_data.shape[0]
        augmented_data = np.zeros([train_data.shape[0] * 2 * L, 3, L])

        for j in range(L):
            # shift
            augmented_data[n_data * j : n_data * (j + 1)] = np.roll(
                train_data, j, axis=-1
            )

        # mirror symmetry is more complicated:
        # the second row of each datapoint contains right-hopping terms c_i^\dag c_{i+1}
        # when mirroring, we obrain the left-hopping term c_i^\dag c_{i-1}
        mirrored_data = np.copy(augmented_data[: n_data * L, :, ::-1])
        # we can restore by rolling back only the second row of the ham_terms by one
        mirrored_data[:, 1, :] = np.roll(mirrored_data[:, 1, :], -1, axis=-1)
        augmented_data[n_data * L :] = mirrored_data

        if verbose:
            print(
                f"Applied symmetry augmentation: data shape {train_data.shape} -> {augmented_data.shape} (factor {augmented_data.shape[0]/train_data.shape[0]:.1f})."
            )
        train_data = augmented_data

    train_size, val_size = len(train_data), len(val_data)

    # Convert to torch tensors
    train_data_tensor = torch.FloatTensor(train_data)
    val_data_tensor = torch.FloatTensor(val_data)

    # Initialize normalization parameters in the model
    model.initialize_normalization(train_data_tensor)

    # Create datasets
    train_dataset = TensorDataset(train_data_tensor)
    val_dataset = TensorDataset(val_data_tensor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # *** Initialize config and training Variables to be saved ***

    # Tracking variables
    train_loss_history = []
    val_loss_history = []

    # Dictionary to save training info

    model_config = {
        "input_shape": list(input_shape),
        "flattened_dim": flattened_dim,
        "n_data": len(data),
        "hidden_depth": hidden_depth,
        "latent_dim": latent_dim,
        "min_dim": min_dim,
        "scaling_factor": scaling_factor,
        "interpolation": interpolation,
        "hidden_dims": hidden_dims,
        "enable_batchnorm": enable_batchnorm,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": str(device),
        "val_split": val_split,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_threshold": early_stopping_threshold,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "scheduler_threshold": scheduler_threshold,
        "weight_decay": weight_decay,
        "lambda_latent_norm": lambda_latent_norm,
        "lambda_latent_repulsion": lambda_latent_repulsion,
        "lambda_lipschitz_encoder": lambda_lipschitz_encoder,
        "lambda_lipschitz_decoder": lambda_lipschitz_decoder,
        "latent_epsilon": latent_epsilon,
        "use_statevector": use_statevector,
        "reduction_type": reduction_type,
        "activation": repr(model.activation),
        "use_lipschitz_layers": model.use_lipschitz_layers,
        "linear_encoder": linear_encoder,
        "augment_by_symmetry": augment_by_symmetry,
    }

    with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f)

    training_info = {
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "reconstruction_loss": float("inf"),
        "latent_norm_loss": float("inf"),
        "repulsion_loss": float("inf"),
    }

    # *** Training Loop ***

    epoch_bar = tqdm(range(epochs), desc="Training AE", disable=True)

    for epoch in epoch_bar:
        # *** Training Phase ***

        model.train()
        avg_train_loss_components = (
            np.zeros(6)
            if (lambda_lipschitz_encoder > 0 or lambda_lipschitz_decoder > 0)
            else np.zeros(4)
        )

        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)

            # Forward pass
            optimizer.zero_grad()
            z_encoded = model.encode(x)
            x_reconstructed = model.decode(z_encoded)

            # Compute loss
            log_lipschitz_bounds = (
                model.log_lipschitz_bounds
                if (lambda_lipschitz_encoder > 0 or lambda_lipschitz_decoder > 0)
                else None
            )

            loss_components = loss_function(
                input_vector=x,
                reconstructed_vector=x_reconstructed,
                latent_vector=z_encoded,
                log_lipschitz_bounds=log_lipschitz_bounds,
                lambda_latent_norm=lambda_latent_norm,
                lambda_latent_repulsion=lambda_latent_repulsion,
                lambda_lipschitz_encoder=lambda_lipschitz_encoder,
                lambda_lipschitz_decoder=lambda_lipschitz_decoder,
                epsilon_latent_repulsion=latent_epsilon,
                reduction_type=reduction_type,
                use_statevector=use_statevector,
                return_components=True,
            )

            # Backward pass and optimization
            total_loss = loss_components[0]
            total_loss.backward()

            # grad_stats = check_gradients(model, epoch, batch_idx)

            optimizer.step()

            # Update totals

            if reduction_type == "mean":
                avg_train_loss_components += [
                    comp.item() * len(x) for comp in loss_components
                ]
            else:
                avg_train_loss_components += [comp.item() for comp in loss_components]

        # Calculate average training losses
        avg_train_loss_components /= train_size

        # *** Validation phase ***

        model.eval()
        avg_val_loss_components = np.zeros_like(avg_train_loss_components)

        with torch.no_grad():
            for batch_idx, (x,) in enumerate(val_loader):
                x = x.to(device)

                # Forward pass
                z_encoded = model.encode(x)
                x_reconstructed = model.decode(z_encoded)

                # Compute loss
                log_lipschitz_bounds = (
                    model.log_lipschitz_bounds
                    if (lambda_lipschitz_encoder > 0 or lambda_lipschitz_decoder > 0)
                    else None
                )
                loss_components = loss_function(
                    input_vector=x,
                    reconstructed_vector=x_reconstructed,
                    latent_vector=z_encoded,
                    log_lipschitz_bounds=log_lipschitz_bounds,
                    lambda_latent_norm=lambda_latent_norm,
                    lambda_latent_repulsion=lambda_latent_repulsion,
                    lambda_lipschitz_encoder=lambda_lipschitz_encoder,
                    lambda_lipschitz_decoder=lambda_lipschitz_decoder,
                    epsilon_latent_repulsion=latent_epsilon,
                    reduction_type=reduction_type,
                    use_statevector=use_statevector,
                    return_components=True,
                )

                # Update totals
                if reduction_type == "mean":
                    avg_val_loss_components += [
                        comp.item() * len(x) for comp in loss_components
                    ]
                else:
                    avg_val_loss_components += [comp.item() for comp in loss_components]

        # Calculate average validation losses
        avg_val_loss_components /= val_size

        # Update scheduler with average validation loss
        scheduler.step(avg_val_loss_components[0])

        # Track losses
        train_loss_history.append(avg_train_loss_components)
        val_loss_history.append(avg_val_loss_components)

        # Update tqdm description
        epoch_bar.set_postfix(
            {
                "train_loss": f"{avg_train_loss_components[0]:.4f}",
                "val_loss": f"{avg_val_loss_components[0]:.4f}",
            }
        )

        # Print detailed stats based on verbose parameter
        if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(
                f"  Train Loss: {avg_train_loss_components[0]:.2e} = "
                f"rec_loss: {avg_train_loss_components[1]:.2e} + "
                f"λ_n * norm_loss: {lambda_latent_norm * avg_train_loss_components[2]:.2e} + "
                f"λ_r * repel_loss: {lambda_latent_repulsion * avg_train_loss_components[3]:.2e}"
                + (
                    f" + λ_lip_enc * lip_loss_enc: {lambda_lipschitz_encoder * avg_train_loss_components[4]:.2e}"
                    if lambda_lipschitz_encoder > 0
                    else ""
                )
                + (
                    f" + λ_lip_dec * lip_loss_dec: {lambda_lipschitz_decoder * avg_train_loss_components[5]:.2e}"
                    if lambda_lipschitz_decoder > 0
                    else ""
                )
            )
            print(
                f"  Valid Loss: {avg_val_loss_components[0]:.2e} = "
                f"rec_loss: {avg_val_loss_components[1]:.2e} + "
                f"λ_n * norm_loss : {lambda_latent_norm * avg_val_loss_components[2]:.2e} + "
                f"λ_r * repel_loss : {lambda_latent_repulsion * avg_val_loss_components[3]:.2e}"
                + (
                    f" + λ_lip_enc * lip_loss_enc : {lambda_lipschitz_encoder * avg_val_loss_components[4]:.2e}"
                    if lambda_lipschitz_encoder > 0
                    else ""
                )
                + (
                    f" + λ_lip_dec * lip_loss_dec : {lambda_lipschitz_decoder * avg_val_loss_components[5]:.2e}"
                    if lambda_lipschitz_decoder > 0
                    else ""
                )
            )
            print(f"  Learning Rate: {scheduler.get_last_lr()[-1]:.2e}")

        # Early stopping check
        improvement = best_val_loss - avg_val_loss_components[0]

        # Always save if current val loss is better (even marginally)
        if improvement > 0:
            best_val_loss = avg_val_loss_components[0]

            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(best_val_loss),
                },
                best_model_path,
            )

            # Update training info
            training_info["best_epoch"] = epoch
            training_info["best_val_loss"] = best_val_loss
            training_info["reconstruction_loss"] = avg_val_loss_components[1]
            training_info["latent_norm_loss"] = avg_val_loss_components[2]
            training_info["repulsion_loss"] = avg_val_loss_components[3]
            if lambda_lipschitz_encoder > 0 or lambda_lipschitz_decoder > 0:
                training_info["lip_loss_encoder"] = avg_val_loss_components[4]
                training_info["lip_loss_decoder"] = avg_val_loss_components[5]

        # Check early stopping with a small epsilon threshold
        if improvement > early_stopping_threshold:  # HERE
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    # *** Collect and save training information ***

    training_info["epochs_trained"] = epoch

    # Save training info
    with open(os.path.join(checkpoint_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f)

    plot_and_save_training_curves(
        train_loss_history,
        val_loss_history,
        lambda_latent_norm,
        lambda_latent_repulsion,
        lambda_lipschitz_encoder,
        lambda_lipschitz_decoder,
        checkpoint_dir,
    )

    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    gc.collect()

    return model, training_info


def plot_and_save_training_curves(
    train_loss_history,
    val_loss_history,
    lambda_latent_norm,
    lambda_latent_repulsion,
    lambda_lipschitz_encoder,
    lambda_lipschitz_decoder,
    checkpoint_dir,
):
    train_loss_history = np.array(train_loss_history)
    val_loss_history = np.array(val_loss_history)

    ncols = 3 if (lambda_lipschitz_encoder > 0 or lambda_lipschitz_decoder > 0) else 2

    fig, axes = plt.subplots(2, ncols, figsize=(12, 8), sharex=True)

    plt.sca(axes[0, 0])
    plt.title("Total Loss")
    plt.plot(train_loss_history[:, 0], label="Train")
    plt.plot(val_loss_history[:, 0], label="Validation")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.sca(axes[0, 1])
    plt.title("Reconstruction Loss")
    plt.plot(train_loss_history[:, 1], label="Train")
    plt.plot(val_loss_history[:, 1], label="Validation")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.sca(axes[1, 0])
    plt.title(f"Latent Norm Loss ($\lambda_n={lambda_latent_norm}$)")
    plt.plot(train_loss_history[:, 2], label="Train")
    plt.plot(val_loss_history[:, 2], label="Validation")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)

    plt.sca(axes[1, 1])
    plt.title(f"Latent Repulsion Loss ($\lambda_r={lambda_latent_repulsion}$)")
    plt.plot(train_loss_history[:, 3], label="Train")
    plt.plot(val_loss_history[:, 3], label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)

    if lambda_lipschitz_encoder > 0 or lambda_lipschitz_decoder > 0:
        plt.sca(axes[0, 2])
        plt.title(f"Lipschitz Loss Encoder ($\lambda_e={lambda_lipschitz_encoder}$)")
        plt.plot(train_loss_history[:, 4], label="Train")
        plt.plot(val_loss_history[:, 4], label="Validation")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.3)

        plt.sca(axes[1, 2])
        plt.title(f"Lipschitz Loss Decoder ($\lambda_d={lambda_lipschitz_decoder}$)")
        plt.plot(train_loss_history[:, 5], label="Train")
        plt.plot(val_loss_history[:, 5], label="Validation")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "training_curves.png"))
    plt.close()


def load_model(checkpoint_dir, device="cpu"):
    """
    Load a trained autoencoder model from the specified checkpoint directory.

    Args:
        checkpoint_dir (str): Directory containing the model checkpoint.
        device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        model (AE): The loaded autoencoder model with normalization parameters.
    """
    # Load model config
    with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
        model_config = json.load(f)

    model_path = os.path.join(checkpoint_dir, "best_ae_model.pt")

    # Get data from config
    input_shape = model_config.get("input_shape")
    input_shape = tuple(input_shape)  # Convert list back to tuple

    activation_dict = {repr(nn.Softplus()): nn.Softplus()}
    try:
        activation = activation_dict[model_config["activation"]]
    except KeyError:
        activation = nn.Softplus()
        Warning(
            f"Unknown activation function [{model_config.get('activation', 'None')}] saved in model_config. using Softplus instead."
        )

    use_lipschitz_layers = (
        model_config["lambda_lipschitz_encoder"] > 0
        or model_config["lambda_lipschitz_decoder"] > 0
    )

    # Initialize model with the same configuration
    model = AE(
        input_dim=input_shape,
        hidden_dims=model_config["hidden_dims"],
        latent_dim=model_config["latent_dim"],
        activation=activation,
        use_lipschitz_layers=use_lipschitz_layers,
        use_batchnorm=model_config.get("enable_batchnorm", False),
        linear_encoder=model_config.get("linear_encoder", False),
    ).to(device)

    # Load model weights (including normalization parameters)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.normalization_initialized = True  # Set normalization initialized flag
    model.eval()

    return model
