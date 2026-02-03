import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Tuple, Union, Dict, List
from dftqml.autoencoders import autoencoder_utils
import h5py
from dftqml.autoencoders.lipschitz import LipschitzLinear


def _loss_norm_latent_well(latent_vector, well_radius=2.0, reduction_type="mean"):
    """
    Compute a well-based loss for the latent space.

    The loss is 0 for latent norms within a radius, and increases quadratically outside the well.

    Args:
        latent_vector: The latent representation tensor
        well_radius: The radius of the well (default 1.0 for [-1, 1] range)
        reduction_type: Type of reduction to apply

    Returns:
        The well-based latent loss
    """
    # Compute norm of the latent vector
    norm = torch.norm(latent_vector, p=2, dim=1)

    # subtract the well radius and apply ReLU to clamp negative values to zero
    loss = torch.relu(norm - well_radius)

    # Apply reduction
    if reduction_type == "sum":
        loss = torch.sum(loss)
    elif reduction_type == "mean":
        loss = torch.mean(loss)
    elif reduction_type is None or reduction_type == "none":
        loss = loss
    else:
        raise ValueError(f"Invalid reduction_type: {reduction_type}")

    return loss


def _loss_latent_contrastive_repulsion(
    latent_vector, input_vector, epsilon=1e-6, reduction_type="mean"
):
    """
    Computes a contrastive repulsion loss between input and latent representations.

    This loss encourages the pairwise distances in the latent space to reflect those in the input space,
    by penalizing cases where points that are far apart in the input space are mapped too close together
    in the latent space. The loss is computed as the ratio of squared input distances to squared latent
    distances (plus a small epsilon for numerical stability), averaged or summed over all unique pairs.

    Args:
        latent_vector: Latent representations of shape (batch_size, latent_dim).
        input_vector: Input representations of shape (batch_size, input_dim).
        epsilon: Small value to avoid division by zero. Default is 1e-6.
        reduction_type: Specifies the reduction to apply to the output: "mean" | "sum".

    Returns:
        torch.Tensor: The computed contrastive repulsion loss (scalar).
    """

    if input_vector.dim() > 2:
        input_vector = input_vector.flatten(start_dim=1)

    input_distances = torch.cdist(input_vector, input_vector, p=2)
    latent_distances = torch.cdist(latent_vector, latent_vector, p=2)

    # Exclude self-pairs (i = j)
    mask = torch.triu(torch.ones_like(input_distances, dtype=torch.bool), diagonal=1)

    repulsion_terms = (input_distances[mask] ** 2) / (
        latent_distances[mask] ** 2 + epsilon
    )

    # normalize by average input distance
    avg_input_distance = (input_distances[mask] ** 2).mean().item()
    repulsion_terms /= avg_input_distance

    # Apply reduction
    if reduction_type == "sum":
        loss = torch.sum(repulsion_terms)
    elif reduction_type == "mean":
        loss = torch.mean(repulsion_terms)
    else:
        raise ValueError(f"Invalid reduction_type: {reduction_type}")

    return loss


def loss_function(
    input_vector,
    reconstructed_vector,
    latent_vector,
    log_lipschitz_bounds=None,
    lambda_latent_norm=0.0,
    lambda_latent_repulsion=0.0,
    lambda_lipschitz_encoder=0.0,
    lambda_lipschitz_decoder=0.0,
    epsilon_latent_repulsion=1e-8,
):
    """
    Compute the total loss for the autoencoder, including reconstruction loss, latent space regularization,
    latent repulsion, and optional Lipschitz regularization.

    Args:
        input_vector (torch.Tensor): Original input data.
        reconstructed_vector (torch.Tensor): Reconstructed data from the autoencoder.
        latent_vector (torch.Tensor): Encoded (latent) representation of the input data.
        lipschitz_bound (tuple or None): Lipschitz regularization losses for encoder and decoder (default None).
        lambda_latent_norm (float): Regularization coefficient for the latent norm loss (default 0.0).
        lambda_latent_repulsion (float): Coefficient for the latent repulsion loss (default 0.0).
        lambda_lipschitz (float): Coefficient for the Lipschitz regularization loss (default 0.0).
        epsilon_latent_repulsion (float): Small constant to avoid division by zero in repulsion loss (default 1e-8).
        reduction_type (str): Reduction to apply to all loss components ("mean", "sum", or "none").
        use_statevector (bool): If True, use statevector loss (infidelity + normalization) for reconstruction.
        return_components (bool): If True, return individual loss components.
        use_well_loss (bool): If True, use well-based latent norm loss.
        use_contrastive_repulsion (bool): If True, use contrastive repulsion loss.

    Returns:
        torch.Tensor: Total loss for the autoencoder.
        or
        tuple: (total_loss, reconstruction_loss, latent_norm_loss, latent_repulsion_loss, lipschitz_bound[0], lipschitz_bound[1])
            if return_components is True.
    """
    # Reconstruction loss

    reduction_type = "mean"
    reconstruction_loss = F.mse_loss(
        input_vector, reconstructed_vector, reduction=reduction_type
    )
    total_loss = reconstruction_loss.clone()

    latent_norm_loss = _loss_norm_latent_well(
        latent_vector, reduction_type=reduction_type
    )

    total_loss += lambda_latent_norm * latent_norm_loss

    latent_repulsion_loss = _loss_latent_contrastive_repulsion(
        latent_vector,
        input_vector,
        epsilon=epsilon_latent_repulsion,
        reduction_type=reduction_type,
    )

    total_loss += lambda_latent_repulsion * latent_repulsion_loss

    if log_lipschitz_bounds != None:
        lipschitz_encoder_loss = log_lipschitz_bounds[0]
        lipschitz_decoder_loss = log_lipschitz_bounds[1]
        total_loss += lambda_lipschitz_encoder * lipschitz_encoder_loss
        total_loss += lambda_lipschitz_decoder * lipschitz_decoder_loss
        return (
            total_loss,
            reconstruction_loss,
            latent_norm_loss,
            latent_repulsion_loss,
            log_lipschitz_bounds[0],
            log_lipschitz_bounds[1],
        )
    else:
        return total_loss, reconstruction_loss, latent_norm_loss, latent_repulsion_loss


class AE(nn.Module):
    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        hidden_dims: list,
        latent_dim: int,
        use_lipschitz_layers=True,
        linear_model=False,
    ):
        super(AE, self).__init__()

        # model parameters
        self.activation = nn.Softplus()
        self.use_lipschitz_layers = use_lipschitz_layers
        self.linear_model = linear_model

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
        if self.linear_model:
            # Single dense layer encoder, never using Lipschitz layers nor batchnorm
            self.encoder = nn.Linear(self.flattened_dim, latent_dim)
            self.decoder = nn.Linear(latent_dim, self.flattened_dim)
        else:
            encoder_layers = []
            prev_dim = self.flattened_dim
            for h_dim in hidden_dims:
                encoder_layers.append(linear_layer(prev_dim, h_dim))

                encoder_layers.append(self.activation)
                prev_dim = h_dim
            # Final encoder layer to latent dimension
            encoder_layers.append(linear_layer(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder
            decoder_layers = []
            # First decoder layer
            decoder_layers.append(linear_layer(latent_dim, hidden_dims[-1]))
            decoder_layers.append(self.activation)
            prev_dim = hidden_dims[-1]
            # Hidden decoder layers
            for h_dim in reversed(hidden_dims[:-1]):
                decoder_layers.append(linear_layer(prev_dim, h_dim))
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
        if self.linear_model:
            log_bound_model = torch.tensor(0.0, device=next(self.parameters()).device)
            return log_bound_model, log_bound_model
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
    scaling_factor=20.0,
    linear_model=False,
    # Training parameters
    batch_size=256,
    epochs=300,
    learning_rate=0.004,
    val_split=0.1,
    early_stopping_patience=30,
    early_stopping_threshold=1e-10,
    scheduler_patience=10,
    scheduler_factor=0.5,
    scheduler_threshold=1e-8,
    # Regularization parameters
    lambda_latent_norm=0.0,
    lambda_latent_repulsion=0.0,
    lambda_lipschitz_encoder=0.0,
    lambda_lipschitz_decoder=0.0,
    latent_epsilon=1e-8,
    verbose=10,
    device="cuda",
    checkpoint_dir="checkpoints",
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

        lambda_norm (float): Regularization coefficient for the latent space.
        lambda_repel (float): Coefficient for the repulsion term in the latent space.
        lambda_lipschitz (float): Coefficient for Lipschitz regularization.
            If set to zero, an unregularized model will be used.
        epsilon (float): Small constant to avoid division by zero in the repulsion term.

        device (str): Device to use for training ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.

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

    hidden_dims = autoencoder_utils.geometric_hidden_dims(
        input_dim=flattened_dim,
        min_dim=min_dim,
        hidden_depth=hidden_depth,
        scaling_factor=scaling_factor,
    )

    # Initialize model

    model = AE(
        input_shape,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        linear_model=linear_model,
    ).to(device)

    if verbose:
        print(model)
        print(
            f"Lipschitz bounds: {model.log_lipschitz_bounds[0].item():.2e} (encoder) and {model.log_lipschitz_bounds[1].item():.2e} (decoder)"
        )

    # *** Optimizer, Scheduler and Early-stopping ***

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        "hidden_dims": hidden_dims,
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
        "lambda_latent_norm": lambda_latent_norm,
        "lambda_latent_repulsion": lambda_latent_repulsion,
        "lambda_lipschitz_encoder": lambda_lipschitz_encoder,
        "lambda_lipschitz_decoder": lambda_lipschitz_decoder,
        "latent_epsilon": latent_epsilon,
        "activation": repr(model.activation),
        "use_lipschitz_layers": model.use_lipschitz_layers,
        "linear_model": linear_model,
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
            )

            # Backward pass and optimization
            total_loss = loss_components[0]
            total_loss.backward()

            # grad_stats = check_gradients(model, epoch, batch_idx)

            optimizer.step()

            # Update totals
            avg_train_loss_components += [
                comp.item() * len(x) for comp in loss_components
            ]

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
                )

                # Update totals

                avg_val_loss_components += [
                    comp.item() * len(x) for comp in loss_components
                ]

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
        if improvement > early_stopping_threshold:
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
        use_lipschitz_layers=use_lipschitz_layers,
        linear_model=model_config.get("linear_model", False),
    ).to(device)

    # Load model weights (including normalization parameters)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.normalization_initialized = True  # Set normalization initialized flag
    model.eval()

    return model


def run_ablation_study(
    data: np.ndarray,
    L: int = 6,
    n_runs: int = 5,
    base_checkpoint_dir: str = "ablation_results",
    device: str = "cpu",
    # Shared hyperparameters
    hidden_depth: int = 4,
    min_dim: int = 6**2,
    scaling_factor: float = 20.0,
    batch_size: int = 256,
    epochs: int = 300,
    learning_rate: float = 0.004,
    # Full model regularization parameters
    lambda_latent_norm: float = 1e-7,
    lambda_latent_repulsion: float = 1e-7,
    lambda_lipschitz_encoder: float = 1e-9,
    lambda_lipschitz_decoder: float = 1e-8,
    **train_kwargs,
) -> Dict[str, Dict[str, List]]:
    """
    Perform systematic ablation study on autoencoder architecture and regularization.

    Args:
        data: Training data array
        L: System size (latent dimension will be L-1)
        n_runs: Number of independent runs per configuration
        base_checkpoint_dir: Base directory for saving results
        device: Device for training ('cuda' or 'cpu')
        hidden_depth: Number of hidden layers
        min_dim: Minimum hidden dimension
        scaling_factor: Scaling factor for hidden dimensions
        batch_size: Training batch size
        epochs: Maximum training epochs
        learning_rate: Learning rate
        lambda_latent_norm: Full model well loss coefficient (α)
        lambda_latent_repulsion: Full model repulsion loss coefficient (β)
        lambda_lipschitz_encoder: Full model encoder Lipschitz coefficient (γ)
        lambda_lipschitz_decoder: Full model decoder Lipschitz coefficient (δ)
        **train_kwargs: Additional training parameters

    Returns:
        Dictionary containing results for each configuration
    """

    # Define ablation configurations
    configs = {
        "full_model": {
            "description": "Full model with all regularization terms",
            "params": {
                "lambda_latent_norm": lambda_latent_norm,
                "lambda_latent_repulsion": lambda_latent_repulsion,
                "lambda_lipschitz_encoder": lambda_lipschitz_encoder,
                "lambda_lipschitz_decoder": lambda_lipschitz_decoder,
                "linear_model": False,
            },
        },
        "reconstruction_only": {
            "description": "Only reconstruction loss (α=β=γ=δ=0)",
            "params": {
                "lambda_latent_norm": 0.0,
                "lambda_latent_repulsion": 0.0,
                "lambda_lipschitz_encoder": 0.0,
                "lambda_lipschitz_decoder": 0.0,
                "linear_model": False,
            },
        },
        "no_lipschitz": {
            "description": "No Lipschitz regularization (γ=δ=0)",
            "params": {
                "lambda_latent_norm": lambda_latent_norm,
                "lambda_latent_repulsion": lambda_latent_repulsion,
                "lambda_lipschitz_encoder": 0.0,
                "lambda_lipschitz_decoder": 0.0,
                "linear_model": False,
            },
        },
        "no_well_loss": {
            "description": "No well confinement (α=0)",
            "params": {
                "lambda_latent_norm": 0.0,
                "lambda_latent_repulsion": lambda_latent_repulsion,
                "lambda_lipschitz_encoder": lambda_lipschitz_encoder,
                "lambda_lipschitz_decoder": lambda_lipschitz_decoder,
                "linear_model": False,
            },
        },
        "no_repulsion": {
            "description": "No contrastive repulsion (β=0)",
            "params": {
                "lambda_latent_norm": lambda_latent_norm,
                "lambda_latent_repulsion": 0.0,
                "lambda_lipschitz_encoder": lambda_lipschitz_encoder,
                "lambda_lipschitz_decoder": lambda_lipschitz_decoder,
                "linear_model": False,
            },
        },
        "linear_model": {
            "description": "Linear encoder/decoder (single layer)",
            "params": {
                "lambda_latent_norm": lambda_latent_norm,
                "lambda_latent_repulsion": lambda_latent_repulsion,
                "lambda_lipschitz_encoder": 0.0,  # No Lipschitz for linear model
                "lambda_lipschitz_decoder": 0.0,
                "linear_model": True,
            },
        },
    }

    # Set latent dimension
    latent_dim = L - 1

    # Storage for results
    results = {
        config_name: {
            "reconstruction_loss": [],
            "total_loss": [],
            "latent_norm_loss": [],
            "repulsion_loss": [],
            "best_epoch": [],
            "training_time": [],
        }
        for config_name in configs.keys()
    }

    # Create base directory
    os.makedirs(base_checkpoint_dir, exist_ok=True)

    # Save configuration info
    ablation_config = {
        "L": L,
        "latent_dim": latent_dim,
        "n_runs": n_runs,
        "hidden_depth": hidden_depth,
        "min_dim": min_dim,
        "scaling_factor": scaling_factor,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "data_shape": list(data.shape),
        "configurations": {k: v["description"] for k, v in configs.items()},
    }

    with open(os.path.join(base_checkpoint_dir, "ablation_config.json"), "w") as f:
        json.dump(ablation_config, f, indent=2)

    # Run experiments
    print("=" * 80)
    print(f"Starting Ablation Study with L={L}, latent_dim={latent_dim}")
    print(f"Number of runs per configuration: {n_runs}")
    print("=" * 80)

    for config_name, config_info in configs.items():
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config_name}")
        print(f"Description: {config_info['description']}")
        print(f"{'=' * 80}\n")

        config_params = config_info["params"]

        for run_idx in range(n_runs):
            print(f"\n--- Run {run_idx + 1}/{n_runs} for {config_name} ---")

            # Create checkpoint directory for this run
            run_checkpoint_dir = os.path.join(
                base_checkpoint_dir, config_name, f"run_{run_idx}"
            )
            os.makedirs(run_checkpoint_dir, exist_ok=True)

            # Set random seed for reproducibility
            torch.manual_seed(42 + run_idx)
            np.random.seed(42 + run_idx)

            # Train model
            import time

            start_time = time.time()

            try:
                model, training_info = train_ae(
                    data,
                    hidden_depth=hidden_depth,
                    latent_dim=latent_dim,
                    min_dim=min_dim,
                    scaling_factor=scaling_factor,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    device=device,
                    checkpoint_dir=run_checkpoint_dir,
                    **config_params,
                    **train_kwargs,
                )

                training_time = time.time() - start_time

                # Store results
                results[config_name]["reconstruction_loss"].append(
                    training_info["reconstruction_loss"]
                )
                results[config_name]["total_loss"].append(
                    training_info["best_val_loss"]
                )
                results[config_name]["latent_norm_loss"].append(
                    training_info["latent_norm_loss"]
                )
                results[config_name]["repulsion_loss"].append(
                    training_info["repulsion_loss"]
                )
                results[config_name]["best_epoch"].append(training_info["best_epoch"])
                results[config_name]["training_time"].append(training_time)

                print(f"✓ Completed in {training_time:.2f}s")
                print(f"  Best validation loss: {training_info['best_val_loss']:.6e}")
                print(
                    f"  Reconstruction loss: {training_info['reconstruction_loss']:.6e}"
                )

            except Exception as e:
                print(f"✗ Run {run_idx} failed with error: {e}")
                # Store NaN for failed runs
                for key in results[config_name].keys():
                    results[config_name][key].append(np.nan)

            # Clear memory
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # Compute statistics and save results
    print("\n" + "=" * 80)
    print("Computing Statistics and Saving Results")
    print("=" * 80)

    summary = compute_ablation_statistics(results, configs)
    save_ablation_results(results, summary, base_checkpoint_dir)
    plot_ablation_results(summary, base_checkpoint_dir)

    print(f"\n✓ Ablation study completed. Results saved to: {base_checkpoint_dir}")

    return results, summary


def compute_ablation_statistics(
    results: Dict[str, Dict[str, List]], configs: Dict[str, Dict]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Compute mean and standard deviation for each metric across runs.

    Args:
        results: Raw results from all runs
        configs: Configuration descriptions

    Returns:
        Dictionary with statistics for each configuration
    """
    summary = {}

    for config_name in results.keys():
        summary[config_name] = {
            "description": configs[config_name]["description"],
            "params": configs[config_name]["params"],
            "statistics": {},
        }

        for metric_name, values in results[config_name].items():
            # Convert to numpy array and remove NaNs
            values_array = np.array(values)
            valid_values = values_array[~np.isnan(values_array)]

            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                summary[config_name]["statistics"][metric_name] = {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "n_valid": int(len(valid_values)),
                    "all_values": valid_values.tolist(),
                }
            else:
                summary[config_name]["statistics"][metric_name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "n_valid": 0,
                    "all_values": [],
                }

    return summary


def save_ablation_results(results: Dict, summary: Dict, base_checkpoint_dir: str):
    """Save ablation results to JSON files."""

    # Save raw results
    with open(os.path.join(base_checkpoint_dir, "ablation_raw_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save summary statistics
    with open(os.path.join(base_checkpoint_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Create readable text summary
    with open(os.path.join(base_checkpoint_dir, "ablation_summary.txt"), "w") as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for config_name, config_data in summary.items():
            f.write(f"{config_name.upper()}\n")
            f.write(f"{config_data['description']}\n")
            f.write("-" * 80 + "\n")

            stats = config_data["statistics"]
            f.write(
                f"Reconstruction Loss: {stats['reconstruction_loss']['mean']:.6e} "
                f"± {stats['reconstruction_loss']['std']:.6e}\n"
            )
            f.write(
                f"Total Loss:          {stats['total_loss']['mean']:.6e} "
                f"± {stats['total_loss']['std']:.6e}\n"
            )
            f.write(
                f"Best Epoch:          {stats['best_epoch']['mean']:.1f} "
                f"± {stats['best_epoch']['std']:.1f}\n"
            )
            f.write(
                f"Training Time:       {stats['training_time']['mean']:.2f}s "
                f"± {stats['training_time']['std']:.2f}s\n"
            )
            f.write(f"Valid Runs:          {stats['reconstruction_loss']['n_valid']}\n")
            f.write("\n")


def plot_ablation_results(summary: Dict, base_checkpoint_dir: str):
    """Create visualization of ablation study results."""

    config_names = list(summary.keys())
    n_configs = len(config_names)

    # Extract data for plotting
    metrics = ["reconstruction_loss", "total_loss", "best_epoch", "training_time"]
    metric_labels = [
        "Reconstruction Loss",
        "Total Loss",
        "Best Epoch",
        "Training Time (s)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        means = []
        stds = []
        labels_list = []

        for config_name in config_names:
            stats = summary[config_name]["statistics"][metric]
            means.append(stats["mean"])
            stds.append(stats["std"])
            # Create short labels
            label_map = {
                "full_model": "Full",
                "reconstruction_only": "Rec Only",
                "no_lipschitz": "No Lip",
                "no_well_loss": "No Well",
                "no_repulsion": "No Rep",
                "linear_model": "Linear",
            }
            labels_list.append(label_map.get(config_name, config_name))

        x_pos = np.arange(n_configs)
        ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color=plt.cm.viridis(np.linspace(0, 0.9, n_configs)),
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list, rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")

        # Use log scale for loss metrics
        if "loss" in metric.lower():
            ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(base_checkpoint_dir, "ablation_comparison.png"), dpi=300)
    plt.close()

    # Create detailed comparison table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    table_data = [["Config", "Reconstruction Loss", "Total Loss", "Best Epoch"]]

    for config_name in config_names:
        stats = summary[config_name]["statistics"]
        row = [
            config_name.replace("_", " ").title(),
            f"{stats['reconstruction_loss']['mean']:.2e} ± {stats['reconstruction_loss']['std']:.2e}",
            f"{stats['total_loss']['mean']:.2e} ± {stats['total_loss']['std']:.2e}",
            f"{stats['best_epoch']['mean']:.1f} ± {stats['best_epoch']['std']:.1f}",
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("Ablation Study Results Summary", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(
        os.path.join(base_checkpoint_dir, "ablation_table.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

if __name__ == "__main__":
    
    # Example with synthetic data
    L = 6
    data_path = f"L{L}-N{L}-U4.0_reduced.hdf5"
    with h5py.File(data_path, "r") as f:
        data = f["hamiltonian_terms"][:90000]

    # Run ablation study
    results, summary = run_ablation_study(
        data=data,
        L=6,
        n_runs=5,
        base_checkpoint_dir="ablation_results_L6",
        device="cuda" if torch.cuda.is_available() else "cpu",
        hidden_depth=4,
        min_dim=6**2,
        scaling_factor=20.0,
        batch_size=256,
        epochs=300,
        learning_rate=0.004,
        # Full model regularization parameters
        lambda_latent_norm=1e-7,
        lambda_latent_repulsion=1e-7,
        lambda_lipschitz_encoder=1e-9,
        lambda_lipschitz_decoder=1e-8,
        # Additional training parameters
        val_split=0.1,
        early_stopping_patience=30,
        verbose=10,
    )

    print("\nAblation study completed successfully!")
