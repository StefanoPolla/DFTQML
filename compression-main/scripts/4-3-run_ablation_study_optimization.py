"""
Script to perform energy optimization across all ablation study models.

This script:
1. Loads all trained models from the ablation study
2. Performs latent space optimization for ground state energies
3. Compares optimization performance across configurations
4. Generates comprehensive comparison plots

Usage:
    python ablation_optimization.py --ablation_dir ablation_results_L6 --data_path L6-N6-U4.0_reduced.hdf5
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, List, Union, Tuple

from dftqml.hubbard_model.fhchain import FermiHubbardChain
import torch.nn as nn

from dftqml.autoencoders import autoencoder_utils
from dftqml.autoencoders.lipschitz import LipschitzLinear

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

def load_model(checkpoint_dir, device="cpu"):
    """
    Load a trained autoencoder model from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing model checkpoint
        device: Device to load model onto

    Returns:
        Loaded AE model
    """
    with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
        model_config = json.load(f)

    model_path = os.path.join(checkpoint_dir, "best_ae_model.pt")

    input_shape = tuple(model_config.get("input_shape"))

    activation_dict = {repr(nn.Softplus()): nn.Softplus()}
    activation = activation_dict.get(model_config["activation"], nn.Softplus())

    use_lipschitz_layers = (
        model_config.get("lambda_lipschitz_encoder", 0.0) > 0
        or model_config.get("lambda_lipschitz_decoder", 0.0) > 0
    )

    model = AE(
        input_dim=input_shape,
        hidden_dims=model_config["hidden_dims"],
        latent_dim=model_config["latent_dim"],
        use_lipschitz_layers=use_lipschitz_layers,
        linear_model=model_config.get("linear_model", False),
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle state_dict with Lipschitz layers when model doesn't use them
    state_dict = checkpoint["model_state_dict"]

    if not use_lipschitz_layers:
        # Remove Lipschitz-specific parameters (c, gamma) if present
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not (k.endswith(".c") or k.endswith(".gamma"))
        }

    # Load with strict=False to handle any remaining mismatches
    model.load_state_dict(state_dict, strict=False)
    model.normalization_initialized = True
    model.eval()

    return model


class LatentEnergyOptimizer:
    """Optimizer for finding ground state energies in latent space."""

    def __init__(self, checkpoint_dir, system_params, device="cpu"):
        """
        Args:
            checkpoint_dir: Path to trained AE checkpoint
            system_params: Dict with keys n_sites, n_particles, u, t, etc.
            device: "cpu" or "cuda"
        """
        self.device = device
        self.model = load_model(checkpoint_dir, device=device)
        self.system = FermiHubbardChain(**system_params)

        with open(f"{checkpoint_dir}/model_config.json", "r") as f:
            model_config = json.load(f)

        if len(model_config["input_shape"]) == 4:
            self.observable_type = "two_rdm"
            self._energy_from_observable = self.system.ground_energy_from_rdm
        elif len(model_config["input_shape"]) == 2:
            self.observable_type = "ham_terms"
            self._energy_from_observable = (
                self.system.ground_energy_from_hamiltonian_terms
            )
        else:
            raise ValueError("Unknown input shape in model_config")

    def energy_from_latent(self, z, potential=None):
        """Compute energy from latent vector."""
        observable = self.model.decode(z.unsqueeze(0)).squeeze(0)
        energy = self._energy_from_observable(observable, potential)
        return energy

    def minimize(
        self,
        potential,
        *,
        init_z=None,
        opt=None,
        lr=1e-2,
        max_iter=100,
        energy_tol=1e-6,
        well_size=2.0,
        well_penalty_strength=50.0,
        verbose=False,
        print_interval=1,
        return_loss_history=False,
    ):
        """
        Optimize latent vector to minimize energy.

        Args:
            potential: Potential array, shape (L, L)
            init_z: Initial latent vector
            opt: Optimizer class (defaults to Adam)
            lr: Learning rate
            max_iter: Maximum iterations
            energy_tol: Convergence tolerance
            well_size: Size of latent space well
            well_penalty_strength: Penalty for leaving well
            verbose: Print progress
            print_interval: Interval for printing
            return_loss_history: Return loss history

        Returns:
            z_opt: Optimized latent vector
            energy_opt: Optimized energy
            loss_history (optional): Loss history
        """
        loss_history = []

        # Get latent dimension from model
        if hasattr(self.model.encoder, "__getitem__"):
            # Sequential model
            latent_dim = None
            for layer in reversed(list(self.model.encoder)):
                if hasattr(layer, "out_features"):
                    latent_dim = layer.out_features
                    break
        else:
            # Linear model
            latent_dim = self.model.encoder.out_features

        # Initialize latent vector
        if init_z is None:
            z = torch.zeros((1, latent_dim), requires_grad=True, device=self.device)
        else:
            z = torch.tensor(
                init_z, dtype=torch.float32, requires_grad=True, device=self.device
            )

        if opt is None:
            optimizer = torch.optim.Adam([z], lr=lr)
        else:
            optimizer = opt([z], lr=lr)

        def cost_fn(z_vec):
            base_loss = self.energy_from_latent(z_vec, potential)
            if well_penalty_strength > 0:
                norm = z_vec.norm(p=2)
                penalty = well_penalty_strength * torch.relu(norm - well_size) ** 2
                return base_loss + penalty
            return base_loss

        def closure():
            optimizer.zero_grad()
            loss = cost_fn(z)
            loss.backward()
            loss_history.append(loss.item())
            return loss

        cost_old = float("inf")

        # Optimization loop
        for step in range(max_iter):
            loss_tensor = optimizer.step(closure)
            cost = loss_tensor.item()

            if verbose and step % print_interval == 0:
                print(
                    f"Step {step}: cost = {cost:.10f}, grad_norm = {z.grad.norm().item():.10f}"
                )

            # Check convergence
            if step > 2 and abs(cost - cost_old) < energy_tol:
                if verbose:
                    print(f"Convergence reached at step {step}.")
                break

            cost_old = cost

        opt_energy = self.energy_from_latent(z, potential).detach().item()
        z_opt = z.detach().cpu().numpy()

        if return_loss_history:
            return z_opt, opt_energy, loss_history
        else:
            return z_opt, opt_energy


def run_optimization_for_model(
    checkpoint_dir: str,
    potentials: np.ndarray,
    ground_energies: np.ndarray,
    system_params: Dict,
    optimization_kwargs: Dict,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run optimization for a single model checkpoint.

    Args:
        checkpoint_dir: Path to model checkpoint
        potentials: Array of potentials to optimize
        ground_energies: Reference ground energies
        system_params: System parameters
        optimization_kwargs: Optimization parameters
        device: Device for computation
        verbose: Print progress

    Returns:
        Dictionary with optimization results
    """
    # Initialize optimizer
    latentopt = LatentEnergyOptimizer(
        checkpoint_dir, system_params=system_params, device=device
    )

    n_potentials = len(potentials)
    latent_dim = (
        optimization_kwargs["init_z"][0]
        if isinstance(optimization_kwargs["init_z"], tuple)
        else len(optimization_kwargs["init_z"])
    )

    # Storage for results
    results = {
        "z_opt": np.zeros((n_potentials, latent_dim), dtype=np.float32),
        "energy_opt": np.zeros(n_potentials, dtype=np.float32),
        "energy_error": np.zeros(n_potentials, dtype=np.float32),
        "opt_steps": np.zeros(n_potentials, dtype=np.int32),
        "converged": np.zeros(n_potentials, dtype=bool),
    }

    # Run optimization for each potential
    iterator = tqdm(
        enumerate(potentials),
        total=n_potentials,
        desc=f"Optimizing {Path(checkpoint_dir).parent.name}",
        disable=not verbose,
    )

    for i, potential in iterator:
        try:
            z_opt, energy_opt, loss_history = latentopt.minimize(
                potential, **optimization_kwargs
            )

            results["z_opt"][i] = z_opt
            results["energy_opt"][i] = energy_opt
            results["energy_error"][i] = energy_opt - ground_energies[i]
            results["opt_steps"][i] = len(loss_history)

            # Check if converged (energy error within tolerance)
            results["converged"][i] = abs(results["energy_error"][i]) < 1e-4

        except Exception as e:
            print(f"Warning: Optimization failed for potential {i}: {e}")
            results["energy_opt"][i] = np.nan
            results["energy_error"][i] = np.nan
            results["converged"][i] = False

    return results


def run_ablation_optimization_study(
    ablation_dir: str,
    data_path: str,
    start_idx: int = 90000,
    end_idx: int = 100000,
    n_runs: int = 5,
    device: str = "cpu",
    optimization_kwargs: Optional[Dict] = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run optimization study across all ablation configurations.

    Args:
        ablation_dir: Directory containing ablation study results
        data_path: Path to HDF5 data file
        start_idx: Starting index for potentials
        end_idx: Ending index for potentials
        n_runs: Number of runs per configuration
        device: Device for computation
        optimization_kwargs: Optimization parameters
        save_results: Whether to save results
        verbose: Print progress

    Returns:
        Dictionary with results for all configurations
    """
    # Load data
    print(f"Loading data from {data_path}...")
    with h5py.File(data_path, "r") as f:
        potentials = f["potentials"][start_idx:end_idx]
        ground_energies = f["ground_energies"][start_idx:end_idx]

    print(f"Loaded {len(potentials)} potentials")

    # Load ablation config
    with open(os.path.join(ablation_dir, "ablation_config.json"), "r") as f:
        ablation_config = json.load(f)

    L = ablation_config["L"]
    latent_dim = ablation_config["latent_dim"]

    # Set up system params
    system_params = {"n_sites": L, "n_particles": L, "u": 4.0}

    # Default optimization parameters
    if optimization_kwargs is None:
        optimization_kwargs = {
            "init_z": [0.0] * latent_dim,
            "opt": torch.optim.LBFGS,
            "lr": 0.1,
            "max_iter": 100,
            "energy_tol": 1e-8,
            "well_size": 3.0,
            "well_penalty_strength": 50.0,
            "verbose": False,
            "print_interval": 50,
            "return_loss_history": True,
        }

    # Get list of configurations
    config_names = [
        d
        for d in os.listdir(ablation_dir)
        if os.path.isdir(os.path.join(ablation_dir, d)) and d not in ["__pycache__"]
    ]

    print(f"\nFound {len(config_names)} configurations: {config_names}")

    # Storage for all results
    all_results = {}

    # Process each configuration
    for config_name in config_names:
        print(f"\n{'='*80}")
        print(f"Processing configuration: {config_name}")
        print(f"{'='*80}")

        config_dir = os.path.join(ablation_dir, config_name)
        config_results = {"runs": [], "aggregate": {}}

        # Process each run
        run_dirs = sorted(
            [
                d
                for d in os.listdir(config_dir)
                if d.startswith("run_") and os.path.isdir(os.path.join(config_dir, d))
            ]
        )

        for run_dir in run_dirs[:n_runs]:
            run_path = os.path.join(config_dir, run_dir)

            # Check if model exists
            model_path = os.path.join(run_path, "best_ae_model.pt")
            if not os.path.exists(model_path):
                print(f"Warning: No model found at {model_path}, skipping...")
                continue

            print(f"\nProcessing {run_dir}...")

            try:
                # Run optimization
                results = run_optimization_for_model(
                    checkpoint_dir=run_path,
                    potentials=potentials,
                    ground_energies=ground_energies,
                    system_params=system_params,
                    optimization_kwargs=optimization_kwargs,
                    device=device,
                    verbose=verbose,
                )

                config_results["runs"].append(results)

                # Save individual run results if requested
                if save_results:
                    output_path = os.path.join(run_path, "optimization_results.hdf5")
                    save_optimization_results(
                        output_path,
                        results,
                        potentials,
                        ground_energies,
                        optimization_kwargs,
                    )
                    print(f"Saved results to {output_path}")

                # Print summary
                mean_error = np.nanmean(np.abs(results["energy_error"]))
                max_error = np.nanmax(np.abs(results["energy_error"]))
                convergence_rate = np.mean(results["converged"]) * 100

                print(f"  Mean |error|: {mean_error:.6e}")
                print(f"  Max |error|:  {max_error:.6e}")
                print(f"  Convergence:  {convergence_rate:.1f}%")

            except Exception as e:
                print(f"Error processing {run_dir}: {e}")
                continue

        # Aggregate statistics across runs
        if config_results["runs"]:
            config_results["aggregate"] = aggregate_run_statistics(
                config_results["runs"]
            )

        all_results[config_name] = config_results

    # Save aggregate results
    if save_results:
        save_aggregate_results(all_results, ablation_dir)

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_optimization_comparison(all_results, ablation_dir)

    return all_results


def aggregate_run_statistics(runs: List[Dict]) -> Dict:
    """
    Aggregate statistics across multiple runs.

    Args:
        runs: List of result dictionaries

    Returns:
        Dictionary with aggregate statistics
    """
    # Stack results from all runs
    energy_errors = np.stack([r["energy_error"] for r in runs])
    converged = np.stack([r["converged"] for r in runs])
    opt_steps = np.stack([r["opt_steps"] for r in runs])

    # Compute statistics
    aggregate = {
        # Mean absolute error
        "mae_mean": np.nanmean(np.abs(energy_errors)),
        "mae_std": np.nanstd(np.abs(energy_errors)),
        "mse_mean": np.nanmean(energy_errors**2),
        "mse_std": np.nanstd(energy_errors**2),
        # Maximum error
        "max_error_mean": np.nanmean(np.nanmax(np.abs(energy_errors), axis=1)),
        "max_error_std": np.nanstd(np.nanmax(np.abs(energy_errors), axis=1)),
        # Convergence rate
        "convergence_rate_mean": np.mean(converged) * 100,
        "convergence_rate_std": np.std(np.mean(converged, axis=1)) * 100,
        # Optimization steps
        "opt_steps_mean": np.mean(opt_steps),
        "opt_steps_std": np.std(opt_steps),
        # Number of runs
        "n_runs": len(runs),
        "n_samples": energy_errors.shape[1],
    }

    return aggregate


def save_optimization_results(
    output_path: str,
    results: Dict,
    potentials: np.ndarray,
    ground_energies: np.ndarray,
    optimization_kwargs: Dict,
):
    """Save optimization results to HDF5 file."""
    with h5py.File(output_path, "w") as f:
        # Save results
        f.create_dataset("z_opt", data=results["z_opt"])
        f.create_dataset("energy_opt", data=results["energy_opt"])
        f.create_dataset("energy_error", data=results["energy_error"])
        f.create_dataset("opt_steps", data=results["opt_steps"])
        f.create_dataset("converged", data=results["converged"])

        # Save reference data
        f.create_dataset("potentials", data=potentials)
        f.create_dataset("ground_energies", data=ground_energies)

        # Save optimization parameters as attributes
        for key, value in optimization_kwargs.items():
            try:
                if key == "opt":
                    f.attrs[key] = str(value)
                else:
                    f.attrs[key] = value
            except:
                pass


def save_aggregate_results(all_results: Dict, ablation_dir: str):
    """Save aggregate statistics to JSON."""
    output_path = os.path.join(ablation_dir, "optimization_summary.json")

    # Convert to serializable format
    summary = {}
    for config_name, config_data in all_results.items():
        if "aggregate" in config_data and config_data["aggregate"]:
            summary[config_name] = {}
            for k, v in config_data["aggregate"].items():
                # Convert numpy types to Python types
                if isinstance(v, (np.floating, np.integer)):
                    summary[config_name][k] = float(v)
                elif isinstance(v, np.ndarray):
                    summary[config_name][k] = v.tolist()
                elif isinstance(v, (int, float, str, bool)):
                    summary[config_name][k] = v
                else:
                    # Skip non-serializable types
                    continue

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved aggregate results to {output_path}")

    # Also save text summary
    text_path = os.path.join(ablation_dir, "optimization_summary.txt")
    with open(text_path, "w") as f:
        f.write("ABLATION STUDY OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for config_name, stats in summary.items():
            f.write(f"{config_name.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Mean Absolute Error: {stats['mae_mean']:.6e} ± {stats['mae_std']:.6e}\n"
            )
            f.write(
                f"Max Error:           {stats['max_error_mean']:.6e} ± {stats['max_error_std']:.6e}\n"
            )
            f.write(
                f"Convergence Rate:    {stats['convergence_rate_mean']:.1f}% ± {stats['convergence_rate_std']:.1f}%\n"
            )
            f.write(
                f"Optimization Steps:  {stats['opt_steps_mean']:.1f} ± {stats['opt_steps_std']:.1f}\n"
            )
            f.write(f"Number of Runs:      {stats['n_runs']}\n")
            f.write("\n")


def plot_optimization_comparison(all_results: Dict, ablation_dir: str):
    """
    Generate comprehensive comparison plots for optimization results.

    Args:
        all_results: Dictionary with results for all configurations
        ablation_dir: Directory to save plots
    """
    # Extract aggregate statistics
    config_names = list(all_results.keys())

    # Short labels for plotting
    label_map = {
        "full_model": "Full",
        "reconstruction_only": "Rec Only",
        "no_lipschitz": "No Lip",
        "no_well_loss": "No Well",
        "no_repulsion": "No Rep",
        "linear_model": "Linear",
    }

    labels = [label_map.get(name, name) for name in config_names]

    # Extract metrics
    mae_mean = []
    mae_std = []
    max_error_mean = []
    max_error_std = []
    conv_rate_mean = []
    conv_rate_std = []
    opt_steps_mean = []
    opt_steps_std = []

    for config_name in config_names:
        if (
            "aggregate" in all_results[config_name]
            and all_results[config_name]["aggregate"]
        ):
            agg = all_results[config_name]["aggregate"]
            mae_mean.append(agg["mae_mean"])
            mae_std.append(agg["mae_std"])
            max_error_mean.append(agg["max_error_mean"])
            max_error_std.append(agg["max_error_std"])
            conv_rate_mean.append(agg["convergence_rate_mean"])
            conv_rate_std.append(agg["convergence_rate_std"])
            opt_steps_mean.append(agg["opt_steps_mean"])
            opt_steps_std.append(agg["opt_steps_std"])
        else:
            # Placeholder for missing data
            mae_mean.append(np.nan)
            mae_std.append(np.nan)
            max_error_mean.append(np.nan)
            max_error_std.append(np.nan)
            conv_rate_mean.append(np.nan)
            conv_rate_std.append(np.nan)
            opt_steps_mean.append(np.nan)
            opt_steps_std.append(np.nan)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x_pos = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))

    # Plot 1: Mean Absolute Error
    ax = axes[0, 0]
    ax.bar(x_pos, mae_mean, yerr=mae_std, capsize=5, alpha=0.7, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Ground State Energy Error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Maximum Error
    ax = axes[0, 1]
    ax.bar(
        x_pos, max_error_mean, yerr=max_error_std, capsize=5, alpha=0.7, color=colors
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Maximum Absolute Error")
    ax.set_title("Worst-Case Performance")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Convergence Rate
    ax = axes[1, 0]
    ax.bar(
        x_pos, conv_rate_mean, yerr=conv_rate_std, capsize=5, alpha=0.7, color=colors
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Optimization Convergence")
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Optimization Steps
    ax = axes[1, 1]
    ax.bar(
        x_pos, opt_steps_mean, yerr=opt_steps_std, capsize=5, alpha=0.7, color=colors
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of Steps")
    ax.set_title("Optimization Efficiency")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, "optimization_comparison.png"), dpi=300)
    plt.close()

    # Create detailed error distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error distribution across samples
    ax = axes[0]
    for i, config_name in enumerate(config_names):
        if all_results[config_name]["runs"]:
            # Collect errors from all runs
            all_errors = []
            for run in all_results[config_name]["runs"]:
                all_errors.extend(np.abs(run["energy_error"]))
            all_errors = np.array(all_errors)
            all_errors = all_errors[~np.isnan(all_errors)]

            if len(all_errors) > 0:
                ax.hist(all_errors, bins=50, alpha=0.5, label=labels[i])

    ax.set_xlabel("|Energy Error|")
    ax.set_ylabel("Frequency")
    ax.set_title("Energy Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot comparison
    ax = axes[1]
    box_data = []
    for config_name in config_names:
        if all_results[config_name]["runs"]:
            all_errors = []
            for run in all_results[config_name]["runs"]:
                all_errors.extend(np.abs(run["energy_error"]))
            all_errors = np.array(all_errors)
            all_errors = all_errors[~np.isnan(all_errors)]
            box_data.append(np.log10(all_errors) if len(all_errors) > 0 else [])
        else:
            box_data.append([])

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("|Energy Error|")
    ax.set_title("Error Distribution by Configuration")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, "optimization_distributions.png"), dpi=300)
    plt.close()

    print(f"Saved comparison plots to {ablation_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run optimization study across ablation configurations"
    )
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default="ablation_results_L6",
        help="Directory containing ablation study results",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="L6-N6-U4.0_reduced.hdf5",
        help="Path to HDF5 data file",
    )
    parser.add_argument(
        "--start_idx", type=int, default=99000, help="Starting index for potentials"
    )
    parser.add_argument(
        "--end_idx", type=int, default=100000, help="Ending index for potentials"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of runs per configuration to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (cuda or cpu)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save individual run results"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress"
    )

    args = parser.parse_args()

    # Run optimization study
    results = run_ablation_optimization_study(
        ablation_dir=args.ablation_dir,
        data_path=args.data_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        n_runs=args.n_runs,
        device=args.device,
        save_results=not args.no_save,
        verbose=args.verbose,
    )

    print("\n" + "=" * 80)
    print("Optimization study completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
