"""
Cluster-ready script to optimize latent variables for all potentials in a dataset using a trained AE model.

This script loads a trained autoencoder and a dataset of potentials, then iterates through all potentials,
optimizing the latent variables for each, and saves the results.
Usage example:
    python optimize.py 4 4 4.0 --input_type ham_terms --hidden_depth 4 --latent_dim 3 --min_dim 16 --scaling_factor 20.0 --interpolation geometric --models_dir models --data_dir data --overwrite --verbose 1
    python optimize.py 4 4 4.0 --input_type ham_terms --no-save --verbose 1  # run without writing output file
"""

import argparse
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

from dftqml.autoencoders.optimizer import LatentEnergyOptimizer
from dftqml.autoencoders import autoencoder_utils

def main():
    parser = argparse.ArgumentParser(
        description="Optimize latent variables for all potentials in a dataset using a trained AE model."
    )
    parser.add_argument("L", type=int, help="Number of sites")
    parser.add_argument("N", type=int, help="Number of electrons")
    parser.add_argument("U", type=float, help="Coulomb repulsion")
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["ham_terms", "two_rdms"],
        required=True,
        help="Data type the AE was trained on",
    )

    parser.add_argument(
        "--iteration", type=int, default=0, help="Iteration number for the training run"
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Latent dimension of the autoencoder. If not set, it will default to L-1.",
    )
    parser.add_argument(
        "--hidden_depth", type=int, default=4, help="Number of hidden layers"
    )
    parser.add_argument(
        "--min_dim", type=int, default=None, help="Minimum dimension for hidden layers"
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=None,
        help="Scaling factor for first hidden layer (no upscaling if not set).",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=["geometric", "linear"],
        default="geometric",
        help="Interpolation method for hidden layer dimensions. Default is 'geometric'.",
    )
    parser.add_argument(
        "--enable_batchnorm",
        action="store_true",
        help="Enable batch normalization between hidden layers in the autoencoder. (By default batch normalization is disabled).",
    )
    parser.add_argument(
        "--linear_encoder",
        action="store_true",
        help="Use linear encoder in the autoencoder. (By default non-linear encoder is used).",
    )
    parser.add_argument(
        "--augment_by_symmetry",
        action="store_true",
        help="Augment training data by symmetry operations. (By default no augmentation is used).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "lbfgs"],
        default="adam",
        help="Optimizer to use for latent variable optimization.",
    )
    parser.add_argument(
        "--use_grad_tol",
        action="store_true",
        help="Use gradient norm tolerance for convergence instead of energy tolerance.",
    )

    parser.add_argument(
        "--lambda_latent_norm",
        type=float,
        default=1e-7,
        help="Weight for latent space norm regularization",
    )
    parser.add_argument(
        "--lambda_latent_repulsion",
        type=float,
        default=1e-7,
        help="Weight for latent space repulsion regularization",
    )
    parser.add_argument(
        "--lambda_lipschitz_encoder",
        type=float,
        default=1e-9,
        help="Weight for Lipschitz regularization on the encoder",
    )
    parser.add_argument(
        "--lambda_lipschitz_decoder",
        type=float,
        default=1e-8,
        help="Weight for Lipschitz regularization on the decoder",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay (L2) for model checkpoints (default 0.0)",
    )
    parser.add_argument(
        "--n_data",
        type=int,
        default=90000,
        help="Number of data points the AE was trained on (needed for checkpoint path)",
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the input data files",
    )

    parser.add_argument(
        "--output_name", type=str, default="energy_opt", help="Name for the output file"
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for optimization"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=500,
        help="Maximum optimization steps per potential",
    )
    parser.add_argument(
        "--energy_tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for energy",
    )
    parser.add_argument(
        "--grad_tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for gradient norm",
    )
    parser.add_argument(
        "--lr", type=float, default=0.15, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--well_size",
        type=float,
        default=2.0,
        help="Radius of the well within which to limit optimization",
    )
    parser.add_argument(
        "--well_penalty_strength",
        type=float,
        default=50.0,
        help="Penalty strength for the well constraint",
    )
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument(
        "--print_interval", type=int, default=1, help="Print progress every N steps"
    )
    parser.add_argument(
        "--start_idx", type=int, default=90000, help="Start index for test potentials"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="End index for test potentials"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output HDF5 file if it already exists.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save any output file; just run optimization and print a summary.",
    )

    args = parser.parse_args()

    # ********* Process input arguments **********

    # Theoretical minimum latent dimension
    args.latent_dim = args.latent_dim or (args.L - 1)

    # Programmatic hyperparameter choices derived from optuna search
    args.min_dim = args.min_dim or args.L**2
    args.scaling_factor = args.scaling_factor or (
        20.0 if args.input_type == "ham_terms" else 1.0
    )

    # Add one print statement to show the parsed arguments
    print(f"Parsed and processed arguments:\n{args}")

    # ********* Print arguments **********
    print("========== Optimization Run Arguments ==========")
    for arg, value in vars(args).items():
        if arg != "lambda_lipschitz_encoder":
            print(f"{arg:25}: {value}")
        else:
            print(
                f"{arg:25}: {args.lambda_lipschitz_encoder if not args.linear_encoder else 0.0}"
            )
    print("===============================================")

    optimization_kwargs = dict(
        init_z=[0.0] * args.latent_dim,
        opt=None if args.optimizer == "adam" else torch.optim.LBFGS,
        lr=args.lr,
        max_iter=args.max_iter,
        energy_tol=args.energy_tol,
        grad_tol=args.grad_tol,
        well_size=args.well_size,
        well_penalty_strength=args.well_penalty_strength,
        verbose=args.verbose,
        print_interval=args.print_interval,
        return_loss_history=True,
        use_grad_tol=args.use_grad_tol,
    )

    # Prepare checkpoint/model directory
    checkpoint_dir = autoencoder_utils.checkpoint_dir_path(
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
        args.lambda_lipschitz_encoder if not args.linear_encoder else 0.0,
        args.lambda_lipschitz_decoder,
        args.iteration,
        linear_encoder=args.linear_encoder,
        augment_by_symmetry=args.augment_by_symmetry,
        n_data=args.n_data,
    )

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} does not exist."
        )

    # Output path and check
    h5_path = os.path.join(checkpoint_dir, args.output_name + ".hdf5")
    if not args.no_save and os.path.exists(h5_path):
        if args.overwrite:
            print(f"Output file {h5_path} exists. Overwriting as requested.")
            try:
                os.remove(h5_path)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to remove existing output file {h5_path}: {e}"
                )
        else:
            print(
                f"Output file {h5_path} already exists. Exiting (use --overwrite or --no-save)."
            )
            exit(0)

    # Load potentials and reference ground energies
    input_file = os.path.join(args.data_dir, f"L{args.L}-N{args.N}-U{args.U}.hdf5")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist. Trying reduced...")
        input_file = os.path.join(
            args.data_dir, f"L{args.L}-N{args.N}-U{args.U}_reduced.hdf5"
        )
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} does not exist.")

    with h5py.File(input_file, "r") as f:
        potentials = f["potentials"][:]
        ground_energies = f["ground_energies"][:]

    n_potentials = potentials.shape[0]
    start_idx = args.start_idx
    end_idx = args.end_idx or n_potentials

    # Continuation mode removed: we always start fresh (unless --overwrite not set and file exists, then exit)

    # Set up system params for optimizer
    system_params = {"n_sites": args.L, "n_particles": args.N, "u": args.U}

    # Initialize optimizer
    latentopt = LatentEnergyOptimizer(
        checkpoint_dir, system_params=system_params, device=args.device
    )

    save_interval = 100
    print(
        f"Optimizing latent variables for potentials {start_idx} to {end_idx-1}..."
        + (" (no-save mode)" if args.no_save else "")
    )

    if args.no_save:
        energies = []
        for i in tqdm(range(start_idx, end_idx), desc="Optimizing potentials"):
            potential = potentials[i]
            z_opt, energy_opt, loss_history = latentopt.minimize(
                potential, **optimization_kwargs
            )
            energies.append(energy_opt)
            if args.verbose and (
                (i - start_idx + 1) % save_interval == 0 or i == end_idx - 1
            ):
                print(
                    f"Completed {i - start_idx + 1} potentials; latest energy {energy_opt:.6f}"
                )
        energies = np.asarray(energies)
        print("No-save run complete.")
        print(f"Potentials processed: {len(energies)}")
        print(f"Mean optimized energy: {energies.mean():.6f}")
        print(f"Std optimized energy : {energies.std():.6f}")
    else:
        # Open HDF5 file in append mode and create datasets
        with h5py.File(h5_path, "a") as f:
            for key, value in optimization_kwargs.items():
                try:
                    if key == "opt":
                        f.attrs[key] = args.optimizer
                    else:
                        f.attrs[key] = value
                except Exception as e:
                    raise RuntimeError(
                        f"Warning: Could not save attribute {key} = {value}: {e}"
                    )

            f.attrs["start_idx"] = start_idx
            f.attrs["end_idx"] = end_idx

            f.create_dataset(
                "z_opt",
                shape=(end_idx - args.start_idx, args.latent_dim),
                dtype=np.float32,
            )
            f.create_dataset(
                "energy_opt", shape=(end_idx - args.start_idx,), dtype=np.float32
            )
            f.create_dataset(
                "ref_energy", data=ground_energies[args.start_idx : end_idx]
            )
            f.create_dataset("potentials", data=potentials[args.start_idx : end_idx])
            f.create_dataset(
                "opt_steps", shape=(end_idx - args.start_idx,), dtype=np.int32
            )
            vlen_float32 = h5py.vlen_dtype(np.float32)
            f.create_dataset(
                "loss_history", shape=(end_idx - args.start_idx,), dtype=vlen_float32
            )

            for i in tqdm(range(start_idx, end_idx), desc="Optimizing potentials"):
                potential = potentials[i]
                z_opt, energy_opt, loss_history = latentopt.minimize(
                    potential, **optimization_kwargs
                )
                opt_steps = len(loss_history)
                idx = i - args.start_idx
                f["z_opt"][idx] = z_opt
                f["energy_opt"][idx] = energy_opt
                f["opt_steps"][idx] = opt_steps
                f["loss_history"][idx] = np.asarray(loss_history, dtype=np.float32)
                if ((idx + 1) % save_interval == 0) or (i == end_idx - 1):
                    f.flush()
                    print(f"Progress saved up to potential index {i}.")
        print(f"Optimization results saved to {h5_path}")


if __name__ == "__main__":
    main()
