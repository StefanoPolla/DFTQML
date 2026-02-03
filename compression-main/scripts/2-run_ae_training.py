"""
Train a deterministic Autoencoder (AE) on Hamiltonian terms and save the trained models.

This script is designed to run directly and trains the AE for various latent dimensions.
"""

from dftqml.autoencoders.ae import train_ae
from dftqml.autoencoders import autoencoder_utils
import torch
import numpy as np
import os
import argparse
import json

def main():
    # *** Parse input arguments ***
    parser = argparse.ArgumentParser(description="Train a deterministic autoencoder.")
    parser.add_argument("L", type=int, help="Number of sites")
    parser.add_argument("N", type=int, help="Number of electrons")
    parser.add_argument("U", type=float, help="Coulomb repulsion")
    parser.add_argument(
        "--n_data",
        type=int,
        default=90000,
        help="Number of training data points to use. Note: some datapoints should be left for testing.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["ham_terms", "two_rdms"],
        default="two_rdms",
        help="Data to be compressed",
    )

    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration number for the training run. If not set, it will default to None.",
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
        help="Use a linear encoder (single dense layer) instead of a neural network encoder.",
    )

    parser.add_argument(
        "--epochs", type=int, default=300, help="Max number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="L2 weight regularization strength (optimizer weight decay). Default 0.0 (disabled)",
    )
    parser.add_argument(
        "--patience", type=int, default=30, help="Patience for early stopping"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation split fraction"
    )

    parser.add_argument(
        "--lambda_latent_norm",
        type=float,
        default=1e-7,
        help="Weight for latent space norm regularization-",
    )
    parser.add_argument(
        "--lambda_latent_repulsion",
        type=float,
        default=1e-7,
        help="Weight for latent space repulsion regularization.",
    )
    parser.add_argument(
        "--lambda_lipschitz_encoder",
        type=float,
        default=1e-9,
        help="Weight for encoder Lipschitz regularization.",
    )
    parser.add_argument(
        "--lambda_lipschitz_decoder",
        type=float,
        default=1e-8,
        help="Weight for decoder Lipschitz regularization.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the input data files (single-source mode)",
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        help="Multiple data directories (multi-source mode)",
    )
    parser.add_argument(
        "--n_data_each",
        type=int,
        nargs="+",
        help="Number of samples to take from each directory (multi-source mode)",
    )
    parser.add_argument(
        "--augment_by_symmetry",
        action="store_true",
        help="Apply translational + mirror symmetry augmentation to training data (only supported for input_type='ham_terms').",
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )

    parser.add_argument("--verbose", type=int, default=5, help="Verbosity level")
    args = parser.parse_args()

    # ********* Process input arguments **********

    # Validate augmentation compatibility
    if args.augment_by_symmetry and args.input_type != "ham_terms":
        raise NotImplementedError(
            "--augment_by_symmetry is only implemented for input_type='ham_terms'."
        )

    # Theoretical minimum latent dimension
    args.latent_dim = args.latent_dim or (args.L - 1)

    # Programmatic hyperparameter choices derived from optuna search
    args.min_dim = args.min_dim or args.L**2
    args.scaling_factor = args.scaling_factor or (
        20.0 if args.input_type == "ham_terms" else 1.0
    )
    args.learning_rate = args.learning_rate or 2e-3
    args.batch_size = args.batch_size or 64

    # Add one print statement to show the parsed arguments
    print(f"Parsed and processed arguments:\n{args}")

    # ********* Load and prepare data ************

    # -------- Multi-source / single-source data collection --------
    multi_mode = args.data_dirs is not None or args.n_data_each is not None
    if multi_mode:
        if (args.data_dirs is None) or (args.n_data_each is None):
            raise ValueError(
                "Both --data_dirs and --n_data_each must be provided for multi-source mode."
            )
        if len(args.data_dirs) != len(args.n_data_each):
            raise ValueError(
                f"--data_dirs length ({len(args.data_dirs)}) != --n_data_each length ({len(args.n_data_each)})."
            )
        if any(n <= 0 for n in args.n_data_each):
            raise ValueError("All values in --n_data_each must be positive integers.")
        sources = []
        collected = []
        for d, n_take in zip(args.data_dirs, args.n_data_each):
            use_reduced = False
            if args.input_type == "ham_terms":
                # Choose wether to use reduced or full data file based on file existence
                fname = f"L{args.L}-N{args.N}-U{args.U}_reduced.hdf5"
                if os.path.exists(os.path.join(d, fname)):
                    print(f"Loading data from reduced file {fname} in directory {d}.")
                    use_reduced = True
                else:
                    print(
                        f"Reduced file {fname} not found in directory {d}, falling back to full data file."
                    )
                    fname = f"L{args.L}-N{args.N}-U{args.U}.hdf5"

            fpath = os.path.join(d, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Input file {fpath} does not exist.")
            arr = autoencoder_utils.load_data(
                fpath, n_take, args.input_type, args.N, use_reduced=use_reduced
            )
            collected.append(arr)
            sources.append({"dir": d, "n_data": n_take, "file": fpath})
        training_data = np.concatenate(collected, axis=0)
        total = sum(args.n_data_each)
        args.n_data = total  # overwrite for downstream (checkpoint naming)
        args.num_sources = len(sources)
        data_sources_meta = sources
    else:
        # Single-source behavior
        fname = f"L{args.L}-N{args.N}-U{args.U}.hdf5"
        use_reduced = False
        if args.input_type == "ham_terms":
            # Choose wether to use reduced or full data file based on file existence
            if os.path.exists(os.path.join(args.data_dir, fname)):
                print(
                    f"Loading data from reduced file {fname} in directory {args.data_dir}."
                )
                use_reduced = True
                fname = f"L{args.L}-N{args.N}-U{args.U}_reduced.hdf5"
            else:
                print(
                    f"Reduced file {fname} not found in directory {args.data_dir}, falling back to full data file."
                )
                fname = f"L{args.L}-N{args.N}-U{args.U}.hdf5"
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Input file {fpath} does not exist.")
        training_data = autoencoder_utils.load_data(
            fpath, args.n_data, args.input_type, args.N, use_reduced=use_reduced
        )
        args.num_sources = 1
        data_sources_meta = [
            {"dir": args.data_dir, "n_data": args.n_data, "file": fpath}
        ]

    # ********* Train ************

    # Set up device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use a fixed seed for reproducibility
    torch.manual_seed(args.iteration or 0)

    # Prepare checkpoint directory
    checkpoint_dir = autoencoder_utils.prepare_checkpoint_dir(args)
    # Write data_sources provenance JSON (retro-compatible addition)
    provenance_path = os.path.join(checkpoint_dir, "data_sources.json")
    try:
        with open(provenance_path, "w") as fh:
            json.dump({"sources": data_sources_meta}, fh, indent=2)
    except Exception as e:
        print(f"Warning: failed to write data_sources.json: {e}")

    # If linear encoder is requested, lipschitz regularization for encoder is not supported
    if getattr(args, "linear_encoder", False) and args.lambda_lipschitz_encoder != 0.0:
        raise ValueError(
            "Lipschitz regularization for encoder is not supported with linear encoder."
        )

    print("Training autoencoder...")
    model, training_info = train_ae(
        data=training_data,
        hidden_depth=args.hidden_depth,
        latent_dim=args.latent_dim,
        min_dim=args.min_dim,
        scaling_factor=args.scaling_factor,
        interpolation=args.interpolation,
        enable_batchnorm=args.enable_batchnorm,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        early_stopping_patience=args.patience,
        lambda_latent_norm=args.lambda_latent_norm,
        lambda_latent_repulsion=args.lambda_latent_repulsion,
        lambda_lipschitz_encoder=args.lambda_lipschitz_encoder,
        lambda_lipschitz_decoder=args.lambda_lipschitz_decoder,
        verbose=args.verbose,
        device=device,
        checkpoint_dir=checkpoint_dir,
        linear_encoder=args.linear_encoder,
        augment_by_symmetry=args.augment_by_symmetry,
    )

    print("Trained and saved succesfully.")


if __name__ == "__main__":
    main()
