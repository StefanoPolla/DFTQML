"""
This script performs hyperparameter optimization for an autoencoder model using Optuna.

Outputs:
--------
- Saves the study results in a SQLite database.
- Prints the best trial number, value (validation loss), and hyperparameters.
- Saves the best hyperparameters as a JSON file in the checkpoint_base directory.
"""

import argparse
import json
import re
import socket
from functools import partial
import os

import h5py
import numpy as np
import optuna

from dftqml.autoencoders.ae import train_ae
from dftqml.hubbard_model.fhchain import hamiltonian_terms_from_two_rdm


def objective(
    trial,
    *,
    data,
    latent_dim,
    device,
    checkpoint_base,
    max_epochs=300,
    data_type,
    enable_batchnorm=False,
):
    hidden_depth = trial.suggest_int("hidden_depth", 2, 6)
    min_dim = trial.suggest_int(
        "min_dim", 2 * (latent_dim + 1), 2 * (latent_dim + 1) ** 2, log=True
    )
    interpolation = trial.suggest_categorical("interpolation", ["linear", "geometric"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    if data_type == "two_rdms":
        scaling_factor = trial.suggest_float("scaling_factor", 0.0, 5.0, step=1.0)
    else:
        scaling_factor = trial.suggest_float("scaling_factor", 1.0, 50.0, log=True)

    trial_id = trial.number
    checkpoint_dir = os.path.join(checkpoint_base, f"trial_{trial_id}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train model with current trial parameters
    _, training_info = train_ae(
        data,
        hidden_depth=hidden_depth,
        latent_dim=latent_dim,
        min_dim=min_dim,
        scaling_factor=scaling_factor,
        interpolation=interpolation,
        disable_batchnorm=(not enable_batchnorm),
        learning_rate=learning_rate,
        epochs=max_epochs,
        batch_size=batch_size,
        use_statevector=False,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        verbose=10,
        lambda_latent_norm=0.0,
        lambda_latent_repulsion=0.0,
    )

    return training_info["best_val_loss"]


def load_data(data_file, input_type, n_data):
    # Load data
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Input file {data_file} does not exist.")

    with h5py.File(data_file, "r") as f:
        two_rdms = f["two_rdms"][:]

    # Select the first `n_data` two-RDMs for training.
    # Some data should be left for testing, but we do not enforce that here.
    if two_rdms.shape[0] < n_data:
        raise ValueError(
            f"Not enough data points in the input file. Found {two_rdms.shape[0]}, "
            f"but requested {n_data}."
        )
    two_rdms = two_rdms[:n_data]

    # Extract L and N from the filename
    match = re.search(r"L(\d+)-N(\d+)-U([\d\.]+)\.hdf5", data_file)
    if not match:
        raise ValueError(f"Could not extract L, N, U from filename: {data_file}")
    N = int(match.group(2))

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


def setup_sqlite_study(study_dir):
    storage_path = "sqlite:///" + os.path.join(study_dir, "ae_search.db")

    study = optuna.create_study(
        study_name="ae_search",
        storage=storage_path,
        direction="minimize",
        load_if_exists=True,
    )

    print(f"Job is up for parallel trials Using QSLite storage at {storage_path}")
    print(f"Running trial on: {socket.gethostname()}")
    return study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument(
        "--input_type", type=str, choices=["ham_terms", "two_rdms"], required=True
    )
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--n_data", type=int, default=90000)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument(
        "--study_dir", type=str, default="models/optuna_studies/default"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Maximum number of epochs for training (can be reduced for testing purposes)",
    )
    parser.add_argument("--enable_batchnorm", action="store_true")

    args = parser.parse_args()

    # Ensure checkpoint base directory exists
    checkpoint_base = args.study_dir
    os.makedirs(checkpoint_base, exist_ok=True)

    # create optuna study
    study = setup_sqlite_study(args.study_dir)

    # Load dataset
    data = load_data(args.data_file, args.input_type, n_data=args.n_data)

    # Run optimization
    study.optimize(
        partial(
            objective,
            data=data,
            latent_dim=args.latent_dim,
            device=args.device,
            checkpoint_base=checkpoint_base,
            max_epochs=args.max_epochs,
            data_type=args.input_type,
            enable_batchnorm=args.enable_batchnorm,
        ),
        n_trials=args.n_trials,
    )

    # Save best params
    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    with open(f"{args.study_dir}/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)


if __name__ == "__main__":
    main()
