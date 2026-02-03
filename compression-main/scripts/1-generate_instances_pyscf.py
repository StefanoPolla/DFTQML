"""
Generate instances of the Fermi-Hubbard chain model with random potentials using PySCF FCI.

Potentials are generated uniformly in [-W, +W] (with W in [0.005, 2.5]) and shifted to zero mean.
Each instance is solved using PySCF's FCI. Saved quantities:
- 'potentials'       : on-site potentials (zero-mean)
- 'ground_energies'  : total energies <GS|T + U + V|GS>
- 'dft_energies'     : <GS|T + U|GS> = ground_energies - v · n
- 'densities'        : site densities n_i = <GS|n_i|GS>
- 'two_rdms' (opt)   : spin-summed spatial 2-RDM Γ_{ijkl} = <GS| a_i† a_j† a_l a_k |GS>
- 'ground_states' (opt): PySCF CI vector (complex128), saved if --save-state-vector is passed

Notes:
- The saved 'ground_states' entries are the PySCF CI vectors (not the OpenFermion/block-basis vectors).
"""

import argparse
import os
import time
import cProfile
import pstats
from functools import partial

import h5py
import numpy as np
from tqdm import tqdm

from dftqml.hubbard_model.pyscf_extension import PySCFFCIHubbardChain
from dftqml.hubbard_model.random_potentials import random_potential_nelson, random_potential_by_volume

N_ATTEMPTS = 100
MU_STD_THRESHOLD = 0.4
W_RANGE = [0.005, 2.5]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

# --- argparse ---
parser = argparse.ArgumentParser()
parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument("ninst", help="number of instances to generate", type=int)
parser.add_argument(
    "--total-spin-penalty", type=float, default=0.3, help="total spin penalty term"
)
parser.add_argument(
    "--sampler",
    help="sampling method to use for choosing the random potential",
    type=str,
    choices=["nelson", "by_volume"],
    default="nelson",
)
parser.add_argument(
    "--save-state-vector",
    help="save the PySCF CI vector in dataset 'ground_states' (complex128)",
    action="store_true",
)
parser.add_argument(
    "--save-two-rdm",
    help="save the two-body reduced density matrix (two-RDM) in the hdf5 file",
    action="store_true",
)
parser.add_argument(
    "--append",
    help="append ninst new data points after last existing index",
    action="store_true",
)
parser.add_argument(
    "--suffix",
    help="append a custom suffix to the filename (e.g., '--suffix test' creates L4-N4-U4.0_test.hdf5)",
    type=str,
    default="",
)
parser.add_argument("--seed", help="RNG seed", type=int, default=None)
parser.add_argument(
    "--data-dir",
    help="Directory to store output HDF5 and profiling results (defaults to DATA_DIR)",
    type=str,
    default=None,
)
parser.add_argument(
    "--float32",
    help="Store real-valued datasets as float32 instead of default float64 (reduces size).",
    action="store_true",
)
args = parser.parse_args()

# --- HDF5 file setup ---
data_dir = os.path.abspath(args.data_dir) if args.data_dir else DATA_DIR
os.makedirs(data_dir, exist_ok=True)
base_filename = f"L{args.L}-N{args.N}-U{args.U}"
if args.suffix:
    base_filename += f"_{args.suffix}"
filename = os.path.join(data_dir, f"{base_filename}.hdf5")

starting_index = 0
if os.path.exists(filename):
    if args.append:
        with h5py.File(filename, "r") as f:
            starting_index = len(f["potentials"])
    else:
        raise Exception(
            f"the file {filename} already exists. Use --append to add more data points, "
            "or delete the file before running this script"
        )
else:
    with h5py.File(filename, "w") as f:
        f.attrs["generator"] = "PySCFFCIHubbardChain"
        f.attrs["L"] = args.L
        f.attrs["N"] = args.N
        f.attrs["U"] = args.U
        # real-valued datasets dtype: default float64, optionally float32 via flag
        real_dtype = "float32" if args.float32 else "float64"
        f.create_dataset(
            "potentials", (0, args.L), maxshape=(None, args.L), dtype=real_dtype
        )
        f.create_dataset("ground_energies", (0,), maxshape=(None,), dtype=real_dtype)
        f.create_dataset("dft_energies", (0,), maxshape=(None,), dtype=real_dtype)
        f.create_dataset(
            "densities", (0, args.L), maxshape=(None, args.L), dtype=real_dtype
        )
        f.create_dataset("s2", (0,), maxshape=(None,), dtype=real_dtype)
        if args.save_two_rdm:
            # Keep only the improved chunking for two_rdms; no explicit compression for compatibility with original.
            m = 8 if args.L > 8 else args.L
            f.create_dataset(
                "two_rdms",
                (0, args.L, args.L, args.L, args.L),
                maxshape=(None, args.L, args.L, args.L, args.L),
                dtype=real_dtype,
                chunks=(1, m, m, m, m),
            )
        # 'ground_states' (CI vectors) will be created lazily on first sample when we know the CI dimension.

# --- utility ---

rng = np.random.default_rng(seed=args.seed)
if args.sampler == "nelson":
    random_potential = partial(random_potential_nelson, args.L, rng)
else:
    random_potential = partial(random_potential_by_volume, args.L, rng)


# --- system (PySCF-backed) ---
system = PySCFFCIHubbardChain(
    n_sites=args.L,
    n_particles=args.N,
    u=args.U,
    t=1.0,
    spin_convention="up_then_down",
    boundary_conditions="periodic",
    total_spin_penalty=args.total_spin_penalty,
)

# --- profiling ---
profiler = cProfile.Profile()
profiler.enable()
start_time = time.time()

ci_dim = None  # will be determined on first iteration if --save-state-vector

for i in tqdm(range(starting_index, args.ninst + starting_index)):
    potential = random_potential()
    try:
        # Ground state with PySCF FCI
        ground_energy, state = system.ground_energy_and_state(potential)

        # One- and two-RDMs as needed
        # spin-summed spatial 1-RDM, shape (L, L)
        rdm1 = system.one_rdm_expectation(state)
        density = np.asarray(rdm1[:, 0], dtype=float)
        dft_energy = float(ground_energy - np.dot(density, potential))
    except Exception as e:
        print(f"Failed to solve instance {i}: {e}")
        print(f"Error at instance {i}: {e}")
        continue

    if args.save_two_rdm:
        two_rdm = system.two_rdm_expectation(state)  # shape (L, L, L, L)

    with h5py.File(filename, "a") as f:
        # Create 'ground_states' dataset lazily once we know CI dimension
        if args.save_state_vector and "ground_states" not in f:
            ci = state.get("ci")
            if ci is None:
                raise RuntimeError(
                    "State dict does not contain 'ci' vector from PySCF."
                )
            ci = np.asarray(ci)
            ci_dim = ci.size
            dset = f.create_dataset(
                "ground_states",
                (0, ci_dim),
                maxshape=(None, ci_dim),
                dtype=np.complex128,
                # Revert to original: let h5py pick default chunking; no explicit compression
            )
            print(f"Created dataset 'ground_states' with CI dimension {ci_dim}.")
            # store some CI metadata for clarity
            f["ground_states"].attrs["ci_dim"] = int(ci_dim)
            # original PySCF CI tensor shape (n_alpha_strings, n_beta_strings)
            f["ground_states"].attrs["ci_shape"] = tuple(int(x) for x in ci.shape)
            f["ground_states"].attrs["nmo"] = int(state.get("nmo", args.L))
            ne = state.get("nelec", None)
            if isinstance(ne, (tuple, list)) and len(ne) == 2:
                f["ground_states"].attrs["nelec_up"] = int(ne[0])
                f["ground_states"].attrs["nelec_down"] = int(ne[1])

        # Write row
        f["potentials"].resize((f["potentials"].shape[0] + 1), axis=0)
        f["potentials"][-1] = potential

        f["ground_energies"].resize((f["ground_energies"].shape[0] + 1), axis=0)
        f["ground_energies"][-1] = ground_energy

        f["dft_energies"].resize((f["dft_energies"].shape[0] + 1), axis=0)
        f["dft_energies"][-1] = dft_energy

        f["densities"].resize((f["densities"].shape[0] + 1), axis=0)
        f["densities"][-1] = density

        # Store s2 value
        s2_value = state.get("s2", np.nan)
        f["s2"].resize((f["s2"].shape[0] + 1), axis=0)
        f["s2"][-1] = s2_value

        if args.save_state_vector:
            # PySCF returns CI as a 2D tensor (n_alpha_strings, n_beta_strings).
            # Flatten to 1D so it matches the dataset shape (ci_dim,).
            ci = np.asarray(state["ci"], dtype=np.complex128).ravel()
            if ci_dim is not None and ci.size != ci_dim:
                raise RuntimeError(
                    f"CI dimension changed within run: expected {ci_dim}, got {ci.size}"
                )
            f["ground_states"].resize((f["ground_states"].shape[0] + 1), axis=0)
            f["ground_states"][-1] = ci

        if args.save_two_rdm:
            f["two_rdms"].resize((f["two_rdms"].shape[0] + 1), axis=0)
            f["two_rdms"][-1] = two_rdm

end_time = time.time()
profiler.disable()

# Save profiling results to a file
profile_dir = os.path.join(data_dir, "profile_results")
os.makedirs(profile_dir, exist_ok=True)
profile_filename = f"profile_results_{args.L}_{args.N}"
if args.suffix:
    profile_filename += f"_{args.suffix}"
profile_path = os.path.join(profile_dir, f"{profile_filename}.txt")
with open(profile_path, "w", encoding="utf-8") as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats("cumulative")
    stats.print_stats()
print(f"Saved profiling stats to {profile_path}")

print(f"Execution time: {end_time - start_time:.2f} seconds")
