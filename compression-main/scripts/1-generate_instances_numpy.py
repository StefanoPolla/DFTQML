"""
Generate instances of the Fermi-Hubbard chain model with random potentials.

The potentials are generated at random, with a uniform distribution in the range [-W, +W],
where W is a random number between 0.005 and 2.5. The potentials are then shifted to have zero mean.

Each problem instance is solved by exact diagonalization. From the ground state |GS>, DFT and RDMFT
input-output quantities are extracted and saved.

The data is saved in a .hdf5 file in the `data` directory, with the filename format
`L{L}-N{N}-U{U}.hdf5`. The file contains the following datasets:
(T - kinetic energy, U - Hubbard interaction energy, V - random local potential energy)
- `potentials`: the random potentials
- `ground_energies`: the ground state energies <GS|T + V + U|GS>
- `dft_energies`: the Hohenberg-Kohn DFT energies <GS|T + U|GS>
- `densities`: the density vector <GS|c_i^dagger c_i|GS>
- `two_rdms`: the two-body reduced density matrices <GS|c_i^dagger c_j^dagger c_k c_l|GS>
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

from dftqml.hubbard_model.fhchain import FermiHubbardChain

from dftqml.hubbard_model.random_potentials import random_potential_nelson, random_potential_by_volume

# *** parse input ***

parser = argparse.ArgumentParser()
parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument("ninst", help="number of instances to generate", type=int)
parser.add_argument(
    "--sampler",
    help="sampling method to use for choosing the random potential",
    type=str,
    choices=["nelson", "by_volume"],
    default="nelson",
)
parser.add_argument(
    "--save_state_vector",
    help="also save the ground state wavefunction in the hdf5 file",
    action="store_true",
)
parser.add_argument(
    "--save_two_rdm",
    help="save the two-body reduced density matrix (two-RDM) in the hdf5 file",
    action="store_true",
)
parser.add_argument(
    "--append",
    help="append ninst new data points after last existing index",
    action="store_true",
)
parser.add_argument(
    "--data_dir",
    help="base directory for storing generated data file",
    type=str,
    default="data",
)
parser.add_argument("--profile", help="enable cProfile", default=False)
parser.add_argument("--seed", help="RNG seed", type=int, default=None)
args = parser.parse_args()


# *** Manage data directories ***

os.makedirs(args.data_dir, exist_ok=True)

reduced_str = "_reduced" if not args.save_two_rdm else ""
filename = os.path.join(args.data_dir, f"L{args.L}-N{args.N}-U{args.U}{reduced_str}.hdf5")

starting_index = 0
if os.path.exists(filename):
    if args.append:
        with h5py.File(filename) as f:
            if args.save_two_rdm and "two_rdms" not in f:
                raise ValueError(
                    "Appending to existing file incompatible with --save_two_rdms"
                )
            if args.save_state_vector and "ground_states" not in f:
                raise ValueError(
                    "Appending to existing file incompatible with --save_state_vector"
                )

        with h5py.File(filename, "r") as f:
            starting_index = len(f["potentials"])
    else:
        raise FileExistsError(
            f"the file {filename} already exists. Use --append to add more data points,"
            "or delete the file before running this script"
        )
else:
    with h5py.File(filename, "w") as f:
        f.create_dataset("potentials", (0, args.L), maxshape=(None, args.L))
        f.create_dataset("ground_energies", (0,), maxshape=(None,))
        f.create_dataset("dft_energies", (0,), maxshape=(None,))
        f.create_dataset("densities", (0, args.L), maxshape=(None, args.L))
        if args.save_state_vector:
            block_dim = FermiHubbardChain(args.L, args.N, args.U).block_dimension
            f.create_dataset(
                "ground_states", (0, block_dim), maxshape=(None, block_dim)
            )
        if args.save_two_rdm:
            f.create_dataset(
                "two_rdms",
                (0, args.L, args.L, args.L, args.L),
                maxshape=(None, args.L, args.L, args.L, args.L),
            )
        else:
            f.create_dataset("hamiltonian_terms", (0, 3, args.L), maxshape=(None, 3, args.L))


# *** generate and save potentials and exact diagonalization results ***

rng = np.random.default_rng(seed=args.seed)
system = FermiHubbardChain(args.L, args.N, args.U)

if args.sampler == "nelson":
    random_potential = partial(random_potential_nelson, args.L, rng)
else:
    random_potential = partial(random_potential_by_volume, args.L, rng)

if args.profile:
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

start_time = time.time()

for _ in tqdm(range(starting_index, args.ninst + starting_index)):
    potential = random_potential()
    ground_energy, ground_state = system.ground_energy_and_state(potential)

    # Project the ground state back to the full Hilbert space
    ground_state_full = np.zeros((2 ** (2 * args.L)), dtype=ground_state.dtype)
    ground_state_full[system.block_indices] = ground_state

    (
        density,
        dft_energy,
    ) = system.dftio(ground_state)

    if args.save_two_rdm:
        two_rdm = system.two_rdm_expectation(ground_state_full)
    else:
        hamiltonian_terms = system.hamiltonian_terms_expval(ground_state_full)

    with h5py.File(filename, "a") as f:
        f["potentials"].resize((f["potentials"].shape[0] + 1), axis=0)
        f["potentials"][-1] = potential

        f["ground_energies"].resize((f["ground_energies"].shape[0] + 1), axis=0)
        f["ground_energies"][-1] = ground_energy

        f["dft_energies"].resize((f["dft_energies"].shape[0] + 1), axis=0)
        f["dft_energies"][-1] = dft_energy

        f["densities"].resize((f["densities"].shape[0] + 1), axis=0)
        f["densities"][-1] = density

        if args.save_state_vector:
            # hdf5 does not natively save complex vectors. Luckily we can always pick a real gauge.
            # choose the U(1) gauge phase to make the state real, based on the overlap with the
            # alternating-spin state (see `FermiHubbardChain.fix_gauge` for details)
            ground_state = system.fix_gauge(ground_state, check_real=True)

            f["ground_states"].resize((f["ground_states"].shape[0] + 1), axis=0)
            f["ground_states"][-1] = ground_state

        if args.save_two_rdm:
            f["two_rdms"].resize((f["two_rdms"].shape[0] + 1), axis=0)
            f["two_rdms"][-1] = two_rdm
        else:
            f["hamiltonian_terms"].resize((f["hamiltonian_terms"].shape[0] + 1), axis=0)
            f["hamiltonian_terms"][-1] = hamiltonian_terms.T


end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")


if args.profile:
    # Stop profiling
    profiler.disable()

    # Save profiling results to a file
    profile_file = os.path.join(args.data_dir, f"profile_results_{args.L}_{args.N}.txt")
    with open(profile_file, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")
        stats.print_stats()
