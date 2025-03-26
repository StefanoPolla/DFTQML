"""
Generate instances of the Fermi-Hubbard chain model with random potentials.
For each instance, the ground state density and energy (DFT input-output, DFTIO) are computed using
exact diagonalization.
The potentials are saved in the directory `./data/L{L}-N{N}-U{U}/potentials` and the corresponding
exact DFTIO results are saved in `./data/L{L}-N{N}-U{U}/dftio/exact`.

TODO : ADD DETAILS ABOUT RDMFTIO
TODO : find a better way to save the data
"""

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

from dftqml.fhchain import FermiHubbardChain

N_ATTEMPTS = 100
MU_STD_THRESHOLD = 0.4
W_RANGE = [0.005, 2.5]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

# *** parse input ***

parser = argparse.ArgumentParser()
parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument("ninst", help="number of instances to generate", type=int)
parser.add_argument(
    "--append", help="append ninst new data points" "after last existing index", action="store_true"
)
args = parser.parse_args()


# *** Manage data directories ***

os.makedirs(DATA_DIR, exist_ok=True)

filename = os.path.join(DATA_DIR, f"L{args.L}-N{args.N}-U{args.U}.hdf5")

starting_index = 0
if os.path.exists(filename):
    if args.append:
        with h5py.File(filename, "r") as f:
            starting_index = len(f["potentials"])
    else:
        raise Exception(
            f"the file {filename} already exists. Use --append to add more data points,"
            "or delete the file before running this script"
        )
else:
    with h5py.File(filename, "w") as f:
        f.create_dataset("potentials", (0, args.L), maxshape=(None, args.L))
        f.create_dataset("ground_energies", (0,), maxshape=(None,))
        f.create_dataset("dft_energies", (0,), maxshape=(None,))
        f.create_dataset("densities", (0, args.L), maxshape=(None, args.L))
        f.create_dataset("rdmft_energies", (0,), maxshape=(None,))
        f.create_dataset(
            "one_rdms", (0, args.L, args.L // 2 + 1), maxshape=(None, args.L, args.L // 2 + 1)
        )


# *** generate and save potentials and exact diagonalization results ***


def random_potential(rng):
    for _ in range(N_ATTEMPTS):
        W = rng.uniform(*W_RANGE)
        potential = rng.uniform(-W, +W, args.L)

        # reject datapoint if potential has high variance
        if np.std(potential) > 0.4:
            continue

        return potential - np.mean(potential)  # shift potential to have zero mean
    else:
        raise (RuntimeError(f"after {N_ATTEMPTS} tries no valid potential was extracted"))


rng = np.random.default_rng()
system = FermiHubbardChain(args.L, args.N, args.U)

for i in tqdm(range(starting_index, args.ninst + starting_index)):
    potential = random_potential(rng)
    ground_energy, ground_state = system.ground_energy_and_state(potential)

    (
        density,
        dft_energy,
    ) = system.dftio(ground_state)
    one_rdm, rdmft_energy = system.rdmftio(ground_state)

    # TODO implement 2-RDM
    # two_rdm = system.two_rdm(ground_state)

    with h5py.File(filename, "a") as f:
        f["potentials"].resize((f["potentials"].shape[0] + 1), axis=0)
        f["potentials"][-1] = potential

        f["ground_energies"].resize((f["ground_energies"].shape[0] + 1), axis=0)
        f["ground_energies"][-1] = ground_energy

        f["dft_energies"].resize((f["dft_energies"].shape[0] + 1), axis=0)
        f["dft_energies"][-1] = dft_energy

        f["densities"].resize((f["densities"].shape[0] + 1), axis=0)
        f["densities"][-1] = density

        f["rdmft_energies"].resize((f["rdmft_energies"].shape[0] + 1), axis=0)
        f["rdmft_energies"][-1] = rdmft_energy

        f["one_rdms"].resize((f["one_rdms"].shape[0] + 1), axis=0)
        f["one_rdms"][-1] = one_rdm
