"""
Generate instances of the Fermi-Hubbard chain model with random potentials.
For each instance, the ground state density and energy (DFT input-output, DFTIO) are computed using
exact diagonalization.
The potentials are saved in the directory `./data/L{L}-N{N}-U{U}/potentials` and the corresponding
exact DFTIO results are saved in `./data/L{L}-N{N}-U{U}/exact`.
"""

import numpy as np
import argparse
import os
import shutil

from tqdm import tqdm

from dftqml.fhchain import FermiHubbardChain


def harmonic_potential(L, strength):
    return strength * (np.linspace(-1, 1, L)**2 - 1/2)


POTENTIALS = "harmonic_potentials"
EXACT = "harmonic_exact"

# *** parse input ***

parser = argparse.ArgumentParser()
parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument("ninst", help="number of instances to generate", type=int)
parser.add_argument("strength_min", help="min strength of the harmonic potential", type=float)
parser.add_argument("strength_max", help="max strength of the harmonic potential", type=float)
parser.add_argument(
    "--clear",
    help="delete and overwrite all previous data" "for given N, L, U",
    action="store_true",
)
parser.add_argument(
    "--append", help="append ninst new data points" "after last existing index", action="store_true"
)
args = parser.parse_args()

if args.clear and args.append:
    raise Exception("--clear and --append flags cannot be used together")

# *** Manage data directories ***

dirname = f"./data/L{args.L}-N{args.N}-U{args.U}"
potentials_dir = os.path.join(dirname, POTENTIALS)
exact_dir = os.path.join(dirname, EXACT)

starting_index = 0
if os.path.exists(dirname):
    if args.clear:
        shutil.rmtree(dirname)
    elif args.append:
        existing_files = [f for f in os.listdir(potentials_dir) if f.endswith(".dat")]
        starting_index = len(existing_files)
    elif len(os.listdir(potentials_dir)) != 0:
        raise Exception("directory not empty. Call with --clear option to overwrite")

os.makedirs(potentials_dir, exist_ok=True)
os.makedirs(exact_dir, exist_ok=True)

# *** generate and save potentials and exact diagonalization results ***

strengths = np.linspace(args.strength_min, args.strength_max, args.ninst)

system = FermiHubbardChain(args.L, args.N, args.U)

for i in tqdm(range(len(strengths))):
    strength = strengths[i]
    potential = harmonic_potential(args.L, strength)
    density, dft_energy = system.ground_state_dftio(potential)

    np.savetxt(os.path.join(potentials_dir, f"{i}.dat"), np.concatenate([potential, [strength]]))
    np.savetxt(os.path.join(exact_dir, f"{i}.dat"), np.concatenate([density, [dft_energy]]))
