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


N_ATTEMPTS = 100
MU_STD_THRESHOLD = 0.4
W_RANGE = [0.005, 2.5]
POTENTIALS = "potentials"
EXACT = "exact"


# *** parse input ***

parser = argparse.ArgumentParser()
parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument("ninst", help="number of instances to generate", type=int)
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

rng = np.random.default_rng()
system = FermiHubbardChain(args.L, args.N, args.U)

for i in tqdm(range(starting_index, args.ninst + starting_index)):
    for _ in range(N_ATTEMPTS):
        W = rng.uniform(*W_RANGE)
        potential = rng.uniform(-W, +W, args.L)

        # reject datapoint if potential has high variance
        if np.std(potential) > 0.4:
            continue
        else:
            density, dft_energy = system.ground_state_dftio(potential)
            break
    else:
        raise (RuntimeError(f"after {N_ATTEMPTS} tries no valid potential was extracted"))

    np.savetxt(os.path.join(potentials_dir, f"{i}.dat"), potential)
    np.savetxt(os.path.join(exact_dir, f"{i}.dat"), np.concatenate([density, [dft_energy]]))
