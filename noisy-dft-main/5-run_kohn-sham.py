"""!! In development !!
Run Kohn-Sham-like optimization of density for a set of external potentials using a trained CNN
model for the functional.

The external potentials are loaded from `./data-hd5/L{L}-N{N}-U{U}/potentials`.
The model is loaded from `./models/base/L{L}-N{N}-U{U}/{source}/ndata{ndata}/split{split}`.

The optimizer is initialized with either exact density ('cheat'), 'random' density, or 'uniform'
density depending on the value of the `init` argument

The output density and corresponding energy (DFTIO) are saved as h5 files in
`./kohn-sham/{init}/L{L}-N{N}-U{U}/{source}/ndata{ndata}/split{split}.h5`.
"""

import argparse
import logging
import os
import h5py

import numpy as np
from dftqml import tfmodel, utils, ksopt
from tensorflow import autograph

DATA_DIR = "./data-h5"
MODEL_DIR = "./models/base"
KOHN_SHAM_DIR = "./kohn-sham"
N_TEST_POTENTIALS = 1000
FIRST_TEST_POTENTIAL = 1000

# *** suppress autograph warnings ***

autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# *** parse input ***

parser = argparse.ArgumentParser()

parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument(
    "source",
    help="source data subdirectory for model picking" "(e.g. `vqe/depth1` or `exact`)",
    type=str,
)
parser.add_argument("ndata", help="size of the dataset (train + val)", type=int)
parser.add_argument("split", help="index of the fold (from 0 to 4)", type=int)

parser.add_argument(
    "init",
    help="choose which initialization to use for the density. "
    "cheat picks the exact density; "
    "random picks a random density; "
    "uniform picks a uniform density.",
    type=str,
    choices=["cheat", "random", "uniform"],
)

parser.add_argument("--overwrite", help="overwrite existing ouput", action="store_true")

args = parser.parse_args()


# *** Manage data directories and load input ***

model_path = os.path.join(
    MODEL_DIR,
    f"L{args.L}-N{args.N}-U{args.U}",
    args.source,
    f"ndata{args.ndata}",
    f"split{args.split}",
)
potentials_file = os.path.join(DATA_DIR, f"L{args.L}-N{args.N}-U{args.U}", "potentials.h5")
cheat_init_file = os.path.join(DATA_DIR, f"L{args.L}-N{args.N}-U{args.U}", "exact.h5")
output_file = os.path.join(
    KOHN_SHAM_DIR,
    args.init,
    f"L{args.L}-N{args.N}-U{args.U}",
    args.source,
    f"ndata{args.ndata}",
    f"split{args.split}.h5",
)

if not os.path.exists(model_path):
    raise FileNotFoundError("the input model does not exist at path " + model_path)
if not os.path.exists(potentials_file):
    raise FileNotFoundError("the input potentials file does not exist at path " + potentials_file)
if not os.path.exists(cheat_init_file) and args.init == "cheat":
    raise FileNotFoundError(
        "the exact DFTIO file needed for cheat initializaion does not exist "
        "at path " + cheat_init_file
    )

if os.path.exists(output_file):
    if args.overwrite:
        os.remove(output_file)
    else:
        raise FileExistsError(
            "the output file already exists at path "
            + output_file
            + "You can call the script with the --overwrite option."
        )

os.makedirs(os.path.dirname(output_file), exist_ok=True)


# *** Load model, potentials and eventual exact densities ***

model = tfmodel.load_model(model_path)
potentials = utils.load_potentials(potentials_file, N_TEST_POTENTIALS, FIRST_TEST_POTENTIAL)

if args.init == "cheat":
    exact_densities, _ = utils.load_dft_data(
        cheat_init_file, N_TEST_POTENTIALS, FIRST_TEST_POTENTIAL
    )


# *** Run Kohn-Sham optimization ***

indices = np.arange(N_TEST_POTENTIALS) + FIRST_TEST_POTENTIAL
densities = []
dft_energies = []

for j in range(len(potentials)):
    if args.init == "cheat":
        x0 = exact_densities[j]
    elif args.init == "random":
        x0 = np.random.rand(args.L)
        x0 *= args.N / np.sum(x0)
    elif args.init == "uniform":
        x0 = np.ones(args.L) * args.N / args.L

    density, energy = ksopt.optimize_dftio(model, potentials[j], x0)
    densities.append(density)
    dft_energies.append(energy)


# *** Save output ***

with h5py.File(output_file, "w") as hf:
    hf.create_dataset("dft_energies", data=dft_energies)
    hf.create_dataset("densities", data=densities)
    hf.create_dataset("indices", data=indices)
