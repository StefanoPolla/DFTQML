"""
Run expectation value estimation  on the exacty ground state of a Fermi-Hubbardnchain with a
potential loaded from `./data/L{L}-N{N}-U{U}/potentials/{idx}.dat'.

The expectation value estimation of energy uses the split operators approach, where the fourier 
(kinetic) term and real-space term are separately measured, each using {nshots}. The energy is
computed from both terms and the density is computed from the real-space term.

The DFT input-output (DFTIO, density and energy) is saved to the file
`./data/L{L}-N{N}-U{U}/npfabric-vqe/depth{depth}/{idx}.dat'.
"""

import argparse
import os
import numpy as np
from os import path

from dftqml.fhchain import FermiHubbardChain
from dftqml.sampling import DFTIOSampler


POTENTIALS = "potentials"
EXACT = "exact"
SAMPLING = "sampling"


# *** parse input ***

parser = argparse.ArgumentParser()

parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument("idx", help="index of data point to process", type=int)
parser.add_argument("nshots", help="number of shots in each basis", type=int)

parser.add_argument("--overwrite", help="overwrite existing ouput", action="store_true")

args = parser.parse_args()


# *** Manage data directories and load input ***

dirname = f"./data/L{args.L}-N{args.N}-U{args.U}"
input_file = path.join(dirname, POTENTIALS, f"{args.idx}.dat")
output_dir = path.join(dirname, SAMPLING, f"nshots{args.nshots}")
output_file = path.join(output_dir, f"{args.idx}.dat")

if not os.path.exists(input_file):
    raise FileNotFoundError(f"the input file {input_file} does not exist")

os.makedirs(output_dir, exist_ok=True)
if os.path.exists(output_file):
    if args.overwrite:
        os.remove(output_file)
    else:
        raise FileExistsError(
            f"{output_file} exists. " "You can call the script with the --overwrite option."
        )


potential = np.loadtxt(input_file)


# *** Set up sampler ***

system = FermiHubbardChain(args.L, args.N, args.U)
sampler = DFTIOSampler(system, potential)

# *** Compute DFT input-output and save to file ***

nshots_computational_basis = args.nshots
nshots_fourier = args.nshots
density, dft_energy = sampler.dftio(nshots_computational_basis, nshots_fourier)

np.savetxt(output_file, np.concatenate([density, [dft_energy]]))


# *** Log results to standard output ***

print("Potential:", potential)
print("number of shots:", args.nshots)
print()

print("sampling results:")
print("density (DFT input):", density)
print("homogeneous energy (DFT output):", dft_energy)

print()

# print exact results along with sampler log
exact_data = np.loadtxt(path.join(dirname, EXACT, f"{args.idx}.dat"))
print("exact diagonalisation results:")
print("density (DFT input):", exact_data[:-1])
print("homogeneous energy (DFT output):", exact_data[-1])
