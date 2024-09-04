"""
Consolidate all the .dat files within the directory tree of `./data/L{L}-N{N}-U{U}` into .h5 files.

Each directory containing a list of .dat files is consolidated into a single .h5 file, the tree
structure is preserved, under a new root directory `./data-h5/L{L}-N{N}-U{U}`.
"""

import argparse
import os

import h5py
import numpy as np

# *** parse input ***
parser = argparse.ArgumentParser()
parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument(
    "--clear", help="delete and overwrite all hd5 files" "for given N, L, U", action="store_true"
)
args = parser.parse_args()

# *** Manage data directories ***
in_dirname = f"./data/L{args.L}-N{args.N}-U{args.U}"
out_dirname = f"./data-h5/L{args.L}-N{args.N}-U{args.U}"

if not os.path.exists(out_dirname):
    print(f"creating output directory {out_dirname}")
    os.makedirs(out_dirname)

# traverse input directory tree

for root, dirs, files in os.walk(in_dirname):
    if len(files) == 0:
        continue
    if not files[0].endswith(".dat"):
        continue

    rel_path = os.path.relpath(root, in_dirname)
    files = sorted(files, key=lambda x: int(x.replace(".dat", "")))
    out_file = os.path.join(out_dirname, rel_path + ".h5")

    if os.path.exists(out_file) and not args.clear:
        print("output file", out_file, "already exists. Skipping...")
        continue

    print("processing directory <IN_DIR>/" + rel_path)
    print("containing:", files[0], "...", files[-1])
    print("into output file", out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # load from txt
    data = [np.loadtxt(os.path.join(root, f)) for f in files]
    indices = [int(f.replace(".dat", "")) for f in files]

    # save to hdf5
    with h5py.File(out_file, "w") as hf:
        if os.path.split(root)[-1] == "potentials":
            hf.create_dataset("potentials", data=data)
            hf.create_dataset("indices", data=indices)
        else:
            data = np.array(data)
            hf.create_dataset("dft_energies", data=data[:, -1])
            hf.create_dataset("densities", data=data[:, :-1])
            hf.create_dataset("indices", data=indices)
