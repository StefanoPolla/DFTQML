import numpy as np
from dftqml import utils
import os.path
import argparse


# *** parse input ***

parser = argparse.ArgumentParser()

parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument("source", help="which file to unpack, e.g. 'potentials' or 'exact'", type=str)
parser.add_argument("nidcs", help="number of potentials to unpack", type=int)

args = parser.parse_args()


# manage files

sys_str = f'L{args.L}-N{args.N}-U{args.U}'

in_file = os.path.join("data-h5", sys_str, args.source + '.h5')
out_dir = os.path.join("data", sys_str, args.source)

os.makedirs(out_dir, exist_ok=True)

# read H5 file

if args.source == "potentials":
    data = utils.load_potentials(in_file, args.nidcs)
else:
    densities, energies = utils.load_dft_data_h5(in_file, args.nidcs)
    data = np.concatenate((densities, energies[:, None]), axis=1)

# save to txt files

for idx, data in enumerate(data):
    out_file = os.path.join(out_dir, f"{idx}.dat")
    if os.path.exists(out_file):
        raise ValueError('file exists already at ' + out_file)
    np.savetxt(out_file, data)
