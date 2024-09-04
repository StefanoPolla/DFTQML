"""
Run VQE on a Fermi-Hubbard chain with a variational hamitlonian ansatz (VHA) of specified depth 
using the potential from file `./data/L{L}-N{N}-U{U}/potentials/{idx}.dat' to define system and
ansatz.

Extract the DFT input-output (DFTIO, density and energy) and save it to file 
`./data/L{L}-N{N}-U{U}/npfabric-vqe/depth{depth}/{idx}.dat'.
"""

import argparse
import os
import numpy as np
import scipy
import openfermion as of

from dftqml.fhchain import FermiHubbardChain
from dftqml.vha import VariationalHamiltonianAnsatz


POTENTIALS = 'potentials'
EXACT = 'exact'
VQE = 'vha-vqe'
LOGS = 'vha-optres'
TIME_CONST = 0.1  # hand-picked adiabatic time unit for unit 1/sqrt(depth)
COBYLA_OPTIONS = dict(maxiter=100000, rhobeg=0.1)
SLSQP_OPTIONS = dict(maxiter=100000)


# *** parse input ***

parser = argparse.ArgumentParser()

parser.add_argument("L", help="number of sites", type=int)
parser.add_argument("N", help="number of electrons", type=int)
parser.add_argument("U", help="coulomb repulsion", type=float)
parser.add_argument("idx", help="index of data point to process", type=int)
parser.add_argument("depth", help="VHA ansatz depth", type=int)

parser.add_argument("--overwrite",
                    help="overwrite existing ouput",
                    action="store_true")

args = parser.parse_args()


# *** Manage data directories and load input ***

dirname = f'./data/L{args.L}-N{args.N}-U{args.U}'
input_file = os.path.join(dirname, POTENTIALS, f'{args.idx}.dat')
output_dir = os.path.join(dirname, VQE, f'depth{args.depth}')
output_file = os.path.join(output_dir, f'{args.idx}.dat')
log_dir = os.path.join(dirname, LOGS, f'depth{args.depth}')
log_file = os.path.join(log_dir, f'{args.idx}.txt')

if not os.path.exists(input_file):
    raise FileNotFoundError('the input does not exist')

os.makedirs(output_dir, exist_ok=True)
if os.path.exists(output_file):
    if args.overwrite:
        os.remove(output_file)
    else:
        raise FileExistsError(
            f'{output_file} exists. '
            'You can call the script with the --overwrite option.')

os.makedirs(log_dir, exist_ok=True)
if os.path.exists(log_file):
    if args.overwrite:
        os.remove(log_file)
    else:
        raise FileExistsError(
            f'{log_file} exists, although {output_file} does not exist. '
            'You can call the script with the --overwrite option.')


potential = np.loadtxt(input_file)


# *** Set up and run VQE ***

system = FermiHubbardChain(args.L, args.N, args.U)
ansatz = VariationalHamiltonianAnsatz(system, potential)

init_params = ansatz.adiabatic_init_params(
    args.depth, TIME_CONST / np.sqrt(args.depth))

optres = scipy.optimize.minimize(ansatz.energy,
                                 init_params,
                                 method='SLSQP',
                                 options=SLSQP_OPTIONS)


# *** Compute DFT input-output and save to file ***

state = ansatz.state(optres.x)
density, dft_energy = system.dftio(state)

np.savetxt(output_file, np.concatenate([density, [dft_energy]]))
np.savetxt(log_file, optres.x)
with open(log_file, 'a') as log_file_pointer:
    log_file_pointer.write('\n\n' + str(optres))

# *** Log results to standard output ***

print('Potential:', potential)
print('Ansatz depth:', args.depth)
print()

print('optres:')
print(optres)
print()

print('VQE results:')
print('density (DFT input):', density)
print('homogeneous energy (DFT output):', dft_energy)
print()

# print exact results along with VQE log
exact_data = np.loadtxt(os.path.join(dirname, EXACT, f'{args.idx}.dat'))
print('exact diagonalisation results:')
print('density (DFT input):', exact_data[:-1])
print('homogeneous energy (DFT output):', exact_data[-1])

s2_interleaved = of.hamiltonians.s_squared_operator(args.L)
s2_fop = of.transforms.reorder(s2_interleaved, of.up_then_down)
s2 = of.get_sparse_operator(s2_fop, n_qubits=2 * args.L
                            ).A

print("A check if the state is singlet:")
print(f'Should be 0: {(state @ s2 @ state).real:.6f}')
