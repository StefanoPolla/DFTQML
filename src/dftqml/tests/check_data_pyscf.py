"""
Check saved hdf5 files for different system sizes.

Can also check if that states and rdms are consistent with the ground state energy.
"""

import h5py
import argparse
import os

import numpy as np
from pyscf import fci

from dftqml.hubbard_model.pyscf_extension import PySCFFCIHubbardChain


# *** Parse input arguments ***
parser = argparse.ArgumentParser(
    description="Check saved hdf5 files for different system sizes."
)
parser.add_argument("L", type=int, nargs="?", help="Number of sites")
parser.add_argument("N", type=int, nargs="?", help="Number of electrons")
parser.add_argument("U", type=float, nargs="?", help="Coulomb repulsion")
parser.add_argument(
    "--input-file",
    type=str,
    default=None,
    help="Explicit path to the input .hdf5 file (overrides L, N, U if given)",
)
parser.add_argument(
    "--check-rdms",
    help="also check the energy of the two-rdms in the hdf5 file",
    action="store_true",
)
parser.add_argument(
    "--check-states",
    help="also check the energy of the states in the hdf5 file",
    action="store_true",
)
parser.add_argument(
    "--check-hamterms",
    help="also check the energy reconstructed from hamiltonian terms (dataset 'hamiltonian_terms' or 'ham_terms')",
    action="store_true",
)
args = parser.parse_args()


# Add one print statement to show the parsed arguments
print(f"Parsed arguments: {args}")

# *** Load data ***
if args.input_file is not None:
    input_file = args.input_file
else:
    if args.L is None or args.N is None or args.U is None:
        raise ValueError("Must provide either --input-file or all of L, N, U.")
    input_file = f"data/L{args.L}-N{args.N}-U{args.U}.hdf5"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file {input_file} does not exist.")


with h5py.File(input_file, "r") as f:
    print("Datasets found in the file:")
    for name, dataset in f.items():
        print(f"  - {name}: shape={dataset.shape}, dtype={dataset.dtype}")
    # Print file-level metadata (HDF5 attributes) in a clean way
    if len(f.attrs) > 0:
        print("\nFile attributes (metadata):")
        # Sort keys for stable output
        for key in sorted(f.attrs.keys()):
            val = f.attrs[key]
            # Convert numpy scalars/arrays to native Python for prettier printing
            if isinstance(val, np.ndarray):
                # Show small arrays inline, large ones with shape info
                if val.size <= 8:
                    printable = val.tolist()
                else:
                    printable = f"ndarray(shape={val.shape}, dtype={val.dtype})"
            elif isinstance(val, np.generic):
                # numpy scalar (e.g., np.int64, np.float64)
                printable = val.item()
            else:
                printable = val
            print(f"  - {key}: {printable}")
    print("\nSummary:")
    for name, dataset in f.items():
        if dataset.shape and dataset.shape[0] > 0:
            print(f"  {name}: {dataset.shape[0]} datapoints")

    # Auto-fill L,N,U from file attrs if needed for checks
    if (args.check_rdms or args.check_states or args.check_hamterms) and (
        args.L is None or args.N is None or args.U is None
    ):
        try:
            if args.L is None:
                args.L = int(f.attrs["L"])  # type: ignore[assignment]
            if args.N is None:
                args.N = int(f.attrs["N"])  # type: ignore[assignment]
            if args.U is None:
                args.U = float(f.attrs["U"])  # type: ignore[assignment]
            print(
                f"Inferred system parameters from file attributes: L={args.L}, N={args.N}, U={args.U}"
            )
        except KeyError:
            raise RuntimeError(
                "L,N,U not provided and could not be inferred from file attributes; please specify them."
            )

    if args.check_rdms or args.check_states or args.check_hamterms:
        ground_energies_arr = np.asarray(f["ground_energies"][:])  # type: ignore[index]
        potentials_arr = np.asarray(f["potentials"][:])  # type: ignore[index]
        densities_arr = np.asarray(f["densities"][:])  # type: ignore[index]
        ci_arr = (
            # type: ignore[index]
            np.asarray(f["ground_states"][...])
            if args.check_states and "ground_states" in f
            else None
        )
        two_rdms_arr = (
            # type: ignore[index]
            np.asarray(f["two_rdms"][...])
            if args.check_rdms and "two_rdms" in f
            else None
        )
        # Support either 'hamiltonian_terms' (preferred) or legacy 'ham_terms'
        if args.check_hamterms:
            if "hamiltonian_terms" in f:
                ham_terms_arr = np.asarray(
                    f["hamiltonian_terms"][...]
                )  # type: ignore[index]
            elif "ham_terms" in f:
                ham_terms_arr = np.asarray(f["ham_terms"][...])  # type: ignore[index]
            else:
                raise RuntimeError(
                    "Requested --check-hamterms but no 'hamiltonian_terms' or 'ham_terms' dataset found."
                )
        else:
            ham_terms_arr = None

if args.check_states or args.check_rdms or args.check_hamterms:
    # type: ignore[arg-type] (we ensured above they are set)
    system = PySCFFCIHubbardChain(args.L, args.N, args.U)  # type: ignore[arg-type]
    for i, potential in enumerate(potentials_arr):
        # Check if sum of cost_hubbard is the same as ground_energies_arr
        if args.check_states:
            if ci_arr is None:
                raise RuntimeError(
                    "Dataset 'ground_states' not found in input file, but --check-states was requested."
                )
            # Reconstruct the PySCF CI vector for direct_spin0 (singlet) as a (na, na) matrix
            ci_flat = ci_arr[i]
            norb = system.n_sites
            nelec = (args.N // 2, args.N // 2)
            na = fci.cistring.num_strings(norb, nelec[0])
            assert (
                ci_flat.size == na * na
            ), f"Unexpected CI size {ci_flat.size}; expected {na*na} for (na,na) with na={na}"
            if not np.allclose(np.imag(ci_flat), 0):
                raise ValueError(
                    f"CI vector at index {i} contains nonzero imaginary parts."
                )
            ci = ci_flat.real.reshape(na, na).astype(np.float64)

            h1 = system._build_h1(potential)
            eri = system._build_eri()
            # Compute energy directly from CI using PySCF solver API
            cisolver = fci.direct_spin0.FCI()
            check_energy = cisolver.energy(h1, eri, ci, norb, nelec)

            assert np.isclose(
                check_energy, ground_energies_arr[i]
            ), f"Energy mismatch at index {i}: {check_energy} != {ground_energies_arr[i]}"

        if args.check_rdms:
            if two_rdms_arr is None:
                raise RuntimeError(
                    "Dataset 'two_rdms' not found in input file, but --check-rdms was requested."
                )
            two_rdm = two_rdms_arr[i]
            check_energy, hop, interaction, pot = system.ground_energy_from_rdm(
                two_rdm, potential, return_breakdown=True
            )

            # print(f"Check energy:  {np.sum(check_energy)}")
            # print(f"Ground energy: {ground_energies_arr[i]}")
            assert np.isclose(
                check_energy, ground_energies_arr[i]
            ), f"Energy mismatch at index {i}: {check_energy} != {ground_energies_arr[i]}"
        if args.check_hamterms:
            if ham_terms_arr is None:
                raise RuntimeError(
                    "Dataset 'hamiltonian_terms' (or 'ham_terms') not found, but --check-hamterms was requested."
                )
            terms = ham_terms_arr[i]
            # Expect shape (3, L)
            if terms.shape[0] != 3:
                raise RuntimeError(
                    f"Hamiltonian terms first dimension expected 3 (density,hopping,interaction); got {terms.shape[0]} at index {i}"
                )
            # Reconstruct energy; handle possible breakdown tuple
            res = system.ground_energy_from_hamiltonian_terms(
                terms, potential, return_breakdown=True
            )
            if isinstance(res, tuple):
                check_energy = res[0]
            else:
                check_energy = res
            assert np.isclose(
                check_energy, ground_energies_arr[i]
            ), f"Energy mismatch (hamterms) at index {i}: {check_energy} != {ground_energies_arr[i]}"
    print("All checks passed successfully.")
