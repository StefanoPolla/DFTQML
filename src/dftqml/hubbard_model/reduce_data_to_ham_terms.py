"""Reduce a PySCF (or generic) Hubbard-chain HDF5 dataset by replacing
stored 2-RDMs with condensed Hamiltonian terms.

Given an input HDF5 file that (optionally) contains the dataset::

	two_rdms: shape (n, L, L, L, L)

this script computes, for every sample, the 3 per-site Hamiltonian term
rows using :func:`dftqml.data_processing.hamiltonian_terms_from_two_rdm`:

	hamiltonian_terms: shape (n, 3, L)

Ordering (fixed across the codebase):
  row 0: density term (⟨n_i⟩)
  row 1: kinetic / nearest-neighbour hopping contribution (factor 2 applied)
  row 2: on-site interaction contribution

All other datasets (``potentials``, ``ground_energies``, ``dft_energies``,
``densities``, ``ground_states``, ``s2``) are copied verbatim if present.
Dataset ``two_rdms`` is NOT copied to the output file.

Attributes from the input file are copied; an additional attribute
``reduced_from`` (input filename) and ``reduction_tool`` are added.

Usage examples:

  Reduce by explicit input path (auto output name <input> -> <stem>_reduced.hdf5):
	  python reduce_data_to_ham_terms.py --input-file compression-main/data/L10-N10-U4.0_pyscf-merged.hdf5

  Provide (L N U) to infer input path pattern (optionally with --suffix):
	  python reduce_data_to_ham_terms.py 10 10 4.0 --suffix pyscf-merged

  Custom output path, chunked processing of large two_rdms:
	  python reduce_data_to_ham_terms.py --input-file data/L12-N12-U4.0_pyscf-merged.hdf5 \
		  --output-file data/L12-N12-U4.0_pyscf-reduced.hdf5 --chunk-size 8

  Dry-run for shape / counts only:
	  python reduce_data_to_ham_terms.py --input-file data/L8-N4-U4.0_pyscf-merged.hdf5 --dry-run

Notes / Design:
  * Processing is chunked along the sample axis to limit peak memory.
  * By default a chunk-size of 16 is used (tunable via --chunk-size).
  * Output is created afresh; in-place overwrite is avoided for safety.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional
import re

import h5py
import numpy as np
from tqdm import tqdm

from dftqml.hubbard_model.data_processing import hamiltonian_terms_from_two_rdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce an HDF5 dataset by converting 2-RDMs to Hamiltonian terms (and dropping 2-RDMs)."
    )
    parser.add_argument(
        "L",
        type=int,
        nargs="?",
        help="Number of lattice sites (needed if inferring filename)",
    )
    parser.add_argument(
        "N",
        type=int,
        nargs="?",
        help="Number of particles (needed if inferring filename)",
    )
    parser.add_argument(
        "U",
        type=float,
        nargs="?",
        help="On-site interaction U (needed if inferring filename)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix used in the input filename when inferring (e.g. pyscf-merged)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Explicit input .hdf5 path (overrides inferred path)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Explicit output .hdf5 path (default: <input_stem>_reduced.hdf5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Number of samples per processing chunk (RAM / speed trade-off)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect input & planned output then exit without writing",
    )
    parser.add_argument(
        "--delete-input",
        action="store_true",
        help="Delete the original file after successful reduction",
    )
    return parser.parse_args()


def infer_input_path(args: argparse.Namespace) -> str:
    if args.input_file:
        return args.input_file
    if args.L is None or args.N is None or args.U is None:
        raise SystemExit("Must provide either --input-file or positional L N U triple.")
    base = f"L{args.L}-N{args.N}-U{args.U}"
    if args.suffix:
        base += f"_{args.suffix}"
    return os.path.join(os.path.dirname(__file__), "data", base + ".hdf5")


def derive_output_path(input_path: str, provided: Optional[str]) -> str:
    if provided:
        return provided
    root, ext = os.path.splitext(input_path)
    return root + "_reduced" + ext


def open_input(input_path: str) -> h5py.File:
    if not os.path.exists(input_path):
        raise SystemExit(f"Input file does not exist: {input_path}")
    return h5py.File(input_path, "r")


def prepare_output(output_path: str, overwrite: bool):
    if os.path.exists(output_path) and not overwrite:
        raise SystemExit(
            f"Output file already exists: {output_path}. Use --overwrite to replace or choose --output-file."
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    return h5py.File(output_path, "w")


def copy_dataset(src_f: h5py.File, dst_f: h5py.File, name: str):
    if name not in src_f:
        return
    dset = src_f[name]
    # Create and populate new dataset with same shape/dtype
    out = dst_f.create_dataset(name, data=dset[...], dtype=dset.dtype)
    # Copy attributes if any
    for k, v in dset.attrs.items():
        out.attrs[k] = v


def _parse_lnu_from_filename(
    path: str,
) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """Parse L, N, U from a filename like .../L8-N8-U4.0[...].hdf5.

    Returns (L, N, U) where values are None if not found.
    """
    base = os.path.basename(path)
    m = re.search(r"L(\d+)-N(\d+)-U([0-9]+(?:\.[0-9]+)?)", base)
    if not m:
        return None, None, None
    Ls, Ns, Us = m.groups()
    try:
        return int(Ls), int(Ns), float(Us)
    except Exception:
        return None, None, None


def _resolve_particle_number(src_f: h5py.File, fallback_N: Optional[int] = None) -> int:
    """Resolve N (number of particles) from multiple sources robustly.

    Priority:
      1) Explicit fallback_N (e.g., CLI args)
      2) File attribute 'N'
      3) Parse from filename 'L{L}-N{N}-U{U}.hdf5'
      4) Infer from first row of 'densities' dataset (sum over sites)
    """
    # 1) CLI-provided
    if fallback_N is not None and int(fallback_N) > 0:
        return int(fallback_N)

    # 2) HDF5 attribute
    if "N" in src_f.attrs:
        try:
            N_attr = int(src_f.attrs["N"])
            if N_attr > 0:
                return N_attr
        except Exception:
            pass

    # 3) Filename pattern
    if hasattr(src_f, "filename") and src_f.filename:
        _, N_from_name, _ = _parse_lnu_from_filename(src_f.filename)
        if N_from_name is not None and N_from_name > 0:
            return int(N_from_name)

    # 4) Densities dataset (sum over sites equals particle number)
    if "densities" in src_f and src_f["densities"].shape[0] > 0:
        try:
            dens0 = np.asarray(src_f["densities"][0], dtype=float)
            N_inf = int(np.rint(dens0.sum()))
            if N_inf > 0:
                return N_inf
        except Exception:
            pass

    raise SystemExit(
        "Unable to determine number of particles N from attributes, filename, CLI, or densities dataset."
    )


def process_two_rdms_to_hamterms(
    src_f: h5py.File,
    dst_f: h5py.File,
    chunk_size: int,
    *,
    n_particles: Optional[int] = None,
):
    if "two_rdms" not in src_f:
        raise SystemExit("Input file lacks dataset 'two_rdms'; nothing to reduce.")
    dset = src_f["two_rdms"]
    if dset.ndim != 5:
        raise SystemExit(
            f"Expected 'two_rdms' to have 5 dims (n,L,L,L,L); found shape {dset.shape}"
        )
    n_samples, L = dset.shape[0], dset.shape[1]

    # Resolve number of particles robustly (supports legacy files without attributes)
    N_val = _resolve_particle_number(src_f, fallback_N=n_particles)
    print(f"  Resolved N={N_val} (L inferred = {L})")

    # Pre-create output dataset
    ham_shape = (n_samples, 3, L)
    ham_dset = dst_f.create_dataset("hamiltonian_terms", ham_shape, dtype=np.float64)
    # Store ordering as a simple UTF-8 string (readable by h5py without decoding hassles)
    ham_dset.attrs["ordering"] = "density,hopping,interaction"

    # Iterate in chunks
    for start in tqdm(
        range(0, n_samples, chunk_size), desc="Reducing two_rdms", unit="chunk"
    ):
        stop = min(start + chunk_size, n_samples)
        block = dset[start:stop]  # shape (k, L,L,L,L)
        # Compute per-sample terms
        out_block = np.empty((block.shape[0], 3, L), dtype=np.float64)
        for i, two_rdm in enumerate(block):
            # Function returns (3,L)
            terms = hamiltonian_terms_from_two_rdm(two_rdm, N_val)
            if terms.shape != (3, L):
                raise RuntimeError(
                    f"Unexpected ham term shape {terms.shape}; expected (3,{L}) from sample index {start+i}"
                )
            out_block[i] = terms
        ham_dset[start:stop, :, :] = out_block


def main():
    args = parse_args()
    input_path = infer_input_path(args)
    output_path = derive_output_path(input_path, args.output_file)

    # Open input to inspect
    with open_input(input_path) as src_f:
        has_two_rdms = "two_rdms" in src_f
        n_samples = int(src_f["two_rdms"].shape[0]) if has_two_rdms else 0
        L_attr = int(src_f.attrs.get("L", -1)) if "L" in src_f.attrs else -1
        print(f"Input file: {input_path}")
        print(
            f"  Attributes: L={L_attr}, N={src_f.attrs.get('N', '?')}, U={src_f.attrs.get('U', '?')}"
        )
        print(f"  Datasets present: {list(src_f.keys())}")
        if has_two_rdms:
            print(f"  two_rdms shape: {src_f['two_rdms'].shape}")
        else:
            print("  (No two_rdms dataset present)")
        print(f"  Planned output: {output_path}")
        if args.dry_run:
            print("Dry run: exiting without writing output.")
            return

    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise SystemExit(
            "Refusing to overwrite input file in-place. Specify a different --output-file."
        )

    with open_input(input_path) as src_f, prepare_output(
        output_path, args.overwrite
    ) as dst_f:
        # Copy file-level attributes
        for k, v in src_f.attrs.items():
            dst_f.attrs[k] = v
        dst_f.attrs["reduced_from"] = os.path.basename(input_path)
        dst_f.attrs["reduction_tool"] = "reduce_data_to_ham_terms.py"

        # Copy simpler datasets (excluding 'two_rdms')
        for name in [
            "potentials",
            "ground_energies",
            "dft_energies",
            "densities",
            "ground_states",
            "s2",
        ]:
            copy_dataset(src_f, dst_f, name)

        # Create hamiltonian_terms from two_rdms if present
        if "two_rdms" in src_f:
            print("Computing hamiltonian_terms from two_rdms ...")
            # Pass CLI-provided N (if any); resolver will fallback otherwise
            process_two_rdms_to_hamterms(
                src_f, dst_f, args.chunk_size, n_particles=args.N
            )
        else:
            print(
                "Warning: input file lacks 'two_rdms'; only copied existing datasets."
            )

    print(f"Wrote reduced file: {output_path}")

    if args.delete_input:
        try:
            os.remove(input_path)
            print(f"Deleted original input file: {input_path}")
        except OSError as e:
            print(f"Failed to delete input file ({e}); continuing.")


if __name__ == "__main__":  # pragma: no cover
    main()
