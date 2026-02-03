"""
Consolidate per-task HDF5 outputs produced by generate_instance_launcher_pyscf.py
into a single merged HDF5 file per (L, N, U).

This script searches in compression-main/data for files matching:
    L{L}-N{N}-U{U}_<suffix>.hdf5
By default it filters to suffixes starting with "pyscf_A" (the launcher default).

Datasets consolidated (if present in inputs):
  - potentials        (n, L)
  - ground_energies   (n,)
  - dft_energies      (n,)
  - densities         (n, L)
  - ground_states     (n, ci_dim)  [optional]
  - two_rdms          (n, L, L, L, L)  [optional]

Attributes copied to output:
    - generator, L, N, U from the first input file
    - merged_file_count: number of input files merged (instead of listing all filenames)

Usage:
  python consolidate_pyscf_outputs.py L N U \
      [--data-dir compression-main/data] \
      [--suffix-prefix pyscf_A] \
      [--output-suffix pyscf-merged] \
      [--delete-inputs]
"""

import argparse
import os
import re
from typing import List, Any
import numpy as np
import h5py
from tqdm import tqdm


def find_input_files(
    data_dir: str, L: int, N: int, U: float, suffix_prefix: str
) -> List[str]:
    prefix = f"L{L}-N{N}-U{U}_"
    files = []
    for name in os.listdir(data_dir):
        if not name.endswith(".hdf5"):
            continue
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) : -5]
        if suffix_prefix and not rest.startswith(suffix_prefix):
            continue
        files.append(os.path.join(data_dir, name))

    # Stable sort by array indices if present, else lexicographic
    def sort_key(path: str):
        base = os.path.basename(path)
        m = re.search(r"_A(\d+)_a(\d+)$", base[:-5])
        if m:
            return (0, int(m.group(1)), int(m.group(2)))
        return (1, base)

    files.sort(key=sort_key)
    return files


def get_dataset(f: h5py.File, name: str):
    obj = f[name]
    if not isinstance(obj, h5py.Dataset):
        raise RuntimeError(f"Object '{name}' in {f.filename} is not a dataset.")
    return obj


def total_rows(h5path: str) -> int:
    with h5py.File(h5path, "r") as f:
        dset: Any = get_dataset(f, "potentials")
        return int(len(dset))


def file_has_dataset(h5path: str, name: str) -> bool:
    with h5py.File(h5path, "r") as f:
        return name in f


def create_out_datasets(out: h5py.File, L: int, totals, dtypes):
    out.attrs["L"] = L
    # Create datasets with maxshape=None along sample axis
    if "potentials" in totals:
        out.create_dataset(
            "potentials",
            totals["potentials"],
            maxshape=(None, L),
            dtype=dtypes["potentials"],
        )
    if "ground_energies" in totals:
        out.create_dataset(
            "ground_energies",
            totals["ground_energies"],
            maxshape=(None,),
            dtype=dtypes["ground_energies"],
        )
    if "dft_energies" in totals:
        out.create_dataset(
            "dft_energies",
            totals["dft_energies"],
            maxshape=(None,),
            dtype=dtypes["dft_energies"],
        )
    if "densities" in totals:
        out.create_dataset(
            "densities",
            totals["densities"],
            maxshape=(None, L),
            dtype=dtypes["densities"],
        )
    if "ground_states" in totals:
        out.create_dataset(
            "ground_states",
            totals["ground_states"],
            maxshape=(None, totals["ground_states"][1]),
            dtype=dtypes["ground_states"],
        )
    if "two_rdms" in totals:
        out.create_dataset(
            "two_rdms",
            totals["two_rdms"],
            maxshape=(None, L, L, L, L),
            dtype=dtypes["two_rdms"],
        )


def merge_files(files: List[str], out_path: str):
    if not files:
        raise RuntimeError("No input files found to merge.")

    # Determine L, N, U and base attributes from first file
    with h5py.File(files[0], "r") as f0:
        if "L" not in f0.attrs:
            raise RuntimeError(
                "Missing attribute 'L' in input file; cannot determine system size."
            )
        L = int(f0.attrs["L"])  # type: ignore[arg-type]
        base_attrs = {k: f0.attrs[k] for k in f0.attrs.keys()}

    # Discover which datasets exist and their shapes/dtypes, verify consistency
    present = {
        k: False
        for k in (
            "potentials",
            "ground_energies",
            "dft_energies",
            "densities",
            "ground_states",
            "two_rdms",
        )
    }
    # Accumulate total counts
    totals_shape = {}
    counts = []
    for fp in files:
        n = total_rows(fp)
        if n == 0:
            continue
        counts.append((fp, n))
        for name in present.keys():
            if file_has_dataset(fp, name):
                present[name] = True

    if not counts:
        raise RuntimeError("No non-empty input files found.")

    total_n = sum(n for _, n in counts)
    # Compute output shapes
    if present.get("potentials"):
        totals_shape["potentials"] = (total_n, L)
    if present.get("ground_energies"):
        totals_shape["ground_energies"] = (total_n,)
    if present.get("dft_energies"):
        totals_shape["dft_energies"] = (total_n,)
    if present.get("densities"):
        totals_shape["densities"] = (total_n, L)
    # ground_states ci_dim must be consistent across files; check
    if present.get("ground_states"):
        ci_dim = None
        for fp in files:
            with h5py.File(fp, "r") as f:
                if "ground_states" in f:
                    gs = get_dataset(f, "ground_states")
                    # Prefer attribute written by generator
                    if "ci_dim" in gs.attrs:
                        dim = int(gs.attrs["ci_dim"])  # type: ignore[arg-type]
                    else:
                        # Fallback: read first row and infer size
                        if len(gs) > 0:
                            first_row = gs[0]
                            try:
                                dim = int(first_row.size)
                            except Exception:
                                dim = int(len(first_row))
                        else:
                            continue
                    if ci_dim is None:
                        ci_dim = dim
                    elif ci_dim != dim:
                        raise RuntimeError(
                            f"Inconsistent CI dimension across files: {ci_dim} vs {dim} in {fp}"
                        )
        if ci_dim is not None:
            totals_shape["ground_states"] = (total_n, ci_dim)
    if present.get("two_rdms"):
        totals_shape["two_rdms"] = (total_n, L, L, L, L)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as out:
        # Copy base attributes
        for k, v in base_attrs.items():
            out.attrs[k] = v
        # Store only count of files merged to avoid huge attribute size
        out.attrs["merged_file_count"] = len(counts)

        # Determine dtypes by preserving source dataset dtypes (first file that contains it)
        dtypes = {}

        def find_dtype(name: str):
            for fp2 in files:
                with h5py.File(fp2, "r") as f2:
                    if name in f2:
                        obj = f2[name]
                        if isinstance(obj, h5py.Dataset):
                            return obj.dtype
            return None

        for ds_name in [
            "potentials",
            "ground_energies",
            "dft_energies",
            "densities",
            "ground_states",
            "two_rdms",
        ]:
            if ds_name in totals_shape:
                dt = find_dtype(ds_name)
                if dt is None:
                    # Fallback defaults if somehow not found
                    if ds_name == "ground_states":
                        dt = np.complex128
                    else:
                        dt = np.float64
                dtypes[ds_name] = dt

        create_out_datasets(out, L, totals_shape, dtypes)

        # Copy dataset-specific attrs from first file when present (e.g., ground_states metadata)
        with h5py.File(files[0], "r") as f0:
            if "ground_states" in out and "ground_states" in f0:
                src = get_dataset(f0, "ground_states")
                for ak, av in src.attrs.items():
                    out["ground_states"].attrs[ak] = av

        # Append data sequentially with a progress bar over total rows
        cursor = 0
        pbar = tqdm(total=total_n, desc="Merging rows", unit="row")
        for fp, n in counts:
            with h5py.File(fp, "r") as f:
                sl = slice(cursor, cursor + n)
                if "potentials" in out and "potentials" in f:
                    out["potentials"][sl, :] = get_dataset(f, "potentials")[
                        :
                    ]  # type: ignore[index]
                if "ground_energies" in out and "ground_energies" in f:
                    out["ground_energies"][sl] = get_dataset(f, "ground_energies")[
                        :
                    ]  # type: ignore[index]
                if "dft_energies" in out and "dft_energies" in f:
                    out["dft_energies"][sl] = get_dataset(f, "dft_energies")[
                        :
                    ]  # type: ignore[index]
                if "densities" in out and "densities" in f:
                    out["densities"][sl, :] = get_dataset(f, "densities")[
                        :
                    ]  # type: ignore[index]
                if "ground_states" in out and "ground_states" in f:
                    out["ground_states"][sl, :] = get_dataset(f, "ground_states")[
                        :
                    ]  # type: ignore[index]
                if "two_rdms" in out and "two_rdms" in f:
                    # Copy in chunks if desired; here we copy per-file block to avoid holding all in memory
                    out["two_rdms"][sl, ...] = get_dataset(f, "two_rdms")[
                        :
                    ]  # type: ignore[index]
            cursor += n
            pbar.update(n)
        pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("L", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("U", type=float)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "data")),
    )
    parser.add_argument("--suffix-prefix", type=str, default="pyscf_A")
    parser.add_argument("--output-suffix", type=str, default="pyscf-merged")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-inputs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = find_input_files(args.data_dir, args.L, args.N, args.U, args.suffix_prefix)
    if not files:
        raise SystemExit("No matching input files found.")

    out_base = f"L{args.L}-N{args.N}-U{args.U}_{args.output_suffix}.hdf5"
    out_path = os.path.join(args.data_dir, out_base)
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"Output file exists: {out_path}. Use --overwrite to replace.")

    if args.dry_run:
        total = sum(total_rows(f) for f in files)
        print(
            f"Found {len(files)} files; total rows = {total}. Would write -> {out_path}"
        )
        return

    merge_files(files, out_path)
    print(f"Wrote merged file: {out_path}")

    if args.delete_inputs:
        for fp in files:
            try:
                os.remove(fp)
            except OSError:
                pass
        print(f"Deleted {len(files)} input files.")


if __name__ == "__main__":
    main()
