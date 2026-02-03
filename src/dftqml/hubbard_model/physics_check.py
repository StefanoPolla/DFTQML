"""
Physics sanity check for AE issues on L=8/L=12 systems.

- Configure variables at the top (no argparse).
- Loads the first `N_CHECK` potentials from an HDF5 dataset.
- For each potential, runs PySCF FCI without spin fixing and computes
  the lowest singlet (S=0) and triplet (S=1) energies from the first `N_EX` roots.
- Plots the triplet–singlet gap (E_T - E_S) vs std deviation of potential.

This uses the same integral building as PySCFFCIHubbardChain but bypasses
spin penalties and asks for multiple roots to classify states by S^2.

Run this file directly in VS Code. Adjust the variables below as needed.
"""

from __future__ import annotations

import os
from pathlib import Path
import math

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import cast

from pyscf import fci

from dftqml.hubbard_model.pyscf_extension import PySCFFCIHubbardChain


# Physics parameters (set manually here)
# IMPORTANT: L, N, U are NOT read from the HDF5 file
L = 8
N = 8
U = 4.0

# ----------------------- Config (edit here) -----------------------
# Path to the HDF5 file to analyze (only 'potentials' is read)
DATAFILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data", f"L{L}-N{N}-U{U}.hdf5")
)


# Number of potentials to check (starting from the top)
N_CHECK = 10

# Number of FCI roots (states) to compute per potential
N_EX = 8

# Absolute tolerance to classify S from S(S+1)
# e.g. S=0 -> 0, S=1 -> 2, S=2 -> 6
S2_CLASSIFY_TOL = 0.2

# Number of bins for the binned average line
N_BINS = 12

# Save figure to file
SAVE_FIG = True
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# Make plots crisp in VS Code and pop up a window
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 140

# ---------------------- Helpers ----------------------


def classify_spin_from_s2(s2: float, tol: float = S2_CLASSIFY_TOL) -> int | None:
    """Map S^2 ≈ S(S+1) to integer S within tolerance; else None.
    Examples: s2≈0 -> 0, s2≈2 -> 1, s2≈6 -> 2.
    """
    # Candidate S up to 4 is plenty for our small systems
    for S in (0, 1, 2, 3, 4):
        target = S * (S + 1)
        if abs(s2 - target) <= tol:
            return S
    return None


def compute_singlet_triplet_gap(
    system: PySCFFCIHubbardChain, potential: np.ndarray, n_ex: int
) -> tuple[float, float, float, float, float, float]:
    """Return (sigma, e_singlet, e_triplet, gap, s2_singlet_or_nan, s2_triplet_or_nan).

    - sigma: std of the potential.
    - gap: E_T - E_S if both identified, else NaN.
    """
    L = system.n_sites
    nelec = (system.n_particles // 2, system.n_particles // 2)

    # Build integrals like in PySCFFCIHubbardChain, but avoid any spin fixing
    h1 = system._build_h1(potential)
    eri = system._build_eri()

    solver = fci.direct_spin0.FCI()
    # Ask PySCF for multiple roots
    energies, ci = solver.kernel(h1, eri, L, nelec, nroots=n_ex)

    # Normalize shapes
    if np.isscalar(energies):
        energies = np.array([energies], dtype=float)
        ci = np.expand_dims(ci, 0)

    # Compute S^2 for each root and classify S
    s2_vals = []
    S_vals = []
    for k in range(len(energies)):
        s2, mult = fci.spin_square(ci[k], L, nelec)
        s2_vals.append(float(s2))
        S_vals.append(classify_spin_from_s2(float(s2)))

    # Pick minimum energies for S=0 and S=1 among the returned roots
    e_singlet = np.nan
    e_triplet = np.nan
    s2_s = np.nan
    s2_t = np.nan

    for e, s2, S in zip(energies, s2_vals, S_vals):
        if S == 0:
            if np.isnan(e_singlet) or e < e_singlet:
                e_singlet = float(e)
                s2_s = s2
        elif S == 1:
            if np.isnan(e_triplet) or e < e_triplet:
                e_triplet = float(e)
                s2_t = s2

    gap = (
        e_triplet - e_singlet
        if (not np.isnan(e_triplet) and not np.isnan(e_singlet))
        else np.nan
    )
    sigma = float(np.std(potential))
    return sigma, e_singlet, e_triplet, gap, s2_s, s2_t


# ---------------------- Main logic ----------------------


def main():
    data_path = Path(DATAFILE)
    if not data_path.exists():
        raise FileNotFoundError(f"DATAFILE not found: {DATAFILE}")

    # Read only the first N_CHECK potentials (ignore file attrs)
    with h5py.File(DATAFILE, "r") as f:
        dset = cast(h5py.Dataset, f["potentials"])  # type: ignore[assignment]
        n_avail = int(dset.shape[0])
        n_take = min(N_CHECK, n_avail)
        potentials: np.ndarray = dset[:n_take]

    # Sanity check on L vs potential length
    pot_L = int(potentials.shape[1])
    if pot_L != L:
        raise ValueError(
            f"Configured L={L} does not match potentials length {pot_L} in {DATAFILE}"
        )

    if N % 2 != 0:
        raise ValueError(f"Configured N must be even (Sz=0). Got N={N}.")

    # Build system (no spin penalty, periodic BC as in the generator)
    system = PySCFFCIHubbardChain(
        n_sites=L,
        n_particles=N,
        u=U,
        t=1.0,
        spin_convention="up_then_down",
        boundary_conditions="periodic",
        total_spin_penalty=0.0,  # IMPORTANT: do not enforce singlet
    )

    # Iterate potentials and compute gaps
    sigmas = []
    gaps = []
    eS = []
    eT = []
    s2S = []
    s2T = []

    for pot in tqdm(potentials, desc="FCI roots / classification"):
        try:
            sigma, es, et, gap, s2s, s2t = compute_singlet_triplet_gap(
                system, pot, N_EX
            )
        except Exception as err:
            # Robustness: skip failures but keep arrays aligned in length
            sigma, es, et, gap, s2s, s2t = (
                float(np.std(pot)),
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
        sigmas.append(sigma)
        gaps.append(gap)
        eS.append(es)
        eT.append(et)
        s2S.append(s2s)
        s2T.append(s2t)

    sigmas = np.asarray(sigmas, dtype=float)
    gaps = np.asarray(gaps, dtype=float)

    # Filter valid points
    valid = np.isfinite(sigmas) & np.isfinite(gaps)
    x = sigmas[valid]
    y = gaps[valid]

    # Prepare binned average and error bars
    if len(x) >= 4:
        xmin, xmax = np.min(x), np.max(x)
        if math.isclose(xmin, xmax):
            bins = np.linspace(xmin - 1e-6, xmax + 1e-6, 3)
        else:
            bins = np.linspace(xmin, xmax, N_BINS + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        means = np.full_like(centers, np.nan, dtype=float)
        sems = np.full_like(centers, np.nan, dtype=float)
        counts = np.zeros_like(centers, dtype=int)
        for i in range(len(centers)):
            m = (x >= bins[i]) & (x < bins[i + 1])
            if i == len(centers) - 1:
                # include right edge in last bin
                m = (x >= bins[i]) & (x <= bins[i + 1])
            if np.any(m):
                vals = y[m]
                means[i] = float(np.nanmean(vals))
                counts[i] = int(np.sum(m))
                # standard error of the mean
                if counts[i] > 1:
                    sems[i] = float(np.nanstd(vals, ddof=1) / np.sqrt(counts[i]))
                else:
                    sems[i] = 0.0
    else:
        centers = np.array([])
        means = np.array([])
        sems = np.array([])
        counts = np.array([])

    # ---------------------- Plot ----------------------
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    # Scatter of all samples
    ax.scatter(
        x, y, s=14, alpha=0.35, color="#1f77b4", edgecolor="none", label="instances"
    )

    # Binned averages with shaded SEM
    if centers.size:
        ax.plot(
            centers,
            means,
            color="#d62728",
            lw=2.0,
            label=f"binned mean (n={int(np.sum(counts>0))})",
        )
        ax.fill_between(
            centers,
            means - sems,
            means + sems,
            color="#d62728",
            alpha=0.18,
            linewidth=0,
        )
        # Annotate bin counts lightly
        for cx, cy, c in zip(centers, means, counts):
            if c > 0:
                ax.annotate(
                    str(int(c)),
                    xy=(cx, cy),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#444444",
                )

    # Zero reference
    ax.axhline(0.0, color="#888888", lw=1.0, ls="--", alpha=0.7)

    basename = data_path.name
    title = f"Triplet–Singlet Gap vs Potential Std\n{basename} | L={L} N={N} U={U}  nroots={N_EX}  samples={len(x)}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Std. dev of potential, σ_v")
    ax.set_ylabel("Gap Δ = E_T − E_S  [t]")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()

    if SAVE_FIG:
        os.makedirs(PLOT_DIR, exist_ok=True)
        out_name = f"physics_check_{basename.replace('.hdf5','')}.png"
        out_path = os.path.join(PLOT_DIR, out_name)
        fig.savefig(out_path)
        print(f"Saved figure to: {out_path}")

    # Show the figure in VS Code
    plt.show()


if __name__ == "__main__":
    main()
