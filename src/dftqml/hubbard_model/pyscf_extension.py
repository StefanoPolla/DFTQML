"""PySCF extension for the Fermi–Hubbard chain.

This module defines a subclass that replaces explicit sparse-matrix
diagonalization with PySCF's Full CI solver, and builds 1-/2-RDMs directly
from the CI vector (no explicit Hamiltonian matrix construction).
"""

from __future__ import annotations

import numpy as np
import warnings
from attrs import frozen

from pyscf import fci
from pyscf.fci import addons as fci_addons  # add this import

from dftqml.hubbard_model.fhchain import FermiHubbardChain
from dftqml.hubbard_model.data_processing import hamiltonian_terms_from_two_rdm

# Tolerance for singlet check: accept tiny numerical noise
S2_TOL = 1e-5


@frozen
class PySCFFCIHubbardChain(FermiHubbardChain):
    """PySCF-backed variant of the Fermi–Hubbard chain.

    This subclass computes ground-state properties and reduced density matrices
    using PySCF's full configuration interaction (FCI) solver instead of
    explicitly constructing and diagonalizing sparse Hamiltonian matrices via
    OpenFermion.

    Only a very small portion of the parent API is overridden:
      * ground_energy_and_state
      * one_rdm_expectation
      * two_rdm_expectation

    The remaining helper / expectation routines from the base class can still
    be used on the RDMs if needed (e.g. by reconstructing energies from the
    2-RDM).  The "state" object returned by :meth:`ground_energy_and_state`
    is NOT a wavefunction vector in the block basis; instead it is a small
    dictionary carrying the CI vector and the FCI solver context.  The
    overridden expectation functions understand this container.  Passing any
    other object will raise a ValueError.

    Notes
    -----
    1. Spin convention: only the 'up_then_down' ordering is currently
       supported. (Extending to the interleaved convention would require
       permuting integrals accordingly.)
    2. Boundary conditions: 'open' and 'periodic' are supported for the
       kinetic (hopping) term when building the one-electron integral matrix.
    3. Spin penalty: If ``total_spin_penalty > 0`` a post‑hoc penalty
       (penalty * <S^2>) is added to the cost. It is advised to set a smaller
       penalty than the base class (e.g., 1.0).
    4. Performance: For lattice sizes used elsewhere in this project the FCI
       dimension is still modest; for larger systems consider a DMRG or other
       approximate solver.
    """

    # --- Internal helper builders -------------------------------------------------
    def _build_h1(self, potential=None):
        """Return the spin‑independent one‑electron integral matrix h1.

        Parameters
        ----------
        potential : array-like or None
            On-site potential v_i. If None, treated as zeros.
        """
        L = self.n_sites
        h1 = np.zeros((L, L), dtype=float)

        # Nearest-neighbour hopping
        for i in range(L - 1):
            h1[i, i + 1] = h1[i + 1, i] = -self.t
        if self.boundary_conditions == "periodic" and L > 2:
            h1[0, L - 1] = h1[L - 1, 0] = -self.t

        if potential is not None:
            pot = np.asarray(potential, dtype=float)
            if pot.shape != (L,):
                raise ValueError(
                    f"Potential has shape {pot.shape}, expected ({L},)"
                )
            h1 += np.diag(pot)
        return h1

    def _build_eri(self):
        """Return the 2-electron (ij|kl) tensor for on-site Hubbard U.

        Only (i i | i i) = U elements are non-zero; PySCF (chemist's notation)
        expects shape (L, L, L, L).
        """
        L = self.n_sites
        eri = np.zeros((L, L, L, L), dtype=float)
        if self.u != 0.0:
            for i in range(L):
                eri[i, i, i, i] = self.u
        return eri

    # --- Overridden core methods --------------------------------------------------
    def ground_energy_and_state(self, potential=None):
        """Compute ground-state energy and CI vector via PySCF FCI.

        Parameters
        ----------
        potential : array-like or None
            On-site potential to include; if None a homogeneous model is used.

        Returns
        -------
        energy : float
            Ground state energy (optionally spin-penalized).
        state : dict
            Container with keys: 'ci', 'solver', 'nmo', 'nelec', 'h1', 'eri',
            'raw_energy', 's2'. This object is consumed by the overridden
            RDM expectation methods.
        """
        if self.spin_convention != "up_then_down":
            raise NotImplementedError(
                "PySCFFCIHubbardChain currently supports only 'up_then_down' spin convention."
            )
        if self.n_particles % 2 != 0:
            raise ValueError("n_particles must be even (Sz=0 sector)")

        h1 = self._build_h1(potential)
        eri = self._build_eri()
        nelec = (self.n_particles // 2, self.n_particles // 2)

        solver = fci.direct_spin0.FCI()
        # Enforce singlet within the solve (targets S(S+1)=0) if a penalty is provided.
        if self.total_spin_penalty is not None and self.total_spin_penalty > 0:
            solver = fci_addons.fix_spin(
                solver, ss=0, shift=self.total_spin_penalty)

        energy, ci = solver.kernel(h1, eri, h1.shape[0], nelec)

        # Compute <S^2> robustly and enforce singlet
        try:
            s2, mult = fci.spin_square(ci, h1.shape[0], nelec)
        except Exception as err:
            raise RuntimeError(
                "Failed to compute <S^2> for CI state; cannot verify singlet."
            ) from err

        if s2 > S2_TOL:
            warnings.warn(
                (
                    f"Ground state is not a singlet: <S^2>={s2:.3e} exceeds tolerance {S2_TOL}. "
                    "Set a positive total_spin_penalty to enforce S=0 with PySCF's fix_spin."
                ),
                category=RuntimeWarning,
                stacklevel=2,
            )

        state = {
            "ci": ci,
            "solver": solver,
            "nmo": self.n_sites,
            "nelec": nelec,
            "h1": h1,
            "eri": eri,
            "raw_energy": energy,
            "s2": s2,
        }
        return energy, state

    # Helper to extract (rdm1, rdm2) from a state container
    def _extract_rdms(self, state):
        if not isinstance(state, dict) or "ci" not in state:
            raise ValueError(
                "State must be the dict returned by ground_energy_and_state for PySCFFCIHubbardChain"
            )

        # Return cached RDMs if present
        if "rdm1" in state and "rdm2" in state:
            return state["rdm1"], state["rdm2"]

        solver = state["solver"]
        ci = state["ci"]
        nmo = state["nmo"]
        nelec = state["nelec"]

        # PySCF returns spin-summed spatial RDMs
        rdm1, rdm2 = solver.make_rdm12(ci, nmo, nelec)

        # Normalize dtype and cache
        rdm1 = np.asarray(rdm1)
        rdm2 = np.asarray(rdm2)
        state["rdm1"] = rdm1
        state["rdm2"] = rdm2
        return rdm1, rdm2

    def one_rdm_expectation(self, state):
        """Return spin-adapted one-RDM in the project's non-redundant lattice form.

        Output shape: (n_sites, n_sites//2 + 1) with entries a[i, j] = gamma_{i, i+j mod L}.
        This mirrors the ordering produced by `one_rdm_operators` in the base module.
        """
        rdm1, _ = self._extract_rdms(state)
        L = self.n_sites
        max_j = L // 2 + 1
        out = np.zeros((L, max_j), dtype=float)
        for i in range(L):
            for j in range(max_j):
                out[i, j] = np.real_if_close(rdm1[i, (i + j) % L])
        return out

    def two_rdm_expectation(self, state):
        """Return the (spin-summed) two-particle RDM in site basis.

        Shape: (n_sites, n_sites, n_sites, n_sites) in chemist ordering (i,j,k,l)
        matching PySCF's convention for spin-summed spatial RDM.
        """
        _, rdm2 = self._extract_rdms(state)
        # Ensure real if within tolerance
        rdm2 = np.real_if_close(rdm2)
        return rdm2

    # Retain parent implementations for density / energies by reconstructing from RDMs if needed
    def ground_energy_from_rdm(self, two_rdm, potential=None, return_breakdown=False):
        """Energy from a (spin-summed spatial) two-RDM.

        Re-uses the project helper converting 2-RDM -> hamiltonian terms, then
        defers to parent implementation for the final energy expression, so the
        breakdown semantics stay identical.
        """
        hamiltonian_terms = hamiltonian_terms_from_two_rdm(
            two_rdm, self.n_particles
        )
        return super().ground_energy_from_hamiltonian_terms(
            hamiltonian_terms, potential, return_breakdown
        )
