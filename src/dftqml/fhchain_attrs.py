import numpy as np
import scipy.sparse
import itertools
import attrs

import openfermion as of

from numpy.typing import ArrayLike


def n_and_sz_indices(nsites: int, nelec: int, up_then_down: bool = True) -> ArrayLike:
    """
    Generate the indices of basis states with fixed particle number and zero
    total spin along the quantization axis (Sz = 0).

    When up_then_down, assumes Jordan-Wigner and spin order convention
    "all up, then all down". Otherwise, the standard up-down-up-down convention
    will be adhered to.

    Args:
        nsites (int): number of spatial sites/orbitals
        nelec (int): number of electrons, should be even
        up_then_down (bool): JW convention

    Returns:
        ArrayLike: list of indices of the subspace
    """
    if up_then_down:
        iterator = itertools.product(
            itertools.combinations(range(nsites, 2 * nsites), nelec // 2),
            itertools.combinations(range(nsites), nelec // 2),
        )
        indices = [np.sum(2 ** np.ravel(i)) for i in iterator]
    else:
        iterator = itertools.product(
            itertools.combinations(range(1, 2 * nsites, 2), nelec // 2),
            itertools.combinations(range(0, 2 * nsites, 2), nelec // 2),
        )
        indices = [np.sum(2 ** np.ravel(i)) for i in iterator]
    return np.array(indices, dtype=int)


def projector_from_block_to_singlet(
    nsites: int, block_indices: ArrayLike, up_then_down: bool = True
) -> ArrayLike:
    s2_fop = of.hamiltonians.s_squared_operator(nsites)
    if up_then_down:
        s2_fop = of.transforms.reorder(s2_fop, of.up_then_down)
    s2_block = of.get_sparse_operator(s2_fop, n_qubits=2 * nsites)[block_indices, :][
        :, block_indices
    ]
    s2_eigvals, s2_eigvecs = np.linalg.eigh(s2_block.toarray())
    singlet_projector = s2_eigvecs[:, np.isclose(s2_eigvals, 0)].T.conj()
    return singlet_projector


def projector_block_and_singlet(nsites: int, nelec: int, up_then_down: bool = True) -> ArrayLike:
    """
    Generate the projector to the spin-singlet subspace with a fixed number of
    particles. Assumes Jordan-Wigner and spin order convention "all up, then
    all down".

    Args:
        nsites (int): number of spatial sites/orbitals
        nelec (int): number of electrons, should be even

    Returns:
        ArrayLike: (matrix) projector on the subspace.
    """
    block_indices = n_and_sz_indices(nsites, nelec, up_then_down)
    block_projector = scipy.sparse.diags(np.ones(4**nsites), format="csc")[block_indices, :]

    singlet_projector = projector_from_block_to_singlet(nsites, block_indices, up_then_down)

    return singlet_projector @ block_projector


def project_operator(operator: ArrayLike, projector: ArrayLike) -> ArrayLike:
    return projector @ operator @ projector.T.conj()

@frozen
class FermiHubbardChain:
    """
    This class represents a Fermi-Hubbard chain system restricted to a
    symmetry sector aka block (fixed particle number, SZ = 0, and S^2 = 0).
    The tunneling and coulomb energies are specified, while the on-site
    potential is left unspecified (requested as a parameter in some functions.

    This class allows to reuse heavy-to-compute variables that depend on the
    system size and number of electrons, such as the local density operators
    and the symmetry block projector.

    The class defines method to obtain relevant operators for the system,
    and ground state properties (obtained through exact diagonalization).

    All the members of this class assume the spin convention up_then_down.
    """

    def __init__(self, nsites, nelec, coulomb, up_then_down=True):
        self.nsites = nsites
        self.nelec = nelec
        self.coulomb = coulomb

        self.up_then_down = up_then_down

        # sector with fixed total particle number, total Z spin, singlet
        self.block_projector = projector_block_and_singlet(nsites, nelec, up_then_down)
        self.block_dimension = self.block_projector.shape[0]

        # homogeneous hamiltonian = kinetic + coulomb (zero onsite potential)
        self.homogeneous_hamiltonian = of.fermi_hubbard(
            x_dimension=nsites, y_dimension=1, tunneling=1, coulomb=coulomb, periodic=True
        )
        if up_then_down:
            self.homogeneous_hamiltonian = of.transforms.reorder(
                self.homogeneous_hamiltonian, of.up_then_down
            )

        self.block_homogeneous_hamiltonian = self.block_project(self.homogeneous_hamiltonian)
        # electron density regardless of spin

        if up_then_down:
            self.density = [
                of.FermionOperator(
                    (
                        (i, 1),
                        (i, 0),
                    ),
                )
                + of.FermionOperator(
                    (
                        (i + nsites, 1),
                        (i + nsites, 0),
                    ),
                )
                for i in range(nsites)
            ]
        else:
            self.density = [
                of.FermionOperator(
                    (
                        (i, 1),
                        (i, 0),
                    ),
                )
                + of.FermionOperator(
                    (
                        (i + 1, 1),
                        (i + 1, 0),
                    ),
                )
                for i in range(0, 2 * nsites, 2)
            ]
        self.block_density = [self.block_project(fop) for fop in self.density]

    def block_project(self, symbolic_operator):
        return project_operator(
            of.get_sparse_operator(symbolic_operator, n_qubits=2 * self.nsites),
            self.block_projector,
        )

    def _check_and_project_state(self, state):
        if len(state) == self.block_dimension:
            return state
        if len(state) == 4**self.nsites:
            return self.block_projector @ state
        raise ValueError("state size does not match block size nor full Hilbert space size")

    def hamiltonian(self, potential=None) -> of.FermionOperator:
        if potential is None:
            return self.homogeneous_hamiltonian
        else:
            if len(potential) != self.nsites:
                raise ValueError("potential has the wrong lenght")
            return self.homogeneous_hamiltonian + np.dot(potential, self.density)

    def block_hamiltonian(self, potential=None):
        if potential is None:
            return self.block_homogeneous_hamiltonian
        else:
            if len(potential) != self.nsites:
                raise ValueError("potential has the wrong lenght")
            return self.block_homogeneous_hamiltonian + np.tensordot(
                potential, self.block_density, axes=([0], [0])
            )

    def homogeneous_block_gse(self):
        """ground state energy of `block_homogeneous_hamiltonian`
        homogeneous hamiltonian = kinetic + coulomb (zero onsite potential)"""
        return scipy.sparse.linalg.eigsh(self.block_homogeneous_hamiltonian, k=1, which="SA")[0][0]

    def ground_energy_and_state(self, potential: ArrayLike = None):
        """
        Args:
            potential (ArrayLike, optional): chemical potential on each site.

        Returns:
            float: Ground energy
            ArrayLike: Ground state
        """
        ham = self.block_hamiltonian(potential)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
        gs_energy = eigvals[0]
        gstate = eigvecs[:, 0]
        return gs_energy, gstate

    def density_expectation(self, state):
        state = self._check_and_project_state(state)
        return np.real([state.conj() @ nop @ state for nop in self.block_density])

    def homogeneous_energy_expectation(self, state):
        state = self._check_and_project_state(state)
        return np.real(state.conj() @ self.block_homogeneous_hamiltonian @ state)

    def dftio(self, state):
        """
        returns DFT input (density expectation value array) and output
        (homogeneous energy expectation value) on a given state
        """
        state = self._check_and_project_state(state)
        return (self.density_expectation(state), self.homogeneous_energy_expectation(state))

    def ground_state_dftio(self, potential=None):
        """
        returns DFT input (density expectation value array) and output
        (homogeneous energy expectation value) on the ground state of the
        full Hamiltonian for a given potential
        """
        gs_energy, gstate = self.ground_energy_and_state(potential)
        density_expval = self.density_expectation(gstate)

        if potential is None:
            homogeneous_energy = gs_energy
        else:
            homogeneous_energy = gs_energy - density_expval @ potential

        return density_expval, homogeneous_energy
