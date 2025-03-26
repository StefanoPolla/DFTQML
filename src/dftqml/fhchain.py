import itertools
from functools import cached_property

import numpy as np
import openfermion
import scipy.sparse
from attrs import field, frozen
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
    s2_fop = openfermion.hamiltonians.s_squared_operator(nsites)
    if up_then_down:
        s2_fop = openfermion.transforms.reorder(s2_fop, openfermion.up_then_down)
    s2_block = openfermion.get_sparse_operator(s2_fop, n_qubits=2 * nsites)[block_indices, :][
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


def spin_adapted_one_body_operator(
    i: int, j: int, nsites: int, spin_convention: str
) -> openfermion.FermionOperator:
    """
    Returns the spin-adapted one-body operator \sum_s c_{i, s}^{\dagger} c_{j, s}.
    """
    if spin_convention == "up_then_down":
        return openfermion.FermionOperator(((i, 1), (j, 0))) + openfermion.FermionOperator(
            ((i + nsites, 1), (j + nsites, 0))
        )
    elif spin_convention == "interleaved":
        return openfermion.FermionOperator(((2 * i, 1), (2 * j, 0))) + openfermion.FermionOperator(
            ((2 * i + 1, 1), (2 * j + 1, 0))
        )
    else:
        raise ValueError("spin_convention should be 'up_then_down' or 'interleaved'")


def density_operators(nsites: int, spin_convention: str) -> np.ndarray[openfermion.FermionOperator]:
    """
    Returns an array representing the density operator for a given spin convention.
    """
    density = [spin_adapted_one_body_operator(i, i, nsites, spin_convention) for i in range(nsites)]
    return np.array(density)


def one_rdm_operators(nsites: int, spin_convention: str) -> np.ndarray[openfermion.FermionOperator]:
    """
    Translational-invariant symbolic representation of spin-adapted one-body reduced density matrix.

    Returns an array of symbolic operators representing the terms of the spin-adapted one-body
    reduced density matrix operators for a given spin convention.

    Ordering of the operators:
        The returned array has shape (m, n), where n = `nsites` and m = (n//2 + 1)

        The returned array has the following form (for simplicity of notation we avoid the spin
        summation, but all terms are to be conisdered summed over spin species, i.e. spin-adapted)

        a = [[ c_1^{\dagger} c_1, c_2^{\dagger} c_2, ..., c_n^{\dagger} c_n ],
             [ c_1^{\dagger} c_2, c_2^{\dagger} c_3, ..., c_n^{\dagger} c_1 ],
             [ ... ],
             [ c_1^{\dagger} c_{m} c_2^{\dagger} c_(m+1), ..., c_n^{\dagger} c_{m - 1}]]

        This ordering has some useful properties:
            1. The first row a[0] corresponds to the density.
                The second row a[1] (+ Hermitian conjugates) corresponds to the hopping terms.
                The j-th row corresponds to the pair correlator at distance j.
            2. Limiting to maximum distance j < m = (n+1)//2 is enough to capture all the
                information without duplication (except for the last row in case of even n_sites,
                in which case the second half of that row is redundant).
            3. Translation and mirror symmetries are manifest in the structure of the array.
                To translate by k: `np.roll(a, k, axis=0)`. To mirror: `a[::-1]`.
                Note that these are not symmmetries of the operators themselves, but of the
                1-RDMFT functional. This structure makes it easy to adapt convolutaional neural
                networks to learn the 1-RDMFT functional.

    Args:
        nsites (int): number of sites
        spin_convention (str): Spin convention, can be "up_then_down" or "interleaved".
    """
    one_rdm = [
        [
            spin_adapted_one_body_operator(i, (i + j) % nsites, nsites, spin_convention)
            for i in range(nsites)
        ]
        for j in range(nsites // 2 + 1)
    ]
    return np.array(one_rdm)


@frozen
class FermiHubbardChain:
    """
    Fermi-Hubbard chain system restricted to a symmetry sector (block).

    Used symmetries are:
        - fixed particle number
        - zero spin along quantization axis SZ = 0
        - total spin singlet S^2 = 0

    The tunneling and interaction energies are specified, while the on-site potential is left
    unspecified (requested as a parameter in some functions).

    This class allows to reuse heavy-to-compute variables that depend on the
    system size and number of electrons, such as the local density operators
    and the symmetry block projector.

    The class defines method to obtain relevant operators for the system,
    and ground state properties (obtained through exact diagonalization).

    Args:
        n_sites: number of sites
        n_particles: number of particles
        u: on-site repulsion (Hubbard interaction). Defaults to 0.0.
        t: Tunneling energy. Defaults to 1.0.
        spin_convention: Spin convention, can be "up_then_down" (default) or "interleaved".
        boundary_conditions (str, optional): Boundary conditions, can be "periodic" (default) or
            "open".
    """

    n_sites: int = field(converter=int)
    n_particles: int = field(converter=int)
    u: float = field(default=0.0)
    t: float = field(default=1.0)
    spin_convention: bool = field(default="up_then_down")
    boundary_conditions: str = field(default="periodic")

    @n_particles.validator
    def _n_particles_validator(self, attribute, value):
        if value <= 0:
            raise ValueError("n_particles must be positive")
        if value >= 2 * self.n_sites:
            raise ValueError("n_particles must be less than 2 * n_sites")
        if value % 2 != 0:
            raise ValueError("n_particles must be even in order to allow for a singlet state")
            
    @spin_convention.validator
    def _spin_convention_validator(self, attribute, value):
        if value not in ["up_then_down", "interleaved"]:
            raise ValueError("spin_convention must be 'up_then_down', or 'interleaved'")

    @boundary_conditions.validator
    def _boundary_conditions_validator(self, attribute, value):
        if value not in ["periodic", "open"]:
            raise ValueError("boundary_conditions must be 'periodic', or 'open'")
        
    # ******* Symmetry blocking infrastucure ********

    @cached_property
    def block_projector(self):
        up_then_down = self.spin_convention == "up_then_down"
        return projector_block_and_singlet(self.n_sites, self.n_particles, up_then_down)

    @property
    def block_dimension(self):
        return self.block_projector.shape[0]

    def _symop_to_block(self, symbolic_operator):
        sparse_op = openfermion.get_sparse_operator(symbolic_operator, n_qubits=2 * self.n_sites)
        projected_op = project_operator(sparse_op, self.block_projector)
        return projected_op

    def _check_and_project_state(self, state):
        if len(state) == self.block_dimension:
            return state
        if len(state) == 4**self.n_sites:
            return self.block_projector @ state
        raise ValueError("state size does not match block size nor full Hilbert space size")


    # ******* Uniform Hamiltonian terms *******

    @property
    def interaction_hamiltonian(self):
        """Hubbard interaction term only"""
        if self.u == 0:
            return openfermion.FermionOperator()
        ham = openfermion.fermi_hubbard(
            x_dimension=self.n_sites, y_dimension=1, tunneling=0, coulomb=self.u, periodic=True
        ) # periodic is irrelevant here
        if self.spin_convention == "up_then_down":
            ham = openfermion.transforms.reorder(ham, openfermion.up_then_down)
        return ham
    
    @property
    def kinetic_hamiltonian(self):
        """Kinetic term only"""
        periodic = self.boundary_conditions == "periodic"
        ham = openfermion.fermi_hubbard(
            x_dimension=self.n_sites, y_dimension=1, tunneling=1, coulomb=0, periodic=periodic
        )
        if self.spin_convention == "up_then_down":
            ham = openfermion.transforms.reorder(ham, openfermion.up_then_down)
        return ham
    
    @property
    def homogeneous_hamiltonian(self):
        """homogeneous hamiltonian = kinetic + interaction (zero onsite potential)"""
        ham = self.kinetic_hamiltonian + self.interaction_hamiltonian
        return ham

    @cached_property
    def block_interaction_hamiltonian(self):
        return self._symop_to_block(self.interaction_hamiltonian)

    @cached_property
    def block_kinetic_hamiltonian(self):
        return self._symop_to_block(self.kinetic_hamiltonian)

    @cached_property
    def block_homogeneous_hamiltonian(self):
        return self._symop_to_block(self.homogeneous_hamiltonian)
    
    def interaction_energy_expectation(self, state):
        """The state can be in block or full Hilbert space."""
        state = self._check_and_project_state(state)
        return np.real(state.conj() @ self.block_interaction_hamiltonian @ state)
    
    def kinetic_energy_expectation(self, state):
        """The state can be in block or full Hilbert space."""
        state = self._check_and_project_state(state)
        return np.real(state.conj() @ self.block_kinetic_hamiltonian @ state)
    
    def homogeneous_energy_expectation(self, state):
        """The state can be in block or full Hilbert space."""
        state = self._check_and_project_state(state)
        return np.real(state.conj() @ self.block_homogeneous_hamiltonian @ state)

    # ******* Density and RDM block operators *******

    @cached_property
    def block_density_operators(self) -> np.ndarray:
        """block operators for spin-summed electron density. Axis 0 is the site index."""
        density = density_operators(self.n_sites, self.spin_convention)
        return np.array([self._symop_to_block(fop) for fop in density])
    
    @cached_property
    def block_one_rdm_operators(self) -> np.ndarray:
        """Block operators for spin-summed pair corrlelators, see `one_rdm_operators`."""
        one_rdm = one_rdm_operators(self.n_sites, self.spin_convention)
        return np.array([[self._symop_to_block(fop) for fop in row] for row in one_rdm])
    
    def density_expectation(self, state):
        state = self._check_and_project_state(state)
        density = np.einsum("j, ijk, k", state.conj(), self.block_density_operators, state)
        return np.real(density)
    
    def one_rdm_expectation(self, state):
        state = self._check_and_project_state(state)
        one_rdm = np.einsum("k, ijkl, l", state.conj(), self.block_one_rdm_operators, state)
        real_rdm = np.real(one_rdm)
        if np.allclose(one_rdm, real_rdm):
            return real_rdm
        else:
            raise NotImplementedError("Complex one-body reduced density matrix not implemented yet")
    
    def dftio(self, state):
        """
        returns DFT input (density expectation value array) and output
        (homogeneous energy expectation value) on a given state
        """
        state = self._check_and_project_state(state)
        return (self.density_expectation(state), self.homogeneous_energy_expectation(state))
    
    def rdmftio(self, state):
        """
        returns DFT input (density expectation value array) and output
        (homogeneous energy expectation value) on a given state
        """
        state = self._check_and_project_state(state)
        return (self.one_rdm_expectation(state), self.interaction_energy_expectation(state))

    # ******* Hamiltonian for system with potential *******

    def hamiltonian(self, potential=None) -> openfermion.FermionOperator:
        if potential is None:
            return self.homogeneous_hamiltonian
        else:
            if len(potential) != self.n_sites:
                raise ValueError("potential has the wrong lenght")
            return self.homogeneous_hamiltonian + np.dot(potential, self.density)

    def block_hamiltonian(self, potential=None):
        if potential is None:
            return self.block_homogeneous_hamiltonian
        else:
            if len(potential) != self.n_sites:
                raise ValueError("potential has the wrong lenght")
            return self.block_homogeneous_hamiltonian + np.tensordot(
                potential, self.block_density_operators, axes=([0], [0])
            )

    # ******* Exact (sparse) diagonalizartion *******

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

    def ground_state_rdmftio(self, potential=None):
        """
        returns 1RDMFT input (pair correlators expectation value matrix) and output
        (interaction energy expectation value) on the ground state of the
        full Hamiltonian for a given potential
        """
        _, gstate = self.ground_energy_and_state(potential)
        one_rdm_expval = self.one_rdm_expectation(gstate)
        interaction_expval = self.interaction_energy_expectation(gstate)

        return one_rdm_expval, interaction_expval