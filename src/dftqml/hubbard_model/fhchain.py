"""
Main module for the Fermi-Hubbard chain system.
"""
import itertools
from functools import cached_property
import warnings

import numpy as np
import openfermion
import scipy.sparse
import scipy.sparse.linalg
from attrs import field, frozen
from numpy.typing import ArrayLike
from typing import Optional

import torch

from dftqml.hubbard_model.data_processing import hamiltonian_terms_from_two_rdm


# ******** Symmetry blocking infrastructure ********


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


def s_squared_block_operator(
    nsites: int, block_indices: ArrayLike, up_then_down: bool
) -> scipy.sparse.spmatrix:
    """
    Generate the S^2 operator restricted to a symmetry block.

    Args:
        nsites (int): Number of spatial sites/orbitals.
        block_indices (ArrayLike): Indices of the symmetry block.
        up_then_down (bool): JW convention.

    Returns:
        scipy.sparse.spmatrix: Sparse matrix representation of the S^2 operator in the block.
    """
    """Generate the projector to the spin-singlet subspace."""
    s2_fop = openfermion.hamiltonians.s_squared_operator(nsites)
    if up_then_down:
        s2_fop = openfermion.transforms.reorder(
            s2_fop, openfermion.up_then_down)
    sparse_operator = openfermion.get_sparse_operator(
        s2_fop, n_qubits=2 * nsites)

    # Validate block_indices
    max_index = sparse_operator.shape[0]
    if not np.all((0 <= block_indices) & (block_indices < max_index)):
        raise ValueError(
            "block_indices contains values out of the valid range for the sparse operator dimensions."
        )

    s2_block = sparse_operator[np.ix_(block_indices, block_indices)]
    return s2_block


def projector_from_block_to_singlet(
    nsites: int, block_indices: ArrayLike, up_then_down: bool = True
) -> ArrayLike:
    """
    [DEPRECATED] Generate the projector to the spin singlet subspace.

    The singlet subspace is defined as the subspace of states with S^2 = 0, and constructe in
    practice by diagonalizing the S^2 operator. This leaves an ambiguity in the choice of
    basis states. For this reason, this function is deprecated.

    As a temporary solution, we do not project on the singlet subspace, but rather
    add a penalty term when diagonalizing the Hamiltonian. The memory cost of skipping this
    projection is negligible, and avoids the ambiguity as well as the extra cost of diagonalizing
    the S^2 operator.
    """

    warnings.warn(
        "The function `projector_from_block_to_singlet` is deprecated and may be removed in future versions. "
        "Consider using an alternative approach as described in the function's docstring.",
        DeprecationWarning,
        stacklevel=2,
    )

    s2_block = s_squared_block_operator(nsites, block_indices, up_then_down)
    s2_eigvals, s2_eigvecs = np.linalg.eigh(s2_block.toarray())
    singlet_projector = s2_eigvecs[:, np.isclose(s2_eigvals, 0)].T.conj()
    sparse_singlet_projector = scipy.sparse.csr_matrix(singlet_projector)
    return sparse_singlet_projector


def projector_block_and_singlet(nsites: int, nelec: int, up_then_down: bool = True) -> ArrayLike:
    """
    [DEPRECATED] see `projector_from_block_to_singlet`

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
    block_projector = scipy.sparse.diags(
        np.ones(4**nsites), format="csc")[block_indices, :]

    singlet_projector = projector_from_block_to_singlet(
        nsites, block_indices, up_then_down)

    return singlet_projector @ block_projector


def project_operator(operator: ArrayLike, projector: ArrayLike) -> ArrayLike:
    """Project the operator to the subspace defined by the projector."""
    tmp = projector @ operator
    tmp2 = tmp @ projector.conj().T
    return tmp2


# ******** Openfermion SymbolicOperator implementation of RDM operators ********


def spin_adapted_one_body_operator(
    i: int, j: int, nsites: int, spin_convention: str
) -> openfermion.FermionOperator:
    r"""
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
        raise ValueError(
            "spin_convention should be 'up_then_down' or 'interleaved'")


def spin_adapted_two_body_operator(
    i: int, j: int, k: int, l: int, nsites: int, spin_convention: str
) -> openfermion.FermionOperator:
    r"""
    Returns the spin-adapted two-body operator
    \sum_{s,t} c_{i, s}^{\dagger} c_{j, t}^{\dagger} c_{k, t} c_{l, s}.
    """
    tmp = spin_adapted_one_body_operator(
        i, j, nsites, spin_convention) * spin_adapted_one_body_operator(
            k, l, nsites, spin_convention
    )
    if j == k:
        tmp -= spin_adapted_one_body_operator(i, l, nsites, spin_convention)
    return tmp


def density_operators(nsites: int, spin_convention: str) -> np.ndarray[openfermion.FermionOperator]:
    """
    Returns an array representing the density operator for a given spin convention.
    """
    density = [spin_adapted_one_body_operator(
        i, i, nsites, spin_convention) for i in range(nsites)]
    return np.array(density)


def one_rdm_operators(nsites: int, spin_convention: str) -> np.ndarray[openfermion.FermionOperator]:
    r"""
    Translational-invariant symbolic representation of spin-adapted one-body reduced density matrix.

    Returns an array of symbolic operators representing the terms of the spin-adapted one-body
    reduced density matrix operators for a given spin convention.

    Ordering of the operators:
        The returned array has shape (n, m), where n = `nsites` and m = (n//2 + 1)

        The returned array has the following form (for simplicity of notation we avoid the spin
        summation, but all terms are to be conisdered summed over spin species, i.e. spin-adapted)

        a = [[ c_1^{\dagger} c_1, c_1^{\dagger} c_2, ..., c_1^{\dagger} c_m    ],
             [ c_2^{\dagger} c_2, c_2^{\dagger} c_3, ..., c_2^{\dagger} c_{m+1}],
             [ ... ],
             [ c_n^{\dagger} c_n, c_n^{\dagger} c_1, ..., c_n^{\dagger} c_{m-1}]]


        This ordering has some useful properties:
            1. The first column a[:, 0] corresponds to the density.
               The second column a[:, 1] (+ Hermitian conjugates) corresponds to the hopping terms.
               The j-th column corresponds to the pair correlator at distance j.
            2. Limiting to maximum distance j < m = (n+1)//2 is enough to capture all the
               information with minimum duplication. In fact there is no redundant information,
               except for the last column in case of even n_sites, in which case the second half of
               that column is redundant.
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
            spin_adapted_one_body_operator(
                i, (i + j) % nsites, nsites, spin_convention)
            for j in range(nsites // 2 + 1)
        ]
        for i in range(nsites)
    ]
    return np.array(one_rdm)


def interaction_operators(
    nsites: int, spin_convention: str
) -> np.ndarray[openfermion.FermionOperator]:
    """
    Returns the on-site interaction operator for a given spin convention.
    """
    if spin_convention == "up_then_down":
        return np.array(
            [
                openfermion.FermionOperator(
                    ((i, 1), (i, 0), (i + nsites, 1), (i + nsites, 0)))
                for i in range(nsites)
            ]
        )
    elif spin_convention == "interleaved":
        return np.array(
            [
                openfermion.FermionOperator(
                    ((2 * i, 1), (2 * i, 0), (2 * i + 1, 1), (2 * i + 1, 0))
                )
                for i in range(nsites)
            ]
        )
    else:
        raise ValueError(
            "spin_convention should be 'up_then_down' or 'interleaved'")


@frozen
class FermiHubbardChain:
    """
    Fermi-Hubbard chain system restricted to a symmetry sector (block).

    Explicitly enforced symmetry sectors are:
        - fixed particle number
        - zero spin along quantization axis SZ = 0
    The total spin-singlet (S^2 = 0) is not enforced explicitly, but rather through a penalty term
    added when diagonalizing the Hamiltonian. The penalty term magnitude is an input parameter.

    The tunneling and interaction energies are fixed upon instantiation, while the on-site potential
    is left unspecified in the class instance (requested as a parameter in some functions).

    This class allows to reuse heavy-to-compute variables that depend on the system size and number
    of electrons, such as the local density operators and the symmetry block projector.

    The class defines method to obtain relevant operators for the system, and ground state
    properties (obtained through exact diagonalization).

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
    spin_convention: str = field(default="up_then_down")
    boundary_conditions: str = field(default="periodic")
    total_spin_penalty: float = field(default=10.0)

    @n_particles.validator
    def _n_particles_validator(self, attribute, value):
        if value <= 0:
            raise ValueError(f"{attribute.name} must be positive")
        if value >= 2 * self.n_sites:
            raise ValueError(
                f"{attribute.name} must be less than 2 times n_sites")
        if value % 2 != 0:
            raise ValueError(
                f"{attribute.name} must be even in order to allow for a singlet state")

    @spin_convention.validator
    def _spin_convention_validator(self, attribute, value):
        if value not in ["up_then_down", "interleaved"]:
            raise ValueError(
                f"{attribute.name} must be 'up_then_down', or 'interleaved'")

    @boundary_conditions.validator
    def _boundary_conditions_validator(self, attribute, value):
        if value not in ["periodic", "open"]:
            raise ValueError(f"{attribute.name} must be 'periodic', or 'open'")

    # ******* Symmetry blocking infrastucure ********

    @cached_property
    def block_indices(self):
        """Indices of the symmetry block."""
        up_then_down = self.spin_convention == "up_then_down"
        return n_and_sz_indices(self.n_sites, self.n_particles, up_then_down)

    @property
    def block_dimension(self):
        """Dimension of the symmetry block defined by the number of particles and
        the spin-singlet subspace."""
        return self.block_indices.shape[0]

    def _symop_to_block(self, symbolic_operator, keep_sparse=False):
        """Projects a symbolic operator to the symmetry block defined by the
        number of particles and the spin-singlet subspace.
        """
        sparse_op = openfermion.get_sparse_operator(
            symbolic_operator, n_qubits=2 * self.n_sites)
        projected_op = sparse_op[np.ix_(
            self.block_indices, self.block_indices)]
        if scipy.sparse.issparse(projected_op) and not keep_sparse:
            return projected_op.toarray()
        return projected_op

    def _check_and_project_state(self, state):
        """
        Check if the state is in the block or full Hilbert space, and project it to the block
        if it is in the full Hilbert space."""
        if len(state) == self.block_dimension:
            return state
        if len(state) == 4**self.n_sites:
            return state[self.block_indices]
        raise ValueError(
            "state size does not match block size nor full Hilbert space size")

    # ******* Total spin singlet infrastructure *******

    @cached_property
    def block_total_spin_penalty(self):
        """
        Returns the penalty term for the total spin operator in the block.
        """
        up_then_down = self.spin_convention == "up_then_down"
        s2_block = s_squared_block_operator(
            self.n_sites, self.block_indices, up_then_down)
        penalty = self.total_spin_penalty * s2_block
        return penalty

    def check_spin_singlet(self, state):
        """
        Check if the state is a spin singlet.
        """
        s2_expval = state.conj() @ self.block_total_spin_penalty @ state
        if not np.isclose(s2_expval, 0):
            raise ValueError("The state is not a spin singlet.")

    # ******* Uniform Hamiltonian terms *******

    @property
    def interaction_hamiltonian(self):
        """Hubbard interaction term only"""
        if self.u == 0:
            return openfermion.FermionOperator()
        ham = openfermion.fermi_hubbard(
            x_dimension=self.n_sites, y_dimension=1, tunneling=0, coulomb=self.u, periodic=True
        )  # periodic is irrelevant here
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
        """Block operator for the interaction Hamiltonian."""
        return self._symop_to_block(self.interaction_hamiltonian)

    @cached_property
    def block_kinetic_hamiltonian(self):
        """Block operator for the kinetic Hamiltonian."""
        return self._symop_to_block(self.kinetic_hamiltonian)

    @cached_property
    def block_homogeneous_hamiltonian(self):
        """Block operator for the homogeneous Hamiltonian."""
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

    @cached_property
    def block_two_rdm_operators(self) -> np.ndarray:
        """Block operators for spin-summed pair corrlelators, see `one_rdm_operators`."""
        num_ind = self.n_sites
        return np.array([[[[openfermion.get_sparse_operator(
            spin_adapted_two_body_operator(
                        p, q, r, s, self.n_sites, self.spin_convention), n_qubits=2*self.n_sites)
            for s in range(num_ind)] for r in range(num_ind)]
            for q in range(num_ind)] for p in range(num_ind)])

    @cached_property
    def block_ham_terms_operators(self) -> np.ndarray:
        """
        Block operators for the Hamiltonian terms. The first two columns are the density and hopping
        terms, the third row is the on-site interaction.
        """
        first_two_cols = one_rdm_operators(
            self.n_sites, self.spin_convention)[:, 0:2]
        third_col = interaction_operators(self.n_sites, self.spin_convention)
        hamiltonian_terms = np.hstack(
            (first_two_cols, third_col[:, np.newaxis]))
        return np.array([[self._symop_to_block(fop) for fop in row] for row in hamiltonian_terms])

    def density_expectation(self, state):
        """Returns the density expectation value on a given state."""
        state = self._check_and_project_state(state)
        density = np.einsum("j, ijk, k", state.conj(),
                            self.block_density_operators, state)
        return np.real(density)

    def one_rdm_expectation(self, state):
        """
        Returns the one-RDM in local non-redundant form. (See 'dftqml.fhchain.one_rdm_operators').

        The state can be a vector in the spin-particle block or full Hilbert space.
        """
        state = self._check_and_project_state(state)
        one_rdm = np.einsum("k, ijkl, l", state.conj(),
                            self.block_one_rdm_operators, state)
        real_rdm = np.real(one_rdm)
        if np.allclose(one_rdm, real_rdm):
            return real_rdm
        else:
            raise NotImplementedError(
                "Complex one-body reduced density matrix not implemented yet")

    def two_rdm_expectation(self, state):
        r"""
        Generate one- and two-particle reduced density matrices from
        a state :math:`| \Psi \rangle`:

        .. math::
            \gamma = \langle \Psi | E_{pq} | \Psi \rangle\\
            \Gamma = \langle \Psi | e_{pqrs} | \Psi \rangle

        where the single (double) excitation operators :math:`E_{pq}`
        (:math:`e_{pqrs}`) can be either restricted (summed over spin)
        or unrestricted.
        """
        if len(state) != 4**self.n_sites:
            try:
                state_full = np.zeros((4**self.n_sites), dtype=state.dtype)
                state_full[self.block_indices] = state
            except ValueError as exc:
                raise ValueError(
                    "state size does not match block_indices nor full Hilbert space size") from exc
        else:
            state_full = state
        two_rdm = np.zeros([self.n_sites]*4)
        for p, q in itertools.product(range(self.n_sites), repeat=2):
            for r, s in itertools.product(range(self.n_sites), repeat=2):
                two_rdm[p, q, r, s] = (state_full.T.conj() @ (
                    self.block_two_rdm_operators[p][q][r][s] @ state_full)).real
        return two_rdm

    def hamiltonian_terms_expval(self, state):
        r"""
        Returns the expectation value of the Hamiltonian terms on a given state, according to the
        following convention:

        first row: spin-summed density
            c_{i, up}^{\dagger} c_i + c_{i, down}^{\dagger} c_i

        second row: spin-summed nearest-neighbor pair correlators
            c_{i, up}^{\dagger} c_{j, up} + c_{i, down}^{\dagger} c_{j, down}

        third row - on-site interaction
            c_{i, up}^{\dagger} c_{i, up} c_{i, down}^{\dagger} c_{i, down}

        The first two rows match the one-RDM operators.

        The state can be a vector in the spin-particle block or full Hilbert space.
        """
        state = self._check_and_project_state(state)
        hamiltonian_terms_expvals = np.einsum(
            "k, ijkl, l", state.conj(), self.block_ham_terms_operators, state
        )
        real_hamiltonian_terms_expvals = np.real(hamiltonian_terms_expvals)
        if np.allclose(hamiltonian_terms_expvals, real_hamiltonian_terms_expvals):
            return real_hamiltonian_terms_expvals
        else:
            raise NotImplementedError(
                "Complex Hamiltonian terms not implemented yet")

    def ground_energy_from_rdm(self, two_rdm, potential=None, return_breakdown=False):
        r"""
        Compute the expectation value of the 1D Fermi-Hubbard Hamiltonian using the two-RDM.

        Parameters:
            two_rdm (ndarray): Two-body RDM, shape (n_sites, n_sites, n_sites, n_sites)

        Returns:
            float or tuple: If return_breakdown is False, returns total energy.
                            If True, returns (total_energy, hopping_energy, interaction_energy, potential_energy).
        """

        hamiltonian_terms = hamiltonian_terms_from_two_rdm(
            two_rdm, self.n_particles)
        return self.ground_energy_from_hamiltonian_terms(hamiltonian_terms, potential, return_breakdown)

    def ground_energy_from_hamiltonian_terms(self, hamiltonian_terms, potential=None, return_breakdown=False):
        r"""
        Compute the expectation value of the 1D Fermi-Hubbard Hamiltonian using the Hamiltonian
        terms.

        Parameters:
            hamiltonian_terms (ndarray): Hamiltonian terms, shape (3, n_sites)
            potential (ndarray, optional): On-site potential, shape (n_sites,)
            return_breakdown (bool, optional): If True, return hopping, interaction, and potential energies separately. Defaults to False.

        Returns:
            float or tuple: If return_breakdown is False, returns total energy.
                            If True, returns (total_energy, hopping_energy, interaction_energy, potential_energy).
        """

        # Torch branch
        if torch.is_tensor(hamiltonian_terms):
            hopping_energy = -self.t * torch.sum(hamiltonian_terms[1])
            interaction_energy = self.u * torch.sum(hamiltonian_terms[2])
            if potential is not None:
                if not torch.is_tensor(potential):
                    potential = torch.tensor(
                        potential, dtype=hamiltonian_terms.dtype, device=hamiltonian_terms.device)
                potential_energy = torch.dot(potential, hamiltonian_terms[0])
            else:
                potential_energy = torch.tensor(
                    0., dtype=hamiltonian_terms.dtype, device=hamiltonian_terms.device)
            total_energy = hopping_energy + interaction_energy + potential_energy
            if return_breakdown:
                return total_energy, hopping_energy, interaction_energy, potential_energy
            return total_energy

        # Numpy branch
        else:
            hopping_energy = -self.t * np.sum(hamiltonian_terms[1])
            interaction_energy = self.u * np.sum(hamiltonian_terms[2])
            if potential is not None:
                potential_energy = np.dot(potential, hamiltonian_terms[0])
            else:
                potential_energy = 0.
            total_energy = hopping_energy + interaction_energy + potential_energy
            if return_breakdown:
                return total_energy, hopping_energy, interaction_energy, potential_energy
            else:
                return total_energy

    def dftio(self, state):
        """
        Returns DFT input (density expectation value array) and output (homogeneous energy
        expectation value) on a given state.
        """
        state = self._check_and_project_state(state)
        return (self.density_expectation(state), self.homogeneous_energy_expectation(state))

    def rdmftio(self, state):
        """
        Returns DFT input (density expectation value array) and output
        (homogeneous energy expectation value) on a given state
        """
        state = self._check_and_project_state(state)
        return (self.one_rdm_expectation(state), self.interaction_energy_expectation(state))

    # ******* Hamiltonian for system with potential *******

    def hamiltonian(self, potential=None) -> openfermion.FermionOperator:
        """
        Returns the Hamiltonian operator for the system with a given potential.

        Args:
            potential (ArrayLike, optional): Potential on each site. If None, returns the homogeneous Hamiltonian.

        Returns:
            openfermion.FermionOperator: The Hamiltonian operator.
        """
        if potential is None:
            return self.homogeneous_hamiltonian
        else:
            if len(potential) != self.n_sites:
                raise ValueError("potential has the wrong lenght")
            return self.homogeneous_hamiltonian + np.dot(potential, density_operators(self.n_sites, self.spin_convention))

    def block_hamiltonian(self, potential=None):
        """
        Returns the Hamiltonian operator for the system with a given potential, projected to the symmetry block.

        Args:
            potential (ArrayLike, optional): Potential on each site. If None, returns the block homogeneous Hamiltonian.

        Returns:
            ArrayLike: The block Hamiltonian operator.
        """
        if potential is None:
            return self.block_homogeneous_hamiltonian
        else:
            if len(potential) != self.n_sites:
                raise ValueError("potential has the wrong lenght")
            return self.block_homogeneous_hamiltonian + np.tensordot(
                potential, self.block_density_operators, axes=([0], [0])
            )

    # ******* Exact (sparse) diagonalizartion *******

    def ground_energy_and_state(self, potential: Optional[ArrayLike] = None):
        """
        Args:
            potential (ArrayLike, optional): chemical potential on each site.

        Returns:
            float: Ground energy
            ArrayLike: Ground state
        """
        ham = self.block_hamiltonian(potential)
        if self.total_spin_penalty > 0:
            ham += self.block_total_spin_penalty
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
        gs_energy = eigvals[0]
        gstate = eigvecs[:, 0]
        if self.total_spin_penalty > 0:
            self.check_spin_singlet(gstate)
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

    # ******* U(1) gauge fixing *******

    def alternating_spin_basis_state_block_idx(self):
        """
        Returns the block index of the basis state with alternating spins.

        The basis state is defined as follows:
        - For the "up_then_down" convention, the state is |...1010 ...0101>.
        - For the "interleaved" convention, the state is |... 10 01 10 01>.
        """
        idx = 0
        n_particles_or_holes = min(
            self.n_particles, 2 * self.n_sites - self.n_particles)

        if self.spin_convention == "up_then_down":
            for i in range(n_particles_or_holes):
                if i % 2 == 0:
                    idx += 2**i
                else:
                    idx += 2 ** (i + self.n_sites)
        else:
            for i in range(n_particles_or_holes):
                if i % 2 == 0:
                    idx += 2 ** (2 * i)
                else:
                    idx += 2 ** (2 * i + 1)
        if self.n_particles > n_particles_or_holes:
            idx = 2 ** (2 * self.n_sites) - 1 - idx

        block_idx = np.argwhere(self.block_indices == idx)[0, 0]
        return block_idx

    def fix_gauge(self, state, check_real=True):
        """
        Fix the U(1) gauge of the state to have positive overlap with the alternating-spin state.

        The gauge is fixed by multiplying the state by a phase factor.
        The phase factor is chosen such that the overlap with the alternating-spin state is real
        and positive.
        """
        block_idx = self.alternating_spin_basis_state_block_idx()
        phase = np.angle(state[block_idx])
        gauge_fixed_state = state * np.exp(-1j * phase)
        if check_real:
            if not np.allclose(gauge_fixed_state.imag, 0):
                raise ValueError(
                    "Gauge fixing did not result in a real state.")
            return gauge_fixed_state.real
        else:
            return gauge_fixed_state
