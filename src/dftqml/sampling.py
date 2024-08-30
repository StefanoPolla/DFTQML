from .fhchain import FermiHubbardChain, n_and_sz_indices
import numpy as np
from typing import Tuple


def binary_expansion(vector_of_int, length):
    power = np.arange(length)[::-1]
    return vector_of_int[:, np.newaxis] // (2 ** power) % 2


class DFTIOSampler():
    def __init__(self, system: FermiHubbardChain, potential, *,
                 rng=np.random.default_rng()):
        self.system = system
        self.rng = rng
        _, self.ground_state = system.ground_energy_and_state(potential)

        # prepare and store CB states probabilities
        self.block_indices = n_and_sz_indices(system.nsites, system.nelec)
        unprojector = system.block_projector[:, self.block_indices].T.conj()
        unprojected_ground_state = unprojector @ self.ground_state
        self.diagonal_block_probabilites = np.abs(unprojected_ground_state)**2

        # prepate and store tunneling Hamiltonian eigenbasis probabilites
        noninteracting_system = FermiHubbardChain(system.nsites, system.nelec,
                                                  coulomb=0.)
        t_block_hamitonian = noninteracting_system.block_hamiltonian()
        t_eigvals, t_eigvecs = np.linalg.eigh(t_block_hamitonian)
        self.tunneling_eigprobs = np.abs(
            self.ground_state.conj() @ t_eigvecs) ** 2
        self.tunneling_eigenvalues = t_eigvals

    def sample_diagonal_indices(self, nshots):
        return self.rng.choice(self.block_indices, size=nshots,
                               p=self.diagonal_block_probabilites)

    def sample_cb_states(self, nshots):
        return binary_expansion(self.sample_diagonal_indices(nshots),
                                2 * self.system.nsites)

    # def density_from_idx_samples(self, idx_samples):
    #     print('s')
    #     spinful_density_count = np.zeros(2 * self.system.nsites)

    #     for i in range(2 * self.system.nsites):
    #         spinful_density_count[-i - 1] = np.sum(
    #             (idx_samples % (2 ** (i + 1))) // 2 ** (i))

    #     spinful_density = spinful_density_count / len(idx_samples)
    #     density = np.sum(spinful_density.reshape(2, -1), axis=0)
    #     return density

    def density_from_cb_samples(self, cb_samples):
        spinful_density = np.mean(cb_samples, axis=0)
        density = np.sum(spinful_density.reshape(2, -1), axis=0)
        return density

    def onsite_correlator_from_cb_samples(self, cb_samples):
        return np.mean(
            np.prod(cb_samples.reshape(len(cb_samples), 2, -1), axis=1),
            axis=0)

    def average_coulomb_from_cb_samples(self, cb_samples):
        return self.system.coulomb * np.sum(
            self.onsite_correlator_from_cb_samples(cb_samples))

    def sample_tunneling_eigenvalues(self, nshots):
        return self.rng.choice(a=self.tunneling_eigenvalues,
                               size=nshots,
                               p=self.tunneling_eigprobs)

    def average_sampled_tunneling(self, nshots):
        return np.mean(self.sample_tunneling_eigenvalues(nshots))

    def dftio(self, nshots_cb: int, nshots_tunneling: int
              ) -> Tuple[np.ndarray, float]:
        """Returns density and DFT energy (expectation value of kinetic ham. +
        Coulomb interaction) by sampling the true ground state.

        The density and coulomb terms are constructed sampling computational
        basis states with `nshots_cb` shots.
        The kinetic term is constructed sampling in the fourier basis (which
        diagonalizes the tunneling hamiltonian) with `nshots_tunneling` shots.

        Args:
            nshots_cb (int): number of samples in the computational basis, used
                to obtain the density andn the Coulomb interaction energy
            nshots_tunneling (int): number of samples in the fourier basis,
                used to obtain the kinetic energy.

        Returns:
            Tuple[np.ndarray, float]: sampled_density, sampled_dft_energy
        """
        cb_samples = self.sample_cb_states(nshots_cb)
        sampled_density = self.density_from_cb_samples(cb_samples)
        sampled_coulomb = self.average_coulomb_from_cb_samples(cb_samples)
        sampled_tunneling = self.average_sampled_tunneling(nshots_tunneling)
        sampled_dft_energy = sampled_coulomb + sampled_tunneling
        return sampled_density, sampled_dft_energy
