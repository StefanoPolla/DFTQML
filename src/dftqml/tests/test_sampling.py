"""
Unit tests for the :mod:`dftqml.fhchain` module.
"""

import unittest

import numpy as np
import openfermion as of

from dftqml.fhchain import FermiHubbardChain
from dftqml.sampling import DFTIOSampler


def norm(a):
    return np.sqrt(np.sum(np.abs(a) ** 2))


def expectation(state, observable):
    return np.real(state.conj() @ observable @ state)


class TestSampling(unittest.TestCase):
    def setUp(self):
        """this is run before each test"""
        system = FermiHubbardChain(4, 2, 4)
        rng = np.random.default_rng(seed=42)
        potential = [-0.39739741, -0.72495679, -0.44073975, -0.03348764]
        self.sampler = DFTIOSampler(system, potential, rng=rng)
        self.nshots = int(2e5)

    def test_dftio(self):
        target_density, target_energy = self.sampler.system.dftio(self.sampler.ground_state)
        sampled_density, sampled_energy = self.sampler.dftio(self.nshots, self.nshots)
        sampled_density, sampled_energy = self.sampler.dftio(self.nshots, self.nshots)

        density_relative_errors = np.abs(sampled_density - target_density)
        for density_relative_error in density_relative_errors:
            self.assertLess(density_relative_error, 1e-2)

        energy_relative_error = np.abs(sampled_energy - target_energy)
        self.assertLess(energy_relative_error, 1e-2)

    def test_coulomb_sampling(self):
        coulomb_fop = of.transforms.reorder(
            of.fermi_hubbard(self.sampler.system.n_sites, 1, 0, self.sampler.system.u),
            of.up_then_down,
        )
        coulomb_block = self.sampler.system._symop_to_block(coulomb_fop)
        target_coulomb = expectation(self.sampler.ground_state, coulomb_block)

        cb_samples = self.sampler.sample_cb_states(self.nshots)
        sampled_coulomb = self.sampler.average_coulomb_from_cb_samples(cb_samples)

        coulomb_relative_error = np.abs(sampled_coulomb - target_coulomb) / target_coulomb
        self.assertLess(coulomb_relative_error, 1e-2)
