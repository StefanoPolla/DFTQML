"""
Unit tests for the :mod:`dftqml.fhchain` module.
"""

import numpy as np
import openfermion as of
import pytest

from dftqml.fhchain import FermiHubbardChain
from dftqml.sampling import DFTIOSampler


def norm(a):
    return np.sqrt(np.sum(np.abs(a) ** 2))


def expectation(state, observable):
    return np.real(state.conj() @ observable @ state)


@pytest.fixture
def setup_sampler():
    """Fixture to set up the sampler and related parameters."""
    system = FermiHubbardChain(4, 2, 4)
    rng = np.random.default_rng(seed=42)
    potential = [-0.39739741, -0.72495679, -0.44073975, -0.03348764]
    sampler = DFTIOSampler(system, potential, rng=rng)
    nshots = int(2e5)
    return sampler, nshots


def test_dftio(setup_sampler):
    sampler, nshots = setup_sampler
    target_density, target_energy = sampler.system.dftio(sampler.ground_state)
    sampled_density, sampled_energy = sampler.dftio(nshots, nshots)

    density_relative_errors = np.abs(sampled_density - target_density)
    for density_relative_error in density_relative_errors:
        assert density_relative_error < 1e-2

    energy_relative_error = np.abs(sampled_energy - target_energy)
    assert energy_relative_error < 1e-2


def test_coulomb_sampling(setup_sampler):
    sampler, nshots = setup_sampler
    coulomb_fop = of.transforms.reorder(
        of.fermi_hubbard(sampler.system.n_sites, 1, 0, sampler.system.u),
        of.up_then_down,
    )
    coulomb_block = sampler.system._symop_to_block(coulomb_fop)
    target_coulomb = expectation(sampler.ground_state, coulomb_block)

    cb_samples = sampler.sample_cb_states(nshots)
    sampled_coulomb = sampler.average_coulomb_from_cb_samples(cb_samples)

    coulomb_relative_error = np.abs(sampled_coulomb - target_coulomb) / target_coulomb
    assert coulomb_relative_error < 1e-2
