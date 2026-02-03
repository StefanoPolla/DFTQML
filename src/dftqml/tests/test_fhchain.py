"""
Unit tests for the :mod:`dftqml.fhchain` module.
"""

import numpy as np
import pytest

from dftqml.fhchain import (
    n_and_sz_indices,
    projector_block_and_singlet,
    projector_from_block_to_singlet,
    FermiHubbardChain,
    s_squared_block_operator,
)


def norm(a):
    return np.sqrt(np.sum(np.abs(a) ** 2))


def test_n_and_sz_indices():
    result_2_2 = n_and_sz_indices(2, 2)
    expected_2_2 = np.array([4 + 1, 4 + 2, 8 + 1, 8 + 2])
    assert np.all(result_2_2 == expected_2_2)


def test_projector_from_block_to_singlet():
    indices_2_2 = n_and_sz_indices(2, 2)
    result_2 = projector_from_block_to_singlet(2, indices_2_2).toarray()

    for row in result_2:
        assert np.isclose(norm(row), 1)

    expected_2 = np.array(
        [
            [1, 0, 0, 0],
            [0, -1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, 0, 0, 1],
        ]
    )
    assert np.allclose(result_2, expected_2)


def test_projector_block_and_singlet():
    result_2 = projector_block_and_singlet(2, 2).toarray()

    for row in result_2:
        assert np.isclose(norm(row), 1)


@pytest.fixture
def fhchain_system():
    sys = FermiHubbardChain(4, 2, 4)
    potential = [-0.39739741, -0.72495679, -0.44073975, -0.03348764]
    return sys, potential


def test_dftio_uniform(fhchain_system):
    sys, _ = fhchain_system
    density, energy = sys.ground_state_dftio(None)
    expected_density = np.array([0.5, 0.5, 0.5, 0.5])
    expected_energy = -3.4185507188738455
    assert np.allclose(density, expected_density)
    assert np.isclose(energy, expected_energy)


def test_dftio_with_potential(fhchain_system):
    sys, potential = fhchain_system
    density, energy = sys.ground_state_dftio(potential)
    expected_density = np.array([0.49676045, 0.58915719, 0.50880766, 0.40527469])
    expected_energy = -3.3867869516230593
    assert np.allclose(density, expected_density)
    assert np.isclose(energy, expected_energy)


def test_alternating_spin_basis_state_block_idx(fhchain_system):
    # Test the alternating spin basis state for a (4,2) up_then_down system
    sys, _ = fhchain_system
    block_idx = sys.alternating_spin_basis_state_block_idx()
    full_idx = sys.block_indices[block_idx]
    expected_idx = int("00100001", 2)
    assert full_idx == expected_idx

    # Test the alternating spin basis state for a (4,4) interleaved system
    sys = FermiHubbardChain(4, 4, 4, spin_convention="interleaved")
    block_idx = sys.alternating_spin_basis_state_block_idx()
    full_idx = sys.block_indices[block_idx]
    expected_idx = int("10011001", 2)
    assert full_idx == expected_idx


def test_singlet_penalty(fhchain_system):
    # Test the singlet penalty for a (4,2) up_then_down system
    sys, potential = fhchain_system
    energy, state = sys.ground_energy_and_state(potential=potential)
    s2_block = s_squared_block_operator(
        sys.n_sites, sys.block_indices, sys.spin_convention == "up_then_down"
    )
    assert np.isclose(np.sum(np.abs(s2_block @ state) ** 2), 0)


def test_n_particles_validator():
    # Test valid n_particles
    chain = FermiHubbardChain(n_sites=4, n_particles=2, u=1.0, t=1.0)
    assert chain.n_particles == 2

    # Test n_particles <= 0
    with pytest.raises(ValueError, match="n_particles must be positive"):
        FermiHubbardChain(n_sites=4, n_particles=0, u=1.0, t=1.0)

    # Test n_particles >= 2 * n_sites
    with pytest.raises(ValueError, match="n_particles must be less than 2 times n_sites"):
        FermiHubbardChain(n_sites=4, n_particles=8, u=1.0, t=1.0)

    # Test n_particles not even
    with pytest.raises(ValueError, match="n_particles must be even in order to allow for a singlet state"):
        FermiHubbardChain(n_sites=4, n_particles=3, u=1.0, t=1.0)


def test_spin_convention_validator():
    # Test valid spin_convention
    chain = FermiHubbardChain(n_sites=4, n_particles=2, u=1.0, t=1.0, spin_convention="up_then_down")
    assert chain.spin_convention == "up_then_down"

    chain = FermiHubbardChain(n_sites=4, n_particles=2, u=1.0, t=1.0, spin_convention="interleaved")
    assert chain.spin_convention == "interleaved"

    # Test invalid spin_convention
    with pytest.raises(ValueError, match="spin_convention must be 'up_then_down', or 'interleaved'"):
        FermiHubbardChain(n_sites=4, n_particles=2, u=1.0, t=1.0, spin_convention="invalid")


def test_boundary_conditions_validator():
    # Test valid boundary_conditions
    chain = FermiHubbardChain(n_sites=4, n_particles=2, u=1.0, t=1.0, boundary_conditions="periodic")
    assert chain.boundary_conditions == "periodic"

    chain = FermiHubbardChain(n_sites=4, n_particles=2, u=1.0, t=1.0, boundary_conditions="open")
    assert chain.boundary_conditions == "open"

    # Test invalid boundary_conditions
    with pytest.raises(ValueError, match="boundary_conditions must be 'periodic', or 'open'"):
        FermiHubbardChain(n_sites=4, n_particles=2, u=1.0, t=1.0, boundary_conditions="invalid")


def test_default_values():
    # Test default values for u, t, spin_convention, and boundary_conditions
    chain = FermiHubbardChain(n_sites=4, n_particles=2)
    assert chain.u == 0.0
    assert chain.t == 1.0
    assert chain.spin_convention == "up_then_down"
    assert chain.boundary_conditions == "periodic"
    assert chain.total_spin_penalty == 10.0


def test_energy_from_two_rdm(fhchain_system):
    sys, potential = fhchain_system
    energy, state = sys.ground_energy_and_state(potential=potential)
    state = sys.fix_gauge(state)
    # state_full = np.zeros((2**(2*sys.n_sites)), dtype=state.dtype)
    # state_full[sys.block_indices] = state
    two_rdm = sys.two_rdm_expectation(state)
    energy_from_two_rdm = sys.ground_energy_from_rdm(two_rdm, potential)
    assert np.isclose(energy, energy_from_two_rdm)

def test_fix_gauge(fhchain_system):
    sys, potential = fhchain_system
    energy, state = sys.ground_energy_and_state(potential=potential)
    state_fixed = sys.fix_gauge(state)
    assert np.isclose(np.sum(np.abs(state_fixed) ** 2), 1.0)
    assert np.all(np.isreal(state_fixed))
