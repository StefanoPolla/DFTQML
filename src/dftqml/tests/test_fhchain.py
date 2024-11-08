"""
Unit tests for the :mod:`dftqml.fhchain` module.
"""

import unittest

import numpy as np

from dftqml.fhchain import (
    n_and_sz_indices,
    projector_block_and_singlet,
    projector_from_block_to_singlet,
    FermiHubbardChain
)


def norm(a):
    return np.sqrt(np.sum(np.abs(a) ** 2))


class TestFuncions(unittest.TestCase):
    def setUp(self):
        """this is run before each test"""
        pass

    def test_n_and_sz_indices(self):
        result_2_2 = n_and_sz_indices(2, 2)
        expected_2_2 = np.array([4 + 1, 4 + 2, 8 + 1, 8 + 2])
        self.assertTrue(np.all(result_2_2 == expected_2_2))

    def test_projector_from_block_to_singlet(self):
        indices_2_2 = n_and_sz_indices(2, 2)
        result_2 = projector_from_block_to_singlet(2, indices_2_2)

        for row in result_2:
            self.assertAlmostEqual(norm(row), 1)

        expected_2 = np.array(
            [
                [1, 0, 0, 0],
                [0, -1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.allclose(result_2, expected_2))

    def test_projector_block_and_singlet(self):
        result_2 = projector_block_and_singlet(2, 2)

        for row in result_2:
            self.assertAlmostEqual(norm(row), 1)


class TestFHChain(unittest.TestCase):
    def setUp(self):
        """this is run before each test"""
        self.sys = FermiHubbardChain(4, 2, 4)
        self.potential = [-0.39739741, -0.72495679, -0.44073975, -0.03348764]
    
    def test_dftio_uniform(self):
        density, energy = self.sys.ground_state_dftio(None)
        expected_density = np.array([0.5, 0.5, 0.5, 0.5])
        expected_energy = -3.4185507188738455
        self.assertTrue(np.allclose(density, expected_density))
        self.assertAlmostEqual(energy, expected_energy)

    def test_dftio_with_potential(self):
        density, energy = self.sys.ground_state_dftio(self.potential)
        expected_density = np.array([0.49676045, 0.58915719, 0.50880766, 0.40527469])
        expected_energy = -3.3867869516230593
        self.assertTrue(np.allclose(density, expected_density))
        self.assertAlmostEqual(energy, expected_energy)

