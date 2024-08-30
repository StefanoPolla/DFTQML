"""
Unit tests for the :mod:`dftqml.fhchain` module.
"""
import unittest
from dftqml.fhchain import (n_and_sz_indices,
                            projector_from_block_to_singlet,
                            projector_block_and_singlet)
import numpy as np


def norm(a):
    return np.sqrt(np.sum(np.abs(a)**2))


class TestFhchain(unittest.TestCase):

    def setUp(self):
        """this is run before each test
        """
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

        expected_2 = np.array([[1, 0, 0, 0],
                               [0, - 1 / np.sqrt(2), - 1 / np.sqrt(2), 0],
                               [0, 0, 0, 1],
                               ])
        self.assertTrue(np.allclose(result_2, expected_2))

    def test_projector_block_and_singlet(self):
        result_2 = projector_block_and_singlet(2, 2)

        for row in result_2:
            self.assertAlmostEqual(norm(row), 1)