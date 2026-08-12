#!/usr/bin/env python3
"""Dependency-light checks for the strict Cartesian benchmark masks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "ram" / "adapters" / "cartesian_masks.py"
SPEC = importlib.util.spec_from_file_location("cartesian_masks", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MASKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MASKS)


class CartesianMaskTest(unittest.TestCase):
    SETTINGS = ((8, 13, 40), (16, 6, 20), (24, 3, 13))

    def test_equispaced_budgets_and_acs(self) -> None:
        for acceleration, center_columns, expected in self.SETTINGS:
            with self.subTest(acceleration=acceleration):
                mask = MASKS.exact_equispaced_mask(320, acceleration, center_columns)
                self.assertEqual(int(mask.sum()), expected)
                center = MASKS.center_indices(320, center_columns)
                self.assertTrue(np.all(mask[center] == 1))

    def test_variable_density_is_reproducible_and_seeded(self) -> None:
        first = MASKS.polynomial_variable_density_mask(320, 16, 6, 123, 4.0)
        repeated = MASKS.polynomial_variable_density_mask(320, 16, 6, 123, 4.0)
        different = MASKS.polynomial_variable_density_mask(320, 16, 6, 124, 4.0)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))

    def test_variable_density_budgets_and_acs(self) -> None:
        for acceleration, center_columns, expected in self.SETTINGS:
            with self.subTest(acceleration=acceleration):
                mask = MASKS.polynomial_variable_density_mask(
                    320, acceleration, center_columns, 123, 4.0
                )
                self.assertEqual(int(mask.sum()), expected)
                center = MASKS.center_indices(320, center_columns)
                self.assertTrue(np.all(mask[center] == 1))

    def test_polynomial_density_favors_center_statistically(self) -> None:
        distances = []
        uniform_expectation = np.mean(np.abs(np.arange(320) - 159.5))
        center = set(MASKS.center_indices(320, 3).tolist())
        for seed in range(200):
            mask = MASKS.polynomial_variable_density_mask(320, 24, 3, seed, 4.0)
            outer = [index for index in np.flatnonzero(mask) if index not in center]
            distances.extend(abs(index - 159.5) for index in outer)
        self.assertLess(float(np.mean(distances)), 0.55 * uniform_expectation)


if __name__ == "__main__":
    unittest.main()
