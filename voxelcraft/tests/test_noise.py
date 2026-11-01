"""Tests for the Perlin noise implementation."""

from __future__ import annotations

import unittest

import numpy as np

from voxelcraft.noise import PerlinNoise, hash01, smoothstep


class PerlinNoiseTests(unittest.TestCase):
    def setUp(self):
        self.n = PerlinNoise(1234)

    def test_scalar_and_array_inputs(self):
        scalar = self.n.noise2(1.5, 2.5)
        array = self.n.noise2(np.array([1.5]), np.array([2.5]))
        self.assertEqual(np.asarray(scalar).shape, ())
        self.assertEqual(np.asarray(array).shape, (1,))
        self.assertAlmostEqual(float(np.asarray(scalar).reshape(-1)[0]), float(array[0]), places=6)

    def test_deterministic_for_same_seed(self):
        a = PerlinNoise(7)
        b = PerlinNoise(7)
        gx, gz = np.meshgrid(np.linspace(0, 20, 25), np.linspace(0, 20, 25))
        np.testing.assert_array_equal(a.noise2(gx, gz), b.noise2(gx, gz))
        np.testing.assert_array_equal(a.fbm3(gx, gz, gx), b.fbm3(gx, gz, gx))

    def test_different_seeds_differ(self):
        a = PerlinNoise(1)
        b = PerlinNoise(2)
        x = np.linspace(0, 10, 40)
        self.assertFalse(np.allclose(a.noise2(x, x), b.noise2(x, x)))

    def test_output_range(self):
        gx, gz = np.meshgrid(np.linspace(-50, 50, 60), np.linspace(-50, 50, 60))
        n2 = self.n.noise2(gx, gz)
        self.assertGreaterEqual(n2.min(), -1.0)
        self.assertLessEqual(n2.max(), 1.0)
        n3 = self.n.noise3(gx, gz, gx * 0.5)
        self.assertGreaterEqual(n3.min(), -1.0)
        self.assertLessEqual(n3.max(), 1.0)
        self.assertGreater(n2.std(), 0.05)       # it actually varies
        self.assertGreater(n3.std(), 0.05)

    def test_fbm_normalised(self):
        x = np.linspace(0, 30, 200)
        for octaves in (1, 3, 5):
            f = self.n.fbm2(x, x * 0.5, octaves=octaves)
            self.assertLessEqual(np.abs(f).max(), 1.0)
            self.assertGreater(f.std(), 0.01)

    def test_ridged_range(self):
        x = np.linspace(0, 40, 150)
        r = self.n.ridged2(x, x * 0.7, octaves=4)
        self.assertGreaterEqual(r.min(), 0.0)
        self.assertLessEqual(r.max(), 1.0)
        self.assertGreater(r.mean(), 0.25)      # ridged noise is biased high

    def test_smoothness(self):
        """Noise must be continuous: a tiny step gives a tiny change."""
        a = self.n.noise2(3.0, 4.0)
        b = self.n.noise2(3.0 + 1e-4, 4.0)
        self.assertLess(abs(float(a) - float(b)), 1e-3)

    def test_lattice_points_are_zero(self):
        """Perlin noise vanishes on integer lattice points."""
        for x, y in ((0, 0), (3, 7), (-4, 2), (11, -5)):
            self.assertAlmostEqual(float(self.n.noise2(x, y)), 0.0, places=6)

    def test_3d_differs_from_2d(self):
        x = np.linspace(0, 10, 30)
        self.assertFalse(np.allclose(self.n.noise2(x, x), self.n.noise3(x, x, 0.37)))


class HashTests(unittest.TestCase):
    def test_range_and_determinism(self):
        a = hash01(np.arange(1000), seed=5)
        b = hash01(np.arange(1000), seed=5)
        np.testing.assert_array_equal(a, b)
        self.assertGreaterEqual(a.min(), 0.0)
        self.assertLess(a.max(), 1.0)

    def test_uniformish(self):
        values = hash01(np.arange(20000), np.arange(20000) * 3, seed=99)
        counts, _ = np.histogram(values, bins=10, range=(0, 1))
        expected = values.size / 10
        self.assertTrue(np.all(np.abs(counts - expected) < expected * 0.25))

    def test_seed_changes_result(self):
        self.assertFalse(np.allclose(hash01(np.arange(50), seed=1),
                                     hash01(np.arange(50), seed=2)))


class SmoothstepTests(unittest.TestCase):
    def test_edges(self):
        self.assertAlmostEqual(float(smoothstep(0.0, 1.0, -5.0)), 0.0)
        self.assertAlmostEqual(float(smoothstep(0.0, 1.0, 5.0)), 1.0)
        self.assertAlmostEqual(float(smoothstep(0.0, 1.0, 0.5)), 0.5)

    def test_monotonic(self):
        x = np.linspace(-1, 2, 100)
        y = smoothstep(0.0, 1.0, x)
        self.assertTrue(np.all(np.diff(y) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
