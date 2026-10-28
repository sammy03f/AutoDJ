"""Known energy values and signal-based timing behavior, including fallbacks."""

import unittest

import numpy as np

from autodj.analysis import refine_onset_times, short_time_energy


class EnergyTests(unittest.TestCase):
    def test_trailing_mean_square_has_no_lookahead(self):
        np.testing.assert_allclose(short_time_energy([0, 2, -2, 0, 0], window_size=2),
                                   [0, 2, 4, 2, 0])

    def test_short_empty_and_generator_inputs(self):
        np.testing.assert_allclose(short_time_energy((x for x in [2, 0]), window_size=4), [1, 1])
        self.assertEqual(short_time_energy([], window_size=4).size, 0)

    def test_energy_scales_with_gain_squared_without_mutation(self):
        samples = np.array([1., -2., 3.])
        original = samples.copy()
        energy = short_time_energy(samples, window_size=2)
        np.testing.assert_allclose(short_time_energy(3 * samples, window_size=2), 9 * energy)
        np.testing.assert_array_equal(samples, original)

    def test_invalid_inputs(self):
        for samples in ([np.nan], [np.inf], [[1]], [1j], [1e308]):
            with self.subTest(samples=samples), self.assertRaises((ValueError, TypeError)):
                short_time_energy(samples, window_size=2)
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises((ValueError, TypeError)):
                short_time_energy([1], window_size=size)


class RefinementTests(unittest.TestCase):
    def refine(self, energy, times, **kwargs):
        return refine_onset_times(energy, 1000, times, search_radius=0.1, **kwargs)

    def test_click_can_move_early_or_late_candidate_to_sample(self):
        samples = np.zeros(400)
        samples[153] = 1
        energy = short_time_energy(samples, window_size=4)
        for time in (0.12, 0.18):
            np.testing.assert_allclose(self.refine(energy, [time]), [0.153])

    def test_threshold_is_relative_to_local_floor(self):
        # Level = 2 + .25 * (10 - 2) = 4; strict crossing occurs at index 3.
        np.testing.assert_allclose(self.refine([2, 2, 4, 6, 10, 10], [0.004],
                                             relative_threshold=0.25), [0.003])

    def test_slow_ramp_crossing_is_later_than_first_nonzero(self):
        energy = np.r_[np.zeros(10), np.arange(1, 11), np.zeros(10)]
        np.testing.assert_allclose(self.refine(energy, [0.018], relative_threshold=0.2), [0.012])

    def test_flat_falling_and_no_observed_rise_retain_candidate(self):
        for energy in (np.zeros(20), np.ones(20), np.arange(20, 0, -1)):
            np.testing.assert_array_equal(self.refine(energy, [0.0055]), [0.0055])

    def test_search_is_bounded_and_edges_are_clipped(self):
        energy = np.zeros(100)
        energy[2] = 1
        energy[-1] = 2
        np.testing.assert_allclose(refine_onset_times(energy, 1000, [0.001, 0.05, 0.098],
                                                    search_radius=0.004), [0.002, 0.05, 0.099])

    def test_preserves_correspondence_duplicates_and_inputs(self):
        energy = np.zeros(100)
        energy[50] = 1
        times = np.array([0.04, 0.06])
        original = energy.copy()
        result = self.refine(energy, times)
        np.testing.assert_allclose(result, [0.05, 0.05])
        np.testing.assert_array_equal(times, [0.04, 0.06])
        np.testing.assert_array_equal(energy, original)
        self.assertFalse(np.shares_memory(result, times))

    def test_empty_candidates_do_not_create_events(self):
        self.assertEqual(self.refine([], []).size, 0)
        self.assertEqual(self.refine([0, 1, 0], []).size, 0)

    def test_separated_events_refine_independently(self):
        samples = np.zeros(500)
        samples[[100, 350]] = [1, 0.4]
        energy = short_time_energy(samples, window_size=8)
        np.testing.assert_allclose(self.refine(energy, [0.12, 0.33]), [0.1, 0.35])

    def test_earlier_unrelated_rise_is_a_known_limitation(self):
        energy = np.zeros(100)
        energy[20:30] = 0.2
        energy[60:70] = 1
        # With both rises inside the search, the heuristic chooses the first.
        np.testing.assert_allclose(self.refine(energy, [0.06]), [0.02])

    def test_invalid_inputs(self):
        for energy in ([-1], [np.nan], [[0]], [1j]):
            with self.subTest(energy=energy), self.assertRaises((ValueError, TypeError)):
                self.refine(energy, [])
        for times in ([np.nan], [-1], [0.02], [0.005, 0.005], [0.01, 0.005], [[0]], [1j]):
            with self.subTest(times=times), self.assertRaises((ValueError, TypeError)):
                self.refine(np.zeros(20), times)
        for key, values in {"sample_rate": [0, -1, np.inf],
                            "search_radius": [0, -1, np.nan],
                            "relative_threshold": [0, 1, -1, np.nan]}.items():
            for value in values:
                args = dict(sample_rate=1000, search_radius=0.01, relative_threshold=0.1)
                args[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    refine_onset_times([0], candidate_times=[], **args)
