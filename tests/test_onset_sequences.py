"""Scheduled fixtures and independently checked one-to-one event scoring."""

from itertools import combinations
import unittest

import numpy as np

from evals.onset_sequences import (
    HIT_COUNT, METHODS, RATE, evaluate_case, make_sequence, score_sequence, summarize,
)


class SequenceFixtureTests(unittest.TestCase):
    def test_known_click_schedule_and_alternating_amplitudes(self):
        samples, starts, amplitudes = make_sequence("click", 120, 1, "alternating", 0.1, 0, 701)
        np.testing.assert_array_equal(starts, 512 + np.arange(8) * 1024)
        np.testing.assert_allclose(amplitudes, [0.1, 0.035] * 4)
        np.testing.assert_array_equal(np.flatnonzero(samples), starts)
        np.testing.assert_array_equal(samples[starts], amplitudes)
        self.assertEqual(len(samples), starts[-1] + RATE // 2)

    def test_absolute_rounding_does_not_accumulate_drift(self):
        _, starts, _ = make_sequence("click", 180, 4, "steady", 1, 0, 701)
        ideal = 512 + np.arange(HIT_COUNT) * RATE / 12
        self.assertLessEqual(np.max(np.abs(starts - ideal)), 0.5)
        self.assertLess(np.max(np.diff(starts)) / RATE, 0.1)

    def test_noise_repeats_and_is_independent_of_gain(self):
        quiet, starts, _ = make_sequence("sharp_burst", 180, 1, "steady", 0.1, 0.05, 809)
        repeat, _, _ = make_sequence("sharp_burst", 180, 1, "steady", 0.1, 0.05, 809)
        loud, _, _ = make_sequence("sharp_burst", 180, 1, "steady", 1, 0.05, 809)
        clean, _, _ = make_sequence("sharp_burst", 180, 1, "steady", 1, 0, 809)
        np.testing.assert_array_equal(quiet, repeat)
        np.testing.assert_allclose(loud - quiet, 0.9 * clean, atol=1e-15)
        self.assertEqual(np.flatnonzero(clean)[0], starts[0])

    def test_control_has_no_truth_and_counts_every_candidate_as_false(self):
        case = evaluate_case("click", 60, 1, "steady", 0, 0.2, 701, control=True)
        self.assertEqual(case["known_times"], [])
        self.assertGreater(case["metrics"]["fixed"]["candidate_count"], 0)
        for method in METHODS:
            m = case["metrics"][method]
            self.assertEqual(m["false_count"], m["candidate_count"])
            self.assertEqual(m["matched_count"], 0)

    def test_clean_quiet_sequence_and_spacing_stress_classification(self):
        case = evaluate_case("sharp_burst", 60, 1, "steady", 0.1, 0, 701)
        self.assertFalse(case["spacing_stress"])
        self.assertEqual(case["metrics"]["selected_adaptive"]["matched_count"], HIT_COUNT)
        self.assertEqual(case["metrics"]["selected_adaptive"]["false_count"], 0)
        stress = evaluate_case("sharp_burst", 180, 4, "steady", 1, 0, 701)
        self.assertTrue(stress["spacing_stress"])
        for times in stress["times"].values():
            self.assertTrue(np.all(np.diff(times) >= 0.1 - 1e-12))


class SequenceMatchingTests(unittest.TestCase):
    def test_empty_missed_and_false(self):
        self.assertEqual(score_sequence([], [0.2, 0.4])["missed_count"], 2)
        self.assertEqual(score_sequence([0.2, 0.4], [])["false_count"], 2)
        self.assertEqual(score_sequence([], [])["matches"], [])

    def test_maximum_cardinality_beats_nearest_first_greedy(self):
        # First truth's nearest candidate is .14, but reserving it for .18
        # allows both truths to match.
        result = score_sequence([0.055, 0.14], [0.1, 0.18])
        self.assertEqual(result["matched_count"], 2)
        self.assertEqual([p["candidate_index"] for p in result["matches"]], [0, 1])

    def test_minimum_error_breaks_equal_cardinality(self):
        result = score_sequence([0.18, 0.201, 0.25], [0.2])
        self.assertEqual(result["matches"][0]["candidate_index"], 1)
        self.assertEqual(result["false_candidate_indices"], [0, 2])

    def test_duplicate_candidates_cannot_double_count_truth(self):
        result = score_sequence([0.2, 0.2], [0.2])
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["false_count"], 1)
        self.assertEqual(result["matches"][0]["candidate_index"], 0)

    def test_inclusive_tolerance_and_outside_errors(self):
        self.assertEqual(score_sequence([0.25], [0.2])["matched_count"], 1)
        result = score_sequence([0.251], [0.2])
        self.assertEqual(result["missed_truth_indices"], [0])
        self.assertEqual(result["false_candidate_indices"], [0])

    def test_matches_brute_force_oracle_on_short_sequences(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            truth = sorted(rng.choice(np.arange(30), 3, replace=False) / 100)
            times = sorted(rng.choice(np.arange(30), 4, replace=False) / 100)
            best = (0, 0.)
            for count in range(1, 4):
                for ti in combinations(range(3), count):
                    for ci in combinations(range(4), count):
                        errors = [abs(times[j] - truth[i]) for i, j in zip(ti, ci)]
                        if all(error <= 0.05 + 1e-12 for error in errors):
                            best = max(best, (count, -sum(errors)))
            actual = score_sequence(times, truth)
            self.assertEqual(actual["matched_count"], best[0])
            self.assertAlmostEqual(sum(abs(p["signed_error_seconds"]) for p in actual["matches"]), -best[1])

    def test_invalid_times(self):
        for times, truths in (([0.2, 0.1], []), ([], [0.1, 0.1]), ([np.nan], []),
                              ([-1], []), ([], [np.inf]), ([[0.1]], []), ([1j], [])):
            with self.subTest(times=times, truths=truths), self.assertRaises(ValueError):
                score_sequence(times, truths)
        for tolerance in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                score_sequence([], [], tolerance=tolerance)

    def test_summary_uses_hit_counts_and_matched_errors_only(self):
        cases = [{"duration_seconds": 1,
                  "metrics": {method: score_sequence(times, truth) for method in METHODS}}
                 for times, truth in (([0.21, 0.7], [0.2, 0.4]), ([], [0.2]), ([0.3], []))]
        m = summarize(cases)["methods"]["fixed"]
        self.assertEqual(m["event_count"], 3)
        self.assertEqual(m["matched_count"], 1)
        self.assertEqual(m["false_count"], 2)
        self.assertEqual(m["precision"], 1 / 3)
        self.assertEqual(m["recall"], 1 / 3)
        self.assertEqual(m["timing_match_count"], 1)
        self.assertAlmostEqual(m["mean_absolute_error_seconds"], 0.01)
        self.assertIsNone(summarize([])["methods"]["fixed"]["mean_absolute_error_seconds"])
