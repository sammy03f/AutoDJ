"""Check filter stages independently and reproduce the existing sequence baseline."""

import unittest

import numpy as np

from evals import onset_diagnostics as diagnostics
from evals import onset_sequences as sequences


class OnsetDiagnosticsTests(unittest.TestCase):
    def test_threshold_and_spacing_remove_different_candidates(self):
        times = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
        result = diagnostics.diagnose([0, 2, 0, 5, 0, 4, 0], times, [3] * 7, [0.02, 0.06, 0.1])
        self.assertEqual(result["threshold_rejected_frames"], [1])
        self.assertEqual(result["spacing_suppressed_frames"], [5])
        self.assertEqual(result["frame_indices"]["threshold_and_spacing"], [3])
        self.assertEqual(result["match_losses"], {"unmatched_with_all_peaks": 0,
                         "additional_misses_after_threshold": 1, "additional_misses_after_spacing": 1})
        self.assertTrue(diagnostics.check_case(result))

    def test_spacing_only_and_full_survivors_need_not_be_nested(self):
        # Removing the strongest peak first lets a formerly suppressed neighbor survive.
        result = diagnostics.diagnose([0, 5, 0, 4, 0], [0, 0.02, 0.04, 0.06, 0.08],
                                      [0, 6, 0, 0, 0], [0.02, 0.06])
        self.assertEqual(result["frame_indices"]["spacing_only"], [1])
        self.assertEqual(result["frame_indices"]["threshold_and_spacing"], [3])
        self.assertTrue(diagnostics.check_case(result))

    def test_no_filters_still_apply_plateau_and_boundary_rules(self):
        result = diagnostics.diagnose([5, 0, 4, 4, 0, 3], list(range(6)), [0] * 6, [2])
        self.assertEqual(result["frame_indices"]["all_peaks"], [2])
        self.assertTrue(diagnostics.check_case(result))

    def test_all_peaks_can_already_miss_events(self):
        result = diagnostics.diagnose([0, 0, 0], [0, 0.1, 0.2], [1] * 3, [0.1])
        self.assertEqual(result["match_losses"]["unmatched_with_all_peaks"], 1)
        self.assertEqual(result["match_losses"]["additional_misses_after_threshold"], 0)
        self.assertTrue(diagnostics.check_case(result))

    def test_controls_count_extras_without_inventing_match_losses(self):
        result = diagnostics.diagnose([0, 2, 0, 3, 0], [0, 1, 2, 3, 4], [2] * 5, [])
        self.assertEqual(result["metrics"]["all_peaks"]["false_count"], 2)
        self.assertEqual(result["metrics"]["threshold_and_spacing"]["false_count"], 1)
        self.assertEqual(sum(result["match_losses"].values()), 0)

    def test_trace_partitions_candidates_and_preserves_inputs(self):
        values = np.array([0., 1, 0, 5, 0])
        times = np.arange(5, dtype=float)
        thresholds = np.full(5, 2.)
        before = (values.copy(), times.copy(), thresholds.copy())
        result = diagnostics.diagnose(values, times, thresholds, [1, 3])
        self.assertEqual([p["status"] for p in result["peak_trace"]], ["threshold_rejected", "retained"])
        for original, current in zip(before, (values, times, thresholds)):
            np.testing.assert_array_equal(original, current)

    def test_full_branch_matches_sequence_evaluator(self):
        for args in (("sharp_burst", 120, 4, "steady", 1, 0, 701),
                     ("click", 180, 4, "alternating", 0.1, 0.2, 809)):
            with self.subTest(args=args):
                original = sequences.evaluate_case(*args)
                result = diagnostics.evaluate_case(*args)
                self.assertEqual(result["input_sha256"], original["input_sha256"])
                self.assertEqual(result["metrics"]["threshold_and_spacing"], original["metrics"]["selected_adaptive"])
                self.assertEqual(result["frame_indices"]["threshold_and_spacing"], original["frame_indices"]["selected_adaptive"])
                self.assertTrue(diagnostics.check_case(result))

    def test_summary_sums_losses_without_averaging_case_rates(self):
        case = diagnostics.diagnose([0, 2, 0], [0, 1, 2], [3] * 3, [1])
        case["duration_seconds"] = 2
        summary = diagnostics.summarize([case, case])
        self.assertEqual(summary["match_losses"]["additional_misses_after_threshold"], 2)
        self.assertEqual(summary["methods"]["threshold_and_spacing"]["missed_count"], 2)
        self.assertIsNone(diagnostics.summarize([])["methods"]["all_peaks"]["recall"])
        case["candidate_stages"]["survivors"] = 5
        self.assertFalse(diagnostics.check_case(case))
