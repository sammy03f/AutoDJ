"""Verify timing conventions and that missed cases stay in evaluation totals."""

import unittest

import numpy as np

from evals.onset_timing import evaluate_case, make_signal, summarize_cases


class OnsetTimingEvaluationTests(unittest.TestCase):
    def test_original_burst_center_corrects_known_offset(self):
        case = evaluate_case("sharp_burst", 128, 32, 0)
        self.assertEqual(case["frame_indices"], [14])
        self.assertEqual(case["errors_seconds"], {"frame_start": -0.03125, "frame_center": 0})
        self.assertTrue(case["candidate_selection_unchanged"])

    def test_shift_uses_frame_size_and_preserves_selection(self):
        for size, hop in ((64, 16), (256, 64)):
            with self.subTest(size=size):
                case = evaluate_case("click", size, hop, 3)
                self.assertTrue(case["candidate_selection_unchanged"])
                np.testing.assert_allclose(np.array(case["frame_center_times"]) - case["frame_start_times"],
                                           size / 4096, atol=1e-15)

    def test_misses_have_no_timing_error_instead_of_zero_error(self):
        case = evaluate_case("click", 64, 16, 0, threshold=1e6)
        self.assertEqual(case["candidate_count"], 0)
        self.assertIsNone(case["errors_seconds"])
        summary = summarize_cases([case])
        self.assertEqual(summary["no_candidate_cases"], 1)
        self.assertIsNone(summary["methods"]["frame_center"]["mean_absolute_error_seconds"])

    def test_summary_tracks_misses_extras_and_worse_results(self):
        cases = [
            {"candidate_count": 1, "errors_seconds": {"frame_start": -0.04, "frame_center": -0.01}},
            {"candidate_count": 1, "errors_seconds": {"frame_start": -0.01, "frame_center": 0.02}},
            {"candidate_count": 0, "errors_seconds": None},
            {"candidate_count": 2, "errors_seconds": None},
        ]
        summary = summarize_cases(cases)
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["paired_timing_cases"], 2)
        self.assertEqual(summary["center_better_cases"], 1)
        self.assertEqual(summary["center_worse_cases"], 1)
        self.assertEqual(summary["multiple_candidate_cases"], 1)
        self.assertEqual(summary["no_candidate_cases"], 1)
        self.assertAlmostEqual(summary["methods"]["frame_center"]["mean_absolute_error_seconds"], 0.015)
        self.assertAlmostEqual(summary["methods"]["frame_center"]["mean_signed_error_seconds"], 0.005)
        self.assertEqual(summary["within_tolerance"][2]["frame_center"], 2)
        self.assertEqual(summary["within_tolerance"][2]["denominator_all_cases"], 4)

    def test_fixture_onset_is_first_nonzero_sample(self):
        for kind in ("click", "sharp_burst", "slow_attack"):
            with self.subTest(kind=kind):
                samples = make_signal(kind, 519)
                self.assertEqual(np.flatnonzero(samples)[0], 519)
                self.assertEqual(len(samples), 1536)
        with self.assertRaises(ValueError):
            make_signal("unknown", 519)
