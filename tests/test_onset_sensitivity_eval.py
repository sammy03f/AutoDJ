"""Verify reproducible fixtures and honest accounting of misses and extras."""

import unittest

import numpy as np

from evals.onset_sensitivity import (
    METHODS, RATE, SAMPLE_COUNT, evaluate_case, make_noisy_signal, score_times, summarize,
)
from evals.onset_timing import make_signal


class SensitivityEvaluationTests(unittest.TestCase):
    def test_repeatable_noise_and_independent_gain(self):
        first, noise = make_noisy_signal("sharp_burst", 0.25, 0.05, 7, 8)
        repeated, repeated_noise = make_noisy_signal("sharp_burst", 0.25, 0.05, 7, 8)
        loud, loud_noise = make_noisy_signal("sharp_burst", 1, 0.05, 7, 8)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(noise, repeated_noise)
        np.testing.assert_array_equal(noise, loud_noise)
        np.testing.assert_allclose(loud - first, 0.75 * make_signal("sharp_burst", 520), atol=1e-15)
        self.assertFalse(np.array_equal(noise, make_noisy_signal(None, 0, 0.05, 19, 0)[1]))

    def test_clean_signal_and_noise_control(self):
        samples, noise = make_noisy_signal("click", 0.5, 0, 7, 31)
        self.assertEqual(np.flatnonzero(samples).tolist(), [543])
        self.assertEqual(samples[543], 0.5)
        self.assertFalse(noise.any())
        control, noise = make_noisy_signal(None, 0, 0.2, 7, 0)
        np.testing.assert_array_equal(control, noise)
        self.assertEqual(control.size, SAMPLE_COUNT)

    def test_empty_predictions_are_misses_not_zero_error(self):
        metrics = score_times([], 0.25)
        self.assertEqual(metrics["missed_count"], 1)
        self.assertEqual(metrics["false_count"], 0)
        self.assertIsNone(metrics["signed_error_seconds"])

    def test_nearest_match_duplicates_and_tolerance_boundary(self):
        metrics = score_times([0.29, 0.251, 0.251, 0.7], 0.25)
        self.assertEqual(metrics["matched_candidate_index"], 1)
        self.assertEqual(metrics["matched_count"], 1)
        self.assertEqual(metrics["false_count"], 3)
        self.assertAlmostEqual(metrics["signed_error_seconds"], 0.001)
        self.assertEqual(score_times([0.3], 0.25)["matched_count"], 1)
        outside = score_times([0.301], 0.25)
        self.assertEqual(outside["missed_count"], 1)
        self.assertEqual(outside["false_count"], 1)
        self.assertIsNone(outside["signed_error_seconds"])

    def test_controls_have_no_events_and_all_predictions_are_false(self):
        metrics = score_times([0.1, 0.3], None)
        self.assertEqual(metrics["event_count"], 0)
        self.assertEqual(metrics["missed_count"], 0)
        self.assertEqual(metrics["false_count"], 2)
        self.assertIsNone(metrics["signed_error_seconds"])

    def test_summary_aggregates_counts_not_case_precision(self):
        cases = [{"duration_seconds": 0.75,
                  "metrics": {method: score_times(times, truth) for method in METHODS}}
                 for times, truth in (([0.25, 0.6], 0.25), ([], 0.25), ([0.4], None))]
        summary = summarize(cases)["methods"]["frame_center"]
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["matched_count"], 1)
        self.assertEqual(summary["missed_count"], 1)
        self.assertEqual(summary["false_count"], 2)
        self.assertEqual(summary["precision"], 1 / 3)
        self.assertEqual(summary["recall"], 0.5)
        self.assertEqual(summary["false_detections_per_second"], 2 / 2.25)
        self.assertEqual(summary["timing_match_count"], 1)

    def test_undefined_metrics_are_null(self):
        case = {"duration_seconds": 0.75,
                "metrics": {method: score_times([], None) for method in METHODS}}
        for summary in (summarize([]), summarize([case])):
            for metrics in summary["methods"].values():
                for key in ("precision", "recall", "mean_absolute_error_seconds"):
                    self.assertIsNone(metrics[key])

    def test_baseline_detection_and_silence(self):
        case = evaluate_case("sharp_burst", 1, 0, 7, 0)
        self.assertEqual(case["frame_indices"], [14])
        self.assertEqual(case["known_start_seconds"], 512 / RATE)
        self.assertEqual(case["metrics"]["energy_refined"]["signed_error_seconds"], 0)
        self.assertEqual(case, evaluate_case("sharp_burst", 1, 0, 7, 0))
        control = evaluate_case(None, 0, 0, 7, 0)
        for method in METHODS:
            self.assertEqual(control["metrics"][method]["false_count"], 0)
            self.assertEqual(len(case["times"][method]), len(case["frame_indices"]))

    def test_quiet_misses_stay_in_totals(self):
        case = evaluate_case("sharp_burst", 0, 0, 7, 0)
        metrics = summarize([case])["methods"]["energy_refined"]
        self.assertEqual(metrics["event_count"], 1)
        self.assertEqual(metrics["missed_count"], 1)
        self.assertEqual(metrics["recall"], 0)
        self.assertIsNone(metrics["mean_absolute_error_seconds"])

    def test_noisy_control_records_false_detections_for_every_method(self):
        case = evaluate_case(None, 0, 0.2, 7, 0)
        self.assertGreater(len(case["frame_indices"]), 0)
        for method in METHODS:
            metrics = case["metrics"][method]
            self.assertEqual(metrics["false_count"], len(case["frame_indices"]))
            self.assertEqual(metrics["matched_count"], 0)
            self.assertIsNone(metrics["signed_error_seconds"])
