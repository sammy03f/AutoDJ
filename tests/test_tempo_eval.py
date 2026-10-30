"""Fixture schedules, honest octave scoring, and the end-to-end tempo path."""

import unittest

import numpy as np

from evals.tempo import BEAT_COUNT, RATE, evaluate_case, make_rhythm, score_bpm, summarize


class TempoEvaluationTests(unittest.TestCase):
    def test_schedules_and_alternating_amplitudes(self):
        samples, starts = make_rhythm("click", 120, pattern="alternating", gain=0.5, offset=11)
        np.testing.assert_array_equal(starts, RATE // 2 + 11 + np.arange(BEAT_COUNT) * RATE // 2)
        np.testing.assert_allclose(samples[starts], [0.5, 0.1] * (BEAT_COUNT // 2))
        self.assertEqual(len(np.flatnonzero(samples)), BEAT_COUNT)

    def test_subdivision_and_absolute_rounding(self):
        samples, starts = make_rhythm("click", 180, pattern="subdivided")
        self.assertEqual(len(starts), BEAT_COUNT * 2)
        ideal = RATE // 2 + np.arange(BEAT_COUNT * 2) * RATE / 6
        self.assertLessEqual(np.max(np.abs(starts - ideal)), 0.5)
        np.testing.assert_array_equal(np.flatnonzero(samples), starts)

    def test_classification_keeps_half_double_and_misses_separate(self):
        for estimate, expected in ((120, "correct"), (60, "half_time"), (240, "double_time"),
                                   (90, "other"), (None, "no_estimate")):
            self.assertEqual(score_bpm(estimate, 120)["classification"], expected)
        self.assertEqual(score_bpm(123.6, 120)["classification"], "correct")
        self.assertEqual(score_bpm(123.61, 120)["classification"], "other")

    def test_summary_keeps_failed_estimates_in_accuracy_denominator(self):
        rows = [score_bpm(value, 120) for value in (120, 60, None)]
        result = summarize(rows)
        self.assertEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["half_time"], 1)
        self.assertEqual(result["no_estimate"], 1)
        self.assertEqual(result["estimated_case_count"], 2)
        self.assertEqual(result["mean_absolute_error_bpm_when_estimated"], 30)
        self.assertIsNone(summarize([score_bpm(None, 120)])["mean_absolute_error_bpm_when_estimated"])

    def test_end_to_end_known_tempo_and_gain_invariance(self):
        first = evaluate_case("sharp_burst", 120, gain=1)
        quiet = evaluate_case("sharp_burst", 120, gain=0.1)
        self.assertEqual(first["estimated_bpm"], 120)
        self.assertEqual(quiet["estimated_bpm"], 120)
        self.assertEqual(first["classification"], "correct")
        self.assertNotEqual(first["input_sha256"], quiet["input_sha256"])
        np.testing.assert_allclose(first["scores"], quiet["scores"], atol=1e-14)

    def test_annotation_and_event_rate_are_distinct(self):
        case = evaluate_case("click", 120, pattern="subdivided")
        self.assertEqual(case["nominal_bpm"], 120)
        self.assertEqual(case["event_repetition_bpm"], 240)
        self.assertFalse(case["event_repetition_in_search_range"])
        for kind, pattern in (("bad", "steady"), ("click", "bad")):
            with self.assertRaises(ValueError):
                make_rhythm(kind, 120, pattern=pattern)
