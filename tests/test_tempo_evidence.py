"""Optional abstention boundaries, reasons, and comparison accounting."""

import unittest

import numpy as np

from autodj.analysis import estimate_tempo
from evals import tempo as baseline
from evals import tempo_evidence as study


class TempoEvidenceTests(unittest.TestCase):
    def curve(self, gain=1.):
        values = np.zeros(640)
        values[::32] = gain
        return values

    def test_disabled_checks_preserve_original_scores(self):
        values = self.curve()
        default = estimate_tempo(values, 64)
        explicit = estimate_tempo(values, 64, min_peak_strength=0, min_correlation=0)
        self.assertEqual(default.bpm, explicit.bpm)
        self.assertEqual(default.score, explicit.score)
        np.testing.assert_array_equal(default.scores, explicit.scores)
        self.assertIsNone(default.rejection_reason)

    def test_raw_strength_is_checked_before_normalization(self):
        values = self.curve(1e-8)
        self.assertEqual(estimate_tempo(values, 64).bpm, 120)
        result = estimate_tempo(values, 64, min_peak_strength=0.001)
        self.assertIsNone(result.bpm)
        self.assertEqual(result.rejection_reason, "weak_strength")
        np.testing.assert_array_equal(result.scores, np.zeros(result.lags.size))

    def test_equal_strength_threshold_passes(self):
        self.assertEqual(estimate_tempo(self.curve(0.001), 64, min_peak_strength=0.001).bpm, 120)
        result = estimate_tempo(self.curve(0.001), 64, min_peak_strength=np.nextafter(0.001, np.inf))
        self.assertEqual(result.rejection_reason, "weak_strength")

    def test_correlation_boundary_and_rejected_curve_retained(self):
        values = self.curve()
        original = estimate_tempo(values, 64)
        equal = estimate_tempo(values, 64, min_correlation=original.score)
        self.assertEqual(equal.bpm, original.bpm)
        rejected = estimate_tempo(values, 64, min_correlation=np.nextafter(original.score, np.inf))
        self.assertEqual(rejected.rejection_reason, "low_correlation")
        self.assertIsNone(rejected.bpm)
        self.assertIsNone(rejected.lag_frames)
        self.assertIsNone(rejected.score)
        np.testing.assert_array_equal(rejected.scores, original.scores)

    def test_existing_abstention_reasons_and_precedence(self):
        self.assertEqual(estimate_tempo([], 64).rejection_reason, "insufficient_data")
        self.assertEqual(estimate_tempo([1] * 256, 64).rejection_reason, "no_variation")
        self.assertEqual(estimate_tempo([0] * 256, 64, min_peak_strength=1).rejection_reason, "no_variation")
        self.assertEqual(estimate_tempo([0, 1], 1, min_bpm=59, max_bpm=61).rejection_reason, "no_positive_correlation")

    def test_invalid_gate_parameters(self):
        for key, values in (("min_peak_strength", (-1, np.nan, np.inf)),
                            ("min_correlation", (-1, 1.01, np.nan, np.inf))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    estimate_tempo([], 64, **{key: value})

    def test_quiet_signal_and_control_integration(self):
        samples, _ = baseline.make_rhythm("sharp_burst", 120, gain=1e-4)
        case = study.evaluate(samples, {"nominal_bpm": 120, "group": "quiet_stress"})
        self.assertEqual(case["results"]["both"]["estimated_bpm"], 120)
        self.assertTrue(study.check_case(case))
        silence = study.evaluate(np.zeros(baseline.RATE * 2), {})
        self.assertTrue(study.check_case(silence))
        self.assertTrue(all(not r["accepted"] for r in silence["results"].values()))

    def test_summary_keeps_abstentions_in_accuracy_denominator(self):
        samples, _ = baseline.make_rhythm("click", 120, gain=1e-6)
        case = study.evaluate(samples, {"nominal_bpm": 120})
        summary = study.summarize([case])
        self.assertEqual(summary["baseline"]["correct"], 1)
        self.assertEqual(summary["both"]["abstained"], 1)
        self.assertEqual(summary["both"]["nominal_accuracy"], 0)
        self.assertIsNone(study.summarize([])["both"]["acceptance_rate"])
        self.assertEqual(summary["both"]["rejection_counts"], {"weak_strength": 1})

    def test_controls_are_repeatable_and_include_off_bin_tones(self):
        first, second = study.control_signals(), study.control_signals()
        self.assertEqual(len(first), 45)
        for (a, ma), (b, mb) in zip(first, second):
            self.assertEqual(ma, mb)
            np.testing.assert_array_equal(a, b)
        self.assertEqual({m["frequency_hz"] for _, m in first if m["kind"] == "steady_tone"}, {128, 137})
        for seed in study.SEEDS[1:]:
            self.assertNotIn(seed, (7, 19, 43, 101, 211, 307, 401, 503, 607, 701, 809, 1013, 1217, 2027))
