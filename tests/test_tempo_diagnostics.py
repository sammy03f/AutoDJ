"""Known relative-flux values, gain behavior, window support, and baseline parity."""

import unittest

import numpy as np

from evals import tempo_diagnostics as study
from evals import tempo_evidence as evidence


class TempoDiagnosticsTests(unittest.TestCase):
    def test_relative_flux_known_ratio_and_zero_denominator(self):
        self.assertEqual(study.relative_peak_flux(np.array([0., 4.]), np.array([[1j], [3j]])), 2)
        self.assertIsNone(study.relative_peak_flux(np.zeros(2), np.zeros((2, 1))))
        self.assertIsNone(study.relative_peak_flux(np.array([]), np.zeros((0, 1))))

    def test_gain_changes_raw_peak_but_preserves_relative_metric(self):
        samples, _ = study.baseline.make_rhythm("sharp_burst", 120)
        loud = study.diagnose(samples, {})
        quiet = study.diagnose(samples * 1e-6, {})
        self.assertAlmostEqual(quiet["strength_peak"] / loud["strength_peak"], 1e-6)
        self.assertAlmostEqual(quiet["relative_peak_flux"], loud["relative_peak_flux"], places=12)
        self.assertEqual(quiet["full"]["bpm"], loud["full"]["bpm"])
        self.assertIsNone(quiet["gated"]["bpm"])
        self.assertEqual(quiet["gated"]["rejection_reason"], "weak_strength")

    def test_window_agreement_is_not_octave_equivalence(self):
        full = {"bpm": 120}
        self.assertTrue(study.window_agreement(full, [{"bpm": 120}, {"bpm": 123.6}]))
        self.assertFalse(study.window_agreement(full, [{"bpm": 120}, {"bpm": 60}]))
        self.assertIsNone(study.window_agreement(full, [{"bpm": 120}, {"bpm": None}]))
        self.assertIsNone(study.window_agreement({"bpm": None}, [{"bpm": 120}, {"bpm": 120}]))

    def test_short_windows_have_less_lag_support(self):
        strength = np.zeros(128)
        strength[::64] = 1
        full = study.estimate_fields(strength)
        first, second = study.estimate_fields(strength[:64]), study.estimate_fields(strength[64:])
        self.assertEqual(full["bpm"], 120)
        self.assertEqual(first["rejection_reason"], "insufficient_data")
        self.assertIsNone(study.window_agreement(full, [first, second]))

    def test_reuses_control_waveforms_and_frozen_decisions(self):
        fixtures = study.fixtures()
        self.assertEqual(len(fixtures), 51)
        controls = evidence.control_signals()
        for (samples, metadata), (prior_samples, prior_metadata) in zip(fixtures, controls):
            np.testing.assert_array_equal(samples, prior_samples)
            self.assertEqual(metadata["kind"], prior_metadata["kind"])
        samples, metadata = next((s, m) for s, m in fixtures if m.get("frequency_hz") == 137 and m.get("gain") == 1)
        original = evidence.evaluate(samples, metadata)
        new = study.diagnose(samples, metadata)
        self.assertEqual(original["input_sha256"], new["input_sha256"])
        for method, old_method in (("full", "baseline"), ("gated", "both")):
            self.assertEqual(new[method]["bpm"], original["results"][old_method]["estimated_bpm"])
            self.assertEqual(new[method]["score"], original["results"][old_method]["score"])

    def test_summary_keeps_incomplete_windows_distinct(self):
        rows = [{"full": {"bpm": 120}, "gated": {"bpm": 120}, "half_window_agreement": agreement}
                for agreement in (True, False, None)]
        rows.append({"full": {"bpm": 120}, "gated": {"bpm": None}, "half_window_agreement": True})
        result = study.summarize(rows)
        self.assertEqual(result["gated_accepted"], 3)
        self.assertEqual(result["accepted_agreement"], 1)
        self.assertEqual(result["accepted_disagreement"], 1)
        self.assertEqual(result["accepted_incomplete_windows"], 1)
        self.assertEqual(result["gated_abstained"], 1)
