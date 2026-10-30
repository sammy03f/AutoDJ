"""Check split isolation, decision accounting, and physical gain invariance."""

import unittest
import numpy as np

from evals import tempo_relative as study


def row(group, ratio, *, bpm=120, split="tuning"):
    result = {"accepted": True, "estimated_bpm": bpm, "selected_lag_frames": 64,
              "score": .8, "rejection_reason": None}
    data = {"split": split, "group": group, "relative_peak_flux": ratio,
            "results": {"correlation_only": result}}
    if group == "rhythm":
        data["nominal_bpm"] = 120
        result.update(study.baseline.score_bpm(bpm, 120))
    return data


class TempoRelativeTests(unittest.TestCase):
    def test_inclusive_relative_floor_and_no_mutation(self):
        case = row("rhythm", .1)
        self.assertTrue(study.relative_result(case, .1)["accepted"])
        rejected = study.relative_result(case, .5)
        self.assertEqual(rejected["rejection_reason"], "weak_relative_flux")
        self.assertEqual(rejected["classification"], "no_estimate")
        self.assertIsNone(rejected["estimated_bpm"])
        self.assertIsNone(rejected["score"])
        self.assertTrue(case["results"]["correlation_only"]["accepted"])
        case["relative_peak_flux"] = None
        self.assertFalse(study.relative_result(case, .1)["accepted"])

    def test_correlation_rejection_is_preserved(self):
        case = row("control", 10)
        original = case["results"]["correlation_only"]
        original.update(accepted=False, estimated_bpm=None, score=None,
                        selected_lag_frames=None, rejection_reason="low_correlation")
        self.assertEqual(study.relative_result(case, .1), original)

    def test_selection_uses_correct_tempo_and_smallest_tie(self):
        cases = [row("rhythm", 1), row("rhythm", 1, bpm=60), row("control", .005)]
        threshold, search = study.select_threshold(cases)
        self.assertEqual(threshold, .01)
        selected = next(r for r in search if r["threshold"] == threshold)
        self.assertEqual(selected["rhythm_accuracy"], .5)
        self.assertEqual(selected["balanced_utility"], .75)
        self.assertEqual(study.metrics(cases, threshold=2)["rhythm_accuracy"], 0)
        for invalid in ([], [row("rhythm", 1, split="evaluation")]):
            with self.assertRaises(ValueError):
                study.select_threshold(invalid)

    def test_split_parameters_disjoint_and_signals_repeatable(self):
        tuning, evaluation = study.SPLITS.values()
        self.assertFalse(set(tuning["seeds"]) & set(evaluation["seeds"]))
        self.assertFalse(set(tuning["bpms"]) & set(evaluation["bpms"]))
        self.assertNotEqual(tuning["frequency"], evaluation["frequency"])
        for kind in study.RHYTHMS + study.CONTROLS:
            a, events = study.make_signal(kind, 2, 5101, 143, 110 if kind in study.RHYTHMS else None)
            b, repeat_events = study.make_signal(kind, 2, 5101, 143, 110 if kind in study.RHYTHMS else None)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(events, repeat_events)
            self.assertEqual(len(a), 4096)
            self.assertTrue(np.all(np.isfinite(a)))
        _, events = study.make_signal("sharp", 8, 5101, 143, 110)
        np.testing.assert_allclose(np.diff(events), 60 / 110, atol=1 / study.baseline.RATE)

    def test_relative_decision_preserves_quiet_rhythm(self):
        audio, _ = study.make_signal("sharp", 8, 5101, 143, 110)
        ratios = []
        for gain in study.GAINS:
            samples = audio * gain
            case = study.evidence.evaluate(samples, {"group": "rhythm", "nominal_bpm": 110})
            spectrum = study.stft(samples, study.baseline.RATE, frame_size=128, hop_size=16)
            case["relative_peak_flux"] = study.relative_peak_flux(study.spectral_flux(spectrum), spectrum.coefficients)
            ratios.append(case["relative_peak_flux"])
            result = study.relative_result(case, .01)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["classification"], "correct")
            self.assertEqual(case["results"]["both"]["accepted"], gain == 1.)
        self.assertAlmostEqual(*ratios, places=12)
