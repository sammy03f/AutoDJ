"""Check hop comparison parity, timing accounting, and diagnostic lag probes."""

import unittest

import numpy as np

from autodj.analysis import TempoEstimate
from evals import tempo as baseline
from evals import tempo_hops as study


class TempoHopTests(unittest.TestCase):
    def test_original_hop_reproduces_baseline(self):
        samples, _ = baseline.make_rhythm("sharp_burst", 150, offset=11)
        metadata = {"nominal_bpm": 150, "pattern": "steady"}
        actual = study.evaluate_samples(samples, metadata, 32, repeats=1)
        expected = baseline.evaluate_case("sharp_burst", 150, offset=11)
        for key in ("input_sha256", "strength_sha256", "estimated_bpm", "selected_lag_frames",
                    "autocorrelation_score", "scores", "classification"):
            self.assertEqual(actual[key], expected[key])

    def test_same_audio_finer_frame_rate_and_more_frames(self):
        samples, _ = baseline.make_rhythm("click", 120)
        metadata = {"nominal_bpm": 120, "pattern": "steady"}
        rows = [study.evaluate_samples(samples, metadata, hop, repeats=1) for hop in study.HOPS]
        self.assertEqual(len({row["input_sha256"] for row in rows}), 1)
        self.assertEqual([row["strength_frame_rate"] for row in rows], [64, 128, 256])
        self.assertTrue(all(a["strength_frame_count"] < b["strength_frame_count"] for a, b in zip(rows, rows[1:])))
        self.assertEqual([row["estimated_bpm"] for row in rows], [120] * 3)

    def test_probes_bracket_lag_not_bpm_and_do_not_interpolate(self):
        result = TempoEstimate(120., 32, 0.9, np.array([25, 26, 32, 51, 52]), np.array([.6, .7, .9, .8, .4]))
        probe = study.probe_lags(result, 64, 150)
        self.assertAlmostEqual(probe["ideal_lag_frames"], 25.6)
        self.assertEqual([b["lag_frames"] for b in probe["bins"]], [25, 26])
        self.assertEqual(probe["best_bracketing_bin"]["score"], 0.7)
        self.assertEqual(result.bpm, 120)
        self.assertEqual(study.probe_lags(result, 64, 75)["best_bracketing_bin"]["lag_frames"], 51)

    def test_unsupported_targets_are_not_clamped(self):
        result = TempoEstimate(None, None, None, np.array([32]), np.array([0.]))
        outside = study.probe_lags(result, 64, 30)
        self.assertFalse(outside["target_in_search_range"])
        self.assertIsNone(outside["best_bracketing_bin"])
        unavailable = study.probe_lags(result, 64, 150)
        self.assertTrue(unavailable["target_in_search_range"])
        self.assertEqual(unavailable["bins"], [])

    def test_timing_repeats_and_units(self):
        samples = np.zeros(512)
        result = study.evaluate_samples(samples, {"kind": "silence"}, 16, repeats=2)
        self.assertEqual(len(result["timing_trials_ms"]), 2)
        for trial in result["timing_trials_ms"]:
            self.assertGreaterEqual(trial["features_ms"], 0)
            self.assertGreaterEqual(trial["tempo_ms"], 0)
            self.assertAlmostEqual(trial["total_ms"], trial["features_ms"] + trial["tempo_ms"])
        self.assertFalse(result["tempo_returned"])
        for repeats in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                study.evaluate_samples(samples, {}, 16, repeats=repeats)

    def test_timing_summary_retains_variability(self):
        trials = [{"features_ms": v, "tempo_ms": 2 * v, "total_ms": 3 * v} for v in (1, 3, 2)]
        self.assertEqual(study.timing_summary(trials)["total_ms"], {"median": 6, "min": 3, "max": 9})

    def test_fixture_populations_and_control_reuse(self):
        fixtures = study.fixtures()
        self.assertEqual(sum(m["group"] == "primary" for _, m in fixtures), 40)
        self.assertEqual(sum(m["group"] == "challenge" for _, m in fixtures), 20)
        self.assertEqual(sum(m["group"] == "control" for _, m in fixtures), 4)
        controls = baseline.negative_control_signals()
        for samples, metadata in fixtures:
            if metadata["group"] == "control":
                np.testing.assert_array_equal(samples, controls[metadata["kind"]])
