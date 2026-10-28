"""Test threshold selection, including control penalties and held-out isolation."""

import unittest
from unittest.mock import patch

from evals import onset_adaptive as adaptive
from evals import onset_threshold_search as search


def group(matches=8, misses=2, extras=1, control_extras=3):
    event = {"matched_count": matches, "missed_count": misses, "false_count": extras}
    control = {"false_count": control_extras}
    return {"summary": {"methods": {name: event.copy() for name in ("fixed", "adaptive")}},
            "control_summary": {"methods": {name: control.copy() for name in ("fixed", "adaptive")}},
            "by_condition": [], "by_kind": {}, "cases": [], "controls": []}


def trial(radius, multiplier, matches=8, misses=2, extras=1, controls=3):
    return {"radius": radius, "multiplier": multiplier,
            "objective": search.objective(group(matches, misses, extras, controls), "adaptive")}


class ThresholdSearchTests(unittest.TestCase):
    def test_objective_includes_noise_controls(self):
        score = search.objective(group(), "adaptive")
        self.assertEqual(score["false_count_including_controls"], 4)
        self.assertAlmostEqual(score["precision"], 8 / 12)
        self.assertAlmostEqual(score["recall"], 8 / 10)
        self.assertAlmostEqual(score["f1"], 16 / 22)

    def test_no_predictions_are_zero_f1_when_events_exist(self):
        score = search.objective(group(0, 10, 0, 0), "adaptive")
        self.assertIsNone(score["precision"])
        self.assertEqual(score["recall"], 0)
        self.assertEqual(score["f1"], 0)
        empty = search.objective(group(0, 0, 0, 0), "adaptive")
        self.assertIsNone(empty["f1"])

    def test_selects_balance_not_just_matches(self):
        noisy = trial(8, 1.5, matches=10, misses=0, extras=20, controls=0)
        balanced = trial(12, 2, matches=8, misses=2, extras=0, controls=0)
        self.assertIs(search.select_setting([noisy, balanced]), balanced)

    def test_ties_are_deterministic_and_prefer_fewer_false(self):
        # Both F1 values are 0.5, but the second has fewer false predictions.
        first = trial(8, 1.5, matches=5, misses=5, extras=5, controls=0)
        second = trial(16, 3, matches=4, misses=6, extras=2, controls=0)
        self.assertIs(search.select_setting([first, second]), second)
        equal = [trial(16, 3), trial(8, 2), trial(8, 1.5)]
        self.assertIs(search.select_setting(list(reversed(equal))), equal[2])

    def test_empty_or_undefined_objectives_are_rejected(self):
        for trials in ([], [trial(8, 3, 0, 0, 0, 0)]):
            with self.assertRaises(ValueError):
                search.select_setting(trials)

    def test_seed_sets_are_new_and_disjoint(self):
        previous = set(adaptive.SEED_GROUPS["baseline"]) | set(adaptive.SEED_GROUPS["fresh"])
        self.assertFalse(set(search.HELD_OUT_SEEDS) & previous)
        self.assertFalse(set(search.HELD_OUT_SEEDS) & set(search.TUNING_SEEDS))

    def test_only_tuning_selects_setting_and_held_out_runs_afterward(self):
        calls = []

        def evaluate(seeds, *, radius=8, multiplier=3):
            calls.append((seeds, radius, multiplier))
            if seeds == search.TUNING_SEEDS:
                return group(matches=9 if radius == 12 else 2, misses=1 if radius == 12 else 8)
            # Poor selected performance must not trigger reselection.
            return group(matches=0, misses=10)

        with patch.object(search, "GRID", ((8, 2), (12, 2))), \
             patch.object(search.adaptive, "evaluate_group", side_effect=evaluate):
            result = search.run_study()
        self.assertEqual(result["selected_config"], {"radius": 12, "multiplier": 2})
        self.assertEqual(calls, [(search.TUNING_SEEDS, 8, 2), (search.TUNING_SEEDS, 12, 2),
                                 (search.HELD_OUT_SEEDS, 8, 3), (search.HELD_OUT_SEEDS, 12, 2)])

    def test_overlapping_seeds_rejected_before_evaluation(self):
        with patch.object(search, "HELD_OUT_SEEDS", search.TUNING_SEEDS), \
             patch.object(search.adaptive, "evaluate_group") as evaluate:
            with self.assertRaises(ValueError):
                search.run_study()
            evaluate.assert_not_called()

    def test_parameterized_evaluator_preserves_original_defaults(self):
        args = ("sharp_burst", 0.1, 0.05, 101, 8)
        self.assertEqual(adaptive.evaluate_case(*args),
                         adaptive.evaluate_case(*args, radius=8, multiplier=3))
        changed = adaptive.evaluate_case(*args, radius=16, multiplier=2)
        original = adaptive.evaluate_case(*args)
        self.assertEqual(changed["input_sha256"], original["input_sha256"])
        self.assertEqual(changed["metrics"]["fixed"], original["metrics"]["fixed"])
        self.assertNotEqual(changed["adaptive_threshold"], original["adaptive_threshold"])
