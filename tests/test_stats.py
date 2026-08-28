import unittest

from rvi_opd.stats import (
    ci_is_equivalent,
    difference_in_differences,
    holm_bonferroni,
    mcnemar_exact_p,
    paired_cluster_bootstrap,
    paired_seed_problem_bootstrap,
    three_way_action_interaction,
)


class StatisticsTests(unittest.TestCase):
    def test_paired_cluster_bootstrap(self) -> None:
        result = paired_cluster_bootstrap(
            {"a": [0.0, 0.0], "b": [0.0, 0.0], "c": [0.0, 0.0]},
            {"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]},
            resamples=100,
            seed=1,
        )
        self.assertEqual(result.estimate, 1.0)
        self.assertEqual(result.lower, 1.0)
        self.assertEqual(result.upper, 1.0)
        self.assertEqual(result.clusters, 3)

    def test_requires_identical_clusters(self) -> None:
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap({"a": [0.0]}, {"b": [1.0]}, resamples=10)

    def test_requires_equal_paired_observation_counts(self) -> None:
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap({"a": [0.0]}, {"a": [1.0, 1.0]}, resamples=10)

    def test_difference_in_differences(self) -> None:
        self.assertAlmostEqual(difference_in_differences(0.6, 0.6, 0.2, 0.8), 0.6)

    def test_three_way_interaction_uses_all_eight_cells(self) -> None:
        value = three_way_action_interaction(
            dl_low_repair=0.5,
            dl_low_intervene=0.4,
            dl_high_repair=0.5,
            dl_high_intervene=0.5,
            di_low_repair=0.5,
            di_low_intervene=0.5,
            di_high_repair=0.2,
            di_high_intervene=0.8,
        )
        self.assertAlmostEqual(value, 0.5)

    def test_tost_ci_contract(self) -> None:
        self.assertTrue(ci_is_equivalent(-0.04, 0.03, margin=0.05))
        self.assertFalse(ci_is_equivalent(-0.05, 0.03, margin=0.05))
        with self.assertRaises(ValueError):
            ci_is_equivalent(0.1, -0.1, margin=0.05)

    def test_exact_mcnemar(self) -> None:
        self.assertEqual(mcnemar_exact_p([0, 0], [0, 0]), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p([0] * 6, [1] * 6), 0.03125)
        with self.assertRaises(ValueError):
            mcnemar_exact_p([0], [2])

    def test_paired_seed_problem_bootstrap(self) -> None:
        baseline = {
            "13": {"p1": [0, 0], "p2": [0, 0]},
            "17": {"p1": [0, 0], "p2": [0, 0]},
        }
        treatment = {
            "13": {"p1": [1, 1], "p2": [1, 1]},
            "17": {"p1": [1, 1], "p2": [1, 1]},
        }
        result = paired_seed_problem_bootstrap(
            baseline, treatment, resamples=100, seed=3
        )
        self.assertEqual(result.estimate, 1.0)
        self.assertEqual(result.lower, 1.0)
        self.assertEqual(result.upper, 1.0)
        self.assertEqual(result.clusters, 2)

    def test_holm_is_step_down(self) -> None:
        result = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
        self.assertTrue(result["a"])
        self.assertTrue(result["b"])
        self.assertTrue(result["c"])


if __name__ == "__main__":
    unittest.main()
