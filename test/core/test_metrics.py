from test import uniform_binary_dataset, skewed_binary_dataset, uniform_binary_dataset_gt, skewed_binary_dataset_gt
from aequitas.core.metrics import discrete_demographic_parities
from aequitas.core.metrics import discrete_equalised_odds
from aequitas.core.metrics import discrete_disparate_impact
from aequitas.core.metrics import discrete_equal_opportunity
import unittest
import numpy as np

DATASET_SIZE = 10000


class AbstractMetricTestCase(unittest.TestCase):
    def assertInRange(self, value, lower, upper):
        self.assertGreaterEqual(value, lower)
        self.assertLessEqual(value, upper)


class TestDemographicParity(AbstractMetricTestCase):
    def setUp(self) -> None:
        self.fair_dataset = uniform_binary_dataset(rows=DATASET_SIZE)
        self.unfair_dataset = skewed_binary_dataset(rows=DATASET_SIZE, p=0.9)

    def test_parity_on_fair_binary_case(self):
        x = self.fair_dataset[:, 0]
        y = self.fair_dataset[:, 1]
        parities = discrete_demographic_parities(x, y, 1)
        self.assertEqual(parities.shape, (1,))
        self.assertInRange(parities[0], 0.0, 0.005)

    def test_parity_on_unfair_binary_case(self):
        x = self.unfair_dataset[:, 0]
        y = self.unfair_dataset[:, 1]
        parities = discrete_demographic_parities(x, y, 1)
        self.assertEqual(parities.shape, (1,))
        self.assertInRange(parities[0], 0.4, 0.5)


class TestEqualisedOdds(AbstractMetricTestCase):
    def setUp(self) -> None:
        self.fair_dataset = uniform_binary_dataset_gt(rows=DATASET_SIZE)
        self.unfair_dataset = skewed_binary_dataset_gt(rows=DATASET_SIZE, p=0.9)

    def test_equalised_odds_on_fair_binary_case(self):
        x = self.fair_dataset[:, -3]
        y = self.fair_dataset[:, -2]
        y_pred = self.fair_dataset[:, -1]

        differences = discrete_equalised_odds(x, y, y_pred)
        for diff_row in differences:
            for diff in diff_row:
                self.assertInRange(diff, 0.0, 0.1)

    def test_equalised_odds_on_unfair_binary_case(self):
        x = self.unfair_dataset[:, -3]
        y = self.unfair_dataset[:, -2]
        y_pred = self.unfair_dataset[:, -1]

        differences = discrete_equalised_odds(x, y, y_pred)
        for diff_row in differences:
            for diff in diff_row:
                self.assertInRange(diff, 0.3, 1.0)


class TestDisparateImpact(AbstractMetricTestCase):
    def setUp(self) -> None:
        self.fair_dataset = uniform_binary_dataset(rows=DATASET_SIZE)
        self.unfair_dataset = skewed_binary_dataset(rows=DATASET_SIZE, p=0.9)

    def test_disparate_impact_on_fair_dataset(self):
        x = self.fair_dataset[:, 0]
        y = self.fair_dataset[:, 1]

        disparate_impact = discrete_disparate_impact(x, y, 1, 1)
        self.assertInRange(disparate_impact, 0.7, 1.3)

    def test_disparate_impact_on_unfair_dataset(self):
        x = self.unfair_dataset[:, 0]
        y = self.unfair_dataset[:, 1]

        disparate_impact = discrete_disparate_impact(x, y, 1, 1)
        self.assertTrue(disparate_impact < 0.5 or disparate_impact > 1.5)


class TestEqualOpportunity(AbstractMetricTestCase):
    def setUp(self) -> None:
        self.fair_dataset = uniform_binary_dataset_gt(rows=DATASET_SIZE)
        self.unfair_dataset = skewed_binary_dataset_gt(rows=DATASET_SIZE)

    def test_equal_opportunity_on_fair_dataset(self):
        x = self.fair_dataset[:, -3]
        y = self.fair_dataset[:, -2]
        y_pred = self.fair_dataset[:, -1]

        differences = discrete_equal_opportunity(x, y, y_pred, 1)
        for diff in differences:
            self.assertInRange(diff, 0.0, 0.1)

    def test_equal_opportunity_on_unfair_dataset(self):
        x = self.unfair_dataset[:, -3]
        y = self.unfair_dataset[:, -2]
        y_pred = self.unfair_dataset[:, -1]

        differences = discrete_equal_opportunity(x, y, y_pred, 1)
        for diff in differences:
            self.assertInRange(diff, 0.3, 1.0)


# delete this abstract class, so that the included tests are not run
del AbstractMetricTestCase

if __name__ == '__main__':
    unittest.main()
