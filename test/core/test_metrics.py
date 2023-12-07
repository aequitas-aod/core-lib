from test import uniform_binary_dataset, skewed_binary_dataset, uniform_binary_dataset_gt, skewed_binary_dataset_gt
from aequitas.core.metrics import discrete_demographic_parities
from aequitas.core.metrics import discrete_equalised_odds
import unittest
import numpy as np


DATASET_SIZE = 10000


class Common(unittest.TestCase):
    def assertInRange(self, value, lower, upper):
        self.assertGreaterEqual(value, lower)
        self.assertLessEqual(value, upper)

class TestDemographicParity(Common):
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

class TestEqualisedOdds(Common):
    def setUp(self) -> None:
        self.fair_dataset = uniform_binary_dataset_gt(rows=DATASET_SIZE)
        self.unfair_dataset = skewed_binary_dataset_gt(rows=DATASET_SIZE, p=0.9)

    def test_equalised_odds_on_fair_binary_case(self):
        x = self.fair_dataset[:, 0]
        y = self.fair_dataset[:, 1]
        y_pred = self.fair_dataset[:, 2]

        y_values = np.unique(y)

        equalised_odds_fps = discrete_equalised_odds(x, y, y_pred, 0)
        equalised_odds_fns = discrete_equalised_odds(x, y, y_pred, 1)
        self.assertEqual(equalised_odds_fps.shape, y_values.shape)        
        self.assertEqual(equalised_odds_fns.shape, y_values.shape)        
        
        eo1 = abs(equalised_odds_fps[0] - equalised_odds_fps[1])
        eo2 = abs(equalised_odds_fns[0] - equalised_odds_fns[1])

        self.assertInRange(eo1, 0.0, 0.1)
        self.assertInRange(eo2, 0.0, 0.1)

    def test_equalised_odds_on_unfair_binary_case(self):
        x = self.unfair_dataset[:, 0]
        y = self.unfair_dataset[:, 1]
        y_pred = self.unfair_dataset[:, 2]

        y_values = np.unique(y)

        equalised_odds_fps = discrete_equalised_odds(x, y, y_pred, 0)
        equalised_odds_fns = discrete_equalised_odds(x, y, y_pred, 1)
        self.assertEqual(equalised_odds_fps.shape, y_values.shape)        
        self.assertEqual(equalised_odds_fns.shape, y_values.shape)        
        
        eo1 = abs(equalised_odds_fps[0] - equalised_odds_fps[1])
        eo2 = abs(equalised_odds_fns[0] - equalised_odds_fns[1])

        self.assertInRange(eo1, 0.3, 1.0)
        self.assertInRange(eo2, 0.3, 1.0)

if __name__ == '__main__':
    unittest.main()
