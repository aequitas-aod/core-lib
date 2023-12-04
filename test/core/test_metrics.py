from test import uniform_binary_dataset, skewed_binary_dataset
from aequitas.core.metrics import discrete_demographic_parities
import unittest


DATASET_SIZE = 10000


class TestDemographicParity(unittest.TestCase):
    def setUp(self) -> None:
        self.fair_dataset = uniform_binary_dataset(rows=DATASET_SIZE)
        self.unfair_dataset = skewed_binary_dataset(rows=DATASET_SIZE, p=0.9)

    def assertInRange(self, value, lower, upper):
        self.assertGreaterEqual(value, lower)
        self.assertLessEqual(value, upper)

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


if __name__ == '__main__':
    unittest.main()
