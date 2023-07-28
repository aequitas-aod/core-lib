import unittest
from aequitas import *


class TestMyClass(unittest.TestCase):
    # test methods' names should begin with `test_`
    def test_my_method(self):
        x = "Hello World"
        self.assertEqual("Hello World", x)
