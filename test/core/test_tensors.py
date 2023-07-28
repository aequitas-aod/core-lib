import unittest
from aequitas.core import ensure_tensor, ensure_same_tensor_type
import numpy
import eagerpy
import tensorflow


class TestTensorsFromIterables(unittest.TestCase):
    def tensor_from_iterable(self, iterable, *items):
        x = ensure_tensor(iterable(items))
        self.assertIsInstance(x, eagerpy.Tensor)
        self.assertEqual(x.shape, (3,))
        self.assertIsInstance(x.raw, numpy.ndarray)
        self.assertEqual(list(x.raw), list(items))

    def test_tensor_from_list(self):
        self.tensor_from_iterable(list, 1, 2, 3)

    def test_tensor_from_set(self):
        self.tensor_from_iterable(set, 1.0, 2.0, 3.0)

    def test_tensor_from_tuple(self):
        self.tensor_from_iterable(tuple, True, False, True)


class TestTensorsFromTensorflow(unittest.TestCase):
    def tensor_from_tf(self, iterable, *items):
        x = ensure_tensor(tensorflow.Tensor(iterable(items)))
        self.assertIsInstance(x, eagerpy.Tensor)
        self.assertEqual(len(x), 3)
        self.assertIsInstance(x.raw, tensorflow.Tensor)
        self.assertEqual(list(x.raw), list(items))

    def test_tensor_from_list(self):
        self.tensor_from_tf(list, 1, 2, 3)

    def test_tensor_from_set(self):
        self.tensor_from_tf(set, 1.0, 2.0, 3.0)
