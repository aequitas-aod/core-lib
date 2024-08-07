import unittest
from aequitas import Decorator


class TestDecorator(unittest.TestCase):
    class DecoratedInt(Decorator):
        def __init__(self, value: int) -> None:
            super().__init__(value)
            self.prev = value - 1

        def next(self):
            return self._delegate + 1   

    def setUp(self) -> None:
        self._x = self.DecoratedInt(1)
    
    def test_new_method(self):
        self.assertEqual(2, self._x.next())

    def test_new_attribute(self):
        self.assertEqual(0, self._x.prev)

    def test_delegate(self):
        self.assertEqual(1, self._x._delegate)

    def test_delegate_method(self):
        self.assertEqual(1, self._x.conjugate())

    def test_expressions(self):
        self.assertEqual(2, self._x + 1)
        self.assertEqual(0, self._x - 1)
        self.assertEqual(2, self._x * 2)
        self.assertEqual(0.5, self._x / 2)
        self.assertEqual(0, self._x // 2)

    def test_comparison(self):
        self.assertTrue(self._x < 2)
        self.assertTrue(self._x <= 1)
        self.assertTrue(self._x > 0)
        self.assertTrue(self._x >= 1)
        self.assertTrue(self._x == 1)
        self.assertTrue(self._x != 2)

    def test_bitwise(self):
        self.assertEqual(2, self._x << 1)
        self.assertEqual(0, self._x >> 1)
        self.assertEqual(1, self._x & 1)
        self.assertEqual(1, self._x | 0)
        self.assertEqual(0, self._x ^ 1)

    def test_str(self):
        self.assertEqual('1', str(self._x))

    def test_repr(self):
        self.assertEqual('1', repr(self._x))

    def test_dir(self):
        self.assertIn('prev', dir(self._x))
        self.assertIn('_delegate', dir(self._x))
        self.assertIn('next', dir(self._x))
        self.assertIn('conjugate', dir(self._x))

    def test_hash_code(self):
        self.assertEqual(1, hash(self._x))

    def test_not_subscriptable(self):
        with self.assertRaises(TypeError):
            self._x[0]


class TestOverridden(unittest.TestCase):
    class DecoratedInt(Decorator):
        def __init__(self, value: int) -> None:
            super().__init__(value)
        
        def conjugate(self):
            return -self._delegate
    
    def setUp(self) -> None:
        self._x = self.DecoratedInt(1)

    def test_overridden_method(self):
        self.assertEqual(-1, self._x.conjugate())
        self.assertEqual(1, self._x._delegate.conjugate())


class TestDecoratedList(unittest.TestCase):
    class DecoratedList(Decorator):
        def __init__(self, value: list) -> None:
            super().__init__(value)
        
        @property
        def head(self):
            return self[0]

        @property
        def tail(self):
            return self[1:]
        
    def setUp(self) -> None:
        self._x = self.DecoratedList([1, 2, 3])
    
    def test_head(self):
        self.assertEqual(1, self._x.head)
    
    def test_tail(self):
        self.assertEqual([2, 3], self._x.tail)

    def test_delete(self):
        del self._x[0]
        self.assertEqual([2, 3], self._x)

    def test_append(self):
        self._x.append(4)
        self.assertEqual([1, 2, 3, 4], self._x)

    def test_len(self):
        self.assertEqual(3, len(self._x))

    def test_iter(self):
        self.assertEqual((1, 2, 3), tuple(self._x))

    def test_str(self):
        self.assertEqual('[1, 2, 3]', str(self._x))

    def test_repr(self):
        self.assertEqual('[1, 2, 3]', repr(self._x))
    
    def test_not_shiftable(self):
        with self.assertRaises(TypeError):
            self._x << 1


class TestSubtypingWithOrdinaryClasses(unittest.TestCase):
    class A():
        pass

    class DecoratedA(Decorator):
        def __init__(self, value) -> None:
            super().__init__(value)

    def setUp(self):
        self._x = self.DecoratedA(self.A())

    def test_not_instance_of_decorated_type(self):
        self.assertNotIsInstance(self._x, self.A)

    def test_instance_of_decorator_type(self):
        self.assertIsInstance(self._x, self.DecoratedA)
        self.assertIsInstance(self._x, Decorator)
