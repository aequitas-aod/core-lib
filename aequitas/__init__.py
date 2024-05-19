import logging
from typing import Any

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# keep this line at the top of this file
__all__ = ["logger", "isinstance", "Decorator"]


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('aequitas')
"""General logger to be used in all `aequitas*` modules"""


__py_isinstance = isinstance


def isinstance(obj, cls):
    """A version of `isinstance` that takes type unions into account"""

    if hasattr(cls, '__args__') and __py_isinstance(cls.__args__, tuple):
        return any(__py_isinstance(obj, t) for t in cls.__args__)
    return __py_isinstance(obj, cls)


class Decorator:
    def __init__(self, delegate) -> None:
        self._delegate = delegate

    def __getattr__(self, name):
        return getattr(self._delegate, name)
    
    def __setattr__(self, name, value):
        if name == '_delegate':
            super().__setattr__(name, value)
        elif hasattr(self._delegate, name):
            setattr(self._delegate, name, value)
        else:
            super().__setattr__(name, value)
    
    def __delattr__(self, name):
        if hasattr(self._delegate, name):
            delattr(self._delegate, name)
        else:
            super().__delattr__(name)

    def __getitem__(self, key):
        return self._delegate[key]
    
    def __setitem__(self, key, value):
        self._delegate[key] = value

    def __delitem__(self, key):
        del self._delegate[key]

    def __dir__(self):
        return dir(self._delegate) + super().__dir__()
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._delegate(*args, **kwds)
    
    def __len__(self) -> int:
        return len(self._delegate)
    
    def __iter__(self):
        return iter(self._delegate)

    def __eq__(self, value: object) -> bool:
        return self._delegate == value
    
    def __hash__(self) -> int:
        return hash(self._delegate)
    
    def __ne__(self, value: object) -> bool:
        return self._delegate != value
    
    def __lt__(self, value: object) -> bool:
        return self._delegate < value
    
    def __le__(self, value: object) -> bool:
        return self._delegate <= value
    
    def __gt__(self, value: object) -> bool:
        return self._delegate > value
    
    def __ge__(self, value: object) -> bool:
        return self._delegate >= value
    
    def __add__(self, value: object) -> Any:
        return self._delegate + value
    
    def __sub__(self, value: object) -> Any:
        return self._delegate - value
    
    def __mul__(self, value: object) -> Any:
        return self._delegate * value
    
    def __truediv__(self, value: object) -> Any:
        return self._delegate / value
    
    def __floordiv__(self, value: object) -> Any:
        return self._delegate // value
    
    def __mod__(self, value: object) -> Any:
        return self._delegate % value
    
    def __pow__(self, value: object) -> Any:
        return self._delegate ** value
    
    def __lshift__(self, value: object) -> Any:
        return self._delegate << value
    
    def __rshift__(self, value: object) -> Any:
        return self._delegate >> value
    
    def __and__(self, value: object) -> Any:
        return self._delegate & value
    
    def __xor__(self, value: object) -> Any:
        return self._delegate ^ value
    
    def __or__(self, value: object) -> Any:
        return self._delegate | value
    
    def __repr__(self) -> str:
        return repr(self._delegate)
    
    def __str__(self) -> str:
        return str(self._delegate)


# keep this line at the bottom of this file
logger.debug("Module %s correctly loaded", __name__)
