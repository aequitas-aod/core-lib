import logging


# keep this line at the top of this file
__all__ = ["logger", "isinstance"]


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('aequitas')
"""General logger to be used in all `aequitas*` modules"""


__py_isinstance = isinstance


def isinstance(obj, cls):
    """A version of `isinstance` that takes type unions into account"""

    if hasattr(cls, '__args__') and __py_isinstance(cls.__args__, tuple):
        return any(__py_isinstance(obj, t) for t in cls.__args__)
    return __py_isinstance(obj, cls)


# keep this line at the bottom of this file
logger.debug("Module %s correctly loaded", __name__)
