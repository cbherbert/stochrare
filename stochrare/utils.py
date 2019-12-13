"""
Utilities
============

.. currentmodule:: stochrare.utils

This module contains various tools which are not really intended to be used by external code,
but should be used by various other modules in the package.
This includes decorators, but is not restricted to it.

.. autofunction:: pseudorand
"""
import functools
import numpy as np

def pseudorand(fun):
    """
    Decorator for methods of random objects.
    If the object's `__deterministic__` attribute is set to True, the random number generator will
    be seeded with a fixed value (here, 100, chosen arbitrarily) before calling the method.
    Hence the method will behave in a deterministic way, only if the instance was initialized
    with the `__deterministic__` flag.
    This is essentially useful for testing.

    The decorator will raise an error if the object does not have a `__deterministic__` attribute.
    """
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        if args[0].__deterministic__:
            np.random.seed(100)
        retval = fun(*args, **kwargs)
        return retval
    return wrapper
