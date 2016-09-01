from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
import numbers
import numpy as np

from collections import defaultdict


__all__ = ['process_params',
           'get_from_module',
           'epsilon',
           'get_uid',
           'reset_uids',
           'check_random_state',
           'tolist']

_EPSILON = 10e-8
_UIDS = defaultdict(int)


def get_from_module(identifier, fn_dict, defaults,
                    instantiate=False, eval_params=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = fn_dict[identifier]

        if instantiate and kwargs is None:
            return res()
        if instantiate and kwargs is not None:
            return res(**process_params(identifier, kwargs, defaults))
        if eval_params and kwargs:
            process_params(identifier, kwargs, defaults)
        return res
    return identifier


def process_params(identifier, kwargs, defaults):
    if kwargs is None:
        return {}

    default = defaults[identifier]

    if isinstance(kwargs, dict):
        if not all(k in default for k in kwargs.keys()):
            raise Exception('{}: Parameter mismatch.'.format(identifier))
        return kwargs

    kwargs = tolist(kwargs)
    if len(kwargs) > len(default):
        raise Exception('{}: Too many parameters given.'.format(identifier))
    return dict(zip(default.keys(), kwargs))


def epsilon():
    """Return the value for epsilon"""
    return _EPSILON


def get_uid(prefix):
    _UIDS[prefix] += 1
    return _UIDS[prefix] - 1


def reset_uids():
    global _UIDS
    _UIDS = defaultdict(int)


# copy-pasted from scikit-learn utils/validation.py
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def tolist(x):
    """Returns the object as a list"""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()

    return [x]
