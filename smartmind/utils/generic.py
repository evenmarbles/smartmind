from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
from six import iteritems
import sys
import time
import numbers
import numpy as np

from collections import defaultdict


__all__ = ['process_params',
           'get_from_module',
           'epsilon',
           'get_uid',
           'reset_uids',
           'check_random_state',
           'tolist',
           'flatten_list',
           'dotdict',
           'Progbar']

_EPSILON = 10e-8
_UIDS = defaultdict(int)


def get_from_module(identifier, fn_dict, defaults,
                    instantiate=False, eval_params=False, params=None):
    if isinstance(identifier, six.string_types):
        res = fn_dict[identifier]

        if instantiate and params is None:
            return res(**process_params(identifier, {}, defaults))
        if instantiate and params is not None:
            return res(**process_params(identifier, params, defaults))
        if eval_params and params is not None:
            process_params(identifier, params, defaults)
        return res
    elif isinstance(identifier, dict):
        name = identifier.pop('name')
        res = fn_dict[name]
        if res:
            return res(**process_params(name, identifier, defaults))
        else:
            raise Exception('Invalid {}'.format(name))
    elif isinstance(identifier, (list, tuple)):
        name = identifier.pop(0)
        res = fn_dict[name]
        if res:
            # noinspection PyTypeChecker
            return res(**process_params(name, identifier, defaults))
        else:
            raise Exception('Invalid {}'.format(name))
    return identifier


def process_params(identifier, params, defaults):
    if params is None:
        return {}

    default = defaults[identifier]

    if isinstance(params, dict):
        if not all(k in default for k in params.keys()):
            raise Exception('{}: Parameter mismatch.'.format(identifier))
        p = default.copy()
        p.update(params)
        return p

    params = tolist(params)
    if len(params) > len(default):
        raise Exception('{}: Too many parameters given.'.format(identifier))

    params = dict(zip(default.keys(), params))
    for k, v in iteritems(default):
        params.setdefault(k, v)
    return params


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


def tolist(x, convert_tuple=True):
    """Returns the object as a list"""
    if x is None:
        return []
    typelist = (list,)
    if convert_tuple:
        typelist += (tuple,)
    if isinstance(x, typelist):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()

    return [x]


def flatten_list(x):
    """Returns the flattened list"""
    return [item for sublist in x for item in tolist(sublist)]


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Progbar(object):
    def __init__(self, target, width=30, verbose=1, interval=0.01):
        """
            @param target: total number of steps expected
            @param interval: minimum visual progress update interval (in seconds)
        """
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        """
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
            @param force: force visual progress update
        """
        values = values if values is not None else []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if type(self.sum_values[k]) is list:
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=None):
        values = values if values is not None else []
        self.update(self.seen_so_far + n, values)
