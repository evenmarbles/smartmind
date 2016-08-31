from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from collections import OrderedDict

from .utils import to_dtype
from .utils import tolist


def _process_parameters(seed, dtype):
    tf_dtype = to_dtype(dtype)

    if seed is None:
        seed = np.random.randint(10e8)

    return seed, tf_dtype


def uniform(minval=0.0, maxval=1.0, seed=None, dtype='float32'):
    seed, tf_dtype = _process_parameters(seed, dtype)
    return tf.random_uniform_initializer(minval, maxval,
                                         seed=seed,
                                         dtype=tf_dtype)


def normal(mean=0.0, stddev=1.0, seed=None, dtype='float32'):
    seed, tf_dtype = _process_parameters(seed, dtype)
    return tf.random_normal_initializer(mean, stddev=stddev,
                                        seed=seed,
                                        dtype=tf_dtype)


def truncated_normal(mean=0.0, stddev=1.0, seed=None, dtype='float32'):
    seed, tf_dtype = _process_parameters(seed, dtype)
    return tf.truncated_normal_initializer(mean, stddev=stddev,
                                           seed=seed,
                                           dtype=tf_dtype)


def constant(value=0.0, dtype='float32'):
    tf_dtype = to_dtype(dtype)
    return tf.constant_initializer(value, dtype=tf_dtype)


def xavier_uniform(seed=None, dtype='float32'):
    seed, tf_dtype = _process_parameters(seed, dtype)
    return tf.contrib.layers.xavier_initializer(uniform=True,
                                                seed=seed,
                                                dtype=tf_dtype)


def xavier_normal(seed=None, dtype='float32'):
    seed, tf_dtype = _process_parameters(seed, dtype)
    return tf.contrib.layers.xavier_initializer(uniform=False,
                                                seed=seed,
                                                dtype=tf_dtype)


def lecun_uniform(seed=None, dtype='float32'):
    """Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    seed, tf_dtype = _process_parameters(seed, dtype)
    return tf.contrib.layers.variance_scaling_initializer(factor=3.,
                                                          mode='FAN_IN',
                                                          uniform=True,
                                                          seed=seed,
                                                          dtype=tf_dtype)


def he_uniform(seed=None, dtype='float32'):
    seed, tf_dtype = _process_parameters(seed, dtype)
    return tf.contrib.layers.variance_scaling_initializer(factor=6.,
                                                          mode='FAN_IN',
                                                          uniform=True,
                                                          seed=seed,
                                                          dtype=tf_dtype)


def he_normal(seed=None, dtype='float32'):
    seed, tf_dtype = _process_parameters(seed, dtype)
    return tf.contrib.layers.variance_scaling_initializer(factor=2.,
                                                          mode='FAN_IN',
                                                          uniform=True,
                                                          seed=seed,
                                                          dtype=tf_dtype)


def get(name, params, instantiate=False):
    def eval_params(p):
        default = {
            'uniform': OrderedDict([('minval', 0.0), ('maxval', 1.0), ('seed', None), ('dtype', 'float32')]),
            'normal': OrderedDict([('mean', 0.0), ('stddev', 1.0), ('seed', None), ('dtype', 'float32')]),
            'truncated_normal': OrderedDict([('mean', 0.0), ('stddev', 1.0), ('seed', None), ('dtype', 'float32')]),
            'constant': OrderedDict([('value', 0.0), ('dtype', 'float32')]),
            'xavier_uniform': OrderedDict([('seed', None), ('dtype', 'float32')]),
            'xavier_normal': OrderedDict([('seed', None), ('dtype', 'float32')]),
            'lecun_uniform': OrderedDict([('seed', None), ('dtype', 'float32')]),
            'he_uniform': OrderedDict([('seed', None), ('dtype', 'float32')]),
            'he_normal': OrderedDict([('seed', None), ('dtype', 'float32')]),
        }[name]

        if p is None:
            return {}

        if isinstance(p, dict):
            if not all(k in default for k in p.keys()):
                raise Exception('{}: Initializer parameter mismatch.'.format(name))
            return p

        p = tolist(p)
        if len(p) > len(default):
            raise Exception('{}: Too many initializer parameters given.'.format(name))
        return dict(zip(default.keys(), p))

    initializer = {
        'uniform': uniform,
        'normal': normal,
        'truncated_normal': truncated_normal,
        'constant': constant,
        'xavier_uniform': xavier_uniform,
        'xavier_normal': xavier_normal,
        'lecun_uniform': lecun_uniform,
        'he_uniform': he_uniform,
        'he_normal': he_normal
    }[name]

    if instantiate:
        return initializer(**eval_params(params))
    return initializer
