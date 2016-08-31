from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from collections import OrderedDict

from .utils import tolist
from .utils import to_tensor
from .utils import tf_ndims
from .utils import tf_max
from .utils import tf_sum


def linear(x, name=None):
    return tf.identity(x, name=name)


def tanh(x, name=None):
    return tf.nn.tanh(x, name=name)


def sigmoid(x, name=None):
    return tf.nn.sigmoid(x, name=name)


def hard_sigmoid(x, name=None):
    x = (.2 * x) + .5
    zero = to_tensor(0., x.dtype.base_dtype)
    one = to_tensor(1., x.dtype.base_dtype)
    return tf.clip_by_value(x, zero, one, name=name)


def softmax(x, name=None):
    ndims = tf_ndims(x)
    if ndims == 2:
        return tf.nn.softmax(x, name=name)

    if ndims == 3:
        e = tf.exp(x - tf_max(x, axis=-1, keep_dims=True))
        s = tf_sum(e, axis=-1, keep_dims=True)
        return tf.identity(e / s, name=name)

    raise Exception('Softmax only defined for 2D and 3D tensors: ndims=' + str(ndims))


def softplus(x, name=None):
    return tf.nn.softplus(x, name=name)


def softsign(x, name=None):
    tf.nn.softsign(x, name=name)


def relu(x, alpha=0.0, max_value=None, name=None):
    x = tf.nn.relu(x, name=name)
    if max_value is not None:
        max_value = to_tensor(max_value, x.dtype.base_type)
        zero = to_tensor(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value, name=name)
    if alpha != 0:
        alpha = to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * tf.nn.relu(-x)
    return x


def get(name):
    return {
        'linear': linear,
        'tanh': tanh,
        'sigmoid': sigmoid,
        'hard_sigmoid': hard_sigmoid,
        'softmax': softmax,
        'softplus': softplus,
        'softsign': softsign,
        'relu': relu,
    }[name]


def get_default(name):
    return {
        'linear': OrderedDict([('name', None)]),
        'tanh': OrderedDict([('name', None)]),
        'sigmoid': OrderedDict([('name', None)]),
        'hard_sigmoid': OrderedDict([('name', None)]),
        'softmax': OrderedDict([('name', None)]),
        'softplus': OrderedDict([('name', None)]),
        'softsign': OrderedDict([('name', None)]),
        'relu': OrderedDict([('alpha', 0.0), ('max_value', None), ('name', None)]),
    }[name]


def check_params(name, params):
    if params is None:
        return {}

    default = get_default(name)

    if isinstance(params, dict):
        if not all(k in default for k in params.keys()):
            raise Exception('{}: Activations parameter mismatch.'.format(name))
        return params

    params = tolist(params)
    if len(params) > len(default):
        raise Exception('{}: Too many activations parameters given.'.format(name))
    return dict(zip(default.keys(), params))
