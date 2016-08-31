from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf


__all__ = ['to_tensor',
           'to_dtype',
           'tf_ndims',
           'tf_int_shape',
           'tf_max',
           'tf_min',
           'tf_sum',
           'tf_mean',
           'tf_clip',
           'tf_l2_normalize',
           'tf_variable_with_weight_decay']


def _normalize_axis(axis, ndims):
    axis = list(axis)
    for i, a in enumerate(axis):
        if axis is not None and a < 0:
            axis[i] = a % ndims
    if len(axis) == 1:
        return axis[0]
    return axis


def to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def to_dtype(str_dtype):
    if str_dtype == 'float16':
        return tf.float16
    if str_dtype == 'float32':
        return tf.float32
    elif str_dtype == 'float64':
        return tf.float64
    elif str_dtype == 'int16':
        return tf.int16
    elif str_dtype == 'int32':
        return tf.int32
    elif str_dtype == 'int64':
        return tf.int64
    elif str_dtype == 'uint8':
        return tf.int8
    elif str_dtype == 'uint16':
        return tf.uint16
    else:
        raise ValueError('Unsupported dtype:', str_dtype)


def tf_ndims(x):
    return x.get_shape().ndims


def tf_int_shape(x):
    """Returns the shape of a tensor as a tuple of integers
    or None entries."""
    return tuple(x.get_shape().as_list())


def tf_max(x, axis=None, keep_dims=False):
    axis = _normalize_axis(axis, tf_ndims(x))
    return tf.reduce_max(x, reduction_indices=axis, keep_dims=keep_dims)


def tf_min(x, axis=None, keep_dims=False):
    axis = _normalize_axis(axis, tf_ndims(x))
    return tf.reduce_min(x, reduction_indices=axis, keep_dims=keep_dims)


def tf_sum(x, axis=None, keep_dims=False):
    axis = _normalize_axis(axis, tf_ndims(x))
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keep_dims)


def tf_mean(x, axis=None, keep_dims=False):
    """Computes the mean of elements accross dimensions of a tensor."""
    axis = _normalize_axis(axis, tf_ndims(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, tf.float32)
    return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keep_dims)


def tf_clip(x, min_value, max_value):
    """Element-wise clipping of tensor values to a specified min and max."""
    if max_value < min_value:
        max_value = min_value
    min_value = to_tensor(min_value, x.dtype.base_dtype)
    max_value = to_tensor(max_value, x.dtype.base_dtype)
    return tf.clip_by_value(x, min_value, max_value)


def tf_l2_normalize(x, axis):
    """Normalizes along dimension axis."""
    if axis < 0:
        axis %= len(x.get_shape())
    return tf.nn.l2_normalize(x, dim=axis)


def tf_variable_with_weight_decay(name, shape=None, dtype=None, initializer=None, wd=None):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Parameters
    ----------
    name: str
        The name of the new or existing variable.
    shape: list or tuple
        Shape of the new or existing variable.
    dtype:
        Type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer:
        Initializer for the variable if one is created.
    wd: float, optional
        Add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns
    -------
    Variable Tensor
    """
    var = tf.get_variable(name, shape, dtype, initializer=initializer)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

