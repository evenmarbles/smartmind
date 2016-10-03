from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .. import losses

from . import tolist

__all__ = ['to_tensor',
           'to_dtype',
           'tf_ndims',
           'tf_int_shape',
           'tf_concatenate',
           'tf_max',
           'tf_min',
           'tf_sum',
           'tf_mean',
           'tf_argmax',
           'tf_argmin',
           'tf_prod',
           'tf_sqrt',
           'tf_clip',
           'tf_l2_normalize',
           'tf_variable_with_loss',
           'tf_batch_dot',
           # 'tf_batch_set_value',
           ]


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
        return tf.uint8
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


def tf_concatenate(tensors, axis=-1):
    """Concantes a list of tensors alongside the specified axis."""
    if axis < 0:
        if len(tensors[0].get_shape()):
            axis %= len(tensors[0].get_shape())
        else:
            axis = 0
    return tf.concat(axis, tensors)


# ELEMENT-WISE OPERATIONS

def _normalize_axis(axis, ndims):
    if axis is None:
        return

    axis = tolist(axis)
    for i, a in enumerate(axis):
        if axis is not None and a < 0:
            axis[i] = a % ndims
    if len(axis) == 1:
        axis = axis[0]
        if axis < 0:
            axis = axis % ndims
    return axis


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


def tf_argmax(x, axis=-1):
    """Returns the index of the maximum value
    along a tensor axis.
    """
    if axis < 0:
        axis %= len(x.get_shape())
    return tf.argmax(x, axis)


def tf_argmin(x, axis=-1):
    """Returns the index of the minimum value
    along a tensor axis.
    """
    if axis < 0:
        axis %= len(x.get_shape())
    return tf.argmin(x, axis)


def tf_prod(x, axis=None, keep_dims=False):
    """Multiplies the values in a tensor, alongside the specified axis."""
    axis = _normalize_axis(axis, tf_ndims(x))
    return tf.reduce_prod(x, reduction_indices=axis, keep_dims=keep_dims)


def tf_sqrt(x):
    """Element-wise square root."""
    zero = to_tensor(0., x.dtype.base_dtype)
    inf = to_tensor(np.inf, x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, inf)
    return tf.sqrt(x)


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


def tf_variable_with_loss(name, shape=None, dtype=None, initializer=None,
                          loss='l1_loss', weight=None):
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
    loss: str, optional
        Loss function. If None, loss is not added to this Variable
    weight: float
        Loss weight. If None, weight decay is not added for this Variable.

    Returns
    -------
    Variable Tensor
    """
    var = tf.get_variable(name, shape, dtype, initializer=initializer)
    if loss is not None and weight is not None:
        losses.get(loss)(var, weight, name)
    return var


def tf_batch_dot(x, y, axes=None):
    """Batch-wise dot product.

    batch_dot results in a tensor with less dimensions than the input.
    If the number of dimensions is reduced to 1, we use `expand_dims` to
    make sure that ndim is at least 2.

    # Arguments
        x, y: tensors with ndim >= 2
        axes: list (or single) int with target dimensions

    # Returns
        A tensor with shape equal to the concatenation of x's shape
        (less the dimension that was summed over) and y's shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to (batch_size, 1).

    # Examples
        Assume x = [[1, 2], [3, 4]]   and y = [[5, 6], [7, 8]]
        batch_dot(x, y, axes=1) = [[17, 53]] which is the main diagonal
        of x.dot(y.T), although we never have to calculate the off-diagonal
        elements.

        Shape inference:
        Let x's shape be (100, 20) and y's shape be (100, 30, 20).
        If dot_axes is (1, 2), to find the output shape of resultant tensor,
            loop through each dimension in x's shape and y's shape:
        x.shape[0] : 100 : append to output shape
        x.shape[1] : 20 : do not append to output shape,
            dimension 1 of x has been summed over. (dot_axes[0] = 1)
        y.shape[0] : 100 : do not append to output shape,
            always ignore first dimension of y
        y.shape[1] : 30 : append to output shape
        y.shape[2] : 20 : do not append to output shape,
            dimension 2 of y has been summed over. (dot_axes[1] = 2)

        output_shape = (100, 30)
    """
    if type(axes) == int:
        axes = (axes, axes)
    if axes is not None:
        adj_x = None if axes[0] == tf_ndims(x) - 1 else True
        adj_y = True if axes[1] == tf_ndims(y) - 1 else None
    else:
        adj_x = None
        adj_y = None
    out = tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
    if tf_ndims(out) == 1:
        out = tf.expand_dims(out, 1)
    return out

#
# def tf_batch_set_value(tuples):
#     """Sets the values of many tensor variables at once.
#
#     Parameters
#     ----------
#     tuples: a list of tuples `(tensor, value)`.
#         `value` should be a Numpy array.
#     """
#     if tuples:
#         assign_ops = []
#         feed_dict = {}
#         for x, value in tuples:
#             value = np.asarray(value)
#             tf_dtype = to_dtype(x.dtype.name.split('_')[0])
#             assign_placeholder = tf.placeholder(tf_dtype, shape=value.shape)
#             assign_ops.append(x.assign(assign_placeholder))
#             feed_dict[assign_placeholder] = value
#         get_session().run(assign_ops, feed_dict=feed_dict)
