from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .utils import to_tensor
from .utils import tf_max
from .utils import tf_mean
from .utils import tf_argmax
from .utils import tf_clip
from .utils import tf_l2_normalize
from .utils import epsilon


def binary_accuracy(y_true, y_pred):
    return tf_mean(tf.equal(y_true, tf.round(y_pred)))


def categorical_accuracy(y_true, y_pred):
    return tf_mean(tf.equal(tf_argmax(y_true, axis=-1),
                            tf_argmax(y_pred, axis=-1)))


def sparse_categorical_accuracy(y_true, y_pred):
    return tf_mean(tf.equal(tf_max(y_true, axis=-1),
                            tf.cast(tf_argmax(y_pred, axis=-1), tf.float32)))


def mean_squared_error(y_true, y_pred):
    return tf_mean(tf.square(y_pred - y_true))


def mean_absolute_error(y_true, y_pred):
    return tf_mean(tf.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred):
    diff = tf.abs((y_true - y_pred) / tf_clip(tf.abs(y_true), epsilon(), np.inf))
    return 100. * tf_mean(diff)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = tf.log(tf_clip(y_pred, epsilon(), np.inf) + 1.)
    second_log = tf.log(tf_clip(y_true, epsilon(), np.inf) + 1.)
    return tf_mean(tf.square(first_log - second_log))


def squared_hinge(y_true, y_pred):
    return tf_mean(tf.square(tf.maximum(1. - y_true * y_pred, 0.)))


def hinge(y_true, y_pred):
    return tf_mean(tf.maximum(1. - y_true * y_pred, 0.))


def categorical_crossentropy(y_true, y_pred, from_logits=True):
    """Expects a binary class matrix instead of a vector of scalar classes."""
    if from_logits:
        return tf_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true))

    # scale preds so that the class probas of each sample sum to 1
    y_pred /= tf.reduce_sum(y_pred,
                            reduction_indices=len(y_pred.get_shape()) - 1,
                            keep_dims=True)
    # manual computation of cross entropy
    eps = to_tensor(epsilon(), y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, eps, 1. - eps)
    return tf_mean(-tf.reduce_sum(y_true * tf.log(output),
                                  reduction_indices=len(output.get_shape()) - 1))


def sparse_categorical_crossentropy(y_true, y_pred, from_logits=True):
    """expects an array of integer classes.
    Note: labels shape must have the same number of dimensions as output shape.
    If you get a shape error, add a length-1 dimension to labels.
    """
    if not from_logits:
        # transform back to logits
        eps = to_tensor(epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        y_pred = tf.log(y_pred)

    y_pred_shape = y_pred.get_shape()
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.reshape(y_pred, [-1, int(y_pred_shape[-1])]),
                                                         tf.cast(tf.reshape(y_true, [-1]), tf.int64))
    if len(y_pred_shape) == 3:
        # if output includes timesteps, reshape is required
        return tf_mean(tf.reshape(res, [-1, int(y_pred_shape[-2])]))
    return tf_mean(res)


def binary_crossentropy(y_true, y_pred, from_logits=True):
    """Binary cross entropy between an output tensor and a target tensor."""
    if not from_logits:
        # transform back to logits
        eps = to_tensor(epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        y_pred = tf.log(y_pred / (1 - y_pred))

    return tf_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true), axis=-1)


def poisson(y_true, y_pred):
    return tf_mean(y_pred - y_true * tf.log(y_pred + epsilon()))


def cosine_proximity(y_true, y_pred):
    y_true = tf_l2_normalize(y_true, axis=-1)
    y_pred = tf_l2_normalize(y_pred, axis=-1)
    return -tf_mean(y_true * y_pred)


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def get(name):
    return {
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'mape': mean_absolute_percentage_error,
        'msle': mean_squared_logarithmic_error,
        'squared_hinge': squared_hinge,
        'hinge': hinge,
        'categorical_xentropy': categorical_crossentropy,
        'sparse_categorical_xentropy': sparse_categorical_crossentropy,
        'binary_xentropy': binary_crossentropy,
        'poisson': poisson,
        'cosine': cosine_proximity,
    }[name]
