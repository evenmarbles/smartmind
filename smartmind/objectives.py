from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .utils import to_tensor
from .utils import tf_mean
from .utils import tf_sum
from .utils import tf_clip
from .utils import tf_l2_normalize
from .utils import epsilon
from .losses import add_loss

__all__ = ['compute_weighted_objective',
           'get',
           'process_parameters']


def compute_weighted_objective(fn):
    """Computes the weighted loss.

    Notes
    -----
    Transforms an objective function `fn(y_pred, y_target)`
    into a sample-weighted, cost-masked loss function
    `fn(y_pred, y_target, weights, mask)`.
    """

    def _compute_weighted_loss(y_pred, y_target, sample_weight, loss_weight, mask=None):
        losses = fn(y_pred, y_target)
        if mask is not None:
            pass

        # reduce score to same dim as weight array
        if sample_weight is not None:
            input_dtype = losses.dtype
            if sample_weight.get_shape().ndims is None:
                raise ValueError("sample_weight.get_shape().ndims cannot be None")

            start_index = max(0, sample_weight.get_shape().ndims)
            reduction_indices = list(range(start_index, losses.get_shape().ndims))
            losses = tf_mean(losses, axis=reduction_indices)

            # apply sample weighting
            losses *= sample_weight
            losses /= tf_mean(tf.cast(tf.not_equal(sample_weight, 0), input_dtype))
        mean_loss = tf_mean(losses)

        # apply loss weighting
        mean_loss *= loss_weight
        add_loss(mean_loss)
        return mean_loss

    return _compute_weighted_loss


def mean_squared_error(output, target):
    return tf_mean(tf.square(output - target), axis=-1)


def mean_absolute_error(output, target):
    return tf_mean(tf.abs(output - target), axis=-1)


def mean_absolute_percentage_error(output, target):
    diff = tf.abs((target - output) / tf_clip(tf.abs(target), epsilon(), np.inf))
    return 100. * tf_mean(diff, axis=-1)


def mean_squared_logarithmic_error(output, target):
    first_log = tf.log(tf_clip(output, epsilon(), np.inf) + 1.)
    second_log = tf.log(tf_clip(target, epsilon(), np.inf) + 1.)
    return tf_mean(tf.square(first_log - second_log), axis=-1)


def squared_hinge(output, target):
    return tf_mean(tf.square(tf.maximum(1. - target * output, 0.)), axis=-1)


def hinge(output, target):
    return tf_mean(tf.maximum(1. - target * output, 0.), axis=-1)


def categorical_crossentropy(output, target, from_logits=True):
    """Categorical cross entropy between an output tensor and a target tensor,
    where the target is a tensor of the same shape as the output tensor.

    Note: Expects a binary class matrix instead of a vector of scalar classes.
    """
    if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(output, target)

    output /= tf.reduce_sum(output,
                            reduction_indices=len(output.get_shape()) - 1,
                            keep_dims=True)
    # manual computation of cross entropy
    eps = to_tensor(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, eps, 1. - eps)
    return -tf.reduce_sum(target * tf.log(output),
                          reduction_indices=len(output.get_shape()) - 1)


def sparse_categorical_crossentropy(output, target, from_logits=True):
    """Categorical cross entropy between an output tensor and a target tensor,
    where the target is an integer tensor.

    Note: Expects an array of integer classes.
    Note: target shape must have the same number of dimensions as output shape.
    If you get a shape error, add a length-1 dimension to labels.
    """
    if not from_logits:
        # transform back to logits
        eps = to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, eps, 1. - eps)
        output = tf.log(output)

    output_shape = output.get_shape()
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.reshape(output, [-1, int(output_shape[-1])]),
                                                         tf.cast(tf.reshape(target, [-1]), tf.int64))
    if len(output_shape) == 3:
        # if output includes timesteps, reshape is required
        return tf.reshape(res, [-1, int(output_shape[-2])])
    return res


def binary_crossentropy(output, target, from_logits=True):
    """Binary cross entropy between an output tensor and a target tensor."""
    if not from_logits:
        # transform back to logits
        eps = to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, eps, 1. - eps)
        output = tf.log(output / (1 - output))

    return tf_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, target), axis=-1)


def kullback_leibler_divergence(output, target):
    y_true = tf_clip(target, epsilon(), 1)
    output = tf_clip(output, epsilon(), 1)
    return tf_sum(y_true * tf.log(y_true / output), axis=-1)


def poisson(output, target):
    return tf_mean(output - target * tf.log(output + epsilon()), axis=-1)


def cosine_proximity(output, target):
    target = tf_l2_normalize(target, axis=-1)
    output = tf_l2_normalize(output, axis=-1)
    return -tf_mean(target * output, axis=-1)


def vr_class_reward(output, target, scale=1, average_loss=True, baseline_objective='mse'):
    output_shape = output[0].get_shape()
    loss, _, _ = tf.contrib.reinforce.vr_class_reward(tf.reshape(output[0], [-1, int(output_shape[-1])]),
                                                      tf.reshape(output[1], [-1]),
                                                      tf.cast(tf.reshape(target, [-1]), tf.int64),
                                                      scale=scale,
                                                      average=average_loss,
                                                      objective=baseline_objective)
    return loss


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
        'kld': kullback_leibler_divergence,
        'poisson': poisson,
        'cosine': cosine_proximity,
        'vr_class_reward': vr_class_reward,
    }[name]
