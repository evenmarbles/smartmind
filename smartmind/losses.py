from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from collections import OrderedDict

from .utils import get_from_module
from .utils import process_params


LOSSES = tf.GraphKeys().LOSSES


__all__ = ['add_loss',
           'get_losses',
           'get_regularization_losses',
           'get_total_loss',
           'l1_loss',
           'l2_loss',
           'get',
           'process_parameters']


def add_loss(loss):
    """Adds an externally defined loss to collection of losses.

    Parameters
    ----------
    loss: A loss `Tensor`.
    """
    tf.add_to_collection(LOSSES, loss)


def get_losses(scope=None):
    """Gets the list of loss variables stored in the collection of losses.

    Parameters
    ----------
    scope: an optional scope for filtering the losses to return.

    Returns
    -------
    a list of loss variables.
    """
    return tf.get_collection(LOSSES, scope)


def get_regularization_losses(scope=None):
    """Gets the regularization losses.

    Parameters
    ----------
    scope: an optional scope for filtering the losses to return.

    Returns
    -------
    A list of loss variables.
    """
    return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope)


def get_total_loss(add_regularization_losses=True, name="total_loss"):
    """Returns a tensor whose value represents the total loss.

    Notice that the function adds the given losses to the regularization losses.

    Parameters
    ----------
    add_regularization_losses: A boolean indicating whether or not to use the
        regularization losses in the sum.
    name: The name of the returned tensor.

    Returns
    -------
    A `Tensor` whose value represents the total loss.

    Raises
    ------
    ValueError: if `losses` is not iterable.
    """
    losses = get_losses()
    if add_regularization_losses:
        losses += get_regularization_losses()
    return tf.add_n(losses, name=name)


def l1_loss(tensor, weight=1.0, scope=None):
    """Define a L1Loss, useful for regularize, i.e. lasso.

    Parameters
    ----------
    tensor: tensor to regularize.
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.

    Returns
    -------
    the L1 loss op.
    """
    with tf.op_scope([tensor], scope, 'L1Loss'):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.mul(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
        add_loss(loss)
        return loss


def l2_loss(tensor, weight=1.0, scope=None):
    """Define a L2Loss, useful for regularize, i.e. weight decay.

    Parameters
    ----------
    tensor: tensor to regularize.
    weight: an optional weight to modulate the loss.
    scope: Optional scope for op_scope.

    Returns
    -------
    the L2 loss op.
    """
    with tf.op_scope([tensor], scope, 'L2Loss'):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
        add_loss(loss)
        return loss


def _get_defaults():
    return {
        'l1': OrderedDict([('scale', 0.01), ('scope', None)]),
        'l2': OrderedDict([('scale', 0.01), ('scope', None)]),
        # 'l1_l2': OrderedDict([('scale_l1', 0.01), ('scale_l2', 0.01), ('scope', None)]),
        # 'sum': OrderedDict([('regularizer_list', None), ('scope', None)])
    }


def get(name, kwargs=None):
    fn_dict = {
        'l1': l1_loss,
        'l2': l2_loss,
        # 'l1_l2': l1_l2_loss,
        # 'sum': sum_loss
    }

    return get_from_module(name, fn_dict, _get_defaults(), True, True, kwargs)


def process_parameters(name, kwargs):
    return process_params(name, kwargs, _get_defaults())
