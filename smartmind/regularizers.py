from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import logging
import numbers
import tensorflow as tf
from collections import OrderedDict

from .utils import get_from_module
from .utils import process_params

__all__ = ['l1_regularizer',
           'l2_regularizer',
           'l1_l2_regularizer',
           'sum_regularizer']


def l1_regularizer(scale=0.01, scope=None):
    """Returns a function that can be used to apply L1 regularization to weights.

    L1 regularization encourages sparsity.

    Parameters
    ----------
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

    Returns
    --------
    A function with signature `l1(weights)` that apply L1 regularization.

    Raises
    ------
    ValueError: If scale is negative or if scale is not a float.
    """
    if isinstance(scale, numbers.Integral):
        raise ValueError('Scale cannot be an integer: %s' % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' % scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _: None

    def l1(weights):
        """Applies L1 regularization to weights."""
        with tf.name_scope(scope, 'l1_regularizer', [weights]) as name:
            my_scale = tf.convert_to_tensor(scale,
                                            dtype=weights.dtype.base_dtype,
                                            name='scale')
            return tf.mul(my_scale, tf.reduce_sum(tf.abs(weights)), name=name)

    return l1


def l2_regularizer(scale=0.01, scope=None):
    """Returns a function that can be used to apply L2 regularization to weights.

    Small values of L2 can help prevent overfitting the training data.

    Parameters
    ----------
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

    Returns
    --------
    A function with signature `l2(weights)` that applies L2 regularization.

    Raises
    ------
    ValueError: If scale is negative or if scale is not a float.
    """
    if isinstance(scale, numbers.Integral):
        raise ValueError('Scale cannot be an integer: %s' % (scale,))
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                             scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _: None

    def l2(weights):
        """Applies l2 regularization to weights."""
        with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
            my_scale = tf.convert_to_tensor(scale,
                                            dtype=weights.dtype.base_dtype,
                                            name='scale')
            return tf.mul(my_scale, tf.nn.l2_loss(weights), name=name)

    return l2


def l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0, scope=None):
    """Returns a function that can be used to apply L1 L2 regularizations.

    Parameters
    ----------
    scale_l1: A scalar multiplier `Tensor` for L1 regularization.
    scale_l2: A scalar multiplier `Tensor` for L2 regularization.
    scope: An optional scope name.

    Returns
    --------
    A function with signature `l1_l2(weights)` that applies a weighted sum of
    L1 L2  regularization.

    Raises
    ------
    ValueError: If scale is negative or if scale is not a float.
    """
    scope = scope or 'l1_l2_regularizer'
    return sum_regularizer([l1_regularizer(scale_l1),
                            l2_regularizer(scale_l2)],
                           scope=scope)


def sum_regularizer(regularizer_list, scope=None):
    """Returns a function that applies the sum of multiple regularizers.

    Parameters
    ----------
    regularizer_list: A list of regularizers to apply.
    scope: An optional scope name

    Returns
    --------
    A function with signature `sum_reg(weights)` that applies the
    sum of all the input regularizers.
    """
    regularizer_list = [reg for reg in regularizer_list if reg is not None]
    if not regularizer_list:
        return None

    def sum_reg(weights):
        """Applies the sum of all the input regularizers."""
        with tf.name_scope(scope, 'sum_regularizer', [weights]) as name:
            regularizer_tensors = [reg(weights) for reg in regularizer_list]
            return tf.add_n(regularizer_tensors, name=name)

    return sum_reg


def _get_defaults():
    return {
        'l1': OrderedDict([('scale', 0.01), ('scope', None)]),
        'l2': OrderedDict([('scale', 0.01), ('scope', None)]),
        'l1_l2': OrderedDict([('scale_l1', 0.01), ('scale_l2', 0.01), ('scope', None)]),
        'sum': OrderedDict([('regularizer_list', None), ('scope', None)])
    }


def get(name, kwargs=None):
    fn_dict = {
        'l1': l1_regularizer,
        'l2': l2_regularizer,
        'l1_l2': l1_l2_regularizer,
        'sum': sum_regularizer
    }

    return get_from_module(name, fn_dict, _get_defaults(), True, True, kwargs)


def process_parameters(name, kwargs):
    return process_params(name, kwargs, _get_defaults())
