from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from ..framework import Layer


class LRN(Layer):
    """Local Response Normalization layer"""
    def __init__(self, depth_radius=None, bias=None, alpha=None, beta=None, trainable=True,
                 input_shape=None, input_dtype=None, batch_size=None, name=None):

        self._depth_radius = depth_radius
        self._bias = bias
        self._alpha = alpha
        self._beta = beta

        super(LRN, self).__init__(trainable, name, input_shape=input_shape,
                                  input_dtype=input_dtype, batch_size=batch_size)

    def _call(self, x):
        return tf.nn.local_response_normalization(x, self._depth_radius,
                                                  bias=self._bias,
                                                  alpha=self._alpha,
                                                  beta=self._beta,)


class BatchNormalization(object):
    pass
