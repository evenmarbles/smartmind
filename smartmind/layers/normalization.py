from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from ..framework import Layer


class LRN(Layer):
    def __init__(self, depth_radius=None, bias=None, alpha=None, beta=None, input_shape=None,
                 input_dtype='float32', batch_size=None, input_default=None, sparse_input=False,
                 reader_input=None, name=None):
        super(LRN, self).__init__(input_shape, input_dtype, batch_size, input_default,
                                  sparse_input, reader_input, name)

        self._depth_radius = depth_radius
        self._bias = bias
        self._alpha = alpha
        self._beta = beta

    def _call(self, x):
        return tf.nn.local_response_normalization(x, self._depth_radius,
                                                  bias=self._bias,
                                                  alpha=self._alpha,
                                                  beta=self._beta,)


class BatchNormalization(object):
    pass
