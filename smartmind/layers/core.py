from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .. import initializers
from .. import activations

from ..framework import Layer
from ..utils import tf_variable_with_weight_decay
from ..utils import tf_int_shape


class Activation(Layer):

    def __init__(self, activation, activation_params=None, input_shape=None, input_dtype='float32',
                 batch_size=None, input_default=None, sparse_input=False, reader_input=None, name=None):
        super(Activation, self).__init__(input_shape, input_dtype, batch_size, input_default,
                                         sparse_input, reader_input, name)

        self._activation = activations.get(activation)
        self._activation_params = activations.check_params(activation, activation_params)

    def _call(self, x):
        return self._activation(x)


class FullyConnected(Layer):
    def __init__(self, output_dim, init='xavier_uniform', init_params=None, activation='linear',
                 activation_params=None, bias=True, bias_init=0.0, weight_decay=None, input_shape=None,
                 input_dtype='float32', batch_size=None, input_default=None, sparse_input=False,
                 reader_input=None, name=None):
        super(FullyConnected, self).__init__(input_shape, input_dtype, batch_size, input_default,
                                             sparse_input, reader_input, name)

        self._output_dim = output_dim

        self._init = initializers.get(init, init_params, True)

        self._activation = activations.get(activation)
        self._activation_params = activations.check_params(activation, activation_params)

        self._bias = bias
        self._bias_init = bias_init
        self._weight_decay = weight_decay

        self._w = None
        self._b = None

    def build(self, shape):
        super(FullyConnected, self).build(shape)

        with tf.variable_scope(self.name):
            self._w = tf_variable_with_weight_decay('Matrix', [shape[1], self._output_dim], tf.float32,
                                                    initializer=self._init,
                                                    wd=self._weight_decay)
            if self._bias:
                self._b = tf.get_variable('bias', [self._output_dim],
                                          initializer=initializers.constant(self._bias_init))

    def _call(self, x):
        output = tf.matmul(x, self._w)
        if self._bias:
            output = tf.nn.bias_add(output, self._b)
        return self._activation(output)

    def _get_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self._output_dim


class Reshape(Layer):
    def __init__(self, target_shape, input_shape=None, input_dtype='float32',
                 batch_size=None, input_default=None, sparse_input=False,
                 reader_input=None, name=None):
        super(Reshape, self).__init__(input_shape, input_dtype, batch_size, input_default,
                                         sparse_input, reader_input, name)

        self._target_shape = target_shape

    def _call(self, x):
        # In case the target shape is not fully defined,
        # we need access to the shape of x.
        # solution:
        # 1) rely on x._keras_shape
        # 2) fallback: tf_int_shape
        target_shape = self._target_shape
        if -1 in target_shape:
            # target shape not fully defined
            if hasattr(x, '_sm_shape'):
                input_shape = x._sm_shape
            else:
                input_shape = tf_int_shape(x)

            target_shape = self._get_output_shape(input_shape)

        if target_shape[0] is None:
            target_shape = (-1,) + target_shape[1:]
        return tf.reshape(x, target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Find and replace a single missing dimension in an output shape
        given an input shape.

        A near direct port of the internal Numpy function _fix_unknown_dimension
        in numpy/core/src/multiarray/shape.c

        Parameters
        ----------
        input_shape:
            Shape of array being reshaped

        output_shape:
            Desired shape of the array with at most
            a single -1 which indicates a dimension that should be
            derived from the input shape.

        Returns
        -------
        The new output shape with a -1 replaced with its computed value.

        Raises a ValueError if the total array size of the output_shape is
        different then the input_shape, or more then one unknown dimension
        is specified.
        """
        output_shape = list(output_shape)

        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('can only specify one unknown dimension')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def _get_output_shape(self, input_shape):
        return (input_shape[0],) + self._fix_unknown_dimension(input_shape[1:], self._target_shape)
