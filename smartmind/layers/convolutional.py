from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from .. import initializers
from .. import activations
from ..framework import Layer
from ..utils import tf_variable_with_weight_decay
from ..utils import conv_output_length


class Conv2d(Layer):

    def __init__(self, output_dim, kernel_size, strides, init='xavier_uniform', init_params=None,
                 activation='linear', activation_params=None, padding='VALID', data_format='NHWC',
                 bias=True, bias_init=0.0, weight_decay=None, input_shape=None, input_dtype='float32',
                 batch_size=None, input_default=None, sparse_input=False, reader_input=None, name=None):
        super(Conv2d, self).__init__(input_shape, input_dtype, batch_size, input_default,
                                     sparse_input, reader_input, name)

        self._output_dim = output_dim
        self._kernel_size = kernel_size
        self._strides = strides

        self._init = initializers.get(init, init_params, True)

        self._activation = activations.get(activation)
        self._activation_params = activations.check_params(activation, activation_params)

        if padding not in {'VALID', 'SAME'}:
            raise Exception('Invalid padding for Conv2d:', padding)
        self._padding = padding

        if data_format not in {'NCHW', 'NHWC'}:
            raise Exception('Invalid data format for Conv2d: {}'.format(data_format))
        self._data_format = data_format

        self._bias = bias
        self._bias_init = bias_init
        self._weight_decay = weight_decay

        self._w = None
        self._b = None

    def build(self, shape):
        super(Conv2d, self).build(shape)

        with tf.variable_scope(self.name):
            if self._data_format == 'NCHW':
                self._strides = [1, 1, self._strides[0], self._strides[1]]
                kernel_shape = [self._kernel_size[0], self._kernel_size[1], shape[1], self._output_dim]
            else:
                self._strides = [1, self._strides[0], self._strides[1], 1]
                kernel_shape = [self._kernel_size[0], self._kernel_size[1], shape[-1], self._output_dim]

            self._w = tf_variable_with_weight_decay('w', kernel_shape, tf.float32,
                                                    initializer=self._init,
                                                    wd=self._weight_decay)

            if self._bias:
                self._b = tf.get_variable('b', [self._output_dim], initializer=initializers.constant(self._bias_init))

    def _call(self, x):
        output = tf.nn.conv2d(x, self._w, self._strides, self._padding, data_format=self._data_format)
        if self._bias:
            output = tf.nn.bias_add(output, self._b, self._data_format)
        return self._activation(output, **self._activation_params)

    def _get_output_shape(self, input_shape):
        """Computes the output shape of the layer given an input shape.
        This function assumes that the layer will be built to match the input shape.

        Parameters
        ----------
        input_shape: tuple[int] or list[tuple[int]]
            The input shape(s) for the layer. Shape tuples can include
            None for free dimensions, instead of integer.
        """
        if self._data_format == 'NCHW':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_output_length(rows, self._kernel_size[0], self._padding, self._strides[0])
        cols = conv_output_length(cols, self._kernel_size[1], self._padding, self._strides[1])

        if self._data_format == 'NCHW':
            return input_shape[0], self._output_dim, rows, cols
        else:
            return input_shape[0], rows, cols, self._output_dim
