from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from .. import initializers
from .. import activations
from .. import regularizers

from ..framework import Layer
from ..utils import conv_output_length


class Conv2d(Layer):

    def __init__(self, output_dim, kernel_size, strides=None, init='xavier_uniform', init_params=None,
                 initial_weights=None, w_regularizer=None, bias=True, b_regularizer=None, trainable=True,
                 activation='linear', activation_params=None, padding='VALID', data_format='NHWC',
                 input_shape=None, input_dtype=None, batch_size=None, name=None):

        self._output_dim = output_dim
        self._kernel_size = kernel_size
        self._strides = strides if strides is not None else [1, 1]

        self._init = initializers.get(init, kwargs=init_params)
        self._initial_weights = initial_weights

        self._activation = activations.get(activation, activation_params)
        self._activation_params = activations.process_parameters(activation, activation_params)

        self._bias = bias

        self._w_regularizer = regularizers.get(w_regularizer)
        self._b_regularizer = regularizers.get(b_regularizer)

        if padding not in {'VALID', 'SAME'}:
            raise Exception('Invalid padding for Conv2d:', padding)
        self._padding = padding

        if data_format not in {'NCHW', 'NHWC'}:
            raise Exception('Invalid data format for Conv2d: {}'.format(data_format))
        self._data_format = data_format

        self._w = None
        self._b = None

        super(Conv2d, self).__init__(trainable, name, input_shape=input_shape, input_dtype=input_dtype,
                                     batch_size=batch_size)

    def build(self, shape):
        super(Conv2d, self).build(shape)

        with tf.variable_scope(self.name):
            if self._data_format == 'NCHW':
                self._strides = [1, 1, self._strides[0], self._strides[1]]
                kernel_shape = [self._kernel_size[0], self._kernel_size[1], shape[1], self._output_dim]
            else:
                self._strides = [1, self._strides[0], self._strides[1], 1]
                kernel_shape = [self._kernel_size[0], self._kernel_size[1], shape[-1], self._output_dim]

            if self._initial_weights is not None and self._initial_weights[0] is not None:
                self._w = tf.get_variable('w', kernel_shape, tf.float32,
                                          initializer=initializers.constant(self._initial_weights[0]),
                                          regularizer=self._w_regularizer)
            else:
                self._w = tf.get_variable('w', kernel_shape, tf.float32,
                                          initializer=self._init,
                                          regularizer=self._w_regularizer)

            if self._bias:
                initial_bias = 0.0
                if self._initial_weights is not None and self._initial_weights[1] is not None:
                    initial_bias = self._initial_weights[1]
                self._b = tf.get_variable('b', [self._output_dim],
                                          initializer=initializers.constant(initial_bias),
                                          regularizer=self._b_regularizer)

    def _call(self, inputs):
        output = tf.nn.conv2d(inputs, self._w, self._strides, self._padding, data_format=self._data_format)
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
