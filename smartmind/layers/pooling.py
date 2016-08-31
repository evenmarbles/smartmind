from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from ..framework import Layer
from ..utils import conv_output_length


def _pool2d(x, ksize, strides, padding, data_format, mode='max'):
    strides = [1, strides[0], strides[1], 1]
    ksize = [1, ksize[0], ksize[1], 1]

    if data_format == 'NCHW':
        x = tf.transpose(x, (2, 3, 1, 0))

    if mode == 'max':
        x = tf.nn.max_pool(x, ksize, strides, padding=padding)
    elif mode == 'avg':
        x = tf.nn.avg_pool(x, ksize, strides, padding=padding)
    else:
        raise Exception('Invalid pooling mode: {}'.format(mode))

    if data_format == 'NCHW':
        x = tf.transpose(x, (0, 3, 1, 2))
    return x


class _Pooling(Layer):
    def __init__(self, ksize, strides, padding='VALID', data_format='NHWC', input_shape=None,
                 input_dtype='float32', batch_size=None, input_default=None, sparse_input=False,
                 reader_input=None, name=None):
        super(_Pooling, self).__init__(input_shape, input_dtype, batch_size, input_default,
                                       sparse_input, reader_input, name)
        self._ksize = ksize
        self._strides = strides

        if padding not in {'VALID', 'SAME'}:
            raise Exception('Invalid padding for Conv2d:', padding)
        self._padding = padding

        if data_format not in {'NCHW', 'NHWC'}:
            raise Exception('Invalid data format for Conv2d: {}'.format(data_format))
        self._data_format = data_format

    def _pooling_function(self, inputs):
        raise NotImplementedError

    def _call(self, x):
        return self._pooling_function(x)

    def _get_2d_output_shape(self, input_shape):
        if self._data_format == 'NCHW':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_output_length(rows, self._ksize[0],
                                  self._padding, self._strides[0])
        cols = conv_output_length(cols, self._ksize[1],
                                  self._padding, self._strides[1])

        if self._data_format == 'NCHW':
            return input_shape[0], input_shape[1], rows, cols
        else:
            return input_shape[0], rows, cols, input_shape[3]


class MaxPooling2D(_Pooling):
    def __init__(self, ksize, strides, padding='VALID', data_format='NHWC', input_shape=None,
                 input_dtype='float32', batch_size=None, input_default=None, sparse_input=False,
                 reader_input=None, name=None):
        super(MaxPooling2D, self).__init__(ksize, strides, padding, data_format, input_shape,
                                           input_dtype, batch_size, input_default, sparse_input,
                                           reader_input, name)

    def _pooling_function(self, inputs):
        return _pool2d(inputs, ksize=self._ksize,
                       strides=self._strides,
                       padding=self._padding,
                       data_format=self._data_format,
                       mode='max')

    def _get_output_shape(self, input_shape):
        return self._get_2d_output_shape(input_shape)


class AveragePooling2D(_Pooling):
    def __init__(self, ksize, strides, padding='VALID', data_format='NHWC', input_shape=None,
                 input_dtype='float32', batch_size=None, input_default=None, sparse_input=False,
                 reader_input=None, name=None):
        super(AveragePooling2D, self).__init__(ksize, strides, padding, data_format, input_shape,
                                           input_dtype, batch_size, input_default, sparse_input,
                                           reader_input, name)

    def _pooling_function(self, inputs):
        return _pool2d(inputs, ksize=self._ksize,
                       strides=self._strides,
                       padding=self._padding,
                       data_format=self._data_format,
                       mode='avg')

    def _get_output_shape(self, input_shape):
        return self._get_2d_output_shape(input_shape)
