from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
import tensorflow as tf

from ..framework import Layer
from ..utils import tf_int_shape


class SpatialGlimpse(Layer):
    def __init__(self, size, depth, scale, trainable=True, data_format='NHWC', input_shape=None, input_dtype=None,
                 batch_size=None, name=None):

        if isinstance(size, (list, tuple)):
            self._height, self._width = size
        else:
            self._height = self._width = size

        self._depth = depth
        self._scale = scale

        height = self._height
        width = self._width
        self._sizes = [np.array([height, width])]

        for _ in range(1, self._depth):
            height *= self._scale
            width *= self._scale
            self._sizes.append(np.array([height, width]))

        if data_format not in {'NCHW', 'NHWC'}:
            raise Exception('Invalid data format for SpatialGlimpse: {}'.format(data_format))
        self._data_format = data_format

        super(SpatialGlimpse, self).__init__(trainable, name, input_shape=[input_shape, (2,)],
                                             input_dtype=input_dtype, batch_size=batch_size)

    def _call(self, inputs):
        super(SpatialGlimpse, self)._call(inputs)

        if not isinstance(inputs, list):
            raise Exception('SpatialGlimpse takes exactly two inputs')

        offsets = inputs[1]
        inputs = inputs[0]

        output = []
        for d in range(self._depth):
            o = tf.image.extract_glimpse(inputs, tf.constant(self._sizes[d], dtype=tf.int32, shape=[2]), offsets)
            if d > 0:
                kernel_size = self._sizes[d] / self._sizes[0]
                assert (np.any(kernel_size % 2) == 0.)
                o = tf.nn.avg_pool(o,
                                   ksize=[1, kernel_size[0], kernel_size[1], 1],
                                   strides=[1, kernel_size[0], kernel_size[1], 1],
                                   padding='VALID')
            output.append(o)

        input_shape = tf_int_shape(inputs)
        batch_size = (-1,) if input_shape[0] is None else (input_shape[0],)
        output_shape = batch_size + (self._height, self._width) + (self._depth,)

        return tf.reshape(tf.pack(output, axis=4), shape=output_shape)

    def _get_output_shape(self, input_shape):
        """Computes the output shape of the layer given an input shape.
        This function assumes that the layer will be built to match the input shape.

        Parameters
        ----------
        input_shape: tuple[int] or list[tuple[int]]
            The input shape(s) for the layer. Shape tuples can include
            None for free dimensions, instead of integer.
        """
        input_shape = input_shape[0]
        return (input_shape[0],) + (self._height, self._width) + (input_shape[3] * self._depth,)
