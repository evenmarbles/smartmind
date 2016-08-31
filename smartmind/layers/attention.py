from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from six.moves import range

import numpy as np
import tensorflow as tf

from ..framework import Layer
from ..utils import tf_int_shape


class SpatialGlimpse(Layer):
    def __init__(self, size, depth, scale, input_shape=None, input_dtype='float32',
                 batch_size=None, input_default=None, sparse_input=False, reader_input=None, name=None):
        super(SpatialGlimpse, self).__init__([input_shape, (2,)], input_dtype, batch_size,
                                             input_default, sparse_input, reader_input, name)

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

    def _call(self, x):
        super(SpatialGlimpse, self)._call(x)

        if not isinstance(x, list):
            raise Exception('SpatialGlimpse takes exactly two inputs')

        offsets = x[1]
        x = x[0]

        output = []
        for d in range(self._depth):
            o = tf.image.extract_glimpse(x, tf.constant(self._sizes[d], dtype=tf.int32, shape=[2]), offsets)
            if d > 0:
                kernel_size = self._sizes[d] / self._sizes[0]
                assert (np.any(kernel_size % 2) == 0.)
                o = tf.nn.avg_pool(o,
                                   ksize=[1, kernel_size[0], kernel_size[1], 1],
                                   strides=[1, kernel_size[0], kernel_size[1], 1],
                                   padding='VALID')
            output.append(o)

        output_shape = (-1,) if self._batch_size is None else (self._batch_size,)
        output_shape += (self._height, self._width, self._batch_input_shape[0][3] * self._depth)

        output = tf.pack(output, axis=4)
        output = tf.reshape(output, shape=output_shape)
        return output

    def _get_output_shape(self, input_shape):
        """Computes the output shape of the layer given an input shape.
        This function assumes that the layer will be built to match the input shape.

        Parameters
        ----------
        input_shape: tuple[int] or list[tuple[int]]
            The input shape(s) for the layer. Shape tuples can include
            None for free dimensions, instead of integer.
        """

        return self._batch_size, self._height, self._width, self._batch_input_shape[0][3] * self._depth
