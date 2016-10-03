from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .. import initializers
from .. import activations
from .. import regularizers

from ..framework import Layer
from ..framework import Model
from ..utils import tf_int_shape
from ..utils import tf_concatenate
from ..utils import tf_prod
from ..utils import tf_sqrt
from ..utils import tf_batch_dot
from ..utils import epsilon


class Activation(Layer):
    def __init__(self, activation, activation_params=None, trainable=True, input_shape=None,
                 input_dtype=None, batch_size=None, name=None):
        self._activation = activations.get(activation)
        self._activation_params = activations.process_parameters(activation, activation_params)

        super(Activation, self).__init__(trainable, name, input_shape=input_shape,
                                         input_dtype=input_dtype, batch_size=batch_size)

    def _call(self, x):
        return self._activation(x)


class FullyConnected(Layer):
    def __init__(self, output_dim, init='xavier_uniform', init_params=None, initial_weights=None,
                 w_regularizer=None, bias=True, b_regularizer=None, trainable=True, activation='linear',
                 activation_params=None, input_shape=None, input_dtype=None, batch_size=None, name=None):

        self._output_dim = output_dim

        self._init = initializers.get(init, kwargs=init_params)
        self._initial_weights = initial_weights

        self._activation = activations.get(activation)
        self._activation_params = activations.process_parameters(activation, activation_params)

        self._bias = bias

        self._w_regularizer = regularizers.get(w_regularizer)
        self._b_regularizer = regularizers.get(b_regularizer)

        self._w = None
        self._b = None

        super(FullyConnected, self).__init__(trainable, name, input_shape=input_shape,
                                             input_dtype=input_dtype, batch_size=batch_size)

    def build(self, shape):
        super(FullyConnected, self).build(shape)

        with tf.variable_scope(self.name):
            if self._initial_weights is not None and self._initial_weights[0] is not None:
                self._w = tf.get_variable('Matrix', [shape[1], self._output_dim], tf.float32,
                                          initializer=initializers.constant(self._initial_weights[0]),
                                          regularizer=self._w_regularizer)
            else:
                self._w = tf.get_variable('Matrix', [shape[1], self._output_dim], tf.float32,
                                          initializer=self._init,
                                          regularizer=self._w_regularizer)

            if self._bias:
                initial_bias = 0.0
                if self._initial_weights is not None and self._initial_weights[1] is not None:
                    initial_bias = self._initial_weights[1]
                self._b = tf.get_variable('bias', [self._output_dim],
                                          initializer=initializers.constant(initial_bias),
                                          regularizer=self._b_regularizer)

    def _call(self, x):
        output = tf.matmul(x, self._w)
        if self._bias:
            output = tf.nn.bias_add(output, self._b)
        return self._activation(output)

    def _get_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self._output_dim


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    Example
    -------

    ```python
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, trainable=True, input_shape=None, input_dtype=None, batch_size=None, name=None):
        super(Flatten, self).__init__(trainable, name, input_shape=input_shape,
                                      input_dtype=input_dtype, batch_size=batch_size)

    def _call(self, x):
        return tf.reshape(x, [-1, tf_prod(tf.shape(x)[1:])])

    def _get_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to Flatten is not fully defined'
                            ' (got {}. Make sure to pass a complete input_shape'
                            ' or "batch_input_shape" argument to the first'
                            ' layer in your model.'.format(input_shape[1:]))
        return input_shape[0], np.prod(input_shape[1:])


class Reshape(Layer):
    def __init__(self, target_shape, trainable=True, input_shape=None, input_dtype=None,
                 batch_size=None, name=None):

        self._target_shape = target_shape

        super(Reshape, self).__init__(trainable, name, input_shape=input_shape,
                                      input_dtype=input_dtype, batch_size=batch_size)

    def _call(self, x):
        # In case the target shape is not fully defined,
        # we need access to the shape of x.
        # solution:
        # 1) rely on x._sm_shape
        # 2) fallback: tf_int_shape
        target_shape = self._target_shape
        if -1 in target_shape:
            # target shape not fully defined
            if hasattr(x, '_sm_shape'):
                input_shape = getattr(x, '_sm_shape')
            else:
                input_shape = tf_int_shape(x)

            target_shape = self._get_output_shape(input_shape)

        if target_shape[0] is None:
            target_shape = (-1,) + target_shape[1:]
        return tf.reshape(x, target_shape)

    # noinspection PyMethodMayBeStatic
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


class AddBias(Layer):
    def __init__(self, output_dim, init='xavier_uniform', init_params=None, initial_weights=None,
                 b_regularizer=None, trainable=True, activation='linear', activation_params=None,
                 input_shape=None, input_dtype=None, batch_size=None, name=None):
        self._output_dim = output_dim

        self._init = initializers.get(init, kwargs=init_params)
        self._initial_weights = initial_weights

        self._initial_weights = initial_weights

        self._activation = activations.get(activation)
        self._activation_params = activations.process_parameters(activation, activation_params)

        self._b_regularizer = regularizers.get(b_regularizer)
        self._b = None

        super(AddBias, self).__init__(trainable, name, input_shape=input_shape,
                                      input_dtype=input_dtype, batch_size=batch_size)

    def build(self, shape):
        super(AddBias, self).build(shape)

        with tf.variable_scope(self.name):
            if self._initial_weights is not None:
                initial_bias = self._initial_weights
                self._b = tf.get_variable('bias', [self._output_dim],
                                          initializer=initializers.constant(initial_bias),
                                          regularizer=self._b_regularizer)
            else:
                self._b = tf.get_variable('bias', [self._output_dim],
                                          initializer=self._init,
                                          regularizer=self._b_regularizer,
                                          trainable=self._trainable)

    def _call(self, x):
        output = tf.nn.bias_add(x, self._b)
        return self._activation(output)

    def _get_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self._output_dim


class Constant(Layer):
    def __init__(self, value, input_shape=None, input_dtype=None, batch_size=None, name=None):
        self._value = value

        super(Constant, self).__init__(False, name, input_shape=input_shape,
                                       input_dtype=input_dtype, batch_size=batch_size)

    def _call(self, x):
        with tf.variable_scope(self.name):
            tensor = tf.convert_to_tensor(self._value, dtype=self.input_dtype)
            ndims = tensor.get_shape().ndims

            input_shape = x.get_shape()
            for _ in range(input_shape.ndims - ndims):
                tensor = tf.expand_dims(tensor, 0)

            fixed_batch_size = input_shape[0]
            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                batch_size = tf.shape(x)[0]

            return tf.tile(tensor, tf.pack([batch_size, 1]))

    def _get_output_shape(self, input_shape):
        return input_shape[0], 1


class Merge(Layer):
    def __init__(self, layers=None, mode='sum', axis=-1, output_shape=None,
                 node_indices=None, tensor_indices=None, name=None):

        self._layers = layers
        self._mode = mode
        self._axis = axis
        self._output_shape = output_shape

        super(Merge, self).__init__(name=name)

        if layers:
            if tensor_indices is None:
                tensor_indices = [0 for _ in range(len(layers))]

            if not node_indices:
                # by default we connect to
                # the 1st output stream in the input layer
                node_indices = [0 for _ in range(len(layers))]
            self._process_params(layers, mode, node_indices, tensor_indices)
            self.add_inbound_layer(layers, tensor_indices)
            self._built = True
        else:
            self._built = False

    def __call__(self, inbound_layers=None):
        """We disable successive calls to __call__ for Merge layers.
        Although there is no technical obstacle to
        making it possible to __call__ a Merge instance many times
        (it is just a layer), it would make for a rather inelegant API.
        """
        if not isinstance(inbound_layers, list):
            raise Exception('Merge can only be called on a list of tensors, '
                            'not a single tensor. Received: ' + str(inbound_layers))
        if self._built:
            raise Exception('A Merge layer cannot be used more than once, '
                            'please use ' +
                            'the "merge" function instead: ' +
                            '`merged_tensor = merge([tensor_1, tensor2])`.')

            # all_sm_tensors = True
            # for in_ in x:
            #     if not hasattr(in_, '_sm_history'):
            #         all_sm_tensors = False
            #         break
            #
            # if all_sm_tensors:
            #     layers = []
            #     tensor_indices = []
            #     for in_ in x:
            #         layer, tensor_index = getattr(in_, '_sm_history')
            #         layers.append(layer)
            #         tensor_indices.append(tensor_index)
            #     self._process_params(layers, self._mode, tensor_indices)
            #     self.built = True
            #     self.add_inbound_node(layers, node_indices, tensor_indices)
            #
            #     outputs = self.inbound_nodes[-1].output_tensors
            #     return outputs[0]  # merge only returns a single tensor
            # else:
            #     return self.call(inputs, mask)

    def _call(self, x):
        if not isinstance(x, list) or len(x) <= 1:
            raise Exception('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(x))
        # case: "mode" is a lambda or function.
        if hasattr(self._mode, '__call__'):
            # TODO: consider making it possible to
            # pass custom arguments to lambda.
            arguments = {}
            return self._mode(x, **arguments)

        if self._mode == 'sum' or self._mode == 'ave':
            s = x[0]
            for i in range(1, len(x)):
                s += x[i]
            if self._mode == 'ave':
                s /= len(x)
            return s

        elif self._mode == 'concat':
            return tf_concatenate(x, axis=self._axis)

        elif self._mode == 'mul':
            s = x[0]
            for i in range(1, len(x)):
                s *= x[i]
            return s
        elif self._mode == 'max':
            s = x[0]
            for i in range(1, len(x)):
                s = tf.maximum(s, x[i])
            return s
        elif self._mode == 'dot':
            l1 = x[0]
            l2 = x[1]
            output = tf_batch_dot(l1, l2, self._axis)
            return output

        elif self._mode == 'cos':
            l1 = x[0]
            l2 = x[1]
            denominator = tf_sqrt(tf_batch_dot(l1, l1, self._axis) *
                                  tf_batch_dot(l2, l2, self._axis))
            denominator = tf.maximum(denominator, epsilon())
            output = tf_batch_dot(l1, l2, self._axis) / denominator
            output = tf.expand_dims(output, 1)
            return output
        else:
            raise Exception('Unknown merge mode.')

    def _process_params(self, layers, mode, node_indices, tensor_indices):
        """Process user arguments"""
        if not hasattr(mode, '__call__'):
            if mode not in {'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'}:
                raise Exception('Invalid merge mode: ' + str(mode))
        if not isinstance(layers, (list, tuple)) or len(layers) < 2:
            raise Exception('A Merge should only be applied to a list of '
                            'layers with at least 2 elements. Found: ' + str(layers))

        input_shapes = []
        for i, layer in enumerate(layers):
            layer_output_shape = layer.get_output_shape_at(node_indices[i])
            if isinstance(layer_output_shape, list):
                # case: the layer has multiple output tensors
                # and we only need a specific one
                layer_output_shape = layer_output_shape[tensor_indices[i]]
            input_shapes.append(layer_output_shape)

        if mode in {'sum', 'mul', 'ave', 'cos', 'max'}:
            input_shapes_set = set(input_shapes)
            if len(input_shapes_set) > 1:
                raise Exception('Only layers of same output shape can '
                                'be merged using ' + mode + ' mode. ' +
                                'Layer shapes: %s' % input_shapes)
        if mode in {'cos', 'dot'}:
            if len(layers) > 2:
                raise Exception(mode + ' merge takes exactly 2 layers')
            shape1 = input_shapes[0]
            shape2 = input_shapes[1]
            n1 = len(shape1)
            n2 = len(shape2)
            if isinstance(self._axis, int):
                if self._axis < 0:
                    self._axis = [self._axis % n1, self._axis % n2]
                else:
                    self._axis = [n1 - self._axis, n2 - self._axis]
            if not isinstance(self._axis, (list, tuple)):
                raise Exception('Invalid type for axis - should be a list.')
            if len(self._axis) != 2:
                raise Exception('Invalid format for axes - should contain two elements.')
            if type(self._axis[0]) is not int or type(self._axis[1]) is not int:
                raise Exception('Invalid format for axes - list elements should be "int".')
            if shape1[self._axis[0]] != shape2[self._axis[1]]:
                raise Exception('Dimension incompatibility using dot mode: ' +
                                '%s != %s. ' % (shape1[self._axis[0]], shape2[self._axis[1]]) +
                                'Layer shapes: %s, %s' % (shape1, shape2))
        elif mode == 'concat':
            if not isinstance(self._axis, int):
                raise Exception('Invalid type for axis - should be integer.')
            reduced_inputs_shapes = [list(shape) for shape in input_shapes]
            shape_set = set()
            for i in range(len(reduced_inputs_shapes)):
                del reduced_inputs_shapes[i][self._axis]
                shape_set.add(tuple(reduced_inputs_shapes[i]))
            if len(shape_set) > 1:
                raise Exception('"concat" mode can only merge layers with matching ' +
                                'output shapes except for the concat axis. ' +
                                'Layer shapes: %s' % input_shapes)

    def _get_output_shape(self, input_shape):
        """Computes the output shape of the layer given an input shape.
        This function assumes that the layer will be built to match the input shape.

        Parameters
        ----------
        input_shape: tuple[int] or list[tuple[int]]
            The input shape(s) for the layer. Shape tuples can include
            None for free dimensions, instead of integer.
        """
        assert type(input_shape) is list  # must have multiple input shape tuples
        # case: callable self._output_shape
        if hasattr(self._mode, '__call__'):
            if hasattr(self._output_shape, '__call__'):
                output_shape = self._output_shape(input_shape)
                return output_shape
            elif self._output_shape is not None:
                return (input_shape[0][0],) + tuple(self._output_shape)
            else:
                # TODO: consider shape auto-inference with TF
                raise Exception('The Merge layer ' + self.name +
                                ' has a callable `mode` argument, ' +
                                'and we cannot infer its output shape because ' +
                                'no `output_shape` argument was provided.' +
                                'Make sure to pass a shape tuple (or a callable) ' +
                                '`output_shape` to Merge.')
        # pre-defined merge modes
        input_shapes = input_shape
        if self._mode in ['sum', 'mul', 'ave', 'max']:
            # all tuples in input_shapes should be the same
            return input_shapes[0]
        elif self._mode == 'concat':
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                if output_shape[self._axis] is None or shape[self._axis] is None:
                    output_shape[self._axis] = None
                    break
                output_shape[self._axis] += shape[self._axis]
            return tuple(output_shape)
        elif self._mode in ['dot', 'cos']:
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            shape1.pop(self._axis[0])
            shape2.pop(self._axis[1])
            shape2.pop(0)
            output_shape = shape1 + shape2
            if len(output_shape) == 1:
                output_shape += [1]
            return tuple(output_shape)
