from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from . import optimizers
from . import objectives
from .objectives import weighted_objective

from .utils import get_uid
from .utils import tf_int_shape
from .utils import to_dtype
from .utils import tolist


class Layer(object):
    """Abstract base layer class"""

    @property
    def input_tensors(self):
        return self._input_tensors

    @property
    def output_tensors(self):
        return self._output_tensors

    def __init__(self, input_shape=None, input_dtype='float32', batch_size=None, input_default=None,
                 sparse_input=False, reader_input=None, name=None):
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(get_uid(prefix))
        self.name = name

        self._built = False

        self._inbound_layers = []
        self._outbound_layers = []

        self._input_tensors = []
        self._output_tensors = []

        # input
        if reader_input is not None:
            self._batch_input_shape = tf_int_shape(reader_input)
            self._batch_size = self._batch_input_shape[0]
            self._input_shape = self._batch_input_shape[1:]
            self._input_dtype = reader_input.dtype
            self._reader_input = reader_input
        else:
            self._batch_size = batch_size
            self._input_shape = input_shape

            self._batch_input_shape = []
            try:
                if not any(isinstance(i, (list, tuple)) for i in input_shape):
                    input_shape = [input_shape]
                for shape in input_shape:
                    self._batch_input_shape.append((batch_size,) + tuple(shape))
            except TypeError:
                try:
                    # attempt input shape inference
                    # TODO do we need to allow multiple input defaults?
                    self._batch_input_shape.append((batch_size,) + tf_int_shape(input_default))
                except AttributeError:
                    pass

            self._input_dtype = input_default.dtype if input_default is not None else to_dtype(input_dtype)
        self._input_default = input_default
        self._sparse_input = sparse_input

    def __call__(self, input_layer=None):
        try:
            self._input_tensors = input_layer.output_tensors
        except AttributeError:
            input_tensors = None
            input_shapes = [()]
        else:
            input_tensors = self._input_tensors
            input_shapes = []
            for x_elem in tolist(input_tensors):
                if hasattr(x_elem, '_sm_shape'):
                    input_shapes.append(getattr(x_elem, '_sm_shape'))
                else:
                    input_shapes.append(tf_int_shape(x_elem))

            self._inbound_layers.append(input_layer)
            input_layer._outbound_layers.append(self)

        if len(input_shapes) == 1:
            input_shapes = input_shapes[0]

        if not self._built:
            self.build(input_shapes)

        if input_tensors is not None and len(input_tensors) == 1:
            input_tensors = input_tensors[0]
        self._output_tensors = tolist(self._call(input_tensors))

        if not self._output_tensors:
            raise Exception('The `_call` method of layer ' + self.name + ' should return a tensor.')

        if input_tensors is not None:
            for i, tensor in enumerate(self._output_tensors):
                tensor._sm_shape = self._get_output_shape(input_shapes)
                tensor._sm_history = (self, i)

        return self._output_tensors

    def create_input_layer(self, name=None):
        if not name:
            prefix = self.__class__.__name__.lower() + '_input'
            name = prefix + '_' + str(get_uid(prefix))

        input_layer = Input(self._input_shape, self._input_dtype, self._batch_size, self._input_default,
                            self._sparse_input, self._reader_input, name)
        # create the input layer
        input_layer()
        # create the current layer
        self(input_layer)

    def build(self, shape):
        self._built = True

    def _call(self, x):
        return x

    def _get_output_shape(self, input_shape):
        """Computes the output shape of the layer given an input shape.
        This function assumes that the layer will be built to match the input shape.

        Parameters
        ----------
        input_shape: tuple[int] or list[tuple[int]]
            The input shape(s) for the layer. Shape tuples can include
            None for free dimensions, instead of integer.
        """
        return input_shape


class Input(Layer):
    def __init__(self, input_shape=None, input_dtype='float32', batch_size=None, input_default=None,
                 sparse_input=False, reader_input=None, name=None):
        super(Input, self).__init__(input_shape, input_dtype, batch_size, input_default, sparse_input,
                                    reader_input, name=name)

    def _call(self, *args):
        if self._reader_input is not None:
            self._reader_input._sm_shape = self._batch_input_shape
            self._reader_input._sm_history = (self, 0)
            return self._reader_input

        tensor = []
        for i, shape in enumerate(self._batch_input_shape):
            if self._input_default is not None:
                tensor.append(tf.placeholder_with_default(self._input_default,
                                                          shape=shape,
                                                          name=self.name))
            elif self._sparse_input:
                tensor.append(tf.sparse_placeholder(self._input_dtype,
                                                    shape=shape,
                                                    name=self.name))
            else:
                tensor.append(tf.placeholder(self._input_dtype,
                                             shape=shape,
                                             name=self.name))
            setattr(tensor[i], '_sm_shape', shape)
            setattr(tensor[i], '_sm_history', (self, i))

        if len(tensor) <= 1:
            return tensor[0]
        return tensor


class Container(object):
    """Abstract base container class."""

    def __init__(self, name=None):
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(get_uid(prefix))
        self.name = name

        self._inputs = []
        self._outputs = []

        # list of layers (1-to-1 mapping with self._inputs)
        self._input_layers = []
        self._input_tensor_indices = []
        self._input_names = []

        # list of layers (1-to-1 mapping with self._outputs)
        self._output_layers = []
        self._output_tensor_indices = []
        self._output_names = []

        self._internal_input_shapes = []
        self._internal_output_shapes = []

    def process(self):
        if not self._inputs or not self._outputs:
            raise Exception(self.__class__.__name__ + ' model cannot be build: model is empty.')

        # check for redundancy in inputs
        input_set = set(self._inputs)
        if len(input_set) != len(self._inputs):
            raise Exception('There are redundant inputs passed to the model. Each input should only'
                            ' occur once: ' + str(self._inputs))

        for x in self._inputs:
            if not hasattr(x, '_sm_history'):
                raise Exception('Input tensors to the ' + self.__class__.__name__ +
                                ' model must contain SmartMind metadata.')

            # check that x is an input tensor
            layer, tensor_index = getattr(x, '_sm_history')
            if layer._inbound_layers:
                raise Exception('Input tensors to the ' + self.__class__.__name__ +
                                ' model must be an Input layer and cannot be the output'
                                ' of a previous non-Input layer. Here, the tensor was'
                                ' generated by layer ' + layer.name)

            self._input_layers.append(layer)
            self._input_tensor_indices.append(tensor_index)
            self._input_names.append(layer.name)
            self._internal_input_shapes.append(getattr(x, '_sm_shape'))

        for x in self._outputs:
            if not hasattr(x, '_sm_history'):
                raise Exception('Output tensors to the ' + self.__class__.__name__ +
                                ' model must contain SmartMind metadata.')

            layer, tensor_index = getattr(x, '_sm_history')
            self._output_layers.append(layer)
            self._output_tensor_indices.append(tensor_index)
            self._output_names.append(layer.name)
            self._internal_output_shapes.append(getattr(x, '_sm_shape'))


class Model(Container):
    def __init__(self, name=None):
        super(Model, self).__init__(name)

        self._optimizer = None
        self._sample_weight_mode = None
        self._loss = None
        self._loss_functions = None
        self._loss_weights = None

        self._targets = None
        self._total_loss = None
        self._sample_weights = None

    def compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None):
        self.process()

        self._optimizer = optimizers.get(optimizer)
        self._loss = loss
        self._loss_weights = loss_weights

        # prepare loss weights
        if loss_weights is None:
            loss_weights_list = [1.] * len(self._outputs)
        elif isinstance(loss_weights, dict):
            for name in loss_weights:
                if name not in self._output_names:
                    raise Exception('Unknown key in loss_weights dictionary: ' + name +
                                    '. Expecting the following keys: ' + str(self._output_names))
            loss_weights_list = []
            for name in self._output_names:
                loss_weights_list.append(loss_weights.get(name, 1.))
        elif isinstance(loss_weights, list):
            if len(loss_weights) != len(self._outputs):
                raise Exception('When passing a lists as loss_weights, an entry per model output'
                                ' must be given. The model has ' + str(len(self._outputs)) +
                                ' outputs, but number of elements in loss_weights is: ' +
                                str(len(loss_weights)))
            loss_weights_list = loss_weights
        else:
            raise Exception('Could not interpret loss_weights argument: ' + str(loss_weights))

        # prepare loss functions
        if isinstance(loss, dict):
            for name in loss:
                if name not in self._output_names:
                    raise Exception('Unknown key in loss dictionary: ' + name +
                                    '. Expecting the following keys: ' + str(self._output_names))
            loss_functions = []
            for name in self._output_names:
                if name not in loss:
                    raise Exception('Output ' + name + ' missing from loss dictionary.')
                loss_functions.append(objectives.get(loss[name]))
        elif isinstance(loss, list):
            if len(loss) != len(self._outputs):
                if len(loss_weights) != len(self._outputs):
                    raise Exception('When passing a lists as loss, an entry per model output'
                                    ' must be given. The model has ' + str(len(self._outputs)) +
                                    ' outputs, but number of elements in loss is: ' +
                                    str(len(loss_weights)))
            loss_functions = [objectives.get(l) for l in loss]
        else:
            loss_function = objectives.get(loss)
            loss_functions = [loss_function for _ in range(len(self._outputs))]
        self._loss_functions = loss_functions
        weighted_losses = [weighted_objective(fn) for fn in loss_functions]

        # prepare targets of model
        self._targets = []
        for i, o in enumerate(self._outputs):
            shape = getattr(o, '_sm_shape')
            name = self._output_names[i]
            target = tf.placeholder(tf.float32, shape=tuple([None for _ in len(shape)]), name=name + '_target')
            target._sm_shape = shape
            self._targets.append(target)

        total_loss = None
        for i in range(len(self._outputs)):
            y_pred = self._outputs[i]
            y_target = self._targets[i]
            weighted_loss = weighted_losses[i]
            sample_weight = None
            mask = None
            loss_weight = loss_weights_list[i]
            output_loss = weighted_loss(y_pred, y_target, sample_weight, mask)
            if total_loss is None:
                total_loss = loss_weight * output_loss
            else:
                total_loss += loss_weight * output_loss
        self._total_loss = total_loss

        # functions for train, test and predict, will be compiled lazily in case user does not
        # use all functions
        setattr(self, '_fn_train', None)
        setattr(self, '_fn_test', None)
        setattr(self, '_fn_predict', None)

