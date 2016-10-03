from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# noinspection PyUnresolvedReferences
from six.moves import range

import time
import copy
import numpy as np
import tensorflow as tf
from datetime import datetime

from . import optimizers
from . import objectives
from . import metrics as metrics_module
from . import callbacks as cbks
from .objectives import compute_weighted_objective
from .losses import get_total_loss
from .losses import get_losses

from .utils import Progbar
from .utils import get_uid
from .utils import tolist
from .utils import tf_ndims
from .utils import tf_int_shape
from .utils import tf_function
from .utils import tf_get_session
from .utils import tf_training_phase
from .utils import dotdict

__all__ = ['InputSpec',
           'Layer',
           'Input',
           'Model']


def _parse_input_params(shape=None, dtype=None, batch_size=None, default=None,
                        sparse=False, tensor=None, reader_op=None):
    if all(i is not None for i in [tensor, reader_op]):
        raise ValueError("Please provide InputDef either tensor OR reader_op, "
                         "but not both.")

    is_reader_op = False

    dtype = tolist(dtype)
    default = tolist(default)
    tensor = tolist(tensor)

    batch_shape = []
    if reader_op is not None:
        is_reader_op = True
        default = None
        tensor = [reader_op]
        dtype = [reader_op.dtype]
        batch_shape = [tf_int_shape(reader_op)]

    if tensor:
        for t in tensor:
            batch_shape.append(tf_int_shape(t))
        default = None

    if not batch_shape:
        try:
            if not any(isinstance(i, (list, tuple)) for i in shape):
                shape = [shape]

            for s in shape:
                batch_shape.append((batch_size,) + tuple(s))
        except TypeError:
            try:
                # shape is not given, try to infer it from default
                for d in default:
                    batch_shape.append((batch_size,) + tf_int_shape(d))
            except AttributeError:
                raise ValueError("Please provide InputDef with an input shape.")

    if not dtype:
        tensor_ = tensor if tensor else default
        if tensor_:
            dtype = []
            for t in tensor_:
                dtype.append(t.dtype)
        else:
            dtype = [tf.float32]

    n = len(batch_shape)
    if len(dtype) < n:
        dtype *= n

    if default and len(default) < n:
        default *= batch_shape

    return dotdict({'batch_shape': batch_shape,
                    'dtype': dtype,
                    'default': default,
                    'sparse': sparse,
                    'tensor': tensor,
                    'is_reader_op': is_reader_op})


def batch_shuffle(index_array, batch_size):
    """This shuffles an array in a batch-wise fashion.
    Useful for shuffling HDF5 arrays
    (where one cannot access arbitrary indices).
    """
    batch_count = int(len(index_array) / batch_size)
    # to reshape we need to be cleanly divisible by batch size
    # we stash extra items and reappend them after shuffling
    last_batch = index_array[batch_count * batch_size:]
    index_array = index_array[:batch_count * batch_size]
    index_array = index_array.reshape((batch_count, batch_size))
    np.random.shuffle(index_array)
    index_array = index_array.flatten()
    return np.append(index_array, last_batch)


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices)."""
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, nb_batch)]


def slice_X(X, start=None, stop=None):
    '''This takes an array-like, or a list of
    array-likes, and outputs:
        - X[start:stop] if X is an array-like
        - [x[start:stop] for x in X] if X in a list

    Can also work on list/array of indices: `slice_X(x, indices)`

    # Arguments:
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.
    '''
    if type(X) == list:
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]


def collect_metrics(metrics, output_names):
    if not metrics:
        return [[] for _ in output_names]
    if type(metrics) is list:
        # we then apply all metrics to all outputs.
        return [copy.copy(metrics) for _ in output_names]
    elif type(metrics) is dict:
        nested_metrics = []
        for name in output_names:
            output_metrics = metrics.get(name, [])
            if type(output_metrics) is not list:
                output_metrics = [output_metrics]
            nested_metrics.append(output_metrics)
        return nested_metrics
    else:
        raise Exception('Type of `metrics` argument not understood. '
                        'Expected a list or dictionary, found: ' +
                        str(metrics))


# noinspection PyProtectedMember
class Node(object):
    """A `Node` describes the connectivity between two layers.

    Each time a layer is connected to some new input, a node is added to
    `layer.inbound_nodes`. Each time the output of a layer is used by another layer,
    a node is added to `layer.outbound_nodes`.

    Attributes
    ----------
    outbound_layer: the layer that takes
        `input_tensors` and turns them into `output_tensors`.
    inbound_layers: a list of layers, the same length as `input_tensors`,
        the layers from where `input_tensors` originate.
    node_indices: a list of integers, the same length as `inbound_layers`.
        `node_indices[i]` is the origin node of `input_tensors[i]`
        (necessary since each inbound layer might have several nodes,
        e.g. if the layer is being shared with a different data stream).
    tensor_indices: a list of integers, the same length as `inbound_layers`.
        `tensor_indices[i]` is the index of `input_tensors[i]` within the
        output of the inbound layer (necessary since each inbound layer might
        have multiple tensor outputs, with each one being
        independently manipulable).
    input_tensors: list of input tensors.
    output_tensors: list of output tensors.
    input_shapes: list of input shape tuples.
    output_shapes: list of output shape tuples.

    Notes
    -----
    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:

    `input_tensors[i] == inbound_layers[i].inbound_nodes[node_indices[i]].output_tensors[tensor_indices[i]]`

    A node from layer A to layer B is added to:
        A.outbound_nodes
        B.inbound_nodes
    """

    def __init__(self, outbound_layer,
                 inbound_layers, node_indices, tensor_indices,
                 input_tensors, output_tensors,
                 input_shapes, output_shapes):
        # layer instance (NOT a list).
        # this is the layer that takes a list of input tensors
        # and turns them into a list of output tensors.
        # the current node will be added to the inbound_nodes of outbound_layer
        self.outbound_layer = outbound_layer

        # the following 3 properties describe where
        # the input tensors come from: which layers,
        # and for each layer, which node and which
        # tensor output of each node.
        self.inbound_layers = inbound_layers  # list of layer instances
        self.node_indices = node_indices  # list of integers, 1:1 mapping with inbound_layers
        self.tensor_indices = tensor_indices  # list of integers, 1:1 mapping with inbound_layers

        # tensor inputs and outputs of outbound_layer
        self.input_tensors = input_tensors  # list of tensors. 1:1 mapping with inbound_layers
        self.output_tensors = output_tensors  # list of tensors, created by outbound_layer._call()

        # input and output shapes
        self.input_shapes = input_shapes  # list of shape tuples, shapes of input_tensors
        self.output_shapes = output_shapes  # list of shape tuples, shapes of output_tensors

        # add nodes to all layers involved.
        for layer in inbound_layers:
            if layer is not None:
                layer.outbound_nodes.append(self)
        outbound_layer.inbound_nodes.append(self)

    @classmethod
    def create_node(cls, outbound_layer,
                    inbound_layers, node_indices=None, tensor_indices=None):
        if not node_indices:
            node_indices = [0 for _ in range(len(inbound_layers))]
        else:
            assert len(node_indices) == len(inbound_layers)
        if not tensor_indices:
            tensor_indices = [0 for _ in range(len(inbound_layers))]

        input_tensors = []
        input_shapes = []

        for inbound_layer, node_index, tensor_index in zip(inbound_layers, node_indices, tensor_indices):
            inbound_node = inbound_layer.inbound_nodes[node_index]
            input_tensors.append(inbound_node.output_tensors[tensor_index])
            input_shapes.append(inbound_node.output_shapes[tensor_index])

        assert len(input_shapes) == len(input_tensors)

        if len(input_tensors) == 1:
            input_tensors = input_tensors[0]
            input_shapes = input_shapes[0]

        output_tensors = tolist(outbound_layer._call(input_tensors))
        # TODO: try to auto-infer shape if exception is raised by _get_output_shape
        output_shapes = tolist(outbound_layer._get_output_shape(input_shapes), convert_tuple=False)

        if not output_tensors or output_tensors[0] is None:
            raise Exception('The `_call` method of layer "{}" should return a tensor. '
                            'Found: {}'.format(outbound_layer.name, output_tensors[0]))
        if len(output_tensors) != len(output_shapes):
            raise Exception('The `_get_output_shape` method of layer "{}" should '
                            'return one shape tuple per output tensor of the layer. '
                            'Found: {}'.format(outbound_layer.name, output_shapes))

        for i in range(len(output_tensors)):
            output_tensors[i]._sm_shape = output_shapes[i]
            output_tensors[i]._sm_history = (outbound_layer, len(outbound_layer.inbound_nodes), i)

        return cls(outbound_layer,
                   inbound_layers, node_indices, tensor_indices,
                   tolist(input_tensors), output_tensors,
                   tolist(input_shapes), output_shapes)


class InputSpec(object):
    """This specifies the ndims, dtype and shape of every input to a layer.
    Every layer should expose (if appropriate) an `input_spec` attribute:
    a list of instances of InputSpec (one per input tensor).

    A None entry in a shape is compatible with any dimension,
    a None shape is compatible with any shape.
    """

    def __init__(self, dtype=None, shape=None, ndims=None):
        if isinstance(ndims, str):
            assert '+' in ndims, 'When passing a str "ndims", it should have the form "2+", "3+", etc.'
            int_ndims = ndims[:ndims.find('+')]
            assert int_ndims.isdigit(), 'When passing a str "ndims", it should have the form "2+", "3+", etc.'
        if shape is not None:
            self.ndims = len(shape)
        else:
            self.ndims = ndims
        self.dtype = dtype
        self.shape = shape


# noinspection PyProtectedMember
class Layer(object):
    """Abstract base layer class

    Parameters
    ----------
    trainable: bool
    name: str

    Other Parameters
    ---------------------
    input_shape: tuple or list[tuple]
    input_dtype:
    batch_size: int
    """

    def __init__(self, trainable=True, name=None, **kwargs):
        # these properties should have been set
        # by the child class, as appropriate.
        if not hasattr(self, '_input_spec'):
            self._input_spec = None
        if not hasattr(self, 'uses_training_phase'):
            self.uses_training_phase = False

        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(get_uid(prefix))
        self.name = name

        input_kwargs = {'input_shape',
                        'input_dtype',
                        'batch_size'}
        for kwarg in kwargs.keys():
            assert kwarg in input_kwargs, 'Keyword argument not understood: ' + kwarg

        self.batch_size = kwargs.get('batch_size', None)
        self.input_shape = kwargs.get('input_shape', None)
        self.input_dtype = kwargs.get('input_dtype', tf.float32)

        self._trainable = trainable
        self._built = False

        # these lists will be filled via successive calls to self.add_inbound_node()
        self.inbound_nodes = []
        self.outbound_nodes = []

    def __call__(self, inputs=None):
        """

        Parameters
        ----------
        inputs: Tensor or list[Tensors]
            The input into the layer

        Returns
        -------
        Tensor or list[Tensors]
        """
        # raise exceptions in case the input is not compatible
        # with the input_spec defined in the layer constructor
        self._verify_input_compatibility(inputs)

        input_tensors = tolist(inputs)

        if not self._built:
            # collect input shapes to build layer
            input_shapes = []
            for x_elem in input_tensors:
                if hasattr(x_elem, '_sm_shape'):
                    input_shapes.append(getattr(x_elem, '_sm_shape'))
                else:
                    input_shapes.append(tf_int_shape(x_elem))

            if len(input_shapes) == 1:
                input_shapes = input_shapes[0]
            self.build(input_shapes)
            self._built = True

        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for input_tensor in input_tensors:
            if hasattr(input_tensor, '_sm_history'):
                previous_layer, node_index, tensor_index = input_tensor._sm_history
                inbound_layers.append(previous_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers = None
                break

        input_added = False
        if inbound_layers:
            # this will call layer.build() if necessary
            self.add_inbound_layer(inbound_layers, node_indices, tensor_indices)
            input_added = True

        if input_added:
            # output was already computed when calling self.add_inbound_node
            outputs = self.inbound_nodes[-1].output_tensors
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        else:
            # this is the case when `inputs` is a tensor without SmartMind meta data
            return self._call(inputs)

    def create_input_layer(self, input_shape=None, input_dtype=None, batch_size=None,
                           default=None, sparse=False, tensor=None, reader_op=None, name=None):
        if not name:
            prefix = self.__class__.__name__.lower() + '_input'
            name = prefix + '_' + str(get_uid(prefix))

        if input_dtype is None:
            input_dtype = tf.float32

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.input_dtype = input_dtype

        # instantiate the input layer
        tensor = Input(input_shape, input_dtype, batch_size,
                       default, sparse, tensor, reader_op, name)

        # this will build the current layer and create the node connecting
        # the current layer to the input layer we just created.
        self(tensor)

    def add_inbound_layer(self, inbound_layers, node_indices=None, tensor_indices=None):
        """
        Parameters
        ----------
        inbound_layers: can be a layer instance
            or a list/tuple of layer instances.
        node_indices: integer (or list of integers).
            The input layer might have a number of
            parallel output streams;
            this is the index of the stream (in the input layer)
            where to connect the current layer.
        tensor_indices: integer or list of integers.
            The output of the inbound node might be a list/tuple
            of tensor, and we might only be interested in one specific entry.
            This index allows you to specify the index of the entry in the output list
            (if applicable). `None` means that we take all outputs (as a list).
        """
        inbound_layers = tolist(inbound_layers)
        if not node_indices:
            node_indices = [0 for _ in range(len(inbound_layers))]
        else:
            node_indices = tolist(node_indices)
            assert len(node_indices) == len(inbound_layers)
        if tensor_indices is None:
            tensor_indices = [0 for _ in range(len(inbound_layers))]
        else:
            tensor_indices = tolist(tensor_indices)

        if not self._built:
            # collect input_shapes for call to build()
            input_shapes = []
            for layer, node_index, tensor_index in zip(inbound_layers, node_indices, tensor_indices):
                input_shapes.append(layer.inbound_nodes[node_index].output_shapes[tensor_index])
            # call build()
            if len(input_shapes) == 1:
                input_shapes = input_shapes[0]
            self.build(shape=input_shapes)
            self._built = True

        # creating the node automatically updates self.inbound_nodes
        # as well as outbound_nodes on inbound layers.
        Node.create_node(self, inbound_layers, node_indices, tensor_indices)

    def build(self, shape):
        self._built = True

    def _verify_input_compatibility(self, inputs):
        """This checks that the tensor(s) `inputs` verify the input
        assumptions of the layer (if any). If not, exceptions are raised.
        """
        if not self._input_spec:
            return True
        assert isinstance(self._input_spec, list), ('input_spec must be a list of InputSpec instances. '
                                                    'Found: {}'.format(self._input_spec))
        inputs = tolist(inputs)
        if len(self._input_spec) > 1:
            if len(inputs) != len(self._input_spec):
                raise Exception('Layer {} expects {} inputs, but it received {} input tensors. '
                                'Input received: {}'.format(self.name, len(self._input_spec), len(inputs), inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, self._input_spec)):
            if spec is None:
                continue

            # check ndims
            if spec.ndims is not None:
                if isinstance(spec.ndims, str):
                    int_ndims = spec.ndims[:spec.ndims.find('+')]
                    ndims = int(int_ndims)
                    if tf_ndims(x) < ndims:
                        raise Exception('Input {} is incompatible with layer {}: expected ndims >= {}, '
                                        'found ndims={}'.format(input_index, self.name, ndims, tf_ndims(x)))
                else:
                    if tf_ndims(x) != spec.ndims:
                        raise Exception('Input {} is incompatible with layer {}: expected ndims >= {}, '
                                        'found ndims={}'.format(input_index, self.name, spec.ndims, tf_ndims(x)))
            if spec.dtype is not None:
                if x.dtype().name != spec.dtype:
                    raise Exception('Input {} is incompatible with layer {}: expected dtype = {}, '
                                    'found dtype={}'.format(input_index, self.name, spec.dtype, x.dtype().name))
            if spec.shape is not None:
                if hasattr(x, '_sm_shape'):
                    x_shape = x._sm_shape
                else:
                    # shape inference
                    x_shape = tf_int_shape(x)
                for spec_dims, dims in zip(spec.shape, x_shape):
                    if spec_dims is not None:
                        if spec_dims != dims:
                            raise Exception('Input {} is incompatible with layer {}: expected shape = {}, '
                                            'found shape = {}'.format(input_index, self.name, spec.shape, x_shape))

    def _call(self, inputs):
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: input tensor, or list/tuple of input tensors.

        Returns
        -------
        A tensor or list/tuple of tensors.
        """
        return inputs

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

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Retrieves an attribute (e.g. input_tensors) from a node.

        Parameters
        ----------
        node_index: integer index of the node from which
            to retrieve the attribute
        attr: exact node attribute name
        attr_name: human-readable attribute name, for error messages
        """
        if not self.inbound_nodes:
            raise Exception('The layer has never been called '
                            'and thus has no defined {}.'.format(attr_name))
        if not len(self.inbound_nodes) > node_index:
            raise Exception('Asked to get {} at node {}, but the layer has only ' +
                            '{} inbound nodes.'.format(attr_name, node_index, len(self.inbound_nodes)))
        values = getattr(self.inbound_nodes[node_index], attr)
        if len(values) == 1:
            return values[0]
        return values

    def get_input_shape_at(self, node_index):
        """Retrieves the input shape(s) of a layer at a given node."""
        return self._get_node_attribute_at_index(node_index,
                                                 'input_shapes',
                                                 'input shape')

    def get_output_shape_at(self, node_index):
        """Retrieves the output shape(s) of a layer at a given node."""
        return self._get_node_attribute_at_index(node_index,
                                                 'output_shapes',
                                                 'output shape')

    def get_input_at(self, node_index):
        """Retrieves the input tensor(s) of a layer at a given node."""
        return self._get_node_attribute_at_index(node_index,
                                                 'input_tensors',
                                                 'input')

    def get_output_at(self, node_index):
        """Retrieves the output tensor(s) of a layer at a given node."""
        return self._get_node_attribute_at_index(node_index,
                                                 'output_tensors',
                                                 'output')


class InputLayer(Layer):
    def __init__(self, shape=None, dtype=None, batch_size=None, default=None, sparse=False, input_tensor=None,
                 reader_op=None,
                 name=None, **kwargs):

        super(InputLayer, self).__init__(name, **kwargs)
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(get_uid(prefix))
        self.name = name

        self.uses_training_phase = False

        self._input_spec = None
        self._trainable = False

        # these lists will be filled via successive calls to self.add_inbound_node()
        self.inbound_nodes = []
        self.outbound_nodes = []

        processed_params = _parse_input_params(shape, dtype, batch_size, default,
                                               sparse, input_tensor, reader_op)

        self._batch_input_shape = processed_params.batch_shape
        self._input_dtype = processed_params.dtype
        self._sparse = processed_params.sparse
        self._is_reader_op = processed_params.is_reader_op

        input_tensor = []
        if not processed_params.tensor:
            for i, (shape, dtype) in enumerate(zip(self._batch_input_shape, self._input_dtype)):
                if processed_params.default and processed_params.default[i] is not None:
                    t = tf.placeholder_with_default(processed_params.default[i], shape=shape, name=self.name)
                elif processed_params.sparse:
                    t = tf.sparse_placeholder(dtype, shape=shape, name=self.name)
                else:
                    t = tf.placeholder(dtype, shape=shape, name=self.name)

                t._sm_shape = shape
                t._sm_history = (self, 0, i)
                input_tensor.append(t)
        else:
            for i, (t, dtype) in enumerate(zip(processed_params.tensor, self._input_dtype)):
                t = tf.cast(t, dtype=dtype, name=self.name)
                t._sm_shape = self._batch_input_shape
                t._sm_history = (self, 0, i)
                input_tensor.append(t)

        self._built = True

        # create an input node to add to self.outbound_node
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=input_tensor,
             output_tensors=input_tensor,
             input_shapes=self._batch_input_shape,
             output_shapes=self._batch_input_shape)


def Input(shape=None, dtype=None, batch_size=None, default=None, sparse=False,
          tensor=None, reader_op=None, name=None):
    input_layer = InputLayer(shape, dtype, batch_size, default, sparse,
                             tensor, reader_op, name)

    # return tensor including _sm_shape and _sm_history
    # note that in this case train_output and test_output are the same pointer.
    outputs = input_layer.inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    return outputs


# noinspection PyProtectedMember
class Container(Layer):
    """Abstract base container class."""

    @property
    def updates(self):
        updates = []
        for layer in self._layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates
        return updates

    @property
    def input_spec(self):
        specs = []
        for layer in getattr(self, 'input_layers', []):
            if layer.input_spec is None:
                specs.append(None)
            else:
                if not isinstance(layer.input_spec, list):
                    raise Exception('Layer {} has an input_spec attribute that is not a list. '
                                    'We expect a list. Found input_spec = {}'.format(layer.name, layer.input_spec))
                specs += layer.input_spec
        return specs

    @property
    def uses_training_phase(self):
        """True if any layer in the graph uses it."""
        return any([layer.uses_training_phase for layer in self._layers])

    # noinspection PyMissingConstructor
    def __init__(self, input_=None, output=None, name=None):
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(get_uid(prefix))
        self.name = name

        # Container-specific properties
        self._inputs = tolist(input_)
        self._outputs = tolist(output)

        # all layers in order of horizontal graph traversal.
        # Entries are unique. Includes input and output layers.
        self._layers = []

        # container_nodes: set of nodes included in the graph
        # (not all nodes included in the layers are relevant to the current graph).
        self._container_nodes = set()  # ids of all nodes relevant to the Container
        self._nodes_by_depth = {}
        self._layers_by_depth = {}

        # list of layers (1-to-1 mapping with self._inputs)
        self._input_layers = []
        # TODO: probably useless because input layers must be Input layers (node_indices = [0], tensor_indices = [0])
        self._input_layers_node_indices = []
        self._input_tensor_indices = []
        self._input_names = []

        # list of layers (1-to-1 mapping with self._outputs)
        self._output_layers = []
        # TODO: probably useless
        self._output_layers_node_indices = []
        self._output_tensor_indices = []
        self._output_names = []

        self._internal_input_shapes = []
        self._internal_output_shapes = []

        # This is for performance optimization when calling the Container on new inputs.
        # Every time the Container is called on a set on input tensors, we compute the
        # output tensors, and output shapes in one pass, then cache them here. When one
        # of these output is queried later, we retrieve it from there instead of
        # recomputing it.
        self._output_tensor_cache = {}
        self._output_shape_cache = {}

        # layer parameters
        self.inbound_nodes = []  # will be appended to below, and by future calls to __call__
        self.outbound_nodes = []  # will be appended to by future calls to __call__

        self._input_spec = None
        self._built = False

        if self._inputs or self._outputs:
            self.process_model()

    def process_model(self):
        if not self._inputs or not self._outputs:
            raise Exception(self.__class__.__name__ + ' model cannot be build: model is empty.')

        # check for redundancy in inputs:
        inputs_set = set(self._inputs)
        if len(inputs_set) != len(self._inputs):
            raise Exception('There are redundant elements in the list of inputs passed '
                            'to the model. All inputs should only occur once. '
                            'Found: {}'.format(self._inputs))

        for x in self._inputs:
            if not hasattr(x, '_sm_history'):
                raise Exception('Input tensors to {} must contain SmartMind metadata. '
                                'Found: {}'.format(self.__class__.__name__, x))

            # check that x is an input tensor
            layer, node_index, tensor_index = x._sm_history
            if len(layer.inbound_nodes) > 1 or (layer.inbound_nodes and layer.inbound_nodes[0].inbound_layers):
                raise Exception('Input tensors to {} must be an Input layer and cannot be the output '
                                'of a previous non-Input layer. Here, the tensor was generated by layer {}.\n'
                                'Note that input tensors are instantiated via `tensor = Input(shape)`\n'
                                'The tensor causing the issue was: {}'.format(self.__class__.__name__,
                                                                              layer.name,
                                                                              x.name))

            # build self.input_layers:
            # it's supposed to be an input layer, so only one node
            # and one tensor output
            assert node_index == 0
            # assert tensor_index == 0

            self._input_layers.append(layer)
            self._input_layers_node_indices.append(node_index)
            self._input_tensor_indices.append(tensor_index)
            self._input_names.append(layer.name)
            self._internal_input_shapes.append(x._sm_shape)

        for x in self._outputs:
            if isinstance(x, list):
                layer, node_index, tensor_index, layer_name, outputs_shape = [], [], [], [], []
                for xx in x:
                    if not hasattr(xx, '_sm_history'):
                        raise Exception('Output tensors to {} must contain SmartMind metadata. '
                                        'Fount: {}'.format(self.__class__.__name__, xx))
                    l, ni, ti = getattr(xx, '_sm_history')
                    layer.append(l)
                    node_index.append(ni)
                    tensor_index.append(ti)
                    layer_name.append(l.name)
                    outputs_shape.append(xx._sm_shape)
            else:
                if not hasattr(x, '_sm_history'):
                    raise Exception('Output tensors to {} must contain SmartMind metadata. '
                                    'Fount: {}'.format(self.__class__.__name__, x))
                # build self.output_layers:
                layer, node_index, tensor_index = x._sm_history
                layer_name = layer.name
                outputs_shape = x._sm_shape

            self._output_layers.append(layer)
            self._output_layers_node_indices.append(node_index)
            self._output_tensor_indices.append(tensor_index)
            self._output_names.append(layer_name)
            self._internal_output_shapes.append(outputs_shape)

        # container_nodes: set of nodes included in the graph
        # (not all nodes included in the layers are relevant to the current graph).
        container_nodes = set()  # ids of all nodes relevant to the Container
        nodes_depths = {}  # map {node: depth value}
        layers_depths = {}  # map {layer: depth value}

        def make_node_marker(node_, depth_):
            return str(id(node_)) + '-' + str(depth_)

        def build_map_of_graph(layer_, node_index_, seen_nodes, depth_=0):
            """This recursively updates the layers_depths map.
            Does not try to detect cycles in graph (TODO?)

            Parameters
            ----------
            layer_: layer from which `tensor` comes from.
            node_index_: node index from which `tensor` comes from.
            seen_nodes: set of layer ids ("{layer_}-{depth}")
                of layers seen so far. Useful to prevent infinite loops.
            depth_: current depth in the graph (0 = last output).
            """
            node_ = layer_.inbound_nodes[node_index_]

            # use layer_marker to prevent cycles
            node_marker = make_node_marker(node_, depth_)
            if node_marker in seen_nodes:
                return

            # prevent cycles
            seen_nodes.add(node_marker)

            node_key = layer_.name + '_ib-' + str(node_index_)
            # update container_nodes
            container_nodes.add(node_key)
            # update nodes_depths
            node_depth = nodes_depths.get(node_)
            if node_depth is None:
                nodes_depths[node_] = depth_
            else:
                nodes_depths[node_] = max(depth_, node_depth)

            previously_seen_depth = layers_depths.get(layer_)
            if previously_seen_depth is None:
                current_depth = depth_
            else:
                current_depth = max(depth_, previously_seen_depth)
            layers_depths[layer_] = current_depth

            # propagate to all previous tensors connected to this node
            for i in range(len(node_.inbound_layers)):
                previous_layer = node_.inbound_layers[i]
                node_index_ = node_.node_indices[i]
                build_map_of_graph(previous_layer, node_index_, seen_nodes, current_depth + 1)

        for x in self._outputs:
            if isinstance(x, list):
                for xx in x:
                    layer, node_index, _ = xx._sm_history
                    build_map_of_graph(layer, node_index, seen_nodes=set())
            else:
                layer, node_index, _ = x._sm_history
                build_map_of_graph(layer, node_index, seen_nodes=set())

        # build a map {depth: list of nodes with this depth}
        nodes_by_depth = {}
        for node, depth in nodes_depths.items():
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)

        # get sorted list of node depths
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        # check that all tensors required are computable.
        # computable_tensors: all tensors in the graph
        # that can be computed from the inputs provided
        computable_tensors = []
        for x in self._inputs:
            computable_tensors.append(x)

        layers_with_complete_input = []  # to provide a better error msg
        for depth in depth_keys:
            for node in nodes_by_depth[depth]:
                layer = node.outbound_layer
                if layer:
                    for x in node.input_tensors:
                        if x not in computable_tensors:
                            raise Exception('Graph disconnected: cannot obtain value for tensor {} '
                                            'at layer "{}". The following previous layers were accessed '
                                            'without issue: {}'.format(x, layer.name, layers_with_complete_input))
                    for x in node.output_tensors:
                        computable_tensors.append(x)
                    layers_with_complete_input.append(layer.name)

        # set self.nodes and self.nodes_by_depth
        self._container_nodes = container_nodes
        self._nodes_by_depth = nodes_by_depth

        # build a map {depth: list of layers with this depth}
        layers_by_depth = {}
        for layer, depth in layers_depths.items():
            if depth not in layers_by_depth:
                layers_by_depth[depth] = []
            layers_by_depth[depth].append(layer)

        # get sorted list of layer depths
        depth_keys = list(layers_by_depth.keys())
        depth_keys.sort(reverse=True)

        # set self.layers and self.layers_by_depth
        layers = []
        for depth in depth_keys:
            layers_for_depth = layers_by_depth[depth]
            # container.layers needs to have a deterministic order
            layers_for_depth.sort(key=lambda x: x.name)
            for layer in layers_for_depth:
                layers.append(layer)
        self._layers = layers
        self._layers_by_depth = layers_by_depth

        # ensure name unicity, which will be crucial for serialization
        # (since serialized nodes refer to layers by their name).
        all_names = [layer.name for layer in self._layers]
        for name in all_names:
            if all_names.count(name) != 1:
                raise Exception('The name "{}" is used {} times in the model. '
                                'All layer names should be unique.'.format(name, all_names.count(name)))

        # the new container starts with a single inbound node
        # for its inputs, and no outbound nodes.
        # create the node linking internal inputs to internal outputs
        outputs_shape = []
        for x in self._outputs:
            if isinstance(x, list):
                shape = []
                for xx in x:
                    shape.append(xx._sm_shape)
                outputs_shape.append(shape)
            else:
                outputs_shape.append(x._sm_shape)

        Node(outbound_layer=self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=self._inputs,
             output_tensors=self._outputs,
             input_shapes=[x._sm_shape for x in self._inputs],
             output_shapes=outputs_shape)
        self._built = True

    def _call(self, inputs):
        """Reapplies all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        It is callable on non-SmartMind tensors.

        Parameters
        ----------
        inputs: a tensor or list of tensors.

        Returns
        -------
        A tensor if there is a single output, or
        a list of tensors if there are more than one outputs.
        """
        inputs = tolist(inputs)

        if not self._built:
            # collect input shapes to build layer
            input_shapes = []
            for x_elem in inputs:
                if hasattr(x_elem, '_sm_shape'):
                    input_shapes.append(getattr(x_elem, '_sm_shape'))
                else:
                    input_shapes.append(tf_int_shape(x_elem))

            if len(input_shapes) == 1:
                input_shapes = input_shapes[0]
            self.build(input_shapes)
            self._built = True

        cache_key = ','.join([str(id(x)) for x in inputs])
        if cache_key in self._output_tensor_cache:
            return self._output_tensor_cache[cache_key]
        else:
            output_tensors, _ = self._run_internal_graph(inputs)
            return output_tensors

    def _get_output_shape(self, input_shape):
        input_shapes = tolist(input_shape, convert_tuple=False)
        if len(input_shapes) != len(self._input_layers):
            raise Exception('Invalid input_shape argument {}: model has {} '
                            'tensor inputs.'.format(input_shape, len(self._input_layers)))

        cache_key = ','.join([str(x) for x in input_shapes])
        if cache_key in self._output_shape_cache:
            output_shapes = self._output_shape_cache[cache_key]
            if isinstance(output_shapes, list) and len(output_shapes) == 1:
                return output_shapes[0]
            return output_shapes
        else:
            # bad luck, have to run the graph manually
            layers_to_output_shapes = {}
            for i in range(len(input_shapes)):
                layer = self._input_layers[i]
                input_shape = input_shapes[i]
                # it's an input layer: _get_output_shape is identity,
                # and there is only one node and one tensor output.
                shape_key = layer.name + '_0_0'
                layers_to_output_shapes[shape_key] = input_shape

            depth_keys = list(self._nodes_by_depth.keys())
            depth_keys.sort(reverse=True)
            # iterate over nodes, by depth level
            if len(depth_keys) > 1:
                for depth in depth_keys:
                    nodes = self._nodes_by_depth[depth]
                    for node in nodes:
                        # this is always a single layer, never a list
                        layer = node.outbound_layer
                        if layer in self._input_layers:
                            # we've already covered the input layers
                            # a few lines above
                            continue
                        # potentially redundant list,
                        # same size of node.input_tensors
                        input_shapes = []
                        for j in range(len(node.inbound_layers)):
                            inbound_layer = node.inbound_layers[j]
                            node_index = node.node_indices[j]
                            tensor_index = node.tensor_indices[j]
                            shape_key = inbound_layer.name + '_%s_%s' % (node_index, tensor_index)
                            input_shape = layers_to_output_shapes[shape_key]
                            input_shapes.append(input_shape)

                        if len(input_shapes) == 1:
                            input_shapes = input_shapes[0]
                        output_shape = layer._get_output_shape(input_shapes)

                        output_shapes = tolist(output_shape, convert_tuple=False)
                        node_index = layer.inbound_nodes.index(node)
                        for j in range(len(output_shapes)):
                            shape_key = layer.name + '_%s_%s' % (node_index, j)
                            layers_to_output_shapes[shape_key] = output_shapes[j]

            # read final output shapes from layers_to_output_shapes
            output_shapes = []
            output_shape_keys = []
            for i in range(len(self._output_layers)):
                layer = self._output_layers[i]
                node_index = self._output_layers_node_indices[i]
                tensor_index = self._output_layers_node_indices[i]
                shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
                output_shape_keys.append(shape_key)

            for i, key in enumerate(output_shape_keys):
                assert key in layers_to_output_shapes
                output_shapes.append(layers_to_output_shapes[key])
            # store in cache
            self._output_shape_cache[cache_key] = output_shapes
            if type(output_shapes) is list and len(output_shapes) == 1:
                return output_shapes[0]
            return output_shapes

    def _run_internal_graph(self, inputs):
        """Computes output tensors for new inputs.

        Parameters
        ----------
        inputs: list of tensors

        Returns
        -------
        Three lists: output_tensors, output_masks, output_shapes

        Notes
        -----
        - expects `inputs` to be a list (potentially with 1 element).
        - can be run on non-SmartMind tensors.
        """
        assert isinstance(inputs, list)

        # dictionary mapping reference tensors to tuples (computed tensor, compute mask)
        # we assume a 1:1 mapping from tensor to mask
        # TODO: raise exception when a .compute_mask does not return a list the same size as call
        tensor_map = {}
        for x, y in zip(self._inputs, inputs):
            tensor_map[str(id(x))] = y

        depth_keys = list(self._nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            nodes = self._nodes_by_depth[depth]
            for node in nodes:
                # this is always a single layer, never a list
                layer = node.outbound_layer

                reference_input_tensors = node.input_tensors
                reference_output_tensors = node.output_tensors

                # if all previous input tensors are available in tensor_map,
                # then call node.inbound_layer on them
                computed_tensors = []
                for x in reference_input_tensors:
                    if str(id(x)) in tensor_map:
                        computed_tensors.append(tensor_map[str(id(x))])
                if len(computed_tensors) == len(reference_input_tensors):
                    # call layer
                    if len(computed_tensors) == 1:
                        computed_tensors = computed_tensors[0]
                    output_tensors = tolist(layer._call(computed_tensors))

                    # update _sm_shape
                    computed_tensors = tolist(computed_tensors)
                    if all([hasattr(x, '_sm_shape') for x in computed_tensors]):
                        tensor_shapes = [x._sm_shape for x in computed_tensors]
                        if len(tensor_shapes) == 1:
                            tensor_shapes = tensor_shapes[0]
                        shapes = tolist(layer._get_output_shape(tensor_shapes), convert_tuple=False)

                        for x, s in zip(output_tensors, shapes):
                            x._sm_shape = s

                    # update tensor_map
                    for x, y in zip(reference_output_tensors, output_tensors):
                        tensor_map[str(id(x))] = y

        output_tensors = []
        output_shapes = []
        for x in self._outputs:
            # todo: better error msg
            assert str(id(x)) in tensor_map, 'Could not compute output ' + str(x)
            tensor = tensor_map[str(id(x))]
            if hasattr(tensor, '_sm_shape') and output_shapes is not None:
                shape = tensor._sm_shape
                output_shapes.append(shape)
            else:
                output_shapes = None
            output_tensors.append(tensor)

        # update cache; keys are based on ids on input tensors and inputs masks
        cache_key = ','.join([str(id(x)) for x in inputs])

        if len(output_tensors) == 1:
            output_tensors = output_tensors[0]
            self._output_tensor_cache[cache_key] = output_tensors
        else:
            self._output_tensor_cache[cache_key] = output_tensors

        if output_shapes is not None:
            input_shapes = [x._sm_shape for x in inputs]
            cache_key = ','.join([str(x) for x in input_shapes])
            if len(output_shapes) == 1:
                output_shapes = output_shapes[0]
                self._output_shape_cache[cache_key] = output_shapes
            else:
                self._output_shape_cache[cache_key] = output_shapes
        return output_tensors, output_shapes


class Model(Container):
    def __init__(self, input_=None, output=None, name=None):
        super(Model, self).__init__(input_, output, name)

        self._optimizer = None
        self._sample_weight_mode = None
        self._loss = None
        self._loss_functions = None
        self._loss_weights = None

        self._targets = None
        self._total_loss = None
        self._sample_weights = None
        self._sample_weight_modes = None

        self._validation_data = None

    # noinspection PyAttributeOutsideInit
    def compile(self, optimizer, loss, targets=None, metrics=None, loss_weights=None, sample_weight_mode=None):
        if not self._built:
            self.process_model()

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
        weighted_losses = [compute_weighted_objective(fn) for fn in loss_functions]

        self._sample_weights = self._prepare_sample_weights(sample_weight_mode)

        # prepare targets of model
        if targets is not None:
            self._targets = tolist(targets)
        else:
            self._targets = []
            for i in range(len(self._outputs)):
                shape = self._internal_output_shapes[i]
                name = self._output_names[i]
                if isinstance(shape, list):
                    shape = shape[0]
                    name = name[0]
                shape = tuple([None for _ in range(len(shape))])
                target = tf.placeholder(tf.float32,
                                        shape=shape,
                                        name=name + '_target')
                target._sm_shape = shape
                self._targets.append(target)

        # prepare metrics
        self._metrics = metrics
        self._metrics_names = ['loss']
        self._metrics_tensors = []

        for i in range(len(self._outputs)):
            y_pred = self._outputs[i]
            y_target = self._targets[i]
            weighted_loss = weighted_losses[i]
            sample_weight = self._sample_weights[i]
            mask = None
            loss_weight = loss_weights_list[i]
            output_loss = weighted_loss(y_pred, y_target, sample_weight, loss_weight, mask)

            if len(self._outputs) > 1:
                self._metrics_tensors.append(output_loss / loss_weight)
                self._metrics_names.append("_".join(tolist(self._output_names[i])) + '_loss')

        # add regularization penalties to the loss
        self._total_loss = get_total_loss(add_regularization_losses=True)

        # list of same size as output_names.
        # contains tuples (metrics for output, names of metrics)
        nested_metrics = collect_metrics(metrics, self._output_names)
        for i in range(len(self._outputs)):
            y_true = self._targets[i]
            y_pred = self._outputs[i]
            output_metrics = nested_metrics[i]

            for metric in output_metrics:
                if metric == 'accuracy' or metric == 'acc':
                    # custom handling of accuracy (because of class mode duality)
                    output_shape = self._internal_output_shapes[i]
                    if output_shape[-1] == 1 or self._loss_functions[i] == objectives.binary_crossentropy:
                        # case: binary accuracy
                        self._metrics_tensors.append(metrics_module.binary_accuracy(y_true, y_pred))
                    elif self._loss_functions[i] == objectives.sparse_categorical_crossentropy:
                        # case: categorical accuracy with sparse targets
                        self._metrics_tensors.append(
                            metrics_module.sparse_categorical_accuracy(y_true, y_pred))
                    elif self._loss_functions[i] == objectives.vr_class_reward:
                        # case: categorical accuracy with sparse targets
                        self._metrics_tensors.append(
                            metrics_module.sparse_categorical_accuracy(y_true, y_pred[0]))
                    else:
                        # case: categorical accuracy with dense targets
                        self._metrics_tensors.append(metrics_module.categorical_accuracy(y_true, y_pred))
                    if len(self._output_names) == 1:
                        self._metrics_names.append('acc')
                    else:
                        self._metrics_names.append("_".join([l.name for l in tolist(self._output_layers[i])]) + '_acc')
                else:
                    metric_fn = metrics_module.get(metric)
                    self._metrics_tensors.append(metric_fn(y_true, y_pred))
                    if len(self._output_names) == 1:
                        self._metrics_names.append(metric_fn.__name__)
                    else:
                        self._metrics_names.append(
                            "_".join([l.name for l in tolist(self._output_layers[i])]) + '_' + metric_fn.__name__)

        # functions for train, test and predict, will be compiled lazily in case user does not
        # use all functions
        self._fn_train = None
        self._fn_test = None
        self._fn_predict = None

    def fit(self, x, y, batch_size=32, n_epoch=10, verbose=1, callbacks=None,
            validation_split=0, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None):
        # validate user data
        x, y, sample_weights = self._standardize_user_data(x, y,
                                                           sample_weight=sample_weight,
                                                           class_weight=class_weight,
                                                           check_batch_dim=False,
                                                           batch_size=batch_size)

        # prepare validation data
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise
            val_x, val_y, val_sample_weights = self._standardize_user_data(val_x, val_y,
                                                                           sample_weight=val_sample_weight,
                                                                           check_batch_dim=False,
                                                                           batch_size=batch_size)
            self._make_test_function()
            val_f = self._fn_test

            val_ins = val_x + val_y + val_sample_weights
            if self.uses_training_phase and type(tf_training_phase()) is not int:
                val_ins += [0.]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
            y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
            sample_weights, val_sample_weights = (
                slice_X(sample_weights, 0, split_at), slice_X(sample_weights, split_at))
            self._make_test_function()
            val_f = self._fn_test

            val_ins = val_x + val_y + val_sample_weights
            if self.uses_training_phase and type(tf_training_phase()) is not int:
                val_ins += [0.]
        else:
            do_validation = False
            val_f = None
            val_ins = None

        ins = x + y + sample_weights
        if self.uses_training_phase and not isinstance(tf_training_phase(), int):
            ins.append(1.)
        self._make_train_function()
        f = self._fn_train

        # prepare display labels
        out_labels = self._metrics_names

        # rename duplicated metrics name
        # (can happen with an output layer shared among multiple dataflows)
        deduped_out_labels = []
        for i, label in enumerate(out_labels):
            new_label = label
            if out_labels.count(label) > 1:
                dup_idx = out_labels[:i].count(label)
                new_label += '_' + str(dup_idx + 1)
            deduped_out_labels.append(new_label)
        out_labels = deduped_out_labels

        if do_validation:
            callback_metrics = copy.copy(out_labels) + ['val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)

        # delegate logic to _fit_loop
        return self._fit_loop(f, ins, out_labels=out_labels,
                              batch_size=batch_size, n_epoch=n_epoch,
                              verbose=verbose, callbacks=callbacks,
                              val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                              callback_metrics=callback_metrics)

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        """Returns the loss value and metrics values for the model
        in test mode. Computation is done in batches.

        # Arguments
            x: Numpy array of test data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named, you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named, you can also pass a dictionary
                mapping output names to Numpy arrays.
            batch_size: integer. Number of samples per gradient update.

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """
        # validate user data
        x, y, sample_weights = self._standardize_user_data(x, y,
                                                           sample_weight=sample_weight,
                                                           check_batch_dim=False,
                                                           batch_size=batch_size)
        # prepare inputs, delegate logic to _test_loop
        ins = x + y + sample_weights
        if self.uses_training_phase and not isinstance(tf_training_phase(), int):
            ins += [0.]

        self._make_test_function()
        f = self._fn_test
        return self._test_loop(f, ins,
                               batch_size=batch_size,
                               verbose=verbose)

    def predict(self, x, batch_size=32, verbose=0):
        """Generates output predictions for the input samples,
        processing the samples in a batched way.

        # Arguments
            x: the input data, as a Numpy array
                (or list of Numpy arrays if the model has multiple outputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A Numpy array of predictions.
        """
        # validate user data
        x = standardize_input_data(x, self._input_names,
                                   self._internal_input_shapes,
                                   check_batch_dim=False)
        # if self.stateful:
        #     if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
        #         raise Exception('In a stateful network, '
        #                         'you should only pass inputs with '
        #                         'a number of samples that can be '
        #                         'divided by the batch size. Found: ' +
        #                         str(x[0].shape[0]) + ' samples. '
        #                         'Batch size: ' + str(batch_size) + '.')

        # prepare inputs, delegate logic to _predict_loop
        ins = x
        if self.uses_training_phase and not isinstance(tf_training_phase(), int):
            ins += [0.]

        self._make_predict_function()
        f = self._fn_predict
        return self._predict_loop(f, ins,
                                  batch_size=batch_size, verbose=verbose)

    def fit_reader_op(self, batch_size=32, n_epoch=10, moving_average_decay=0.9999, sample_weight=None,
                      class_weight=None):

        # sample_weights = standardize_sample_or_class_weights(sample_weight,
        #                                                      self._output_names,
        #                                                      'sample_weight')
        # sample_weights = [standardize_weights(ref, sw, cw, mode)
        #                   for (ref, sw, cw, mode)
        #                   in zip(y, sample_weights, class_weight, self._sample_weight_modes)]
        # ins =
        if self.uses_training_phase and not isinstance(tf_training_phase(), int):
            ins = [1.]

        init = tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables())

        self._make_train_function()

        sess = tf_get_session()
        sess.run(init)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                # Run training steps or whatever
                start_time = time.time()
                output = self._fn_train()
                duration = time.time() - start_time

                assert not np.isnan(output[0]), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, output[0],
                                        examples_per_sec, sec_per_batch))

                # if step % 100 == 0:
                #     summary_str = sess.run(summary_op)
                #     summary_writer.add_summary(summary_str, step)
                #
                # # Save the model checkpoint periodically.
                # if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                #     checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                #     saver.save(sess, checkpoint_path, global_step=step)
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

    def _fit_loop(self, f, ins, out_labels=None, batch_size=32,
                  n_epoch=100, verbose=1, callbacks=None,
                  val_f=None, val_ins=None, shuffle=True,
                  callback_metrics=None):
        out_labels = out_labels if out_labels is not None else []
        callbacks = callbacks if callbacks is not None else []
        callback_metrics = callback_metrics if callback_metrics is not None else []

        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print('Train on %d samples, validate on %d samples' %
                      (len(ins[0]), len(val_ins[0])))

        n_train_sample = len(ins[0])
        index_array = np.arange(n_train_sample)

        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self
        # (used by Sequential models)
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self

        callbacks._set_model(callback_model)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': n_epoch,
            'nb_sample': n_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()
        callback_model.stop_training = False
        self.validation_data = val_ins

        init = tf.initialize_all_variables()

        sess = tf_get_session()
        sess.run(init)

        for epoch in range(n_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(n_train_sample, batch_size)

            epoch_logs = {}
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(ins[-1], float):
                        # do not slice the training phase flag
                        ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = slice_X(ins, batch_ids)
                except TypeError:
                    raise Exception('TypeError while preparing batch. '
                                    'If using HDF5 input data, pass shuffle="batch".')
                batch_logs = {
                    'batch': batch_index,
                    'size': len(batch_ids)
                }
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(ins_batch)
                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o
            callbacks.on_epoch_end(epoch, epoch_logs)
            if callback_model.stop_training:
                break

        callbacks.on_train_end()
        return self.history

    def _test_loop(self, f, ins, batch_size=32, verbose=0):
        """Abstract method to loop over some data in batches.

        # Arguments
            f: Keras function returning a list of tensors.
            ins: list of tensors to be fed to `f`.
            batch_size: integer batch size.
            verbose: verbosity mode.

        # Returns
            Scalar loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """
        nb_sample = len(ins[0])
        outs = []
        progbar = None
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)

        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if isinstance(ins[-1], float):
                # do not slice the training phase flag
                ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(ins_batch)
            if isinstance(batch_outs, list):
                if batch_index == 0:
                    for _ in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

            if verbose == 1:
                progbar.update(batch_end)
        for i, out in enumerate(outs):
            outs[i] /= nb_sample
        if len(outs) == 1:
            return outs[0]
        return outs

    def _predict_loop(self, f, ins, batch_size=32, verbose=0):
        """Abstract method to loop over some data in batches.

        # Arguments
            f: Keras function returning a list of tensors.
            ins: list of tensors to be fed to `f`.
            batch_size: integer batch size.
            verbose: verbosity mode.

        # Returns
            Array of predictions (if the model has a single output)
            or list of arrays of predictions
            (if the model has multiple outputs).
        """
        nb_sample = len(ins[0])
        outs = []
        progbar = None
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if isinstance(ins[-1], float):
                # do not slice the training phase flag
                ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(ins_batch)
            if not isinstance(batch_outs, list):
                batch_outs = [batch_outs]
            if batch_index == 0:
                for batch_out in batch_outs:
                    shape = (nb_sample,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape, dtype=tf.float32))

            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if verbose == 1:
                progbar.update(batch_end)
        if len(outs) == 1:
            return outs[0]
        return outs

    def _make_train_function(self):
        if not hasattr(self, '_fn_train'):
            raise Exception('Model must be compiled before using it to train.')
        if self._fn_train is None:
            inputs = []
            for tensor in self._inputs + self._targets:
                # noinspection PyProtectedMember
                if tensor._op._op_def.name == u'Placeholder':
                    inputs.append(tensor)
            inputs += self._sample_weights
            if self.uses_training_phase and not isinstance(tf_training_phase(), int):
                inputs.append(tf_training_phase())

            outputs = [self._total_loss]
            with tf.control_dependencies(outputs):
                grads = self._optimizer.compute_gradients(self._total_loss)
            apply_gradient_op = self._optimizer.apply_gradients(grads)

            # # Track the moving averages of all trainable variables.
            # variable_averages = tf.train.ExponentialMovingAverage(
            #     moving_average_decay, self._optimizer.global_step)
            # variables_averages_op = variable_averages.apply(tf.trainable_variables())

            with tf.control_dependencies([apply_gradient_op]):
                train_op = tf.no_op(name='train')

            outputs += self._metrics_tensors
            self._fn_train = tf_function(inputs, outputs,
                                         ops=[train_op],
                                         updates=self.updates)

    def _make_test_function(self):
        if not hasattr(self, '_fn_test'):
            raise Exception('You must compile your model before using it.')
        if self._fn_test is None:
            inputs = []
            for tensor in self._inputs + self._targets:
                # noinspection PyProtectedMember
                if tensor._op._op_def.name == u'Placeholder':
                    inputs.append(tensor)
            inputs += self._sample_weights
            if self.uses_training_phase and not isinstance(tf_training_phase(), int):
                inputs.append(tf_training_phase())

            # return loss and metrics, no gradient updates.
            # Does update the network states.
            outputs = [self._total_loss] + self._metrics_tensors
            self._fn_test = tf_function(inputs, outputs, ops=[], updates=[])

    def _make_predict_function(self):
        if not hasattr(self, '_fn_predict'):
            self._fn_predict = None
        if self._fn_predict is None:
            inputs = []
            for tensor in self._inputs:
                # noinspection PyProtectedMember
                if tensor._op._op_def.name == u'Placeholder':
                    inputs.append(tensor)
            inputs += self._sample_weights
            if self.uses_training_phase and not isinstance(tf_training_phase(), int):
                inputs.append(tf_training_phase())

            # returns network outputs. Does not update weights.
            # Does update the network states.
            self._fn_predict = tf_function(inputs, self._outputs, ops=[], updates=[])

    def _prepare_sample_weights(self, sample_weight_mode):
        # if sample_weight_mode is None:
        #     return [None] * len(self._output_names)

        if isinstance(sample_weight_mode, dict):
            for name in sample_weight_mode:
                if name not in self._output_names:
                    raise Exception('Unknown entry in sample_weight_mode dictionary: {}.'
                                    ' Only expected the following keys: {}'.format(name, self._output_names))
            sample_weights = []
            sample_weight_modes = []
            for name in self._output_names:
                if name not in sample_weight_mode:
                    raise Exception('Output {} missing from sample_weight_modes dictionary'.format(name))
                scope_name = name + '_sample_weights'
                if sample_weight_mode.get(name) == 'temporal':
                    weight = tf.placeholder(tf.float32, shape=tuple([None] * 2), name=scope_name)
                    sample_weight_modes.append('temporal')
                else:
                    weight = tf.placeholder(tf.float32, shape=tuple([None]), name=scope_name)
                    sample_weight_modes.append(None)
                sample_weights.append(weight)
        elif isinstance(sample_weight_mode, list):
            if len(sample_weight_mode) != len(self._outputs):
                raise Exception('When passing a list as sample_weight_mode, it should'
                                ' have one entry per model outputs. The model has {} outputs,'
                                ' but you passed sample_weight_mode={}'.format(len(self._outputs), sample_weight_mode))
            sample_weights = []
            sample_weight_modes = []
            for mode, name in zip(sample_weight_mode, self._output_names):
                scope_name = name + '_sample_weights'
                if mode == 'temporal':
                    weight = tf.placeholder(tf.float32, shape=tuple([None] * 2), name=scope_name)
                    sample_weight_modes.append('temporal')
                else:
                    weight = tf.placeholder(tf.float32, shape=tuple([None]), name=scope_name)
                    sample_weight_modes.append(None)
                sample_weights.append(weight)
        else:
            if sample_weight_mode == 'temporal':
                sample_weights = [tf.placeholder(tf.float32, shape=tuple([None] * 2), name=name + '_sample_weights')
                                  for name in self._output_names]
                sample_weight_modes = ['temporal'] * len(self._output_names)
            else:
                sample_weights = [tf.placeholder(tf.float32, shape=tuple([None]), name="_".join(tolist(name)) + '_sample_weights')
                                  for name in self._output_names]
                sample_weight_modes = [None] * len(self._output_names)

        self._sample_weight_modes = sample_weight_modes
        return sample_weights

    def _standardize_user_data(self, x, y,
                               sample_weight=None, class_weight=None,
                               check_batch_dim=True, batch_size=None):
        if not hasattr(self, '_optimizer'):
            raise Exception('You must compile a model before training/testing.'
                            ' Use `model.compile(optimizer, loss)`.')

        output_shapes = []
        for output_shape, loss_fn in zip(self._internal_output_shapes, self._loss_functions):
            if loss_fn.__name__ == 'sparse_categorical_crossentropy':
                output_shapes.append(output_shape[:-1] + (1,))
            elif loss_fn.__name__ == 'vr_class_reward':
                output_shapes.append(output_shape[0][:-1] + (1,))
            elif getattr(objectives, loss_fn.__name__, None) is None:
                output_shapes.append(None)
            else:
                output_shapes.append(output_shape)
        x = standardize_input_data(x, self._input_names,
                                   self._internal_input_shapes,
                                   check_batch_dim=False,
                                   exception_prefix='model input')
        y = standardize_input_data(y, self._output_names,
                                   output_shapes,
                                   check_batch_dim=False,
                                   exception_prefix='model target')
        sample_weights = standardize_sample_or_class_weights(sample_weight,
                                                             self._output_names,
                                                             'sample, weight')
        class_weights = standardize_sample_or_class_weights(class_weight,
                                                            self._output_names,
                                                            'class_weight')
        sample_weights = [standardize_weights(ref, sw, cw, mode)
                          for (ref, sw, cw, mode)
                          in zip(y, sample_weights, class_weights, self._sample_weight_modes)]
        check_array_lengths(x, y, sample_weights)
        check_loss_and_target_compatibility(y, self._loss_functions, self._internal_output_shapes)
        # if self.stateful and batch_size:
        #     if x[0].shape[0] % batch_size != 0:
        #         raise Exception('In a stateful network, '
        #                         'you should only pass inputs with '
        #                         'a number of samples that can be '
        #                         'divided by the batch size. Found: ' +
        #                         str(x[0].shape[0]) + ' samples')
        return x, y, sample_weights

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses.

        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

        Parameters
        ----------
        total_loss: Total loss from loss().

        Returns
        -------
        loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = get_losses()
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name + ' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op


def standardize_input_data(data, names, shapes=None,
                           check_batch_dim=True,
                           exception_prefix=''):
    """Users may pass data as a list of arrays, dictionary of arrays,
    or as a single array. We normalize this to an ordered list of
    arrays (same order as `names`), while checking that the provided
    arrays have shapes that match the network's expectations.
    """
    if type(data) is dict:
        arrays = []
        for name in names:
            if name not in data:
                raise Exception('No data provided for "' +
                                name + '". Need data for each key in: ' +
                                str(data.keys()))
            arrays.append(data[name])
    elif type(data) is list:
        if len(data) != len(names):
            if len(data) > 0 and hasattr(data[0], 'shape'):
                raise Exception('Error when checking ' + exception_prefix +
                                ': the list of Numpy arrays '
                                'that you are passing to your model '
                                'is not the size the model expected. '
                                'Expected to see ' + str(len(names)) +
                                ' arrays but instead got '
                                'the following list of ' + str(len(data)) +
                                ' arrays: ' + str(data)[:200] +
                                '...')
            else:
                if len(names) == 1:
                    data = [np.asarray(data)]
                else:
                    raise Exception('Error when checking ' + exception_prefix +
                                    ': you are passing a list as '
                                    'input to your model, '
                                    'but the model expects '
                                    'a list of ' + str(len(names)) +
                                    ' Numpy arrays instead. '
                                    'The list you passed was: ' +
                                    str(data)[:200])
        arrays = data
    else:
        if not hasattr(data, 'shape'):
            raise Exception('Error when checking ' + exception_prefix +
                            ': data should be a Numpy array, '
                            'or list/dict of Numpy arrays. '
                            'Found: ' + str(data)[:200] + '...')
        if len(names) != 1:
            # case: model expects multiple inputs but only received
            # a single Numpy array
            raise Exception('The model expects ' + str(len(names)) +
                            ' input arrays, but only received one array. '
                            'Found: array with shape ' + str(data.shape))
        arrays = [data]

    # make arrays at least 2D
    for i in range(len(names)):
        array = arrays[i]
        if len(array.shape) == 1:
            array = np.expand_dims(array, 1)
            arrays[i] = array

    # check shapes compatibility
    if shapes:
        for i in range(len(names)):
            if shapes[i] is None:
                continue
            array = arrays[i]
            if len(array.shape) != len(shapes[i]):
                raise Exception('Error when checking ' + exception_prefix +
                                ': expected ' + names[i] +
                                ' to have ' + str(len(shapes[i])) +
                                ' dimensions, but got array with shape ' +
                                str(array.shape))
            for j, (dim, ref_dim) in enumerate(zip(array.shape, shapes[i])):
                if not j and not check_batch_dim:
                    # skip the first axis
                    continue
                if ref_dim:
                    if ref_dim != dim:
                        raise Exception('Error when checking ' + exception_prefix +
                                        ': expected ' + names[i] +
                                        ' to have shape ' + str(shapes[i]) +
                                        ' but got array with shape ' +
                                        str(array.shape))
    return arrays


def standardize_weights(y, sample_weight=None, class_weight=None,
                        sample_weight_mode=None):
    """Performs weight input validation and standardization
    to a single sample-wise (or timestep-wise) weight array.
    """
    if sample_weight_mode is not None:
        if sample_weight_mode != 'temporal':
            raise Exception('"sample_weight_mode '
                            'should be None or "temporal". '
                            'Found: ' + str(sample_weight_mode))
        if len(y.shape) < 3:
            raise Exception('Found a sample_weight array for '
                            'an input with shape ' +
                            str(y.shape) + '. '
                                           'Timestep-wise sample weighting (use of '
                                           'sample_weight_mode="temporal") is restricted to '
                                           'outputs that are at least 3D, i.e. that have '
                                           'a time dimension.')
        if sample_weight is not None and len(sample_weight.shape) != 2:
            raise Exception('Found a sample_weight array with shape ' +
                            str(sample_weight.shape) + '. '
                                                       'In order to use timestep-wise sample weighting, '
                                                       'you should pass a 2D sample_weight array.')
    else:
        if sample_weight is not None and len(sample_weight.shape) != 1:
            raise Exception('Found a sample_weight array with shape ' +
                            str(sample_weight.shape) + '. '
                                                       'In order to use timestep-wise sample weights, '
                                                       'you should specify sample_weight_mode="temporal" '
                                                       'in compile(). If you just mean to use '
                                                       'sample-wise weights, make sure your '
                                                       'sample_weight array is 1D.')

    if sample_weight is not None:
        assert len(sample_weight.shape) <= len(y.shape)
        # TODO: proper error message
        assert y.shape[:sample_weight.ndim] == sample_weight.shape
        return sample_weight
    elif isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise Exception('class_weight not supported for '
                            '3+ dimensional targets.')
        if y.shape[1] > 1:
            y_classes = y.argmax(axis=1)
        elif y.shape[1] == 1:
            y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y
        weights = np.asarray([class_weight[cls] for cls in y_classes])
        return weights
    else:
        if sample_weight_mode is None:
            return np.ones((y.shape[0],), dtype='float32')
        else:
            return np.ones((y.shape[0], y.shape[1]), dtype='float32')


def standardize_sample_or_class_weights(x_weight, output_names, weight_type):
    if x_weight is None or len(x_weight) == 0:
        return [None for _ in output_names]
    if len(output_names) == 1:
        if type(x_weight) is list and len(x_weight) == 1:
            return x_weight
        if type(x_weight) is dict and output_names[0] in x_weight:
            return [x_weight[output_names[0]]]
        else:
            return [x_weight]
    if type(x_weight) is list:
        if len(x_weight) != len(output_names):
            raise Exception('Provided `' + weight_type + '` was a list of ' +
                            str(len(x_weight)) +
                            ' elements, but the model has ' +
                            str(len(output_names)) + ' outputs. '
                                                     'You should provide one `' + weight_type + '`'
                                                                                                'array per model output.')
        return x_weight
    if type(x_weight) is dict:
        x_weights = []
        for name in output_names:
            x_weights.append(x_weight.get(name))
        return x_weights
    else:
        raise Exception('The model has multiple outputs, so `' +
                        weight_type + '` '
                                      'should be either a list of a dict. '
                                      'Provided `' + weight_type +
                        '` type not understood: ' +
                        str(x_weight))


def check_array_lengths(X, Y, W):
    x_lengths = [x.shape[0] for x in X]
    y_lengths = [y.shape[0] for y in Y]
    w_lengths = [w.shape[0] for w in W]
    set_x = set(x_lengths)
    if len(set_x) != 1:
        raise Exception('All input arrays (x) should have '
                        'the same number of samples.')
    set_y = set(y_lengths)
    if len(set_y) != 1:
        raise Exception('All target arrays (y) should have '
                        'the same number of samples.')
    set_w = set(w_lengths)
    if len(set_w) != 1:
        raise Exception('All sample_weight arrays should have '
                        'the same number of samples.')
    if list(set_x)[0] != list(set_y)[0]:
        raise Exception('Input arrays should have '
                        'the same number of samples as target arrays. Found ' +
                        str(list(set_x)[0]) + ' input samples and ' +
                        str(list(set_y)[0]) + ' target samples.')
    if list(set_x)[0] != list(set_w)[0]:
        raise Exception('Sample_weight arrays should have '
                        'the same number of samples as input arrays. Found ' +
                        str(list(set_x)[0]) + ' input samples and ' +
                        str(list(set_w)[0]) + ' target samples.')


def check_loss_and_target_compatibility(targets, losses, output_shapes):
    assert len(targets) == len(losses) == len(output_shapes)
    key_losses = {'mean_square_error',
                  'binary_crossentropy',
                  'categorical_crossentropy'}
    for y, loss, shape in zip(targets, losses, output_shapes):
        if loss.__name__ == 'categorical_crossentropy':
            if y.shape[1] == 1:
                raise Exception('You are passing a target array of shape ' + str(y.shape) +
                                ' while using as loss `categorical_crossentropy`. '
                                '`categorical_crossentropy` expects '
                                'targets to be binary matrices (1s and 0s) '
                                'of shape (samples, classes). '
                                'If your targets are integer classes, '
                                'you can convert them to the expected format via:\n'
                                '```\n'
                                'from keras.utils.np_utils import to_categorical\n'
                                'y_binary = to_categorical(y_int)\n'
                                '```\n'
                                '\n'
                                'Alternatively, you can use the loss function '
                                '`sparse_categorical_crossentropy` instead, '
                                'which does expect integer targets.')
        if loss.__name__ in key_losses and shape[1] is not None and y.shape[1] != shape[1]:
            raise Exception('A target array with shape ' + str(y.shape) +
                            ' was passed for an output of shape ' + str(shape) +
                            ' while using as loss `' + loss.__name__ + '`. '
                                                                       'This loss expects '
                                                                       'targets to have the same shape '
                                                                       'as the output.')
