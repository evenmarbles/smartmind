from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from .framework import Model
from .framework import Node
from .utils import tolist


class Sequential(Model):

    # @property
    # def uses_training_phase(self):
    #     if not self._built:
    #         self.build()
    #     return super(Sequential, self).uses_training_phase()

    def __init__(self, layers=None, name=None):
        self._built = False

        for layer in tolist(layers):
            self.add(layer)

        super(Sequential, self).__init__(name)

    def add(self, layer):
        """Adds a layer instance on top of the layer stack.

        Parameters
        ----------
        layer: layer instance.
        """
        if not self._outputs:
            # first layer in model: check that it is an input layer
            if len(layer.inbound_nodes) == 0:
                # create input layer
                if layer.input_shape is None:
                    raise Exception('For the first layer in a Sequential model the '
                                    'argument `input_shape` is required.')
                layer.create_input_layer(layer.input_shape, layer.input_dtype, layer.batch_size)

            if len(layer.inbound_nodes) != 1:
                raise Exception('A layer added to a Sequential model must not already be connected '
                                'somewhere else. Model received layer {} which has {} pre-existing '
                                'inbound connections.'.format(layer.name, len(layer.inbound_nodes)))

            if len(layer.inbound_nodes[0].output_tensors) != 1:
                raise Exception('All layers in a Sequential model should have a single output tensor. '
                                'For multi-output layers, use the functional API.')

            self._outputs = layer.inbound_nodes[0].output_tensors
            self._inputs = get_source_inputs(self._outputs[0])

            # We create an input node, which we will keep updated
            # as we add more layers
            Node(outbound_layer=self,
                 inbound_layers=[],
                 node_indices=[],
                 tensor_indices=[],
                 input_tensors=self._inputs,
                 output_tensors=self._outputs,
                 input_shapes=[x._sm_shape for x in self._inputs],
                 output_shapes=[self._outputs[0]._sm_shape])
        else:
            output_tensor = layer(self._outputs[0])
            if isinstance(output_tensor, list):
                raise Exception('All layers in a Sequential model should have a single output '
                                'tensor. For multi-output layers, use the functional API.')
            self._outputs = [output_tensor]
            # update self.inbound_nodes
            self.inbound_nodes[0].output_tensors = self._outputs
            self.inbound_nodes[0].output_shapes = [self._outputs[0]._sm_shape]

        self._layers.append(layer)
        self._built = False

    def build(self, shape=None):
        if not self._inputs or not self._outputs:
            raise Exception('Sequential model cannot be built: model is empty. '
                            'Add some layers first.')
        self.process_model()
        self._built = True

    def predict(self, x, batch_size=32, verbose=0):
        if not self._built:
            self.build()
        return super(Sequential, self).predict(x, batch_size, verbose)


class Parallel(Model):

    def __init__(self):
        super(Parallel, self).__init__()


def get_source_inputs(tensor, layer=None, node_index=None):
    """Returns the list of input tensors
    necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    Parameters
    ----------
    tensor: the tensor to start from.
    layer: origin layer of the tensor. Will be
        determined via tensor._keras_history if not provided.
    """
    if not hasattr(tensor, '_sm_history'):
        raise Exception('Tensor must be a SmartMind tensor. Found: ' + str(tensor))

    if layer is None or node_index:
        layer, node_index, _ = tensor._sm_history
    if not layer.inbound_nodes:
        return [tensor]
    else:
        node = layer.inbound_nodes[node_index]
        if not node.inbound_layers:
            # reached an Input layer, stop recursion
            return node.input_tensors
        else:
            source_tensors = []
            for i in range(len(node.inbound_layers)):
                x = node.input_tensors[i]
                layer = node.inbound_layers[i]
                node_index = node.node_indices[i]
                previous_sources = get_source_inputs(x, layer, node_index)

                # avoid input redundancy
                for x in previous_sources:
                    if x not in source_tensors:
                        source_tensors.append(x)
            return source_tensors
