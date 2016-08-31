from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from .framework import Model
from .utils import tolist


class Sequential(Model):

    def __init__(self, layers=None, name=None):
        super(Sequential, self).__init__(name)

        self._layers = []
        self._built = False

        for layer in tolist(layers):
            self.add(layer)

    def add(self, layer):
        if not self._outputs:
            # first layer in model
            if len(layer._inbound_layers) == 0:
                # create input layer
                if not hasattr(layer, '_batch_input_shape'):
                    raise Exception('For the first layer in a Sequential model `input_shape` is required. '
                                    'Alternatively, the shape is inferred from `default_input` if provided.')
            layer.create_input_layer()

            self._inputs = layer.input_tensors
            self._outputs = layer.output_tensors

            input_layer = []
            for l in layer.input_tensors:
                in_layer, _ = getattr(l, '_sm_history')
                input_layer.append(in_layer)
            if len(input_layer) == 1:
                input_layer = input_layer[0]
            self._layers.append(input_layer)
        else:
            prev_layer = self._layers[-1]
            if isinstance(prev_layer.output_tensors, list) and len(prev_layer.output_tensors) > 1:
                raise Exception('All layers in a Sequential model should have a single output'
                                ' tensor. For multi-output layers use the Parallel model.')
            self._outputs = layer(prev_layer)

        self._layers.append(layer)


class Parallel(Model):

    def __init__(self):
        super(Parallel, self).__init__()
