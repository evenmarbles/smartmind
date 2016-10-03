from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
# noinspection PyUnresolvedReferences
from six.moves import range
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn import _rnn_step

from .. import activations

from ..framework import Model
from ..framework import Layer
from ..utils import get_from_module
from ..utils import process_params
from ..utils import tolist


class Recurrent(Layer):
    def __init__(self, num_units, cell='BasicRNNCell', initial_state=None,
                 activation='linear', sequence_length=None, return_sequences=False,
                 trainable=True, input_shape=None, input_dtype=None,
                 batch_size=None, name=None):
        super(Recurrent, self).__init__(trainable, name, input_shape=input_shape,
                                        input_dtype=input_dtype, batch_size=batch_size)

        self._cell = get_cell(self._update_cell_params(cell, num_units, activation))
        self._output_size = self._cell.output_size

        self._initial_state = initial_state
        self._sequence_length = sequence_length
        self._return_sequences = return_sequences

    def _call(self, x):
        inputs = self._preprocess_input(x)
        outputs = []

    def _preprocess_input(self, x):
        return x

    def _get_output_shape(self, input_shape):
        if self._return_sequences:
            sequence_length = self._sequence_length
            input_shape = tolist(input_shape, convert_tuple=False)

            if sequence_length is None or len(input_shape) > 1:
                sequence_length = len(input_shape)

            return input_shape[0], sequence_length, self._output_size
        else:
            return input_shape[0], self._output_size

    def _update_cell_params(self, identifier, num_units, activation):
        if isinstance(identifier, six.string_types):
            identifier = {
                'name': identifier,
                'num_units': num_units,
                'activation': activations.get(activation)}
        elif isinstance(identifier, dict):
            identifier.update({
                'num_units': num_units,
                'activation': activations.get(activation)})
        elif isinstance(identifier, (list, tuple)):
            if isinstance(identifier, tuple):
                identifier = list(identifier)
            identifier[1] = num_units
            identifier[-1] = activations.get(activation)
        return identifier


class AttentionRNN(Recurrent):
    def __init__(self, num_units, input_model, action_model, cell='BasicRNNCell',
                 activation='linear', initial_state=None, sequence_length=None,
                 return_sequences=False, trainable=True, input_shape=None, input_dtype='float32',
                 batch_size=None, name=None):

        self._input_model = input_model
        assert isinstance(input_model, Model), "input_ must be of type Model"

        self._action_model = action_model
        assert isinstance(action_model, Model), "action must be of type Model"

        self.uses_training_phase = any(
            [layer.uses_training_phase for layer in self._input_model._layers + self._action_model._layers])

        super(AttentionRNN, self).__init__(num_units, cell, initial_state, activation,
                                           sequence_length, return_sequences, trainable,
                                           input_shape, input_dtype, batch_size, name)

    def _call(self, inputs):
        # inputs = self._preprocess_input(inputs)

        actions = []
        outputs = []

        # Create a new scope in which the caching device is either
        # determined by the parent scope, or is set to place the cached
        # Variable using the same placement as for the rest of the RNN.
        with tf.variable_scope(self.name or "AttentionRNN") as varscope:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

            # Obtain the first sequence of the input
            first_input = inputs
            while nest.is_sequence(first_input):
                first_input = first_input[0]

            # Temporarily avoid EmbeddingWrapper and seq2seq badness
            # TODO(lukaszkaiser): remove EmbeddingWrapper
            if first_input.get_shape().ndims != 1:

                input_shape = first_input.get_shape().with_rank_at_least(2)
                fixed_batch_size = input_shape[0]

                flat_inputs = nest.flatten(inputs)
                for flat_input in flat_inputs:
                    input_shape = flat_input.get_shape().with_rank_at_least(2)
                    batch_size, input_size = input_shape[0], input_shape[1:]
                    fixed_batch_size.merge_with(batch_size)
                    for i, size in enumerate(input_size):
                        if size.value is None:
                            raise ValueError(
                                "Input size (dimension %d of inputs) must be accessible via "
                                "shape inference, but saw value None." % i)
            else:
                fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                batch_size = tf.shape(inputs)[0]
            if self._initial_state is not None:
                state = self._initial_state
            else:
                state = self._cell.zero_state(batch_size, inputs.dtype)

            def _create_zero_output(output_size_):
                # convert int to TensorShape if necessary
                size_ = [batch_size] + tolist(output_size_)
                output_ = tf.zeros(tf.pack(size_), inputs.dtype)
                shape = [fixed_batch_size.value] + tolist(output_size_)
                output_.set_shape(tf.TensorShape(shape))
                return output_

            flat_output_size = nest.flatten(self._output_size)
            flat_zero_output = tuple(
                _create_zero_output(size) for size in flat_output_size)
            zero_output = nest.pack_sequence_as(structure=self._output_size,
                                                flat_sequence=flat_zero_output)

            inputs_ = tolist(inputs)

            sequence_length = self._sequence_length
            if sequence_length is None or len(inputs_) > 1:
                sequence_length = len(inputs_)

            if len(inputs_) == 1:
                max_val = sequence_length
                if isinstance(sequence_length, (list, tuple)):
                    max_val = max(sequence_length)
                inputs_ = inputs_ * max_val

            sequence_length = tf.to_int32(sequence_length)
            if sequence_length.get_shape().ndims == 0:
                sequence_length = tf.expand_dims(sequence_length, 0)
                tf.tile(sequence_length, [batch_size])

            min_sequence_length = tf.reduce_min(sequence_length)
            max_sequence_length = tf.reduce_max(sequence_length)

            # sample an initial starting action by forwarding zeros
            # through the action
            output = tf.zeros((batch_size, self._output_size))

            for step, in_ in enumerate(inputs_):
                if step > 0:
                    varscope.reuse_variables()

                action = self._action_model(output)
                actions.append(action)

                input_ = self._input_model([action, in_, action])

                call_cell = lambda: self._cell(input_, state)
                (output, state) = _rnn_step(
                    time=step,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=self._cell.state_size)

                outputs.append(output)

            outputs_ = tf.pack(outputs)
            axes = [1, 0] + list(range(2, len(outputs_.get_shape())))  # (batch_size, sequence_length, input0, ...)
            outputs_ = tf.transpose(outputs_, axes)

            if self._return_sequences:
                return outputs_
            return outputs[-1]


def _get_defaults():
    return {
        'BasicRNNCell': OrderedDict([('num_units', 'required'), ('input_size', None), ('activation', 'tanh')]),
        'GRUCell': OrderedDict([('num_units', 'required'), ('input_size', None), ('activation', 'tanh')]),
        'BasicLSTMCell': OrderedDict([('num_units', 'required'), ('forget_bias', 1.0), ('input_size', None),
                                      ('state_is_tuple', True), ('activation', 'tanh')]),
        'LSTMCell': OrderedDict([('num_units', 'required'), ('input_size', None), ('use_peepholes', False),
                                 ('cell_clip', None), ('initializer', None), ('num_proj', None),
                                 ('proj_clip', None), ('num_unit_shards', 1), ('num_proj_shards', 1),
                                 ('forget_bias', 1.0), ('state_is_tuple', True), ('activation', 'tanh')]),
        'MultiRNNCell': OrderedDict([('cells', 'required'), ('state_is_tuple', True)]),
    }


def get_cell(name, kwargs=None):
    fn_dict = {
        'BasicRNNCell': tf.nn.rnn_cell.BasicRNNCell,
        'GRUCell': tf.nn.rnn_cell.GRUCell,
        'BasicLSTMCell': tf.nn.rnn_cell.BasicLSTMCell,
        'LSTMCell': tf.nn.rnn_cell.LSTMCell,
        'MultiRNNCell': tf.nn.rnn_cell.MultiRNNCell
    }

    return get_from_module(name, fn_dict, _get_defaults(), True, params=kwargs)


def process_parameters(name, kwargs):
    if isinstance(name, tf.nn.rnn_cell.RNNCell):
        name = name.__class__.__name__
    return process_params(name, kwargs, _get_defaults())
