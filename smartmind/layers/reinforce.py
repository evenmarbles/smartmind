from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# noinspection PyUnresolvedReferences
from six.moves import range

import tensorflow as tf

from ..framework import Layer
from ..utils import tf_in_train_phase


class Reinforce(Layer):
    """Abstract class for layers implementing the REINFORCE algorithm

    The reinforce (reward) method is called by a special Reward objective.

    Parameters
    ----------
    stochastic: bool
        If set to True, the layer is stochastic during evaluation and training.
        Else it only is stochastic during training.

    Ref A. http://incompleteideas.net/sutton/williams-92.pdf
    """

    @property
    def stochastic(self):
        return self._stochastic

    @stochastic.setter
    def stochastic(self, value):
        self._stochastic = value

    def __init__(self, stochastic=False, trainable=True, input_shape=None, input_dtype='float32',
                 batch_size=None, name=None):
        self.uses_training_phase = True

        self._stochastic = stochastic

        super(Reinforce, self).__init__(trainable, name, input_shape=input_shape,
                                        input_dtype=input_dtype, batch_size=batch_size)


class ReinforceNormal(Reinforce):
    def __init__(self, stddev=1.0, stochastic=False, trainable=True, input_shape=None,
                 input_dtype='float32', batch_size=None, name=None):
        self._stddev = stddev

        super(ReinforceNormal, self).__init__(stochastic, trainable, input_shape, input_dtype,
                                              batch_size, name)

    def _call(self, x):
        stochastic = tf_in_train_phase(tf.convert_to_tensor(True, dtype=tf.bool),
                                       tf.convert_to_tensor(self._stochastic, dtype=tf.bool))
        return tf.contrib.reinforce.reinforce_normal(x, self._stddev, stochastic)
