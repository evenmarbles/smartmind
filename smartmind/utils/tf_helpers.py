from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import tensorflow as tf

from . import reset_uids


__all__ = ['tf_get_session',
           'tf_set_session',
           'tf_clear_session',
           'tf_function',
           'tf_training_phase',
           'tf_set_training_phase',
           'tf_in_train_phase',
           'tf_in_test_phase']


_SESSION = None
_TRAINING_PHASE = tf.placeholder(dtype='uint8', name='sm_training_phase')  # 0 = test, 1 = train


def tf_set_session(session):
    """Sets the global TF session."""
    global _SESSION
    _SESSION = session


def tf_clear_session():
    global _SESSION
    global _TRAINING_PHASE
    tf.reset_default_graph()
    reset_uids()
    _SESSION = None
    _TRAINING_PHASE = tf.placeholder(dtype='uint8', name='sm_training_phase')


def tf_get_session():
    """Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global SmartMind session.

    If no global SmartMind session exists at this point:
    we will create a new global session.

    Note that you can manually set the global session
    via `set_session(sess)`.
    """
    global _SESSION
    if tf.get_default_session() is not None:
        return tf.get_default_session()
    if _SESSION is None:
        if not os.environ.get('OMP_NUM_THREADS'):
            _SESSION = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        else:
            nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
            _SESSION = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                                        allow_soft_placement=True))
    return _SESSION


def tf_training_phase():
    """Returns the training phase flag.

    The training phase flag is an integer tensor (0 = test, 1 = train)
    to be passed as input to any SmartMind function
    that uses a different behavior at train time and test time.
    """
    return _TRAINING_PHASE


def tf_set_training_phase(value):
    global _TRAINING_PHASE
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    _TRAINING_PHASE = value


def tf_in_train_phase(x, alt):
    """Selects `x` in train phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    """
    if _TRAINING_PHASE is 1:
        return x
    elif _TRAINING_PHASE is 0:
        return alt
    # else: assume learning phase is a placeholder.
    x_shape = copy.copy(x.get_shape())
    x = tf.python.control_flow_ops.cond(tf.cast(_TRAINING_PHASE, 'bool'),
                                        lambda: x,
                                        lambda: alt)
    x.set_shape(x_shape)
    return x


def tf_in_test_phase(x, alt):
    """Selects `x` in test phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    """
    if _TRAINING_PHASE is 1:
        return alt
    elif _TRAINING_PHASE is 0:
        return x
    x_shape = copy.copy(x.get_shape())
    x = tf.python.control_flow_ops.cond(tf.cast(_TRAINING_PHASE, 'bool'),
                                        lambda: alt,
                                        lambda: x)
    x.set_shape(x_shape)
    return x


# GRAPH MANIPULATION

class Function(object):
    def __init__(self, inputs, outputs, ops=None, updates=None):
        assert isinstance(inputs, (list, tuple)), 'Input to a TensorFlow function should be a list or tuple.'
        assert isinstance(outputs, (list, tuple)), 'Output to a TensorFlow function should be a list or tuple.'
        assert isinstance(ops, (list, tuple)), 'Ops in a TensorFlow function should be a list or tuple.'
        assert isinstance(updates, (list, tuple)), 'Updates in a TensorFlow function should be a list or tuple.'

        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.ops = list(ops) if ops is not None else []
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if type(update) is tuple:
                    p, new_p = update
                    updates_ops.append(tf.assign(p, new_p))
                else:
                    # assumed already an op
                    updates_ops.append(update)
            self.updates_op = tf.group(*updates_ops)

    def __call__(self, inputs=None):
        assert isinstance(inputs, (list, tuple, type(None)))
        if inputs is None:
            inputs = []
        names = [getattr(v, 'name', None) for v in self.inputs]
        feed_dict = dict(zip(names, inputs))
        session = tf_get_session()
        updated = session.run(self.ops + self.outputs + [self.updates_op], feed_dict=feed_dict)
        return updated[len(self.ops):len(self.outputs)]

tf_function = Function
