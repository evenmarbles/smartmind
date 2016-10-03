from __future__ import absolute_import

import six
import tensorflow as tf
from collections import OrderedDict

from .utils import get_from_module


class Optimizer(object):
    # Values for gate_gradients.
    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2

    @property
    def global_step(self):
        return self._global_step

    def __init__(self, optimizer, decay_steps=100, decay_rate=0.001, staircase=False,
                 min_learning_rate=None, kwargs=None):
        optimizers = {
            'sgd': tf.train.GradientDescentOptimizer,
            'adadelta': tf.train.AdadeltaOptimizer,
            'adagrad': tf.train.AdagradOptimizer,
            'momentum': tf.train.MomentumOptimizer,
            'adam': tf.train.AdamOptimizer,
            'ftrl': tf.train.FtrlOptimizer,
            'rmsprop': tf.train.RMSPropOptimizer,
            # 'adamx': Adamx,
            # 'nadam': Nadam,
        }
        learning_rate = kwargs.get('learning_rate', 0.1)

        self._global_step = tf.Variable(0., trainable=False)

        lr = tf.train.exponential_decay(learning_rate,
                                        self._global_step,
                                        decay_steps,
                                        decay_rate,
                                        staircase=staircase)
        if min_learning_rate is not None:
            min_rate = tf.convert_to_tensor(min_learning_rate, dtype=tf.float32)
            lr = tf.python.control_flow_ops.cond(lr < min_rate, lambda: min_rate, lambda: lr)

        kwargs['learning_rate'] = lr
        tf.scalar_summary('learning_rate', lr)

        self._op = get_from_module(optimizer, optimizers, _get_defaults(), True, params=kwargs)

    def get_name(self):
        return self._op._name

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        """Add operations to minimize `loss` by updating `var_list`.

        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before applying
        them call `compute_gradients()` and `apply_gradients()` explicitly instead
        of using this function.

        Parameters
        ----------
        loss: A `Tensor` containing the value to minimize.
        global_step: Optional `Variable` to increment by one after the
        variables have been updated.
        var_list: Optional list of `Variable` objects to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
        gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
        aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
        colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
        name: Optional name for the returned operation.
        grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

        Returns
        -------
        An Operation that updates the variables in `var_list`.  If `global_step`
        was not `None`, that operation also increments `global_step`.

        Raises
        ------
        ValueError: If some of the variables are not `Variable` objects.
        """
        return self._op.minimize(loss, global_step, var_list, gate_gradients,
                                 aggregation_method, colocate_gradients_with_ops,
                                 name, grad_loss)

    def compute_gradients(self, loss, var_list=None,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        """Compute gradients of `loss` for the variables in `var_list`.

        This is the first part of `minimize()`.  It returns a list
        of (gradient, variable) pairs where "gradient" is the gradient
        for "variable".  Note that "gradient" can be a `Tensor`, an
        `IndexedSlices`, or `None` if there is no gradient for the
        given variable.

        Parameters
        ----------
        loss: A Tensor containing the value to minimize.
        var_list: Optional list of tf.Variable to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
        gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
        aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
        colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
        grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

        Returns
        -------
        A list of (gradient, variable) pairs.

        Raises
        ------
        TypeError: If `var_list` contains anything else than `Variable` objects.
        ValueError: If some arguments are invalid.
        """
        return self._op.compute_gradients(loss, var_list, gate_gradients,
                                          aggregation_method, colocate_gradients_with_ops,
                                          grad_loss)

    def apply_gradients(self, grads_and_vars, name=None):
        """Apply gradients to variables.

        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.

        Parameters
        ----------
        grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
        variables have been updated.
        name: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.

        Returns
        -------
        An `Operation` that applies the specified gradients. If `global_step`
        was not None, that operation also increments `global_step`.

        Raises
        ------
        TypeError: If `grads_and_vars` is malformed.
        ValueError: If none of the variables have gradients.
        """
        return self._op.apply_gradients(grads_and_vars, self._global_step, name)

    def get_slot(self, var, name):
        """Return a slot named `name` created for `var` by the Optimizer.

        Some `Optimizer` subclasses use additional variables.  For example
        `Momentum` and `Adagrad` use variables to accumulate updates.  This method
        gives access to these `Variable` objects if for some reason you need them.

        Use `get_slot_names()` to get the list of slot names created by the
        `Optimizer`.

        Parameters
        ----------
        var: A variable passed to `minimize()` or `apply_gradients()`.
        name: A string.

        Returns
        -------
        The `Variable` for the slot if it was created, `None` otherwise.
        """
        return self._op.get_slot(var, name)

    def get_slot_names(self):
        """Return a list of the names of slots created by the `Optimizer`.

        See `get_slot()`.

        Returns
        -------
        A list of strings.
        """
        return self._op.get_slot_names()


def _get_defaults():
    return {
        'sgd': OrderedDict([('learning_rate', 0.01), ('use_locking', False), ('name', 'GradientDescent')]),
        'adadelta': OrderedDict(
            [('learning_rate', 0.001), ('rho', 0.95), ('epsilon', 1e-8), ('use_locking', False), ('name', 'Adadelta')]),
        'adagrad': OrderedDict(
            [('learning_rate', 0.01), ('initial_accumulator_value', 0.1), ('use_locking', False), ('name', 'Adagrad')]),
        'momentum': OrderedDict(
            [('learning_rate', 0.01), ('momentum', 0.0), ('use_locking', False), ('name', 'Momentum'),
             ('use_nesterov', False)]),
        'adam': OrderedDict(
            [('learning_rate', 0.001), ('beta1', 0.9), ('beta2', 0.999), ('epsilon', 1e-08), ('use_locking', False),
             ('name', 'Adam')]),
        'ftrl': OrderedDict([('learning_rate', 0.01), ('learning_rate_power', -0.5), ('initial_accumulator_value', 0.1),
                             ('l1_regularization_strength', 0.0), ('l2_regularization_strength', 0.0),
                             ('use_locking', False), ('name', 'Ftrl')]),
        'rmsprop': OrderedDict(
            [('learning_rate', 0.001), ('decay', 0.9), ('momentum', 0.0), ('epsilon', 1e-10), ('use_locking', False),
             ('name', 'RMSProp')]),
        # 'adamx': OrderedDict([('seed', None), ('dtype', 'float32')]),
        # 'nadam': OrderedDict([('seed', None), ('dtype', 'float32')]),
    }


def get(identifier, params=None):
    defaults = {'optimizer': OrderedDict(
        [('optimizer', 'sgd'), ('decay_steps', 100), ('decay_rate', 0.001), ('staircase', False),
         ('min_learning_rate', None), ('kwargs', None)])
    }
    if isinstance(identifier, six.string_types):
        if identifier not in _get_defaults():
            raise Exception('Invalid optimizer.')
        params = params if params is not None else {}
        params['optimizer'] = identifier
        return get_from_module('optimizer', {'optimizer': Optimizer}, defaults, True, params=params)
    elif isinstance(identifier, dict):
        name = identifier.pop('name')
        identifier['optimizer'] = name
        if identifier not in _get_defaults():
            raise Exception('Invalid optimizer.')
        return get_from_module('optimizer', {'optimizer': Optimizer}, defaults, True, params=identifier)
    elif isinstance(identifier, (list, tuple)):
        if identifier[0] not in _get_defaults():
            raise Exception('Invalid optimizer.')
        return get_from_module('optimizer', {'optimizer': Optimizer}, defaults, True, params=identifier)

    return identifier
