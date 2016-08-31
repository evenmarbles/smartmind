from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from ..framework import Layer


class Select(Layer):

    def __init__(self, index, name=None):
        super(Select, self).__init__(name=name)

        self._index = index
