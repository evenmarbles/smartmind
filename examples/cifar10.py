from __future__ import print_function

from smartmind.datasets.cifar10 import distorted_inputs
from smartmind.models import Sequential
from smartmind.layers import Conv2d
from smartmind.layers import Activation
from smartmind.layers import MaxPooling2D
from smartmind.layers import LRN
from smartmind.layers import FullyConnected
from smartmind.layers import Reshape


BATCH_SIZE = 128
NUM_CLASSES = 10

images, label = distorted_inputs(BATCH_SIZE)

model = Sequential()
model.add(Conv2d(64, [5, 5], [1, 1], init='truncated_normal', init_params={'stddev': 5e-2},
                 padding='SAME', bias_init=0.1, weight_decay=0.0, reader_input=images))
model.add(Activation('relu'))
model.add(MaxPooling2D([3, 3], [2, 2], padding='SAME'))
model.add(LRN(4, bias=1.0, alpha=0.001 / 9.0, beta=0.75))

model.add(Conv2d(64, [5, 5], [1, 1], init='truncated_normal', init_params={'stddev': 5e-2},
                 padding='SAME', weight_decay=0.0))
model.add(Activation('relu'))
model.add(LRN(4, bias=1.0, alpha=0.001 / 9.0, beta=0.75))
model.add(MaxPooling2D([3, 3], [2, 2], padding='SAME'))

model.add(Reshape((-1,)))
model.add(FullyConnected(384, init='truncated_normal', init_params={'stddev': 0.04},
                         bias_init=0.1, weight_decay=0.004))
model.add(Activation('relu'))

model.add(FullyConnected(192, init='truncated_normal', init_params={'stddev': 0.04},
                         bias_init=0.1, weight_decay=0.004))
model.add(Activation('relu'))

model.add(FullyConnected(NUM_CLASSES, init='truncated_normal', init_params={'stddev': 1./192.},
                         weight_decay=0.0))

model.compile(optimizer='sgd', loss='sparse_categorical_xentropy', loss_weights={'input': 1.})

