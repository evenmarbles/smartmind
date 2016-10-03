from __future__ import print_function

import os

from smartmind.datasets.cifar10 import distorted_inputs
from smartmind.models import Sequential
from smartmind.layers import Conv2d
from smartmind.layers import Activation
from smartmind.layers import MaxPooling2D
from smartmind.layers import LRN
from smartmind.layers import FullyConnected
from smartmind.layers import Reshape

from smartmind import optimizers

from smartmind.datasets.cifar10 import NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
# from smartmind.datasets.cifar10 import NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# run this on CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

BATCH_SIZE = 128
NUM_CLASSES = 10

images, labels = distorted_inputs(BATCH_SIZE)

model = Sequential()
model.add(Conv2d(64, [5, 5], [1, 1], init='truncated_normal', init_params={'stddev': 5e-2},
                 padding='SAME', reader_input=images))
model.add(Activation('relu'))
model.add(MaxPooling2D([3, 3], [2, 2], padding='SAME'))
model.add(LRN(4, bias=1.0, alpha=0.001 / 9.0, beta=0.75))

model.add(Conv2d(64, [5, 5], [1, 1], init='truncated_normal', init_params={'stddev': 5e-2},
                 padding='SAME'))
model.add(Activation('relu'))
model.add(LRN(4, bias=1.0, alpha=0.001 / 9.0, beta=0.75))
model.add(MaxPooling2D([3, 3], [2, 2], padding='SAME'))

model.add(Reshape((-1,)))
model.add(FullyConnected(384, init='truncated_normal', init_params={'stddev': 0.04},
                         initial_weights=[None, 0.1] , w_regularizer=['l2', 0.004]))
model.add(Activation('relu'))

model.add(FullyConnected(192, init='truncated_normal', init_params={'stddev': 0.04},
                         initial_weights=[None, 0.1] , w_regularizer=['l2', 0.004]))
model.add(Activation('relu'))

model.add(FullyConnected(NUM_CLASSES, init='truncated_normal', init_params={'stddev': 1./192.}))

num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

optimizer = optimizers.Optimizer('sgd', decay_steps, LEARNING_RATE_DECAY_FACTOR, kwargs=0.1)
model.compile(optimizer=optimizer, loss='sparse_categorical_xentropy', targets=labels)

model.fit_reader_op(batch_size=BATCH_SIZE, n_epoch=1)
