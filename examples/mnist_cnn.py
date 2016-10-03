from __future__ import print_function

from smartmind.datasets import mnist

from smartmind.models import Sequential
from smartmind.layers import Conv2d
from smartmind.layers import FullyConnected
from smartmind.layers import Activation
from smartmind.layers import SpatialGlimpse
from smartmind.layers import Reshape
from smartmind.layers import Merge

batch_size = 128
nb_epoch = 12
nb_classes = mnist.NUM_CLASSES

nb_filters = 32  # number of convolutional filters
nb_conv = 3  # convolution kernel size

dataset = mnist.load_data()
input_shape = dataset.train.shape

model1 = Sequential()
model1.add(Conv2d(nb_filters, [nb_conv, nb_conv], input_shape=input_shape))
model1.add(Activation('relu'))

model2 = Sequential()
model2.add(Conv2d(nb_filters, [nb_conv, nb_conv], input_shape=input_shape))
model2.add(Activation('relu'))
