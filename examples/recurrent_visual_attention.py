from smartmind.models import Sequential
from smartmind.layers import FullyConnected
from smartmind.layers import SpatialGlimpse
from smartmind.layers import Activation

locator_hidden_size = 128
glimpse_patch_size = 64
glimpse_depth = 3
glimpse_scale = 2


# create location sensor network
local_sensor = Sequential()
local_sensor.add(FullyConnected(locator_hidden_size, input_shape=(2,)))
local_sensor.add(Activation('relu'))

# # create glimpse sensor
# glimpse_sensor = Sequential()
# glimpse_sensor.add(SpatialGlimpse(glimpse_patch_size, glimpse_depth, glimpse_scale, input_shape=(512, 512, 3)))
#
# glimpse_sensor.compile(optimizer='sgd', loss='categorical_xentropy', loss_weights={'input': 1.})
