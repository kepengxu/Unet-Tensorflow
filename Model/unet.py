import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.convolutional import deconv2d
from tensorflow.python.layers.pooling import max_pool2d
from tensorflow.python.layers.pooling import average_pooling2d
from tensorflow.python.layers.normalization import batch_normalization
from tensorflow.python.layers.core import flatten
from tensorflow.python.layers.core import dense
from tensorflow.python.layers.core import dropout
from tensorflow.python.layers.convolutional import separable_conv2d

from netbase import *


class unet(object):
    def __init__(self):
        pass
    #TODO