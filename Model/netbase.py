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






class net(object):
    def __init__(self):
        trainable_collection=[]
    def loss(self):

        raise NotImplementedError
    def inference(self):

        raise NotImplementedError

    def upsampling(x, block_name, uprate=2, outshape=None, method='bicubic'):
        '''

        :param block_name: The name of operation
        :param uprate: The rate of Upsampling
        :param outshape: The shape of output
        :param method: The method of  interpolation.
        :return: A fearture after upsampling
        '''
        with tf.name_scope(block_name):
            methods = {'bicubic': tf.image.resize_bicubic, 'bilinear': tf.image.resize_bilinear}
            # bilinear
            if uprate:
                x = methods[method](x, size=(x.shape[1] * uprate, x.shape[2] * uprate), align_corners=True,
                                    name=block_name + 'upsampling')
            else:
                if not isinstance(outshape, tuple) or not len(outshape) == 2:
                    raise TypeError('The output shape must be define right')
                x = methods[method](x, size=outshape, align_corners=True, name=block_name)
                return x

    def maxpool(x, block_name, poolsize=2, stride=2):
        with tf.name_scope(block_name):
            x = max_pool2d(x, pool_size=(poolsize, poolsize), strides=(stride, stride), padding='same',
                           name=block_name + 'pooling')
            return x

    def deconv_bn(x, feature_channel, kernel_size, block_name, stride=(1, 1), padding='same', dilation_rate=(1, 1),
                  activation=None):
        with tf.name_scope(block_name):
            x = deconv2d(x, feature_channel, kernel_size, stride=stride, padding=padding, dilation_rate=dilation_rate,
                         activation=activation, name=block_name + 'dconv')
            x = batch_normalization(x, name=block_name + 'bn')
            return x

    def conv_bn(x, feature_channel, kernel_size, block_name, stride=(1, 1), padding='same', dilation_rate=(1, 1),
                activation=None):
        with tf.name_scope(block_name):
            x = conv2d(x, feature_channel, kernel_size, stride=stride, padding=padding, dilation_rate=dilation_rate,
                       activation=activation, name=block_name + 'conv')
            x = batch_normalization(x, name=block_name + 'bn')
            return x

    def leaky_relu(self, x, alpha=0.1, dtype=tf.float32):
        """leaky relu
        if x > 0:
          return x
        else:
          return alpha * x
        Args:
          x : Tensor
          alpha: float
        Return:
          y : Tensor
        """
        x = tf.cast(x, dtype=dtype)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + alpha * (1 - mask) * x
