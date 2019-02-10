import tensorflow as tf
#import tensorflow.contrib.slim as slim

from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.convolutional import deconv2d
from tensorflow.python.layers.pooling import max_pool2d
from tensorflow.python.layers.pooling import average_pooling2d
from tensorflow.python.layers.normalization import batch_normalization
from tensorflow.python.layers.core import flatten
from tensorflow.python.layers.core import dense
from tensorflow.python.layers.core import dropout
from tensorflow.python.layers.convolutional import separable_conv2d

from netbase import net

import numpy as np

class unet(net):
    def __init__(self,common_params,net_params,test=False):
        super(net, self).__init__(common_params,net_params)
        self.image_size=int(common_params['image_size'])
        self.num_classes=int(common_params['num_classes'])
        self.down_number=int(net_params['down_number'])
        self.batch_size=int(common_params['batch_size'])
        self.weight_deacy=float(net_params['weight_deacy'])
        self.channel_base=int(net_params['channnel_base'])
        self.summary_path=str(common_params['summary_path'])

        if not test:
            pass
            # TODO
            # define The loss function param






    def inference(self,Images):
         '''
         Build the Unet model
         :param Images: 4D tensor [batchsize,h,w,channels]
         :return: 4D tensor    [batchsize,h,w,num_classes]
         '''
         x=self.conv_bn(Images, self.channel_base * 1, block_name='Init-1-conv')
         x = self.conv_bn(x, self.channel_base * 2, block_name='Init-2-conv')

         d1 = self.conv_bn(Images, self.channel_base * 1, block_name='down-1-conv',stride=(2,2))
         d2 = self.conv_bn(d1, self.channel_base * 2, block_name='down-2-conv',stride=(2,2))
         d3 = self.conv_bn(d2, self.channel_base * 4, block_name='down-3-conv',stride=(2,2))
         d4 = self.conv_bn(d3, self.channel_base * 8, block_name='down-4-conv',stride=(2,2))
         d5 = self.conv_bn(d4, self.channel_base * 16, block_name='down-5-conv',stride=(2,2))

         med=self.conv_bn(d5,self.channel_base * 16,block_name='med-conv')

         u4 = self.deconv_bn(med, self.channel_base * 8, block_name='up-4-deconv')
         u4 = tf.concat([u4, d4], -1, name='concat-4')

         u3 = self.deconv_bn(u4, self.channel_base * 4, block_name='up-3-deconv')
         u3 = tf.concat([u3, d3], -1, name='concat-3')

         u2 = self.deconv_bn(u3, self.channel_base * 2, block_name='up-2-deconv')
         u2 = tf.concat([u2, d2], -1, name='concat-2')

         u1 = self.deconv_bn(u2, self.channel_base * 1, block_name='up-1-deconv')
         u1 = tf.concat([u1, d1], -1, name='concat-1')

         x=self.conv_bn(u1,self.num_classes,block_name='out',activation='softmax')
         return x


    def _get_cost(self, y, logits,cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1, self.num_classes])
            flat_labels = tf.reshape(y, [-1, self.num_classes])
            if cost_name == "cross_entropy":
                class_weights = cost_kwargs.pop("class_weights", None)

                if class_weights is not None:
                    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                    weight_map = tf.multiply(flat_labels, class_weights)
                    weight_map = tf.reduce_sum(weight_map, axis=1)

                    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                          labels=flat_labels)
                    weighted_loss = tf.multiply(loss_map, weight_map)

                    loss = tf.reduce_mean(weighted_loss)

                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                     labels=flat_labels))
            elif cost_name == "dice_coefficient":
                eps = 1e-5
                #prediction = pixel_wise_softmax(logits)
                prediction=logits
                intersection = tf.reduce_sum(prediction * y)
                union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
                loss = -(2 * intersection / (union))

            else:
                raise ValueError("Unknown cost function: " % cost_name)

            regularizer = cost_kwargs.pop("regularizer", None)
            if regularizer is not None:
                regularizers = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
                loss += regularizers

            return loss

    def Acc(self,y,pre):
        correct=tf.equal(tf.arg_max(y,3),tf.argmax(pre,3))
        acc=tf.reduce_mean(tf.cast(correct,tf.float32),name='acc')
        return acc
    def mIou(self,y,pre):
        score,upop=tf.metrics.mean_iou(tf.argmax(y,3),tf.argmax(pre,3),self.num_classes)
        return score

