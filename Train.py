
import tensorflow as tf
import re
import sys
import time

class Trainer(object):
    '''

    '''

    def __init__(self,net,common_params,Train_params,DataGenerator,DataDir,opt_kwargs={}):
        self.net=net
        self.batch_size=    int(common_params['batch_size'])
        self.h=             int(common_params['image_size'])
        self.w=             int(common_params['image_size'])
        self.norm_grads=        common_params['norm_grads']
        self.optimizer=     str(common_params['optimizer'])
        self.num_classes =  int(common_params['num_classes'])
        self.lr=            int(common_params['lr'])
        #self.pretrain_path= str(Train_params['pretrain_path'])
        self.train_dir=     str(Train_params['train_dir'])
        self.max_iterators=  int(Train_params['max_iterators'])
        self.input_channel=3
        self.TrainGenerator=DataGenerator(batch_size=self.batch_size,Dir=DataDir,K='Train',size=self.h)
        self.ValGenerator=DataGenerator(batch_size=self.batch_size,Dir=DataDir,K='Val',size=self.h)
        self.decay_step=100
    def construct_graph(self):
        self.global_step=tf.Variable(0,trainable=False)
        self.images=tf.placeholder(tf.float32,(self.batch_size,self.h,self.w,self.input_channel))
        self.labels=tf.placeholder(tf.float32,(self.batch_size,self.h,self.w,self.num_classes))
        self.pre=self.net.inference(self.images)
        self.loss=self.net._get_cost(y=self.labels,logits=self.pre,cost_name='cross_entropy')
        self.acc=self.net.Acc(y=self.labels,pre=self.pre)
        self.mIou=self.net.mIou(self.labels,self.pre)
        tf.summary.scalar('Total_loss',self.loss)
        tf.summary.scalar('Acc',self.acc)
        tf.summary.scalar('mIou',self.mIou)

    def get_opt(self,global_step,decay_step):
        self.lr_node = tf.train.polynomial_decay(learning_rate=self.lr,
                                                 global_step=global_step,
                                                 decay_steps=decay_step,
                                                 end_learning_rate=0.0001,
                                                 power=0.5,
                                                 cycle=True,
                                                 name='polynom_decay')

        opt = tf.train.GradientDescentOptimizer(self.lr_node).minimize(self.loss)
        return opt

    def solve(self):

        init=tf.global_variables_initializer()
        summary_op=tf.summary.merge_all()
        sess=tf.Session()
        sess.run(init)

        summary_writer=tf.summary.FileWriter(self.train_dir,sess.graph)

        for step in range(self.max_iterators):
            start_time=time.time()
            Images,Labels=next(self.TrainGenerator)
            #opt=self.GetOptimizer(decay_step=100,global_step=self.max_iterators)
            #_,acc,loss,Iou=sess.run([opt,self.acc,self.loss,])








