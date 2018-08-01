# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

import nets.TinyMobileFaceNet as TinyMobileFaceNet
import nets.MobileFaceNet as MobileFaceNet
import tensorflow as tf
import argparse
import cv2
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


slim = tf.contrib.slim


class mobilefacenet(object):
    def __init__(self):

        with tf.Graph().as_default():
            args = self.get_parser()

            # define placeholder
            self.inputs = tf.placeholder(name='img_inputs',
                                         shape=[None, args.image_size[0], args.image_size[1], 3],
                                         dtype=tf.float32)
            self.phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                                       shape=None,
                                                                       name='phase_train')
            # identity the input, for inference
            inputs = tf.identity(self.inputs, 'input')

            if args.model_type == 0:
                prelogits, net_points = MobileFaceNet.inference(images=inputs,
                                                                phase_train=self.phase_train_placeholder,
                                                                weight_decay=args.weight_decay)
            else:
                prelogits, net_points = TinyMobileFaceNet.inference(images=inputs,
                                                                    phase_train=self.phase_train_placeholder,
                                                                    weight_decay=args.weight_decay)
            self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

            # define sess
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping,
                                    gpu_options=gpu_options,
                                    )
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # saver to load pretrained model or save model
            # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
            saver = tf.train.Saver(tf.trainable_variables())

            # init all variables
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # load pretrained model
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
                saver.restore(self.sess, ckpt.model_checkpoint_path)

    def get_parser(self):
        parser = argparse.ArgumentParser(description='parameters to train net')
        parser.add_argument('--image_size', default=[112, 112], help='the image size')
        parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
        parser.add_argument('--pretrained_model', type=str, default='./output/ckpt_best/tinymobilefacenet_best_ckpt',
                            help='Load a pretrained model before training starts.')
        parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
        parser.add_argument('--model_type', default=1, help='MobileFaceNet or TinyMobileFaceNet')

        args = parser.parse_args()
        return args

    def get_feature(self, inputs):
        inputs = np.expand_dims(inputs, axis=0)
        feed_dict = {self.inputs: inputs, self.phase_train_placeholder: False}
        feature = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return feature


if __name__ == '__main__':
    t1 = time.time()
    model = mobilefacenet()
    t2 = time.time()
    img_num = 5000
    for _ in range(img_num):
        img = cv2.imread('./utils/test.jpg')
        feature = model.get_feature(img)
    t3 = time.time()
    print('init model cost {} s, get image feature speed {} fps'.format((t2 - t1), img_num / (t3 - t2)))
