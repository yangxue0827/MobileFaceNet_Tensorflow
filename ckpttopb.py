# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

from utils.data_process import load_data
from nets.TinyMobileFaceNet import inference
from tensorflow.python.tools import freeze_graph
import tensorflow as tf
import argparse
import os

slim = tf.contrib.slim


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=85164, help='the train images number')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=1)
    parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    # parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--pretrained_model', type=str, default='./output/ckpt',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args = get_parser()

        # define placeholder
        inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None,
                                                              name='phase_train')

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        w_init_method = slim.initializers.xavier_initializer()
        prelogits, net_points = inference(inputs, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping,
                                gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # saver to load pretrained model or save model
        # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        print('Restoring pretrained model: %s' % args.pretrained_model)
        ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
        CKPT_PATH = ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)

        out_dir = './output/ckpt_best'
        save_name = 'mobilefacenet_model.pb'
        tf.train.write_graph(sess.graph_def, out_dir, save_name)
        freeze_graph.freeze_graph('./output/ckpt_best' + '/{}'.format(save_name),
                                  '', False, CKPT_PATH,
                                  'embeddings',
                                  'save/restore_all',
                                  'save/Const:0',
                                  out_dir + '/' + save_name+'frozen_model.pb',
                                  False, "")

        print('dene')





