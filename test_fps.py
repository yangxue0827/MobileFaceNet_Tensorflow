# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

from tensorflow.python.tools import freeze_graph
from utils.data_process import load_data
from nets.TinyMobileFaceNet import inference
from verification import evaluate
from scipy.optimize import brentq
from scipy import interpolate
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os

slim = tf.contrib.slim


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=10, help='epoch to train the network')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=85164, help='the train images number')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.', default=[1, 4, 6, 8])
    parser.add_argument('--train_batch_size', default=90, help='batch size to train network')
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
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', default='./output/ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='./output/ckpt',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=5e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    # prepare validate datasets
    ver_list = []
    ver_name_list = []
    # for db in args.eval_datasets:
    for db in ['lfw']:
        print('begin db %s convert.' % db)
        data_set = load_data(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)

    total_accuracy = {}
    output_graph_path = "./output/ckpt_best/mobilefacenet_model.pbfrozen_model.pb"
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        print('testing...')

        # output_graph_def = tf.GraphDef()
        # with open(output_graph_path, "rb") as f:
        #     output_graph_def.ParseFromString(f.read())
        #     _ = tf.import_graph_def(output_graph_def, name="")

        with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")

        for ver_step in range(len(ver_list)):
            start_time = time.time()
            data_sets, issame_list = ver_list[ver_step]
            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
            nrof_batches = data_sets.shape[0] // args.test_batch_size
            for index in range(nrof_batches):  # actual is same multiply 2, test data total
                start_index = index * args.test_batch_size
                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                img_inputs = sess.graph.get_tensor_by_name('img_inputs:0')
                phase_train = sess.graph.get_tensor_by_name('phase_train:0')
                feed_dict = {img_inputs: data_sets[start_index:end_index, ...], phase_train: False}
                embeddings = sess.graph.get_tensor_by_name('embeddings:0')
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)
            duration = time.time() - start_time

            print("total time %.3f to evaluate %d images of %s" % (duration,
                                                                   data_sets.shape[0],
                                                                   ver_name_list[ver_step]))
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f\n' % eer)

            coord.request_stop()
            coord.join(threads)