# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

from utils.data_process import parse_function, load_data, next_batch
from tensorflow.core.protobuf import config_pb2
from nets.TinyMobileFaceNet import inference
from losses.face_losses import cos_loss
from verification import evaluate
from scipy.optimize import brentq
from utils.common import train
from scipy import interpolate
from datetime import datetime
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os
import re

gpu_group = '0,1,3'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group

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
                        help='Number of images to process in a batch in the test set.', default=100)
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
    parser.add_argument('--saver_maxkeep', default=20, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='',
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


def tower_loss(scope, images, labels, phase_train_placeholder, args):

    logits, net_points = inference(images, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)

    embeddings = tf.nn.l2_normalize(logits, 1, 1e-10, name='embeddings')

    # Norm for the prelogits
    eps = 1e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(logits) + eps, ord=1.0, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * 5e-5)

    inference_loss, logit = cos_loss(logits, labels, args.num_output)

    # calculate accuracy
    pred = tf.nn.softmax(logit)
    correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
    accuracy_op = tf.reduce_mean(correct_prediction)

    tf.add_to_collection('losses', inference_loss)

    # total losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    # tf.summary.scalar('regularization_losses', regularization_losses)
    # tf.summary.scalar('inference_loss', inference_loss)
    # tf.summary.scalar('total_loss', total_loss)

    with tf.device('/cpu:0'):
        for l in losses + [total_loss]:
            loss_name = re.sub('tower_[0-9]*/', '', l.op.name)
            tf.summary.scalar(loss_name, l)
    return total_loss, embeddings, accuracy_op


def average_gradients(tower_grads):
    average_grads = []
    for val_and_grad in zip(*tower_grads):
        grads = []
        for g,_ in val_and_grad:
            grad = tf.expand_dims(g, 0)
            grads.append(grad)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = val_and_grad[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        args = get_parser()

        # create log dir
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
        # define placeholder
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None,
                                                              name='phase_train')

        img_batch, label_batch = next_batch(batch_size=args.train_batch_size,
                                            pattern=os.path.join(args.tfrecords_file_path, 'tran.tfrecords'))

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [img_batch, label_batch], capacity=2 * len(gpu_group.split(',')))

        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)

        # define the learning rate schedule
        learning_rate = tf.train.piecewise_constant(epoch,
                                                    boundaries=args.lr_schedule,
                                                    values=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                                                    name='lr_schedule')

        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping,
                                gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        if args.optimizer == 'ADAGRAD':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif args.optimizer == 'ADADELTA':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif args.optimizer == 'ADAM':
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif args.optimizer == 'RMSPROP':
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif args.optimizer == 'MOM':
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        # train op
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(len(gpu_group.split(','))):
                with tf.device('/gpu:%s' % gpu_group.split(',')[i]):
                    with tf.name_scope('tower_%d' % i) as scope:
                        image_batch, label_batch = batch_queue.dequeue()
                        total_loss, embeddings, accuracy_op = tower_loss(scope, image_batch, label_batch,
                                                                         phase_train_placeholder, args)
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)

        if len(tower_grads) > 1:
            grads = average_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        update_ops = tf.get_collection((tf.GraphKeys.UPDATE_OPS))
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')

        # saver to load pretrained model or save model
        # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        # summary
        summary_op = tf.summary.merge(summaries)
        summary = tf.summary.FileWriter(args.summary_path, sess.graph)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # output file path
        if not os.path.exists(args.log_file_path):
            os.makedirs(args.log_file_path)
        if not os.path.exists(args.ckpt_best_path):
            os.makedirs(args.ckpt_best_path)

        total_accuracy = {}
        _ = sess.run(inc_epoch_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for i in range(args.max_epoch):
            # sess.run(iterator.initializer)
            count = 0
            while True:
                try:
                    images_train, labels_train = sess.run([img_batch, label_batch])

                    feed_dict = {phase_train_placeholder: True}
                    start = time.time()

                    _, total_loss_val, _, acc_val = \
                        sess.run([train_op, total_loss, inc_global_step_op,
                                  accuracy_op],
                                 feed_dict=feed_dict)
                    end = time.time()
                    pre_sec = args.train_batch_size * len(gpu_group.split(',')) / (end - start)

                    count += 1
                    # print training information
                    if count > 0 and count % args.show_info_interval == 0:
                        print('epoch %d, total_step %d, total loss is %.2f , '
                              'training accuracy is %.6f, time %.3f samples/sec' %
                              (i, count, total_loss_val, acc_val, pre_sec))

                    # save summary
                    if count > 0 and count % args.summary_interval == 0:
                        feed_dict = {phase_train_placeholder: True}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # save ckpt files
                    if count > 0 and count % args.ckpt_interval == 0:
                        filename = 'MobileFaceNet_iter_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)

                    if count > 0 and args.train_batch_size * count * len(gpu_group.split(',')) > 3804846:
                        break

                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break
        coord.request_stop()
        coord.join(threads)