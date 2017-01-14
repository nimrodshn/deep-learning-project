""" 
Main script
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import copy
import os
import datetime
import time
import argparse
from cellSegmentation import CellSegmentation
from data.DataHandeling import DataSets

"""
FLAGS - an easy way to share constants variables between functions
"""
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 5000000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
flags.DEFINE_float('regularization_weight',5e-4, 'L2 Norm regularization weight.')
flags.DEFINE_integer('mini_batch_size', 10, 'Size of mini batch')
flags.DEFINE_integer('print_test', 1000, 'Print test frequency')
flags.DEFINE_integer('print_train', 100, 'Print train frequency')

# Please do not change those two flags
flags.DEFINE_string('train_dir',
                           './train_results/',
                           """Directory where to write event logs """
                           """and checkpoints.""")
flags.DEFINE_string('data_dir',
                           './data/',
                           """Directory of input data for the network """)

file_names = ['train', 'test', 'val']

DIMS_IN = (64, 64, 1)
DIMS_OUT = (64, 64, 1)
TEST_AMOUNT = 478

# File for stdout
logfile = open(os.path.join(FLAGS.train_dir, 'results_%s.log' % datetime.datetime.now()), 'w')


class Net(object):
    """
    Takes CellSeegmentation and create full network graph
    """
    # Init inputs
    def __init__(self, inputs, train_phase, name):
        """
        :param inputs: inputs from queue
        :param train_phase: Bool, true for training only
        :param name: prefix for summery writer names 
        """
        self.x_input = inputs[0]
        self.y_input = inputs[1]
        self.train_phase = tf.constant(train_phase, dtype=tf.bool)
        self.base_name = name
        
        # Get model
        self.network = CellSegmentation(input=self.x_input, labels=self.y_input, dims_in=np.array(DIMS_IN), dims_out=np.array(DIMS_OUT),
                                   regularization_weight=FLAGS.regularization_weight, name=self.base_name)

        # Connect nodes and create graph
        with tf.name_scope('model'):
            #self.model, self.reg = self.network.model(self.train_phase)
            self.model  = self.network.model(self.train_phase)

        with tf.name_scope('loss'):
            self.loss = self.network.loss(predict=self.model) # reg=self.reg

        with tf.name_scope('train'):
            # Train and update weights using the solver
            self.train_step = self.network.training(s_loss=self.loss, learning_rate=FLAGS.learning_rate)

        with tf.name_scope('evaluation'):
            # Evaluate performance
            self.evaluation = self.network.evaluation(predict=self.model, labels=self.y_input)


def run_evaluation(sess, eval_op, step, summary_op, writer):
    """
    Run evaluation and save checkpoint
    :param sess: tf session
    :param step: global step
    :param summary_op: summary operation
    :param eval_op: evaluate operation
    :param writer:
    :return:
    """
    result = sess.run([summary_op, eval_op])
    summary_str = result[0]
    acc = result[1]
    writer.add_summary(summary_str, step)
    print('Validation:  Time: %s , Evaluation at step %s: %s' % (datetime.datetime.now(), step, acc))
    logfile.writelines('Validation: Time: %s , Evaluation at step %s: %s\n' % (datetime.datetime.now(), step, acc))
    logfile.flush()


def save_checkpoint(sess, saver, step):
    """
    Dump checkpoint
    :param sess: tf session
    :param saver: saver op
    :param step: global step
    :return:
    """
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)


def train_model(mode, checkpoint=None):
    """
    Train the model
    If checkpoint exsits, resume from that point
    :param mode: "train", "resume"
    """    
    # Create DataSets handel
    with tf.name_scope('DataSets') as scope:
        data_sets = DataSets(filenames=file_names, base_folder=FLAGS.data_dir, image_size=DIMS_IN)
        data_set_train = data_sets.data['train'].get_batch(batch_size=FLAGS.mini_batch_size)
        data_set_val = data_sets.data['val'].get_batch(batch_size=FLAGS.mini_batch_size)
        data_set_test = data_sets.data['test'].get_batch(batch_size=1)

    # Init network graph
    with tf.name_scope('Net') as scope:
        net = Net(inputs=data_set_train, train_phase=True, name="train")
        tf.get_variable_scope().reuse_variables()
        net_val = Net(inputs=data_set_val, train_phase=False, name="val")
        tf.get_variable_scope().reuse_variables()
        net_test = Net(inputs=data_set_test, train_phase=False, name="test")

    # Create a saver and keep all checkpoints
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

    # Merge all the summaries and write them out to FLAGS.train_dir
    merged = tf.merge_all_summaries()

    # Init session, initialize all variables and create writer
    sess = tf.Session()

    init = tf.initialize_all_variables()
    writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    if mode == 'resume':
        # Load weights from checkpoint
        saver.restore(sess, checkpoint)
    else:
        # Initialize random weights
        sess.run(init)

    # Init queue runner (for data set readers)
    tf.train.start_queue_runners(sess)

    # Please do not remove this lines. It needed for the evaluate script
    tf.add_to_collection('net_eval', net_test.evaluation)
    tf.add_to_collection('net_predict', net_test.model)

    # main loop
    for i in range(FLAGS.max_steps):
        
        if (i % FLAGS.print_test == 0):
            # Display evaluation, write it into a log file and save checkpoint
            run_evaluation(sess, step=i, summary_op=merged, eval_op=net_val.evaluation, writer=writer)
            save_checkpoint(sess=sess, saver=saver, step=i)
        else:
            t_step, loss_value = sess.run([net.train_step, net.loss])
            if i % FLAGS.print_train == 0:
                print('TRAIN: Time: %s , Loss value at step %s: %s' % (datetime.datetime.now(), i, loss_value))
                logfile.writelines('TRAIN: Time: %s , Loss value at step %s: %s\n' % (datetime.datetime.now(), i, loss_value))
                logfile.flush()

    logfile.close()


def evaluate_checkpoint(checkpoint=None, output_file=None):
    """
    Evaluate checkpoint on Test data
    :param checkpoint: path to checkpoint
    :param output_file: If not None, the output will write to this path
    :return:
    """

    merged = tf.merge_all_summaries()

    # Create a saver and keep all checkpoints
    saver = tf.train.import_meta_graph('%s.meta' % checkpoint)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, checkpoint)
    net_evaluation = tf.get_collection('net_eval')
    net_predict = tf.get_collection('net_predict')

    tf.train.start_queue_runners(sess)

    all_acc = []
    predict_counter = 0
    if output_file is not None:
        f_out = open(output_file, 'w')
    print("Evaluate Model using checkpoint: %s, data=%s" % (checkpoint, "Test"))

    # Go over all data once
    while predict_counter < TEST_AMOUNT:

        predict, result = sess.run([net_predict, net_evaluation])
        
        # Save into list for averaging
        # 2.0 is a singular value 2 * (0 + EPS) / (0 + 0 + EPS) = 2
        res = np.array(result)
        if res != 2.0:
            all_acc.append(res)

        print('Time: %s , Performance evaluation for mini_batch is: %s' % (datetime.datetime.now(), res))
        if output_file is not None:
            f_out.write(np.array(predict).ravel())

        predict_counter += 1
        print("Done - " + str(predict_counter))

    if output_file is not None:
        f_out.close()
    print("Average performance is: %f" % np.array(all_acc).mean())


def main(args):

    if args.mode == 'train' or args.mode == 'resume':
        train_model(args.mode, args.checkpoint)
    elif args.mode == 'evaluate':
        evaluate_checkpoint(checkpoint=args.checkpoint, output_file=args.output_file)

if __name__ == '__main__':
    """
    Parse command line for main function.
    Please read about python argparser for more information
    In general, this code is resposible for parsing your inputs using shell command.
    :params mode: 'train' - train your model from scratch
                  'resume' - resume training from spesific checkpoint
                  'evaluate' - feed forward data into your model and evaluate performance, if 'output_file' is given, 
                   the network outputs will be dumped as binary files.
    """
    parser = argparse.ArgumentParser(description='Main script for train Cell segmentation')
    parser.add_argument('--mode', dest='mode', choices=['train', 'evaluate', 'resume'], type=str, help='mode')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, help='checkpoint full path')
    parser.add_argument('--output_file', dest='output_file', default=None, type=str, help='Output file for predict')
    args = parser.parse_args()

    if args.mode == 'evaluate':
        assert args.checkpoint, "Must have checkpoint for evaluate"
    elif args.mode == 'resume':
        assert args.checkpoint, "Must have checkpoint for resume"

    main(args)
