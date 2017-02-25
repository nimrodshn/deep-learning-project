import copy
import os
import datetime
import time
import argparse
from kerasNN import runNet, evalModelOnValAndTest
import tensorflow as tf
import numpy as np
from data.DataHandeling import DataSets
from keras import backend as K
import argparse

# Command Line Args Parser

parser = argparse.ArgumentParser(description='Arguments for running the segmentation model for Final Project')
parser.add_argument('--mode' ,type=str , help='flag for test / train modes')
parser.add_argument('--plot-activations' ,type=bool , help='flag for plotting activation functions')
parser.add_argument('--plot-weights' ,type=bool , help='flag for plotting weights')
parser.add_argument('--net-type', type=str, help='type of net to use for model. options to select from: "FCN", "ResNetFCN", "EncoderDecoder", "EncoderDecoderResNet", "DeeperResNetFCN"')
mode = parser.parse_args().mode
activations_flag = parser.parse_args().plot_activations
weights_flag = parser.parse_args().plot_weights
net_type = parser.parse_args().net_type
K.set_image_dim_ordering('tf')

"""
FLAGS - an easy way to share constants variables between functions
"""
flags = tf.app.flags
FLAGS = flags.FLAGS

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

if mode == 'train':
	runNet(net_type, activations_flag, weights_flag)
else:
	if (net_type == 'ResNetFCN'):
		best_model_folder = 'ResNetFCN2017-02-20_04-20-05'
	if (net_type == 'DeeperResNetFCN'):
		best_model_folder = 'DeeperResNetFCN2017-02-19_04-00-57'
	evalModelOnValAndTest(net_type, best_model_folder)
	#evalModelOnValAndTest('DeeperResNetFCN', 'DeeperResNetFCN2017-02-19_04-00-57')
