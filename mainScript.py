import copy
import os
import datetime
import time
import argparse
from kerasNN import runNet
import tensorflow as tf
import numpy as np
from data.DataHandeling import DataSets
from keras import backend as K

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

runNet()
