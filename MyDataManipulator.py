import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imread

def read_train_data():
    images_train = np.zeros([4815, 64, 64, 1])
    labels_train = np.zeros([4815, 64, 64, 1])

    raw_path_train ='data/Train/RAW/'
    seg_path_train ='data/Train/SEG/'

    k = 0

    for f in os.listdir(raw_path_train):
        img = imread(os.path.join(raw_path_train,f))
        label = imread(os.path.join(seg_path_train,f))
        images_train[k, :, :, 0] = img
        labels_train[k, :, :, 0] = label
        k = k + 1

    return  images_train, labels_train

def read_val_data():
    images_val = np.zeros([466, 64, 64, 1])
    labels_val = np.zeros([466, 64, 64, 1])

    raw_path_val = 'data/Val/RAW/'
    seg_path_val = 'data/Val/SEG/'

    k  = 0

    for f in os.listdir(raw_path_val):
        img = imread(os.path.join(raw_path_val,f))
        label = imread(os.path.join(seg_path_val,f))
        images_val[k, :, :, 0] = img
        labels_val[k, :, :, 0] = label
        k = k + 1

    return  images_val, labels_val

def read_test_data():
    images_test = np.zeros([478, 64, 64, 1])
    labels_test = np.zeros([478, 64, 64, 1])


    raw_path_test = 'data/Test/RAW/'
    seg_path_test = 'data/Test/SEG/'

    k = 0

    for f in os.listdir(raw_path_test):
        img = imread(os.path.join(raw_path_test,f))
        label = imread(os.path.join(seg_path_test,f))
        images_test[k, :, :, 0] = img
        labels_test[k, :, :, 0] = label
        k = k + 1


    return  images_test, labels_test

def plot_pairwise_train(train_set_numpy, train_label_set_numpy):

    for index in range(train_set_numpy.shape[0]):
        fig=plt.figure()
        img=train_set_numpy[index,:,:,0]
        a=fig.add_subplot(1,2,1)
        plt.imshow(img,interpolation='none') 
        img=train_label_set_numpy[index,:,:,0]
        a=fig.add_subplot(1,2,2)
        plt.imshow(img,cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
        plt.show()

def plot_pairwise_val(val_set_numpy, val_label_set):

    for index in range(val_set_numpy.shape[0]):
        fig=plt.figure()
        img=val_set_numpy[index,:,:,0]
        a=fig.add_subplot(1,2,1)
        plt.imshow(img,interpolation='none') 
        img=val_label_set_numpy[index,:,:,0]
        a=fig.add_subplot(1,2,2)
        plt.imshow(img,cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
        plt.show()
