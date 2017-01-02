import tensorflow as tf
import numpy as np
import ops  # Ops is a file with operations. Currently only conv layer implementation
EPS = 1e-5

class CellSegmentation(object):
    """
    Cell segmentation model class
    """
    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None, regularization_weight=None, name=None):
        """
        :param input: data set images
        :param labels: data set labels
        :param dims_in: list input image size, for example [64,64,1] (W,H,C)
        :param dims_out: list output image size, for example [64,64,1] (W,H,C)
        :param regularization_weight: L2 Norm reg weight
        :param name: model name, used for summary writer sub-names (Must be unique!)
        """
        self.input = input
        self.labels = labels
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.regularization_weight = regularization_weight
        self.base_name = name

    def model(self, train_phase):
        """
        Define the model - The network architecture
        :param train_phase: tf.bool with True for train and False for test
        """
        # Reshape the input for batchSize, dims_in[0] X dims_in[1] image, dims_in[2] channels
        x_image = tf.reshape(self.input, [-1, self.dims_in[0], self.dims_in[1], self.dims_in[2]],
                             name='x_input_reshaped')
        # Dump input image
        tf.image_summary(self.get_name('x_input'), x_image)

        # Model convolutions
        d_out = 1
        conv_1, reg1 = ops.conv2d(x_image, output_dim=d_out, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_1")

        predict = conv_1
        
        reg = reg1 # reg2 + reg3 + reg4
        return predict, reg

    def loss(self, predict, reg=None):
        """
        Return loss value
        :param predict: prediction from the model
        :param reg: regularization
        :return:
        """
        labels_image = tf.reshape(tf.cast(self.labels, tf.float16), [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='y_input_reshape')
        tf.image_summary(self.get_name('Labels'), labels_image)

        # Reshape to flatten tensors
        predict_reshaped = tf.contrib.layers.flatten(predict)
        labels = tf.contrib.layers.flatten(self.labels)
        
        # You need to choose loss function
        loss = -999
        # loss = ?

        tf.scalar_summary(self.get_name('loss without regularization'), loss)

        if reg is not None:
            tf.scalar_summary(self.get_name('regulariztion'), reg)

            # Add the regularization term to the loss.
            loss += self.regularization_weight * reg
            tf.scalar_summary(self.get_name('loss+reg'), loss)
        
        return loss


    def training(self, s_loss, learning_rate):
        """
        :param s_loss:
        :param learning_rate:
        :return:
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(self.get_name(s_loss.op.name), s_loss)
        
        # Here you can change to any solver you want

        # Create Adam optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(s_loss, global_step=global_step)
        return train_op

    def evaluation(self, predict, labels):
        """
        Calcualte dice score
        :param predict: predict tensor
        :param labels: labels tensor
        :return: Dice score [0,1]
        """

        # Please do not change this function

        predict = tf.cast(tf.contrib.layers.flatten(predict > 0), tf.float32)
        labels = tf.contrib.layers.flatten(self.labels)
        
        # Calculate dice score
        intersection = tf.reduce_sum(predict * labels, keep_dims=True) + EPS
        union = tf.reduce_sum(predict, keep_dims=True) + tf.reduce_sum(labels, keep_dims=True) + EPS 
        dice = (2 * intersection) / union

        # Return value and write summary
        ret = dice[0,0]
        tf.scalar_summary(self.get_name("Evaluation"), ret)
        return ret

    def get_name(self, name):
        """
        Get full name with prefix name
        """
        return "%s_%s" % (self.base_name, name)