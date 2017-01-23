import tensorflow as tf
import numpy as np
import ops  # Ops is a file with operations. Currently only conv layer implementation
import pdb
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
        self.batch_labels = None

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
        #d_out = 1
        #conv_1, reg1 = ops.conv2d(x_image, output_dim=d_out, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_1")

        #conv_1_1, reg1_1 = ops.conv2d(x_image, 64, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv_1_1")
        #conv_1_2, reg1_2 = ops.conv2d(conv_1_1, 64, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv_1_2")
        predict, reg_predict = ops.conv2d(x_image, 1, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="predict") #we have a single segmentation map and not 21!!!
        predict = tf.squeeze(predict);
        reg = reg_predict
        print("model was called")
        predict =  tf.cond(train_phase , lambda:self.reshape_pred_labels_train(predict,self.labels), lambda:self.reshape_pred_labels_test(predict,self.labels))
        #pdb.set_trace()
        #predict, labels = reshape_pred_labels(predict,self.labels),

        #res = tf.select(train_phase,  labels)

        #if train_phase:
        #    predict, labels = reshape_pred_labels(predict, self.labels)
        #    self.batch_labels

        #loss = self.loss(predict, reg)
        #print(loss)
        # pool_1, pool_1_argmax = ops.pool_layer(conv_1_2)

        # conv_2_1, reg2_1 = ops.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
        # conv_2_2, reg2_2 = ops.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')
        # reg2 = reg2_2 + reg2_1
        # pool_2, pool_2_argmax = ops.pool_layer(conv_2_2)

        # conv_3_1, reg3_1 = ops.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
        # conv_3_2, reg3_2 = ops.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
        # conv_3_3, reg3_3 = ops.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')
        # reg3 = reg3_1 + reg3_2 + reg3_3
        # pool_3, pool_3_argmax = ops.pool_layer(conv_3_3)

        # conv_4_1, reg4_1 = ops.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
        # conv_4_2, reg4_2 = ops.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
        # conv_4_3, reg4_3 = ops.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')
        # reg4 = reg4_1 + reg4_2 + reg4_3
        # pool_4, pool_4_argmax = ops.pool_layer(conv_4_3)

        # conv_5_1, reg5_1 = ops.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
        # conv_5_2, reg5_2 = ops.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
        # conv_5_3, reg5_3 = ops.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')
        # reg5 = reg5_1 + reg5_2 + reg5_3
        # pool_5, pool_5_argmax = ops.pool_layer(conv_5_3)

        # fc_6, reg6 = ops.conv_layer(pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
        # fc_7, reg7 = ops.conv_layer(fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')

        # deconv_fc_6, reg_fc_6 = ops.deconv_layer(fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')
        # reg_fc = reg6 + reg7 + reg_fc_6
        # unpool_5 = ops.unpool_layer2x2(deconv_fc_6, pool_5_argmax, tf.shape(conv_5_3))

        # deconv_5_3, regd5_3 = ops.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
        # deconv_5_2, regd5_2 = ops.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
        # deconv_5_1, regd5_1 = ops.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')
        # regd5 = regd5_1 + regd5_2 + regd5_3
        # unpool_4 = ops.unpool_layer2x2(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))

        # deconv_4_3, regd4_3 = ops.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
        # deconv_4_2, regd4_2 = ops.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
        # deconv_4_1, regd4_1 = ops.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')
        # regd4 = regd4_1 + regd4_2 + regd4_3
        # unpool_3 = ops.unpool_layer2x2(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))

        # deconv_3_3, regd3_3 = ops.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
        # deconv_3_2, regd3_2 = ops.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
        # deconv_3_1, regd3_1 = ops.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')
        # regd3 = regd3_1 + regd3_2 + regd3_3
        # unpool_2 = ops.unpool_layer2x2(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))

        # deconv_2_2, regd2_2 = ops.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
        # deconv_2_1, regd2_1 = ops.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')
        # regd2 = regd2_1 + regd2_2
        # unpool_1 = ops.unpool_layer2x2(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))

        # deconv_1_2, regd1_2 = ops.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
        # deconv_1_1, regd1_1 = ops.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')
        # regd1 = regd1_1 + regd1_2

        # predict, reg_predict = ops.deconv_layer(deconv_1_1, [1, 1, 1, 32], 1, 'predict') #we have a single segmentation map and not 21!!!

        # reg = reg_predict + regd1 + regd2 + regd3 + regd4 + regd5 + reg_fc + reg5 + reg4 + reg3 + reg2 + reg1



        return predict, reg

    def reshape_pred_labels_train(self,predict,labels):
        # Reshape to flatten tensors
        print('train phase')
        predict = tf.reshape(predict, [10, -1, 1]) # flatten the predicted arr
        self.batch_labels = tf.reshape(self.labels, [10,-1]) # flatten the labels / SEG arr
        self.batch_labels = tf.to_int32(self.batch_labels, name='ToInt32')
        #predict = tf.to_int32(predict_reshaped, name='ToInt32')

        # DEBUG
        print("labels shape:" , self.batch_labels.get_shape().as_list())
        print("pred shape", predict.get_shape().as_list())

        return predict

    def reshape_pred_labels_test(self, predict, labels):
        print('not train phase')
        predict = tf.reshape(predict, [ -1, 1]) # flatten the predicted arr
        self.batch_labels = tf.reshape(self.labels, [-1]) # flatten the labels / SEG arr
        self.batch_labels = tf.to_int32(self.batch_labels, name='ToInt32')
        #predict = tf.to_int32(predict_reshaped, name='ToInt32')
        # DEBUG
        print("labels shape:" , self.batch_labels.get_shape().as_list())
        print("pred shape", predict.get_shape().as_list())
        return predict

    def loss(self, predict, reg=None):
        """
        Return loss value
        :param predict: prediction from the model
        :param reg: regularization
        :return:
        """
        labels_image = tf.reshape(tf.cast(self.labels, tf.float16), [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='y_input_reshape')
        tf.image_summary(self.get_name('Labels'), labels_image)

        # You need to choose loss function
        # Cross Entropy Loss Function = computes the cross entropy between the actoual labels and the predicted map
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(predict, self.batch_labels, name='x_entropy')
        loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

        tf.scalar_summary(self.get_name('loss without regularization'), loss)

        if reg is not None:
            tf.scalar_summary(self.get_name('regulariztion'), reg)

            # Add the regularization term to the loss.
            loss += self.regularization_weight * reg # lambda *( l2 norm of weight + l2 norm of bias )
            tf.scalar_summary(self.get_name('loss+reg'), loss)
        #debug
        return loss


    def training(self, s_loss, learning_rate):
        """
        :param s_loss:
        :param learning_rate:
        :return:
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary('loss', s_loss)

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
