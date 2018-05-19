import tensorflow as tf

from numpy import random
from glob import glob
from custom_models import VGG19
from ImageLoader import *
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68, 102.32]
HPF_risks = []

class COX_model_with_VGG(object, VGG19):
    
    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param rgba: rgba image [batch, height, width, 4]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue, alpha = tf.split(axis=3, num_or_size_splits=4, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [256, 256, 1]
        assert green.get_shape().as_list()[1:] == [256, 256, 1]
        assert blue.get_shape().as_list()[1:] == [256, 256, 1]
        assert alpha.get_shape().as_list()[1:] == [256, 256, 1]
        bgr = tf.concat(axis=4, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
            alpha - VGG_MEAN[3]
        ])
        assert bgr.get_shape().as_list()[1:] == [256, 256, 4]

        self.conv1_1 = self.conv_layer(bgr, 4, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 32768, 1000, "fc6")  
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 1000, 1000, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 1000, 256, "fc8")
        self.risk = self.Cox_layer(self.fc8, 256, 1, "Cox")

        return self.risk
    
    
    def Cox_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            Betas, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            Risk = tf.matmul(x, Betas)

            return Risk
    
    

if __name__=="__main__":
    input_Tensor, SubSets = random_image_tensor()
    is_training = tf.placeholder(tf.bool, name='MODE')
    x = tf.placeholder(tf.float32, shape=(14, 256, 256, 4))
    y = []
    for i in range(0,13):
        individual_x = tf.slice(x, [i, 0, 0, 0], [1, 256, 256, 4])
        individual_y = tf.Variable(tf.zeros([1]), tf.float32)
        print("x = ", x.shape)
        print("sliced x = ", individual_x.shape)
        
    
    individual_y = COX_model_with_VGG()
    y.append(individual_y)
    
    
    # CONVOLUTIONAL NEURAL NETWORK MODEL
    # DEFINE LOSS
    with tf.name_scope("LOSS"):
        likelihood = tf.Variable(tf.zeros([1]), tf.float32)
        for i in range(0,13):
            SetofAtRisk = tf.Variable(tf.zeros([1]), tf.float32)
            if len(SubSets[i]) > 0:
                for j in SubSets[i]:
                    SetofAtRisk = tf.add(SetofAtRisk, tf.exp(fourteenD[j]))
                likelihood = likelihood + y[i] - tf.log(SetofAtRisk)
            else:
                continue
        likelihood = -1.0*likelihood
    

    
    # DEFINE OPTIMIZER
    with tf.name_scope("ADAGRAD"):
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
        1e-3,               # LEARNING_RATE
        batch * batch_size, # GLOBAL_STEP
        train_size,         # DECAY_STEP
        4e-4,               # DECAY_RATE
        staircase=True)     # LR = LEARNING_RATE*DECAY_RATE^(GLOBAL_STEP/DECAY_STEP)
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(likelihood,global_step=batch)
    
    
    # SUMMARIES For TensorBoard
    saver = tf.train.Saver()
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss', loss)    
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())
    print ("MODEL DEFINED.")
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={is_training: True, x: input_Tensor})

