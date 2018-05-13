import tensorflow as tf

from numpy import random
from glob import glob
from custom_models import VGG19
from ImageLoader import *
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]
HPF_risks = []

class COX_model_with_VGG(object, VGG19):
    
    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [256, 256, 1]
        assert green.get_shape().as_list()[1:] == [256, 256, 1]
        assert blue.get_shape().as_list()[1:] == [256, 256, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [256, 256, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
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
    
    def get_Neg_Likelihood(self, HPF, mode):
        
        risk = self.build(HPF) 
        loglikelihood = tf.Variable(tf.zeros([1]), tf.float32)
        for i in range(14):
            loglikelihood = loglikelihood + (tf.gather_nd(risk, [i, 0]) - tf.log(tf.reduce_sum(tf.exp(risk), 0)))
        loglikelihood = -1.0*loglikelihood
        return loglikelihood

if __name__=="__main__":
    # IMAGE SELECTION
    batch_input = []
    num_images = 14
    seed_num = 5
    count = 1
    
    while len(batch_input) < num_images:
        batch = select_batch_images(seed_num*3, num_images)
        batch_input = batch + batch_input
        batch_input = list(set(batch_input))
        del batch
        print(len(batch_input))
        print(batch_input)
        count = count + 1
        if count == 19:
            break
    
    resized_img1 = load_image(batch_input[0], 1)
    resized_img1 = resized_img1.reshape((1, 256, 256, 4))
    for img_num in range(1,len(batch_input)):
        resized_img = load_image(batch_input[img_num], img_num)
        resized_img = resized_img.reshape((1, 256, 256, 4))
        resized_img1 = np.concatenate((resized_img1, resized_img), 0)
    ######################################################################
    SCNN = COX_model_with_VGG()
    HPF = tf.placeholder(tf.float32, shape=(14, 256, 256, 4))
    is_training = tf.placeholder(tf.bool, name='MODE')
    
    # CONVOLUTIONAL NEURAL NETWORK MODEL
    # DEFINE LOSS
    with tf.name_scope("LOSS"):
        loss = SCNN.get_Neg_Likelihood(HPF, is_training)
    
    # DEFINE ACCURACY
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # DEFINE OPTIMIZER
    with tf.name_scope("ADAGRAD"):
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
        1e-3,               # LEARNING_RATE
        batch * batch_size, # GLOBAL_STEP
        train_size,         # DECAY_STEP
        4e-4,               # DECAY_RATE
        staircase=True)     # LR = LEARNING_RATE*DECAY_RATE^(GLOBAL_STEP/DECAY_STEP)
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss,global_step=batch)
    
    
    # SUMMARIES For TensorBoard
    saver = tf.train.Saver()
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', accuracy)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())
    print ("MODEL DEFINED.")
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={is_training: True, HPF: resized_img1})

