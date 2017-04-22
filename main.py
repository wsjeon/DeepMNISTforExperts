import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.slim import conv2d, dropout, max_pool2d, flatten

tf.random_seed(777)
np.random.seed(777)

flags = tf.app.flags

flags.DEFINE_integer(NUM_TOWERS, 1, 'number of towers')
flags.DEFINE_string(PATH_TO_DATASET, '/home/wsjeon/webdav/datasets/MNIST', 'path to dataset')
flags.DEFINE_float(STDDEV, 10, 'standard deviation for weights and biases initialization')
flags.DEFINE_float(LR, 0.001, 'learning rate')
flags.DEFINE_integer(BATCH_SIZE, 50, 'size of the minibatch')

FLAGS = flags.FLAGS

""" Load datasets """

Dataset = input_data.read_data_sets(FLAGS.PATH_TO_DATASET, one_hot = True)
Dataset.train.labels

""" Graph construction """

keep_prob = tf.placeholder_with_default(0.5, ()) # shared across networks
init = tf.random_normal_initializer(stddev = FLAGS.STDDEV) # shared across networks
x_ = tf.placeholder(tf.float32, [None, 784]) # shared across networks
y_ = tf.placeholder(tf.float32, [None, 10]) # shared across networks

def net(x_):
  with slim.arg_scope([max_pool2d], kernel_size = [2, 2], padding = 'SAME'):
    with slim.arg_scope([conv2d, fc], weights_initializer = init, biases_initializer = init):
      net = tf.reshape(x_, [-1, 28, 28, 1])
      net = conv2d(net, 32, [5, 5], scope = 'conv0')
      net = max_pool2d(net, scope = 'max_pool0')
      net = conv2d(net, 64, [5, 5], scope = 'conv1')
      net = max_pool2d(net, scope = 'max_pool1')
      net = flatten(net)
      net = fc(net, 1024)
      net = dropout(net, keep_prob = keep_prob)
      net = fc(net, 10, activation_fn = None)
    return net, tf.nn.softmax(net)

ys = []; softmax_ys = []; losses = []; optimizers = []

for i in range(FLAGS.NUM_TOWERS):
  with tf.variable_scope('network%d' % i):
    y, softmax_y = net(x_)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logiths(logits = y, labels = y_)
    optimizer = tf.train.AdamOptimizer(FLAGS.LR).minimize(loss)
    ys.append(y); softmax_ys.append(softmax_y); losses.append(loss); optimizers.append(optimizer)

""" Summary """
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(10):
    for step in range(max_steps):
      batch_xs, batch_ys = Dataset.train.next_batch(50)
      sess.run(optimizer, feed_dict = {x_: batch_xs, y_: batch_ys})
      if (step % 100) == 0:
        print(step, sess.run(accuracy, feed_dict = \
            {x_: Dataset.test.images, y_: Dataset.test.labels, keep_prob: 1.0}))
