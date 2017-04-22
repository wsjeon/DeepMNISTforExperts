import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.slim import conv2d, dropout, max_pool2d, flatten

tf.set_random_seed(777)
np.random.seed(777)

flags = tf.app.flags

flags.DEFINE_integer('NUM_TOWERS', 16, 'number of towers')
flags.DEFINE_integer('TOT_GPUS', 8, 'the number of gpus available')
flags.DEFINE_string('PATH_TO_DATASET', '/home/wsjeon/webdav/datasets/MNIST', 'path to dataset')
flags.DEFINE_float('STDDEV', 0.1, 'standard deviation for weights and biases initialization')
flags.DEFINE_float('LR', 0.0005, 'learning rate')
flags.DEFINE_integer('BATCH_SIZE', 200, 'size of the minibatch')
flags.DEFINE_integer('TRAINING_STEP', 10000, 'training step')

FLAGS = flags.FLAGS

""" Load datasets """

Dataset = input_data.read_data_sets(FLAGS.PATH_TO_DATASET, one_hot = True)

""" Graph construction """

keep_prob = tf.placeholder_with_default(0.5, ()) # shared across networks
init = tf.random_normal_initializer(stddev = FLAGS.STDDEV) # shared across networks
x_ = tf.placeholder(tf.float32, [None, 784]) # shared across networks
y_ = tf.placeholder(tf.float32, [None, 10]) # shared across networks

def net(x_): # model from TF MNIST Tutorial
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
correct_predictions = []; accuracys = []

# for each tower
for i in range(FLAGS.NUM_TOWERS):
  with tf.device('/gpu:%d' % (i % FLAGS.TOT_GPUS)):
    with tf.variable_scope('network%d' % i):
      y, softmax_y = net(x_)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
      optimizer = tf.train.AdamOptimizer(FLAGS.LR).minimize(loss)
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      ys.append(y); softmax_ys.append(softmax_y); losses.append(loss); optimizers.append(optimizer)
      correct_predictions.append(correct_prediction); accuracys.append(accuracy)

# for ensemble model
sum_probability = softmax_ys[0]
for i in range(1, len(softmax_ys)):
  sum_probability += softmax_ys[i]
correct_prediction = tf.equal(tf.argmax(sum_probability, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(10000):
    batch_xs, batch_ys = Dataset.train.next_batch(FLAGS.BATCH_SIZE)
    accuracy_train = sess.run(accuracys + optimizers, feed_dict = {x_: batch_xs, y_: batch_ys})
    accuracy_train = np.array(accuracy_train[:FLAGS.NUM_TOWERS])
    if (step % 100) == 0:
      print '=' * 50
      print 'step: {}'.format(step)
      print 'training accuracy: {}'.format(accuracy_train[-1])
      print 'test accuracy: {}'.format(
          sess.run(accuracy, feed_dict =\
              {x_: Dataset.test.images, y_: Dataset.test.labels, keep_prob: 1.0}))
