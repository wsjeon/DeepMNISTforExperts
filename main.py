import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.slim import conv2d, dropout, max_pool2d, flatten

""" Utils """
def np_one_hot(indices, depth):
  one_hot = np.zeros((len(indices), depth))
  one_hot[np.arange(len(indices)), indices] = 1
  return one_hot

""" Load datasets """
Dataset = input_data.read_data_sets('/home/wsjeon/webdav/datasets/MNIST')

""" Graph """
x_ = tf.placeholder(tf.float32, [None, 784], name = 'x_')

def net(x_):
  keep_prob = tf.placeholder_with_default(0.5, ())
  with slim.arg_scope([max_pool2d], kernel_size = [2, 2], padding = 'SAME'):
    net = tf.reshape(x_, [-1, 28, 28, 1])
    net = conv2d(net, 32, [5, 5], scope = 'conv0')
    net = max_pool2d(net, scope = 'max_pool0')
    net = conv2d(net, 64, [5, 5], scope = 'conv1')
    net = max_pool2d(net, scope = 'max_pool1')
    net = flatten(net)
    net = fc(net, 1024)
    net = dropout(net, keep_prob = keep_prob)
    net = fc(net, 10, activation_fn = None)
  return keep_prob, net, tf.nn.softmax(net)

keep_prob, y, softmax_y = net(x_)

""" Loss """
y_ = tf.placeholder(tf.float32, [None, 10], name = 'y_')
loss =\
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))

""" Optimizer """
optimizer = tf.train.AdamOptimizer(0.0001)
optimizer = optimizer.minimize(loss)

""" Summary """
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

max_steps = 1000
for epoch in range(10):
  for step in range(max_steps):
    batch_xs, batch_ys = Dataset.train.next_batch(50)
    sess.run(optimizer, feed_dict = {x_: batch_xs, y_: np_one_hot(batch_ys, 10)})
    if (step % 100) == 0:
      print(step, sess.run(accuracy, feed_dict = \
          {x_: Dataset.test.images, y_: np_one_hot(Dataset.test.labels, 10), keep_prob: 1.0}))

sess.close()
