import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../mnist/MNIST_data', one_hot=True)

x = tf.placeholder('float', [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder('float', [None, 10])

cross = -tf.reduce_sum(y_ * tf.log(y))

step = tf.train.GradientDescentOptimizer(0.01).minimize(cross)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(201):
    xs, ys = mnist.train.next_batch(100)
    sess.run(step, feed_dict={x: xs, y_: ys})
    if i % 20 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print(acc)
