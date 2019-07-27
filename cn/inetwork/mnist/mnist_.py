import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b) #激活函数
cross = - tf.reduce_sum(y_ * tf.log(y)) #成本函数
step = tf.train.AdamOptimizer(0.001).minimize(cross) #优化器

rigth = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

persent = tf.reduce_mean(tf.cast(rigth, 'float'))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    step.run(feed_dict={x:batch[0], y_:batch[1]}, session=sess)
    if i % 100 == 0:
        print('step:%s' % str(i))
        # print('w%s,b%s' % (W.eval(session=sess), b.eval(session=sess)))
        # print(sess.run(rigth, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print(sess.run(persent, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


