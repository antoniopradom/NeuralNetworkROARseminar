import tensorflow as tf
import numpy as np

# this is the computation done with tensorflow
a = tf.range(1000)
b = tf.reduce_sum(a)
# this is the computation done with numpy
a_np = np.arange(1000)
b_np = np.sum(a_np)

print('The tensorflow sum from 0 to 999 is: %d \nusing numpy is: %d. \n I hope them being equal is %r' %
      (b, b_np, (b == b_np).numpy()))
