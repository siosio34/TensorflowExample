import tensorflow as tf
import matplotlib.pyplot as plt


# Gradient descent algorithm 하는데 유용한 형태

x = [1., 2., 3.]
y = [1., 2., 3.]
samples_length = len(x)

# set model Weight
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.multiply(x, W)

# Cost function
cost = tf.reduce_sum(tf.pow(hypothesis-y, 2)) / samples_length

# Initializing the variables
init = tf.global_variables_initializer()

# For graphs
W_val = []
cost_val = []

# Launch the graph
sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
    print(i*0.1, sess.run(cost, feed_dict={W: i*0.1})) # feed_dict 얼마만큼 이동할것인가
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

# Graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()




