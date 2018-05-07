import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z)
#plt.show()


cost_plot = [0]
input_num_units = 2
hidden_num_units1 = 3
hidden_num_units2 = 2
output_num_units = 1
tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
tf_exp_q =  tf.placeholder(tf.float32,[1,1],name="Expected_Q_value")
hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.relu)
hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
output_layer = tf.layers.dense(hidden_layer2, output_num_units)
cost = tf.losses.mean_squared_error(tf_exp_q, output_layer)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	ep = 0
	while ep<=40:
		co = 0
		for t in range(800):
			x = np.random.random()
			y = np.random.random()
			qw = (np.append(x,y))
			mn = np.sin(np.sqrt(x**2 + y**2))
			_,c = sess.run([train_op,cost],{tf_x:qw.reshape(1,2),tf_exp_q:mn.reshape(1,1)})
			co += c
		print("cost",c," Ep",ep)
		ep = ep+1
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	#for a in range(10):
		# X = np.random.random()
		# Y = np.random.random()
		# ll = np.append(X,Y)
		# Z = sess.run(output_layer,{tf_x:ll.reshape(1,2)})

		#X,Y = np.meshgrid(X, Y)
		#Z= (0*X+0*Y+4)
		#X,Y = np.meshgrid(range(10), range(10))

	x = np.arange(-5,5,0.25)
	y = np.arange(-5,5,0.25)
	x_m,y_m = np.meshgrid(x,y)

	arr = np.append(x_m.flatten(),y_m.flatten())
	obvT = arr.reshape(2,int(arr.size/2)).transpose()
	z = sess.run(output_layer,{tf_x: obvT})
	X = x_m.reshape(40,40)
	Y = y_m.reshape(40,40)
	Z = z.reshape(40,40)


	surf = ax.plot_surface(X,Y,Z,linewidth=0, antialiased=False)
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	plt.show()
