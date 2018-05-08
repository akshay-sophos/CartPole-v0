import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def hello():
	fig = plt.figure("Actual")
	ax = fig.gca(projection='3d')
	# Make data.
	X = np.arange(-5, 5, 0.25)
	Y = np.arange(-5, 5, 0.25)
	X, Y = np.meshgrid(X, Y)
	Z = np.sin(np.sqrt(X**2 + Y**2))
	surf = ax.plot_surface(X, Y, Z)
	#plt.savefig('C:/Users/admin/Desktop/3Dphoto/actual.png')
	plt.show()


cost_plot = []
seed = 1
input_num_units = 2
hidden_num_units1 = 55
hidden_num_units2 = 55
output_num_units = 1
tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
tf_exp_q =  tf.placeholder(tf.float32,[None,1],name="Expected_Q_value")
# weights = {
#     'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units1], seed=seed)),
#     'hidden2': tf.Variable(tf.random_normal([hidden_num_units1, hidden_num_units2], seed=seed)),
#     'output': tf.Variable(tf.random_normal([hidden_num_units2, output_num_units], seed=seed))
# }
#
# biases = {
#     'hidden1': tf.Variable(tf.random_normal([hidden_num_units1], seed=seed)),
#     'hidden2': tf.Variable(tf.random_normal([hidden_num_units2], seed=seed)),
#     'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
# }
#
# hidden_layer1 = tf.add(tf.matmul(tf_x, weights['hidden1']), biases['hidden1'])
# hidden_layer1 = tf.nn.relu(hidden_layer1)
# hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])
# hidden_layer2 = tf.nn.leaky_relu(hidden_layer2,alpha=0.2)
# output_layer = tf.matmul(hidden_layer2, weights['output']) + biases['output']
hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.relu)
hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
output_layer = tf.layers.dense(hidden_layer2, output_num_units)
cost = tf.losses.mean_squared_error(tf_exp_q, output_layer)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	ep = 0
	final = 50000
	row = 20
	while ep<=final:
		co = 0
		for t in range(row):
			x = random.uniform(-5,5)
			y = random.uniform(-5,5)
			qw = (np.append(x,y))
			mn = np.sin(np.sqrt(x**2 + y**2))
			if t == 0:
				I =qw
				Z = mn
			if t!=0:
				I = np.vstack([I,qw])
				Z = np.vstack([Z,mn])
		_,c = sess.run([train_op,cost],{tf_x:I,tf_exp_q:Z})#.reshape(row,1)})
			#co += c
		#print(I,I.shape,"hey",Z)
		cost_plot = np.append(cost_plot,c)#o)
		print("cost",c," Ep",ep)
		def vizAngleAction():
			fig = plt.figure(ep)
			ax = fig.gca(projection='3d')
			x = np.arange(-5,5,0.25)
			y = np.arange(-5,5,0.25)
			x_m,y_m = np.meshgrid(x,y)
			arr = np.append(x_m.flatten(),y_m.flatten())
			obvT = arr.reshape(2,int(arr.size/2)).transpose()
			z = sess.run(output_layer,{tf_x: obvT})
			X = x_m.reshape(40,40)
			Y = y_m.reshape(40,40)
			Z = z.reshape(40,40)
			surf = ax.plot_surface(X,Y,Z)
			ax.set_xlabel('X axis')
			ax.set_ylabel('Y axis')
		if ep == final:
			vizAngleAction()
			x = 'a'+str(int(ep))+'.png'
			plt.savefig('C:/Users/admin/Desktop/3Dphoto/'+x)
			plt.show()
		if ep == final:
			fig = plt.figure("Cost")
			plt.plot( cost_plot,'ro')
			plt.savefig('C:/Users/admin/Desktop/3Dphoto/cost.png')
		ep = ep+1
