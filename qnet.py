#!/usr/bin/env python
import gym
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
observation = env.reset()
#cart_position,cart_velocity,pole_angle,pole_velocity = observation
e = 0.4     #epsilon for exploration
learn_rate = 0.00003 #learning rate
gamma = 0.99 #discount factor
epoch = 500
epoch_disp = 100#how much epoch to be rendered
seed = 1
neg_rew = -5 #negative reward for loosing
cost_plot = [0]

# number of neurons in each layer
input_num_units = 5
hidden_num_units1 = 3
hidden_num_units2 = 3
output_num_units = 1

# define placeholders
x = tf.placeholder(tf.float32, [1, input_num_units],name="Input")
y = tf.placeholder(tf.float32, [1, output_num_units],name="Output")
Q_value = tf.placeholder(tf.float32,[1,1],name="Q_value")
exp_q =  tf.placeholder(tf.float32,[1,1],name="Expected_Q_value")
weights = {
    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units1, hidden_num_units2], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units2, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units2], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_layer1 = tf.nn.tanh(hidden_layer1)
hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])
hidden_layer2 = tf.nn.leaky_relu(hidden_layer2,alpha=0.2)
output_layer = tf.matmul(hidden_layer2, weights['output']) + biases['output']

cost = tf.square(output_layer-exp_q)
#optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    e = e*0.999
    def Q(observation,act,max):
        a0 = sess.run(output_layer,feed_dict={x:(np.append(observation,0))[np.newaxis]})
        a1 = sess.run(output_layer,feed_dict={x:(np.append(observation,1))[np.newaxis]})
        r = random.randint(1,1001)
        if (max ==0):
            if(a0>a1 and r >= e*1000) or (a0<a1 and r< e*1000):
                act[0] =0
                return a0
            elif (a0>a1 and r < e*1000) or (a0<a1 and r >= e*1000):
                act[0] =1
                return a1
        else:
            if a0>a1:
                return a0
            else:
                return a1
    for ep in range(epoch):
        tot_cost = 0
        tot_rew = 0
        for a in range(200):
            pobs = observation
            act = [0]
            Q_value = Q(observation,act,0)
            action = act[0]
            observation,reward,done,_ = env.step(action)
            if reward != 1:
                reward = neg_rew
            _, c = sess.run([optimizer, cost], feed_dict = {x: np.append(pobs,action)[np.newaxis], exp_q : (reward + (gamma*Q(observation,act,1)))})
            tot_cost += c
            tot_rew +=reward

            #if(ep>epoch-epoch_disp):
                #env.render()
            if done == True:
                env.reset()
                break
        if(epoch%10 ==0):
            cost_plot = np.append(cost_plot,tot_cost/200)
            #print (cost_plot[int(epoch/10)])

            #rew_plot[int(epoch/10)] = tot_rew
        print("Total Cost",tot_cost/200," Total Reward",tot_rew)
    plt.plot(cost_plot)
    plt.show()
    #plt.plot(rew_plot,'go')
    print("\n Training Over")
