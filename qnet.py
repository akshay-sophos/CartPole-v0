import gym
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

#### Learning related constants ####
MIN_EXPLORE_RATE = 0.01 #The min exploration rate; The max is 1
PULL_UP_EXPLORE_LINE = 5 #Increase this to decrease the rate of decrease of epsilon

START_LEARNING_RATE = 1 #The max learning_rate
MIN_LEARNING_RATE = 0.01   #The min learning_rate
PULL_UP_LEARN_RATE = 1

START_DISCOUNT_FACTOR = 0 #The min discount_factor
MAX_DISCOUNT_FACTOR = 0.99  #The max discount_factor
PULL_UP_DISC_FACTOR = 1

TF_LEARN_RATE = 0.01 #Learning Rate for Gradient Descent

#### Defining the simulation related constants ####

#Defines the number of episodes it should perform the increment/decrement of values
NUM_EPISODES = 5000
NUM_EPISODES_PLATEAU_EXPLORE = 3000
NUM_EPISODES_PLATEAU_LEARNING = 2000
NUM_EPISODES_PLATEAU_DISCOUNT = 2000

STREAK_TO_END = 120
SOLVED_T = 199          # anything more than this returns Done = true for the openAI Gym

NEG_REW = -50 #negative reward for fallen pole
DISPLAY_RATES = True #To display the rates as a graph over time
DISPLAY_ENV = False  #To display the render for enviroment
if DISPLAY_ENV ==True:
    from time import sleep

# number of neurons in each layer
input_num_units = 5
hidden_num_units1 = 100
hidden_num_units2 = 100
output_num_units = 1

def pcom(s):
    print(s, end='', flush=True)

def get_explore_rate(t):
    maxValReached = math.log10(NUM_EPISODES_PLATEAU_EXPLORE)
    return max( min(1, 1.0 - math.log10((t+0.1)/PULL_UP_EXPLORE_LINE)/maxValReached), MIN_EXPLORE_RATE)
def get_learning_rate(t):
    maxValReached = math.log10(NUM_EPISODES_PLATEAU_LEARNING)
    return max(min(START_LEARNING_RATE, 1.0 - (math.log10(t+0.1)/PULL_UP_LEARN_RATE)/maxValReached), MIN_LEARNING_RATE)
def get_discount_factor(t):
    maxValReached = math.log10(NUM_EPISODES_PLATEAU_DISCOUNT)
    return min(max(START_DISCOUNT_FACTOR, (math.log10(t+0.1)/PULL_UP_DISC_FACTOR)/maxValReached), MAX_DISCOUNT_FACTOR)
    # return MAX_DISCOUNT_FACTOR


if DISPLAY_RATES:
    numPoints = 100;
    a = np.linspace(0,NUM_EPISODES,numPoints);
    e = np.zeros(numPoints);
    l = np.zeros(numPoints);
    d = np.zeros(numPoints);
    for i in range(numPoints):
        e[i]= get_explore_rate(a[i])
        l[i]= get_learning_rate(a[i])
        d[i]= get_discount_factor(a[i])

    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(a,e)
    axarr[0].set_title('Exploration Factor')

    axarr[1].plot(a,l)
    axarr[1].set_title('Learning Rate of Q function')

    axarr[2].plot(a,d)
    axarr[2].set_title('Discount factor for Q function')
    plt.show()

# define placeholders
tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
#y = tf.placeholder(tf.float32, [1, output_num_units],name="Output")
# tf_qval = tf.placeholder(tf.float32,[1,1],name="Q_value")
tf_exp_q =  tf.placeholder(tf.float32,[None,1],name="Expected_Q_value")

if 0:
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
    hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.tanh)
    hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
    output_layer = tf.layers.dense(hidden_layer2, output_num_units)
else:
    hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.tanh)
    hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
    output_layer = tf.layers.dense(hidden_layer2, output_num_units)

cost = tf.losses.mean_squared_error(tf_exp_q, output_layer)
optimizer = tf.train.AdamOptimizer(TF_LEARN_RATE)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=TF_LEARN_RATE)
train_op = optimizer.minimize(cost)


env = gym.make('CartPole-v0')
cost_plot = [0]
reward_plot = [0]
with tf.Session() as sess:
    # create initialized variables
    sess.run(tf.global_variables_initializer())
    ep = 0
    #for ep in range(NUM_EPISODES):
    while ep<=NUM_EPISODES:
        explore_rate = get_explore_rate(ep)
        learning_rate = get_learning_rate(ep)
        discount_factor = get_discount_factor(ep)
        observation = env.reset()
        if DISPLAY_ENV == True:
            env.render()
        tot_cost = 0
        tot_rew = 0
        # Qlearning is off-policy.
        # if max=True, return the (maxQ, bestAction)
        # if max = False, return the (bestQ, correspondingAction) based on explore_rate
        def Q(observation,max):
            #array returned, make scalar
            a0 = sess.run(output_layer,feed_dict={tf_x:(np.append(observation,0))[np.newaxis]})[0][0]
            a1 = sess.run(output_layer,feed_dict={tf_x:(np.append(observation,1))[np.newaxis]})[0][0]
            if(a0>a1):
                # print ('0000',a0, a1)
                maxA = 0
                maxQ = a0
            else:
                # print ('1111',a0, a1)
                maxA = 1
                maxQ = a1

            if (max ==True):
                return (maxQ, maxA)
            else:
                if(random.random()<explore_rate): # EXPLORE high explore rate => more exploration
                    if( random.randrange(2) ==0):      # a random action
                        return (a0,0)
                    else:
                        return (a1,1)
                else:                             # DONT EXPLORE
                    return (maxQ, maxA)


        for t in range(SOLVED_T):
            pobs = observation
            curQval,action = Q(pobs,False)
            # reward is 1 for all steps except those that are called after a done=True is returned
            # done is True when the pole has fell
            observation,reward,done,_ = env.step(action)
            if DISPLAY_ENV == True:
                env.render()
            # pcom(action)
            # print ("curQval ", curQval, " action ", action)
            nextMaxQval,_ = Q(observation, True)
            if done == True and tot_rew<199:
                reward = NEG_REW
            exp_qVal = (1-learning_rate)* curQval  + learning_rate*( reward + discount_factor*nextMaxQval )
            action_array = np.asarray(action).reshape([1,1])
            exp_qVal_array = np.asarray(exp_qVal).reshape([1,1])
            inpu = np.append(pobs,action_array)[np.newaxis]
            if t==0:
                I = inpu
                Z = exp_qVal_array
            else:
                I = np.vstack([I,inpu])
                Z = np.vstack([Z,exp_qVal_array])
            #_,c = sess.run([train_op,cost], {tf_x: np.append(pobs, action_array)[np.newaxis], tf_exp_q: exp_qVal_array})
            #c1 = sess.run([cost], {tf_x: np.append(pobs, action_array)[np.newaxis], tf_exp_q: exp_qVal_array})
            # print('c c1 ',c, ' ', c1)
            #tot_cost += c
            tot_rew +=reward
            if done == True:
                break
        _,c = sess.run([train_op,cost], {tf_x: I, tf_exp_q: Z})
        #if(tot_rew < NEG_REW+20 and ep >NUM_EPISODES/2):
            #ep =0
            #if(tot_cost >0.2500 and ep >NUM_EPISODES-1000):
                #ep =0
################################################################################
        def vizAngleAction():
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            theta_threshold_radians = 12 * 2 * math.pi / 360 * 1.5 # 1.5 is for safety margin
            theta_d_lim = 75
            x_threshold = 2.4 + 0.1

            x_grid =10
            theta_grid = 9
            x_d_grid = 8
            theta_d_grid = 8
            disp = 2        # we can only display a 3d graph. so disp selects which parameters to display , disp =2 doesn't work yet. FIX IT ?
            if disp == 1:
                x = np.linspace(-x_threshold, x_threshold, x_grid)
                x_d = 0
                theta = np.linspace(-theta_threshold_radians, theta_threshold_radians,theta_grid)
                theta_d = 0.1
            elif disp == 2:
                x = 0.01
                x_d = 0.0
                theta = np.linspace(-theta_threshold_radians, theta_threshold_radians,theta_grid)
                theta_d = np.linspace(-theta_d_lim, theta_d_lim, theta_d_grid)

            for j in range(2):      # 2 actions
                if disp==1:
                    # note the order in which this meshgrid (x,y, ..) is called. This is important when trying to reshape it to print in the surf function
                    # The reshape is the opposite order of meshgrid
                    x_m, theta_m, x_d_m, theta_d_m, a_m = np.meshgrid(x, theta, x_d, theta_d, j) # action i
                elif disp ==2:
                    theta_m, theta_d_m, x_m, x_d_m, a_m = np.meshgrid(theta, theta_d, x, x_d, j) # action i
                arr = np.append(x_m.flatten(), x_d_m.flatten())
                arr = np.append(arr, theta_m .flatten())
                arr = np.append(arr, theta_d_m.flatten())
                arr = np.append(arr, a_m.flatten())
                obvT = arr.reshape(input_num_units, int(arr.size/input_num_units) ).transpose() # returns a (x, input_num_units) matrix

                if j ==0:
                    qval0 = sess.run([output_layer], {tf_x:obvT}) # returns an array
                else:
                    qval1 = sess.run([output_layer], {tf_x:obvT}) # returns an array

            if disp == 1:
                axis1 = x_m.reshape(theta_grid, x_grid)
                axis2 = theta_m.reshape(theta_grid, x_grid)
                axis3 = np.subtract(qval1[0].reshape(theta_grid, x_grid), qval0[0].reshape(theta_grid, x_grid) )
            elif disp == 2:
                axis1 = theta_d_m.reshape(theta_d_grid,theta_grid)
                axis2 = theta_m.reshape(theta_d_grid,theta_grid)
                axis3 = np.subtract( qval1[0].reshape(theta_d_grid,theta_grid), qval0[0].reshape(theta_d_grid,theta_grid) )

                # qval = sess.run([output_layer], {tf_x:[x_m, theta_m, theta_d_m, x_d_m, a_m][:,np.newaxis]})
            surf = ax.plot_surface(axis1, axis2, axis3,cmap= (cm.coolwarm if (j==0) else cm.seismic), linewidth=0, antialiased=False)
                # cmap= cm.seismic, linewidth=0, antialiased=False)

            if disp == 1:
                ax.set_xlabel('x')
                ax.set_ylabel('theta')
            elif disp == 2:
                ax.set_xlabel('theta_d')
                ax.set_ylabel('theta')


            x = 'a'+str(int(ep/500))+'.png'
            plt.savefig('C:/Users/admin/Desktop/TFphoto/'+x)
            #plt.show()
################################################################################
        if(ep==NUM_EPISODES):#%500 ==0):
            if 1:
                vizAngleAction()

        if(ep%10 == 0):
            cost_plot = np.append(cost_plot,c)#tot_cost)
            reward_plot = np.append(reward_plot, tot_rew)
        print(ep, "T_Cost:%.4f" %c,  "T_Reward:%d" %tot_rew)
        ep = ep+1

# To plot Reward and Cost w.r.t time

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(cost_plot)
    axarr[0].set_title('cost_plot')
    axarr[0].set_ylim([0, 1])
    axarr[1].plot(reward_plot)
    axarr[1].set_title('reward_plot')

    plt.show()
    saver = tf.train.Saver()
    saver.save(sess, 'C:/Users/admin/Desktop/Sessio/model')
    print("\n Training Over")
