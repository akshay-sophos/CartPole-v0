#! /usr/bin/python
import gym
import numpy as np

env = gym.make('CartPole-v0')
observation = env.reset()
e = 0.4     #epsilon for exploration
alpha = 0.01 #learning rate
gamma = 0.99 #discount factor
rew = 0 # average reward per episode
brew = 0 #best reward
avre=0#last 10k reqard
avre4=0
#('Max', '4.8::3.40282e+38::0.418879::3.40282e+38') 
#('Min', '-4.8::-3.40282e+38::-0.418879::-3.40282e+38')
cpb = np.linspace(-2.4, 2.4, 9)
cvb = np.linspace(-2, 2, 9) # (-inf, inf) 
pab = np.linspace(-0.4, 0.4,9)
pvb = np.linspace(-3.5, 3.5,9) # (-inf, inf) 
def disc(a,b): #to discretize
	return np.digitize(x=[a],bins=b)[0] #shows the index i such that b[i-1]<a<=b[i]
def make_state(observation):
	cart_position,cart_velocity,pole_angle,pole_velocity = observation
	s = [disc(cart_position,cpb),disc(cart_velocity,cvb),disc(pole_angle,pab),disc(pole_velocity,pvb)]
	return int("".join(map(lambda x: str(int(x)), s))) #converting array to integer

qtable = np.zeros((10**4,2)) # Q table of m x 2 matrix values
episode = 0
while episode < 10000:
	#print(observation) to understand the range of values for bins
	state = make_state(observation)
	if np.random.rand(1) >  e:
		action = qtable[state].argsort()[1]
	else:
		#e *= 0.95 #decrease the eploration rate
		#action = env.action_space.sample()
		action = qtable[state].argsort()[0]
	pst = state  #previous state
	act = action #previous action
	observation,reward,done,_ = env.step(action)
	if done:
		reward = -500
	state = make_state(observation)
	action = qtable[state].argsort()[-1]
	#print(qtable[pst,act])
	qtable[pst,act] += alpha*(reward + (gamma*(max(qtable[state])) - qtable[pst,act]))
	#print(qtable[pst,act])
	#print("done")
	if reward != -500:
		rew += reward
	if done:
		if episode %100 == 0:
				e = 1/(np.sqrt(episode+1))
		if episode >9000: #% 100 == 0:
			#e = 1/(np.sqrt(episode))
			#alpha += 0.0001
			print("Episode Average_Reward Epsilon Alpha",episode,rew,e,alpha)
		#env.render()
		if episode>1000 and episode<=2000:
			avre4 +=rew
		if episode>9000:
			avre +=rew
		episode+=1
		if(rew>brew):
			brew = rew
		rew = 0
		env.reset()
	#if rew>=200:
	#	print("Reached 200 after",episode,"episodes")
print("Best Reward",brew)
print("4th 1K Average_Reward",avre4/1000)
print("Last 1K Average_Reward",avre/1000)
env.close()
#for x in xrange(4443):
#	print(qtable[x])


