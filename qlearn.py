#! usr/bin/python
import gym
import numpy as np
from numpy.random import choice

env = gym.make('CartPole-v0')	#Creating the environment
observation = env.reset()
e = 0.2 #epsilon for exploration
gamma = 0.1
rew = 0
s = observation
q[[s]+[0]]=0
def Q(s,a):
	Q(s,a) += r+(max({Q(env.step(0)[0],0),Q(s,1)})*gamma)
	return Q(s,a)

def main():
	episode = 0
	while episode < 1000:
		env.render()
		action = 0
		if not q.has_key([observation]+[0])
			q[[observation]+[0]]=0

		if Q(observation,action)> Q(observation,action+1):
			action=0
		else:
			action=1
		observation,reward,done,_ = env.step(action)
		state  = {observation,reward}
		rew += reward
		# draw = choice(list_of_candidates, number_of_items_to_pick, p=probability_distribution)

		if done:
			episode+=1
			print("Average Reward",rew)
			env.reset()
			break
		if rew>=200:
			print("Reached 200 after",episode,"episodes")
			break
		else:
			print("Best Reward",rew)


# if abs(random.rand(1) )>  e:
# 	do explore
# else:
# 	bellman update