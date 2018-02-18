#! usr/bin/python
import gym
import numpy as np

env = gym.make('CartPole-v0')	#Creating the environment
observation = env.reset()
episode = 0
rew = 0	#avg reward
bestrew = 0#best avg reward
bestweight = np.random.rand(4)*2 - 1
#bestweight = [-0.16384956,  1.00060299,  1.03389299,  0.97398544]
alpha = 0.1
while episode < 200:
	for i in range(200):	#aim is avg reward 195.0
		env.render()
		done = False
		rndweight = (1-alpha)*bestweight+((np.random.rand(4)*2 - 1))
		x = np.dot(rndweight,observation)
		if x>0:
			action=1
		else:
			action=0
		observation,reward,done,_ = env.step(action)
		rew += reward
		if done:
			episode+=1
			print('Average Reward ',rew)
			observation = env.reset()
	
			break
	if(episode>80):
		if(rew<60):
			#alpha = 0.999 
			bestweight=np.random.rand(4)*2 - 1
		elif(rew > 100 and rew < 200):
			alpha = 0.001
	if(rew>=200):
	 	print('Total episodes = ',episode)
	 	bestrew = rew
		bestweight = rndweight
	 	break
	if rew>bestrew:
		bestrew = rew
		bestweight = rndweight
	rew = 0
print('Best Reward',bestrew)
print('Best Weight',bestweight)
print('The End')
#108 [ 0.03999007,  0.85283089, -0.1495635 ,  0.33778807]
#200 [-0.16384956,  1.00060299,  1.03389299,  0.97398544]