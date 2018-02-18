#! usr/bin/python
import gym
import numpy as np

env = gym.make('CartPole-v0')									#Creating the environment
print('Possible action spaces are:',env.action_space)			#Possible action spaces
print('Cart position  Cart velocity  Pole Angle  Pole Velocity')
print('Max','::'.join(map(str, env.observation_space.high)))
print('Min','::'.join(map(str, env.observation_space.low)))

observation = env.reset()
episode = 0
rew = 0	#avg reward
bestrew = 0#best avg reward
bestweight = [0, 0, 0 ,0]
while episode < 1000:
	for i in range(200):	#aim is avg reward 195.0
		env.render()
		done = False
		rndweight = np.random.rand(4)*2 - 1
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
	if(rew>=195):
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