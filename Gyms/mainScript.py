import gym
import random
import sys
import numpy as np

sys.path.append('..')
import vae_forward_model

env = gym.make('FetchReach-v1')
print(env.observation_space)
vae_forward_model.initialize()

def euclidean(goal, current):
    dist = (goal[0] - current[0])**2 + (goal[1] - current[1])**2 + (goal[2] - current[2])**2
    return dist

for i_episode in range(1000):
    observation2 = env.reset()
    goal = observation2['desired_goal']
    print(observation2)
    for t in range(1000):
        observation1= observation2
        env.render()
        a1 = random.uniform(-1.0,1.0)
        a2 = random.uniform(-1.0, 1.0)
        a3 = random.uniform(-1.0, 1.0)

        current = observation1['observation'][:3]


        dist = euclidean(goal,current)

        action = [a1, a2, a3,0.0]

        currentMod = current
        otherMod = observation1['observation'][5:]
        actionMod = action[:3]
        #print(actionMod)
        #some how this goal mod needs to be split
        goalMod = np.append(goal, dist)


        observation2, reward, done, info = env.step(action)


        currentMod2 = observation2['observation'][:3]
        dist = euclidean(goal, currentMod2)
        otherMod2 = observation2['observation'][5:]
        goalMod2 = np.append(goal, dist)

        input = np.concatenate((currentMod,currentMod2,otherMod,otherMod2,goalMod,goalMod2,actionMod))

        vae_forward_model.train_robot(input_data = input,vae_mode=True, vae_mode_modalities=True, epoch=i_episode)
        #print(observation2['observation'])
        #print(info)


        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


