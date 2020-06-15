import gym
env = gym.make('FetchReach-v1')
for i_episode in range(1):
    observation2 = env.reset()
    print(observation2)
    for t in range(1000):
        observation1= observation2
        env.render()

        action = [-0.10, -0.20, -0.30,0.0]
        print(action)

        observation2, reward, done, info = env.step(action)
        print(observation2['observation'])
        #print(info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
