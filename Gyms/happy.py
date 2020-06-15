import gym
env = gym.make('FetchSlide-v1')
print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)
#> Discrete(2)
print(env.observation_space)


#> Box(4,)