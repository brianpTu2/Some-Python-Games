import gym
import tflearn as tf
import numpy as np

#  YES
# -----
# env = gym.make('CartPole-v0')
# env = gym.make('Acrobot-v1')
# env = gym.make('MountainCar-v0')
# env = gym.make('MountainCarContinuous-v0')
env = gym.make('Pendulum-v0')
# env = gym.make('FrozenLake-v0')
# env = gym.make('FrozenLake8x8-v0')
# env = gym.make('Taxi-v2')

for i_episode in range(10):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Game " + str(i_episode) + " lasted " + str(t) + "steps")
            #print(env.observation_space.high)
            break