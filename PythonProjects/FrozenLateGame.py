import gym
import random

# VARIABLES
env = gym.make("FrozenLake-v0")
score = 0
numGames = 1000
numSteps = 10

env.reset()

print(env.observation_space) # which space the player is currently in
print(env.action_space) # actions the player can take

# 0 = left
# 1 = down
# 2 = right
# 3 = up
#obs, rew, done, info = env.step(env.action_space.sample()) # take an action

def playGame():
    global score
    for i in range(numSteps):
        obs, rew, done, info = env.step(random.randint(1, 2))
        env.render()
        if done:
            score += rew
            break
            # scipy

# Main game loop
for g in range(numGames):
    env.reset()
    playGame()

print(score)