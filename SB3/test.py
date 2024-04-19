import gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

done = np.array([False for _ in range(4)])
rewards = np.array([0 for _ in range(4)])
# done[1] = True
# done[2] = True
for index in np.where(done)[0]:
    rewards[index] = 5
    
print(rewards)