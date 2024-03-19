import sys
sys.path.append('c:/Users/filip/Desktop/politecnico/tesi')

import gym
import pygame
from cartpole.agents import Q_learning_Agent, DQN_Agent
import numpy as np
import math
import time

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Q_learning_Agent(env.action_space, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay_value = 0.99995)
    prior_reward = 0
    total_time = 0
    total_reward = 0

    for episode in range(50001):
        episode_reward = 0
        t0 = time.time()
        observation, info = env.reset()
        done = False

        while not done:
            if np.random.random() > agent.epsilon:
                action = agent.get_action(observation)
            else:
                action = np.random.randint(0, env.action_space.n)

            prev_observation = observation

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            if not done:
                agent.update(prev_observation, action, reward, observation)

        if agent.epsilon > 0.05:
            if episode_reward > prior_reward and episode > 10000:
                agent.epsilon = math.pow(agent.epsilon_decay_value, episode - 10000)

        t1 = time.time()
        episode_time = t1 - t0
        total_time += episode_time
        prior_reward = episode_reward
        total_reward += episode_reward

        if episode % 5000 == 0 and episode > 0:
            print("")

            mean_time = total_time / 5000
            mean_reward = total_reward / 5000

            total_time = 0
            total_reward = 0

            print(f"AFTER {episode} EPISODES")
            print(f"mean time: {mean_time}")
            print(f"mean reward: {mean_reward}")

    agent.q_table.save_table("cache")
    env.close()

    #############################################################################

    # num_episodes_train = 200
    # num_episodes_test = 20

    # # Create the environment
    # env = gym.make('CartPole-v1')

    # # Plot average performance of 5 trials
    # num_seeds = 5
    # l = num_episodes_train // 10
    # res = np.zeros((num_seeds, l))

    # # Loop over multiple seeds
    # for i in range(num_seeds):
    #     reward_means = []

    #     # Create an instance of the DQN_Agent class
    #     agent = DQN_Agent(env, env.observation_space._shape[0], env.action_space.n, learning_rate=5e-4, gamma=0.99, batch_size=32)

    #     # Training loop
    #     for m in range(num_episodes_train):
    #         agent.train()

    #         # Evaluate the agent every 10 episodes during training
    #         if m % 10 == 0:
    #             print("Episode: {}".format(m))

    #             # Evaluate the agent's performance over 20 test episodes
    #             G = np.zeros(num_episodes_test)
    #             for k in range(num_episodes_test):
    #                 g = agent.test()
    #                 G[k] = g

    #             reward_mean = G.mean()
    #             reward_sd = G.std()
    #             print(f"The test reward for episode {m} is {reward_mean} with a standard deviation of {reward_sd}.")
    #             reward_means.append(reward_mean)

    #     res[i] = np.array(reward_means)