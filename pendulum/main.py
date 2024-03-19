import gym
from agents import Q_learning_agent, DDPG_agent
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from collections import deque

tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":
    env = gym.make("Pendulum-v1", g=9.81)

    agent = Q_learning_agent(env.action_space, n_actions=50, learning_rate=0.1, epsilon=1.0, epsilon_decay_value=0.99995, discount_factor=0.95)

    for episode in range(50000):
        observation, _ = env.reset()
        done = False
        episode_max_reward = float('-inf')
        episode_reward = 0

        while not done:
            if random.random() < agent.epsilon:
                action_index = random.choice(range(agent.n_actions+1))
                action = agent.actions[action_index]
            else:
                action_index = agent.get_action(observation)
                action = agent.actions[action_index]

            prev_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_max_reward = max(episode_max_reward, reward)
            episode_reward += reward

            if not done:
                agent.update(action_index, observation, prev_observation, reward)

        if agent.epsilon > 0.05:
            if episode > 10000:
                agent.epsilon = agent.epsilon * agent.epsilon_decay_value

        if episode > 10000 and episode_reward >= -201:
            print(f"{episode_reward} in episode number {episode}")

    #####################################################################################################
    
    # state_shape = env.observation_space.shape
    # action_shape = env.action_space.shape

    # sess = tf.compat.v1.Session()
    # K.set_session(sess)

    # agent = DDPG_agent(state_shape, action_shape)

    # last100scores = deque(maxlen=100)
    # scores = []
    # avgscores = []
    # avg100scores = []

    # for episode in range(3000):
    #     print(f"{episode}")
    #     state = env.reset()
    #     state = np.reshape(state[0], [1, state_shape[0]])
    #     total_reward, done = 0, False

    #     while not done:
    #         action = agent.act(state).reshape((1, *action_shape))
    #         next_state, reward, terminated, truncated, _ = env.step(action)
    #         done = terminated or truncated
    #         next_state = np.array([x.item() for x in next_state]).reshape(1, -1)
    #         agent.remember(state, action, reward, next_state, done)

    #         if episode % 5 == 0 and episode > 1:
    #             agent.train()
    #             agent.update_target_models()

    #         state = next_state
    #         total_reward += reward[0]

    #     scores.append(total_reward)
    #     avgscores.append(np.mean(scores))
    #     last100scores.append(total_reward)
    #     avg100scores.append(np.mean(last100scores))

    #     if episode % 100 == 0 and episode > 0:
    #         print(f'trial: {episode}, scores: {total_reward}, avgscores: {np.mean(scores)}, avg100scores:{np.mean(last100scores)}')

    env.close()