from utils.datastructures import Dictionary
from utils.q_network import QNetwork
from utils.replaymemory import ReplayMemory
import numpy as np
import random
import torch
import collections

Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Q_learning_Agent:
    """Q-Learning agent for the cart-pole problem"""

    def __init__(self, action_space, learning_rate, discount_factor, epsilon, epsilon_decay_value):
        """setup the agent"""
        self.actions = np.arange(0, action_space.n)
        self.q_table = Dictionary(0, action_space.n, filename="./cartpole/utils/cache")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_value = epsilon_decay_value

    def get_action(self, observation):            
        _, action = self.get_max_q(observation)

        return action
    
    def get_max_q(self, observation):
        discrete = self.discretization(observation)

        q_values = self.q_table.get(discrete)
        if q_values[0] == q_values[1]:
            action = np.random.randint(0, len(self.actions))
        else:
            action = np.argmax(q_values)

        return q_values[action], action
    
    def update(self, prev_observation, action, reward, observation):
        prev_discrete = self.discretization(prev_observation)

        max_future_q, _ = self.get_max_q(observation)
        current_q = self.q_table.get(prev_discrete)[action]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table.get(prev_discrete)[action] = new_q

    def discretization(self, observation):
        step_array = np.array([0.2, 0.2, 0.01, 0.1])
        discrete = observation/step_array
        return tuple(discrete.astype(int))
    
class DQN_Agent:
    def __init__(self, environment, input_size, output_size, learning_rate, gamma, batch_size):
        self.learning_rate = learning_rate
        self.environment = environment

        self.policy_net = QNetwork(environment, input_size, output_size, learning_rate)
        self.target_net = QNetwork(environment, input_size, output_size, learning_rate)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())

        self.replay_memory = ReplayMemory(environment)
        self.burn_in_memory()

        self.gamma = gamma
        self.batch_size = batch_size
        self.c = 0

    def burn_in_memory(self):
        cnt = 0
        done = False
        state, _ = self.environment.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while cnt < self.replay_memory.burn_in:
            if done:
                state, _ = self.environment.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            action = torch.tensor(random.sample([0, 1], 1)).reshape(1, 1)
            next_state, reward, terminated, truncated, _ = self.environment.step(action.item())
            done = terminated or truncated
            reward = torch.tensor([reward])

            next_state = None if done else torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            self.replay_memory.append((state, action, next_state, reward))
            state = next_state
            cnt += 1

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        p = random.random()
        if p > epsilon:
            with torch.no_grad():
                return self.greedy_policy(q_values)
        else:
            return torch.tensor([[self.environment.action_space.sample()]], dtype=torch.long)
        
    def greedy_policy(self, q_values):
        return torch.argmax(q_values)
    
    def train(self):
        # Train the Q-network using Deep Q-learning.
        state, _ = self.environment.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False

        while not done:
            with torch.no_grad():
                q_values = self.policy_net.net(state)

            action = self.epsilon_greedy_policy(q_values).reshape(1,1)
            next_state, reward, terminated, truncated, _ = self.environment.step(action.item())
            done = terminated or truncated
            reward = torch.tensor([reward])

            next_state = None if done else torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            self.replay_memory.append((state, action, next_state, reward))
            state = next_state

            # Sample minibatch with size N from memory
            transitions = self.replay_memory.sample_batch(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = self.policy_net.net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(self.batch_size)

            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).max(1)[0]

            # Update the model
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()

            self.c += 1
            if self.c % 50 == 0:
                self.target_net.net.load_state_dict(self.policy_net.net.state_dict())

    def test(self):
        max_t = 1000
        state, _ = self.environment.reset()
        total_reward = 0

        for t in range(max_t):
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net.net(state)
            action = self.greedy_policy(q_values)
            state, reward, terminated, truncated, _ = self.environment.step(action.item())
            total_reward += reward
            if terminated or truncated:
                break

        return np.sum(total_reward)