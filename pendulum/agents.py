from utils.datastructures import Dictionary
import numpy as np
import random

import tensorflow as tf
import numpy as np
from collections import deque
from utils.neural_network import build_actor, build_critic

tf.get_logger().setLevel('ERROR')

class Q_learning_agent:
    def __init__(self, action_space, n_actions, learning_rate, epsilon, epsilon_decay_value, discount_factor):
        self.n_actions = n_actions
        self.actions = self.get_discrete_actions(action_space, n_actions)
        self.q_table = Dictionary(-16.2736044, n_actions+1)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay_value = epsilon_decay_value
        self.discount_factor = discount_factor

    def get_discrete_actions(self, action_space, n_actions):
        step = (abs(action_space.high) + abs(action_space.low)) / n_actions
        return [action_space.low + step * i for i in range(n_actions+1)]
    
    def get_action(self, observation):
        action_index, _ = self.get_max_q(observation)
        return action_index

    def get_max_q(self, observation):
        discrete = self.discretization(observation)
        q_values = self.q_table.get(discrete)

        max_index = np.where(q_values == np.max(q_values))[0]
        index = random.choice(max_index)
        
        return index, q_values[index]
    
    def update(self, action_index, observation, prev_observation, reward):
        prev_discrete = self.discretization(prev_observation)

        _, max_future_q = self.get_max_q(observation)
        current_q = self.q_table.get(prev_discrete)[action_index]

        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table.get(prev_discrete)[action_index] = new_q

    def discretization(self, observation):
        step_array = np.array([0.1, 0.1, 0.8])
        discrete = observation/step_array
        return tuple(discrete.astype(int))
    
# class DDPG_agent:
#     def __init__(self, sess, state_shape, action_shape):
#         self.state_shape = state_shape
#         self.action_shape = action_shape
#         self.sess = sess

#         self.actor_state_input, self.actor = build_actor(state_shape, action_shape)
#         _, self.target_actor = build_actor(state_shape, action_shape)
#         # self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32, [None, self.action_shape])
#         self.actor_critic_grad = tf.Variable(tf.zeros([1, self.action_shape[0]]), dtype=tf.float32)
#         actor_model_weights = self.actor.trainable_weights
#         self.actor_grads = tf.gradients(self.actor.output, actor_model_weights, -self.actor_critic_grad)
#         grads = zip(self.actor_grads, actor_model_weights)
#         self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

#         self.critic_state_input, self.critic_action_input, self.critic = build_critic(state_shape, action_shape)
#         _, _, self.target_critic = build_critic(state_shape, action_shape)
#         self.critic_grads = tf.GradientTape(self.critic.output, self.critic_action_input)
#         self.sess.run(tf.compat.v1.initialize_all_variables())

#         self.memory = deque(maxlen=4000)
#         self.learning_rate = 0.0001
#         self.epsilon = 0.9
#         self.epsilon_decay = 0.99995
#         self.gamma = 0.9
#         self.tau = 0.01

#     def stack_samples(samples):
#         array = np.array(samples)

#         states = np.stack(array[:,0]).reshape((array.shape[0], -1))
#         actions = np.stack(array[:,1]).reshape((array.shape[0], -1))
#         rewards = np.stack(array[:,2]).reshape((array.shape[0], -1))
#         next_states = np.stack(array[:,3]).reshape((array.shape[0], -1))
#         dones = np.stack(array[:,4]).reshape((array.shape[0], -1))

#         return states, actions, rewards, next_states, dones 

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         self.epsilon *= self.epsilon_decay
#         if np.random.random() < self.epsilon:
#             return self.actor.predict(state) * 2 + np.random.normal() 
#         else:
#             return self.actor.predict(state) * 2
    
#     def train_actor(self, samples):
#         states, actions, rewards, new_states, _ = self.stack_samples(samples)
#         predicted_actions = self.actor.predict(states)
#         grads = self.sess.run(self.critic_grads, feed_dict={
#             self.critic_state_input: states,
#             self.critic_action_input: predicted_actions 
#         })[0]

#         self.sess.run(self.optimize, feed_dict={
#             self.actor_state_input: states,
#             self.actor_critic_grad: grads
#         })

#     def train_critic(self, samples):
#         states, actions, rewards, new_states, dones = self.stack_samples(samples)
#         target_actions = self.target_actor.predict(new_states)
#         future_rewards = self.target_critic.predict([new_states, target_actions])

#         rewards += self.gamma * future_rewards * (1 - dones)
        
#         self.critic.fit([states, actions], rewards, verbose=0)

#     def train(self, batch_size=256):
#         # rewards = []
#         samples = random.sample(self.memory, batch_size)

#         self.samples = samples
#         self.train_critic(samples)
#         self.train_actor(samples)

#     def update_target_models(self):
#         actor_weights = self.actor.get_weights()
#         target_actor_weights = self.target_actor.get_weights()
#         critic_weights = self.critic.get_weights()
#         target_critic_weights = self.target_critic.get_weights()

#         for i in range(len(target_actor_weights)):
#             target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        
#         for i in range(len(target_critic_weights)):
#             target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
        
#         self.target_actor.set_weights(target_actor_weights)
#         self.target_critic.set_weights(target_critic_weights)
    
class DDPG_agent:
    def __init__(self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape
        
        self.actor = build_actor(state_shape, action_shape)
        self.target_actor = build_actor(state_shape, action_shape)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.critic = build_critic(state_shape, action_shape)
        self.target_critic = build_critic(state_shape, action_shape)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.memory = deque(maxlen=4000)
        self.gamma = 0.9
        self.tau = 0.01

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        return self.actor.predict(state)
    
    def train(self, batch_size=30):
        samples = random.sample(self.memory, batch_size)

        self.samples = samples
        self.train_critic(samples)
        self.train_actor(samples)

    def train_actor(self, samples):
        # states, _, _, _, _ = self.stack_samples(samples)
        states = np.array([sample[0] for sample in samples])
        # print("Actor input shape:", states.shape)

        with tf.GradientTape() as tape:
            predicted_actions = self.actor(states)
            # print("Predicted actions shape:", predicted_actions.shape)

            q_values = self.critic([states, predicted_actions])
            # print("Q values shape:", q_values.shape)

            actor_loss = -tf.reduce_mean(q_values)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

    def train_critic(self, samples):
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        new_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])

        with tf.GradientTape() as tape:
            target_actions = self.target_actor.predict(new_states)
            future_rewards = self.target_critic([new_states, target_actions])
            target_q_values = rewards + self.gamma * future_rewards * (1 - dones)
            predicted_q_values = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q_values - predicted_q_values))

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

    def update_target_models(self):
        self.update_target_weights(self.target_actor.variables, self.actor.variables)
        self.update_target_weights(self.target_critic.variables, self.critic.variables)

    def update_target_weights(self, target_variables, source_variables):
        for target_var, source_var in zip(target_variables, source_variables):
            target_var.assign(self.tau * source_var + (1 - self.tau) * target_var)