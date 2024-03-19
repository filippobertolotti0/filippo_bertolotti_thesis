import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.layers import Concatenate
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# def build_actor(state_shape, action_shape):
#     state_input = Input(shape=state_shape)
#     h1 = Dense(500, activation='relu')(state_input)
#     h2 = Dense(1000, activation='relu')(h1)
#     h3 = Dense(500, activation='relu')(h2)
#     output = Dense(action_shape[0], activation='tanh')(h3)

#     actor = Model(inputs=state_input, outputs=output)
#     actor.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
#     actor.summary()

#     return (state_input, actor)

# def build_critic(state_shape, action_shape):
#     state_input = Input(shape=state_shape)
#     state_h1 = Dense(500, activation='relu')(state_input)
#     state_h2 = Dense(1000)(state_h1)

#     action_input = Input(shape=action_shape)
#     action_h1 = Dense(500)(action_input)
    
#     merged = Concatenate()([state_h2, action_h1])
#     merged_h1 = Dense(500, activation='relu')(merged)
#     output = Dense(1, activation='linear')(merged_h1)
    
#     critic = Model(inputs=[state_input, action_input], outputs=output)
#     critic.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
#     critic.summary()

#     return state_input, action_input, critic

def build_actor(state_shape, action_shape):
    actor = tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation='relu', input_shape=state_shape),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(action_shape[0], activation='tanh')
    ])
    return actor

def build_critic(state_shape, action_shape):
    state_input = tf.keras.layers.Input(shape=state_shape)
    action_input = tf.keras.layers.Input(shape=action_shape)
    state_h1 = tf.keras.layers.Dense(500, activation='relu')(state_input)
    state_h2 = tf.keras.layers.Dense(1000, activation='relu')(state_h1)
    action_h1 = tf.keras.layers.Dense(500)(action_input)
    merged = tf.keras.layers.Concatenate()([state_h2, action_h1])
    merged_h1 = tf.keras.layers.Dense(500, activation='relu')(merged)
    output = tf.keras.layers.Dense(1, activation='linear')(merged_h1)
    critic = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
    return critic