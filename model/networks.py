import warnings
import numpy as np
import os
import keras
import tensorflow as tf
import tensorflow_probability as tfp

from model.environment import Environment
from collections import deque
from keras.api.layers import Dense
from keras.api.optimizers import Adam
from keras.api.models import Sequential

import warnings
warnings.filterwarnings("ignore")

class ActorCriticNetwork(keras.Model):
    def __init__(
        self,
        n_actions=81,
        cur_chapter="chuong-1",
        name="actor_critic",
        checkpoint_dir="tmp/actor_critic",
    ):
        super(ActorCriticNetwork, self).__init__()

        self.model_name = name
        self.cur_chapter = cur_chapter
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_c{self.cur_chapter.split('-')[-1]}.weights.h5" 
        )

        self.osn = 8
        self.n_actions = n_actions

        """
        Hidden layers
        Output = vector of size 32
        """
        self.model = Sequential([
                Dense(32, input_dim=self.osn, activation="relu"),
                Dense(16, activation="relu"),
            ])

        """
        Critic
        Output = value of state
        Ex: v = 0.85
        """
        self.v = Dense(1, activation=None)
        """
        Actor
        Output = probability distribution of all available actions 
        Ex: pi = [0.7, 0.2, 0.1, ...]
        """
        self.pi = Dense(self.n_actions, activation="softmax")

    def call(self, state):

        value = self.model(state)
        v = self.v(value)
        pi = self.pi(value)

        return v, pi
    

# actor_critic = ActorCriticNetwork(n_actions=485)
# state = [0, 1, 0, 0, 0, 0, 0, 1]
# next_state = [0, 1, 0, 0, 0, 0, 0, 1]
# state = tf.convert_to_tensor([state], dtype=tf.float32)
# next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
# state_value, probs = actor_critic(state)

# print(f"state_value: {state_value}")
# print(f"probs: {probs}")
# state_value = tf.squeeze(state_value)
# print(f"state_value: {state_value}")

# action_probs = tfp.distributions.Categorical(probs=probs)
# print(f"action_probs: {action_probs}")
# action = action_probs.sample()
# print(f"action: {action}")
# action_prob = action_probs.prob(action)
# print(f"action_prob: {action_prob}")
# log_prob = action_probs.log_prob(action)
# print(f"log_prob: {log_prob}")