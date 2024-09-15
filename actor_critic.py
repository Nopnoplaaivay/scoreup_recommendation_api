import tensorflow_probability as tfp
import tensorflow as tf

from environment import Environment
from mongodb import Database
from networks import ActorCriticNetwork
from keras.api.optimizers import Adam

# db = Database()
# env = Environment(db)


class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2, env=None):
        self.gamma = gamma
        self.n_actions = env.action_space.n
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=self.n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        _, probs = self.actor_critic(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        self.action = action
        return action.numpy()[0]

    def save_models(self):
        print("... saving models ...")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print("... loading models ...")
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self, state, reward, next_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            next_state_value, _ = self.actor_critic(next_state)
            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = (
                reward + self.gamma * next_state_value * (1 - int(done)) - state_value
            )
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss
            print(f"Total Loss: {total_loss}")

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(
            zip(gradient, self.actor_critic.trainable_variables)
        )


# agent = Agent(env=env)
# state = [1, 0, 1, 0, 0, 1, 0, 0]
# action = agent.choose_action(state)
# exer = env.get_exercise_by_action(int(action))

# print(exer)
# print(action)
# print(type(int(action)))