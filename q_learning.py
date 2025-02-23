import numpy as np
import random

class QTables:
    def __init__(self, observation_space, action_space, eps_start=1, eps_end=0.1, gamma=0.9, lr=0.1):
        self.observation_length = observation_space[0].shape[0]
        self.action_num = action_space.n

        self.eps = eps_start
        self.eps_end = eps_end
        self.gamma = gamma
        self.lr = lr

        # Single shared Q-table for all agents
        self.q_table = np.zeros((4 ** self.observation_length, self.action_num))

    def obs_to_row(self, obs):
        # Convert observation to a unique row index
        return int(sum([val * (4 ** i) for i, val in enumerate(obs)]))

    def get_action(self, obs, agent_i, shared_obs):
        if random.random() < self.eps:
            return random.randint(0, self.action_num - 1)
        else:
            # Combine personal and shared observations
            combined_obs = np.mean([obs[agent_i]] + shared_obs, axis=0)
            obs_row = self.obs_to_row(combined_obs)
            return np.argmax(self.q_table[obs_row])

    def update_eps(self):
        # Decay epsilon
        if self.eps > self.eps_end:
            self.eps *= 0.99

    def train(self, obs, obs_next, action, reward, done, agent_i):
        obs_row = self.obs_to_row(obs[agent_i])
        obs_next_row = self.obs_to_row(obs_next[agent_i])

        q_current = self.q_table[obs_row][action]
        q_next_max = np.max(self.q_table[obs_next_row])

        if done:
            self.q_table[obs_row][action] = q_current + self.lr * (reward - q_current)
        else:
            self.q_table[obs_row][action] = q_current + self.lr * (reward + self.gamma * q_next_max - q_current)