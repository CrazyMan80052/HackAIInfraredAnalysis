import numpy as np
import random

class QTables:
	def __init__(self, observation_space, action_space, eps_start=1, eps_end=0.1, gamma=0.8, lr=0.1):
		self.observation_length = observation_space.shape[0]
		self.action_num = action_space.n

		self.eps = eps_start
		self.eps_end = eps_end
		self.gamma = gamma
		self.lr = lr
		
		self.decay_rate = 0.999
		# Single shared Q-table for all agents
		self.q_table = np.zeros((4 ** self.observation_length, self.action_num))

	def update_eps(self):
		if(self.eps > self.eps_end):
			self.eps *= self.decay_rate

	def obs_to_row(self, obs):
		""" Convert observation to a unique row index while preventing overflow. """
		obs = np.clip(obs, 0, 3)  # Ensure values are in range [0, 3]
		return int(np.dot(obs, 4 ** np.arange(len(obs))))  # Vectorized computation

	def get_action(self, obs, agent_i, shared_obs):
		""" Selects an action based on ε-greedy Q-learning policy. """
		if random.random() < self.eps:
			return random.randint(0, self.action_num - 1)

		# Ensure obs is a NumPy array
		obs_agent = np.array(obs[agent_i], dtype=np.float32)
		shared_obs = [np.array(s_obs, dtype=np.float32) for s_obs in shared_obs if s_obs.shape == obs_agent.shape]

		# Compute mean of all valid observations
		all_obs = [obs_agent] + shared_obs
		combined_obs = np.mean(all_obs, axis=0)

		obs_row = self.obs_to_row(combined_obs)
		return np.argmax(self.q_table[obs_row])
	
	# def update_eps(self):
	# 	""" Decay epsilon, ensuring it doesn't go below eps_end. """
	# 	self.eps = max(self.eps_end, self.eps * 0.99)

	def train(self, obs, obs_next, action, reward, done, agent_i):
		""" Updates Q-table using Q-learning update rule. """
		obs_row = self.obs_to_row(obs[agent_i])
		obs_next_row = self.obs_to_row(obs_next[agent_i])

		q_current = self.q_table[obs_row][action]
		q_next_max = np.max(self.q_table[obs_next_row])

		# Update Q-value
		self.q_table[obs_row][action] = q_current + self.lr * (
			reward + (0 if done else self.gamma * q_next_max) - q_current
		)

	# def train(self, obs, obs_next, action, reward, done, agent_i):
	# 	obs_row = self.obs_to_row(obs[agent_i])
	# 	obs_next_row = self.obs_to_row(obs_next[agent_i])

	# 	q_current = self.q_table[agent_i][obs_row][action]
	# 	q_next_max = np.max(self.q_table[agent_i][obs_next_row])

	# 	if done:
	# 		self.q_table[agent_i][obs_row][action] = q_current + self.lr * (reward - q_current)
	# 	else:
	# 		self.q_table[agent_i][obs_row][action] = q_current + self.lr * (reward + self.gamma * q_next_max - q_current)
