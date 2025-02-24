import numpy as np
import random
import gym
from gym import spaces

class GridMultiAgent(gym.Env):
	metadata = {'render.modes': ['console']}

	# Action IDs
	XM = 0  # x minus
	XP = 1  # x plus
	YM = 2  # y minus
	YP = 3  # y plus

	# Grid status
	OOE = -2  # out of the environment
	OBS = -1  # obstacle
	POI = 0   # POI that is not mapped
	MAP = 1   # POI that is mapped
	AGT = 2   # agent

	def __init__(self, x_size=10, y_size=10, fov_x=3, fov_y=3, n_agents=3):
		super(GridMultiAgent, self).__init__()

		# Size of the grid
		self.x_size = x_size
		self.y_size = y_size

		# Number of agents
		self.n_agents = n_agents
		self.idx_agents = list(range(n_agents))

		# Initialize grid and agents
		self.init_grid()
		self.init_agent()

		# Define action and observation spaces
		self.action_space = spaces.Discrete(4)  # 4 actions per agent
		self.observation_space = spaces.Box(low=-2, high=2, shape=(fov_x * fov_y,), dtype=np.float32)

	def init_grid(self):
		# Initialize the grid with obstacles and POIs
		self.grid_status = np.zeros((self.x_size, self.y_size))

		# Randomly place obstacles
		n_obstacles = random.randint(0, int(self.x_size * self.y_size * 0.2))  # Up to 20% obstacles
		for _ in range(n_obstacles):
			x, y = random.randint(0, self.x_size - 1), random.randint(0, self.y_size - 1)
			self.grid_status[x, y] = self.OBS

		# Number of POIs
		self.n_poi = np.count_nonzero(self.grid_status == self.POI)

	def init_agent(self):
		# Initialize agent positions
		self.agent_pos = []
		for _ in range(self.n_agents):
			while True:
				x, y = random.randint(0, self.x_size - 1), random.randint(0, self.y_size - 1)
				if self.grid_status[x, y] == self.POI:
					self.agent_pos.append([x, y])
					self.grid_status[x, y] = self.AGT
					break

	def get_coverage(self):
		# Calculate coverage percentage
		mapped_poi = np.count_nonzero(self.grid_status == self.MAP)
		return mapped_poi / self.n_poi

	def reset(self):
		# Reset the environment
		self.init_grid()
		self.init_agent()
		return self.get_agent_obs()

	def step(self, actions):
		rewards = []
		done = False

		for i, action in enumerate(actions):
			x, y = self.agent_pos[i]
			if action == self.XM:
				x -= 1
			elif action == self.XP:
				x += 1
			elif action == self.YM:
				y -= 1
			elif action == self.YP:
				y += 1

			# Check boundaries and obstacles
			if x < 0 or x >= self.x_size or y < 0 or y >= self.y_size or self.grid_status[x, y] == self.OBS:
				reward = -1  # Penalty for hitting boundaries or obstacles
			else:
				if self.grid_status[x, y] == self.POI:
					self.grid_status[x, y] = self.MAP
					reward = 10  # Reward for mapping a new POI
				else:
					reward = -0.1  # Small penalty for revisiting mapped areas
				self.agent_pos[i] = [x, y]

			rewards.append(reward)

		# Check if all POIs are mapped
		done = self.get_coverage() >= 0.95
		return self.get_agent_obs(), rewards, done, {}
	# def step(self, actions):
	# 	rewards = []
	# 	done = False

	# 	for i, action in enumerate(actions):
	# 		x, y = self.agent_pos[i]
	# 		if action == self.XM:
	# 			x -= 1
	# 		elif action == self.XP:
	# 			x += 1
	# 		elif action == self.YM:
	# 			y -= 1
	# 		elif action == self.YP:
	# 			y += 1

	# 		# Check boundaries and obstacles
	# 		if x < 0 or x >= self.x_size or y < 0 or y >= self.y_size or self.grid_status[x, y] == self.OBS:
	# 			reward = 0
	# 		else:
	# 			if self.grid_status[x, y] == self.POI:
	# 				self.grid_status[x, y] = self.MAP
	# 				reward = 10
	# 			else:
	# 				reward = 0
	# 			self.agent_pos[i] = [x, y]

	# 		rewards.append(reward)

	# 	# Check if all POIs are mapped
	# 	done = self.get_coverage() >= 0.95
	# 	return self.get_agent_obs(), rewards, done, {}

	def get_agent_obs(self):
		# Get observations for all agents
		# obs = []
		# for x, y in self.agent_pos:
		#     single_obs = self.grid_status[x - 1:x + 2, y - 1:y + 2].flatten()  # 3x3 FOV
		#     obs.append(np.array(single_obs, dtype=np.float32))
		# return obs
		
		obs = []
		for x, y in self.agent_pos:
			# Initialize a 3x3 grid with -2 (out-of-environment)
			single_obs = np.full((3, 3), -2, dtype=np.float32)

			for i in range(3):
				for j in range(3):
					x_fov = x - 1 + i
					y_fov = y - 1 + j
					if 0 <= x_fov < self.x_size and 0 <= y_fov < self.y_size:
						single_obs[i, j] = self.grid_status[x_fov, y_fov]

			single_obs = single_obs.flatten()
			obs.append(single_obs)

		return obs