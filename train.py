import matplotlib.pyplot as plt
from grid_env import GridMultiAgent
from q_learning import QTables

# Initialize environment and Q-Learning
env = GridMultiAgent(x_size=20, y_size=20, n_agents=5)
q = QTables(env.observation_space, env.action_space)

# Initialize visualization
plt.ion()
fig, ax = plt.subplots()

def visualize_grid(env, ax):
	grid = env.grid_status.copy()
	for x, y in env.agent_pos:
		grid[x, y] = 2  # Mark drones
	ax.clear()
	ax.imshow(grid, cmap='viridis', vmin=-2, vmax=2)
	ax.set_title(f"Coverage: {env.get_coverage() * 100:.2f}%")
	plt.pause(0.1)

# def visualize_grid(env, ax, start_positions):
# 	grid = env.grid_status.copy()

# 	# Mark drones' positions
# 	for x, y in env.agent_pos:
# 		grid[x, y] = 2  # Mark drones

# 	ax.clear()
# 	ax.imshow(grid, cmap='viridis', vmin=-2, vmax=2)

# 	# Overlay start positions with red circles
# 	for x, y in start_positions:
# 		ax.scatter(y, x, s=100, c='red', marker='o', edgecolors='black', label="Start Position")

# 	plt.pause(0.1)


# Training parameters
episodes = 4000
max_steps = 100

for episode in range(episodes):
	state = env.reset()
	total_reward = 0

	for step in range(max_steps):
		# Share observations between drones
		shared_obs = [state[i] for i in range(env.n_agents)]
		
		# shared_obs = []
		# for i in range(env.n_agents):
		# 	if(i != agent_i):
		# 		shared_obs.append([state[j] for j in range(env.n_agents) if j != i])
		
		# Get actions for all drones
		actions = [q.get_action(state, i, shared_obs) for i in range(env.n_agents)]
		
		# Take actions and observe results
		next_state, rewards, done, _ = env.step(actions)
		
		# Train the Q-table
		for i in range(env.n_agents):
			q.train(state, next_state, actions[i], rewards[i], done, i)

		state = next_state
		total_reward += sum(rewards)

		# Visualize the grid
		if(episode % 1000 == 0):
			visualize_grid(env, ax)

		if done:
			break

	q.update_eps()
	if(episode % 100 == 0):
		print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Coverage: {env.get_coverage():.2f}")

plt.ioff()
plt.show()
print("DONE")