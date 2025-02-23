import matplotlib.pyplot as plt
from grid_env import GridMultiAgent
from q_learning import QTables

# Initialize environment and Q-Learning
env = GridMultiAgent(x_size=10, y_size=10, n_agents=3)
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
    plt.pause(0.1)

# Training parameters
episodes = 1000
max_steps = 100

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Share observations between drones
        shared_obs = [state[i] for i in range(env.n_agents)]
        
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
        visualize_grid(env, ax)

        if done:
            break

    q.update_eps()
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Coverage: {env.get_coverage()}")

plt.ioff()
plt.show()