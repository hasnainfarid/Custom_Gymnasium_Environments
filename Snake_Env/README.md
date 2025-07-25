# Snake RL Environment

A custom gymnasium environment for the classic Snake game, designed for reinforcement learning experiments.

## Author

**Hasnain Fareed**  
Email: hasnainfarid7@yahoo.com  
Year: 2025

## Features

- **Gymnasium Interface**: Fully compatible with gymnasium environments
- **Pygame Visualization**: Real-time rendering with pygame
- **Customizable Grid Size**: Configurable game board size
- **Multiple Render Modes**: Human visualization and RGB array output
- **RL-Ready**: Proper observation and action spaces for RL algorithms

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd snake_env
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. The environment will be automatically registered with gymnasium when imported.

## Usage

### Basic Usage

```python
import gymnasium as gym
from snake_env import SnakeEnv

# Create environment
env = SnakeEnv(render_mode="human", grid_size=20)

# Reset environment
observation, info = env.reset()

# Take actions
action = env.action_space.sample()  # Random action
observation, reward, done, truncated, info = env.step(action)

# Render
env.render()

# Close environment
env.close()
```

### Action Space

- `0`: Move Up
- `1`: Move Right  
- `2`: Move Down
- `3`: Move Left

### Observation Space

The observation is a 2D grid where:
- `0`: Empty cell
- `1`: Snake segment
- `2`: Food

### Rewards

- `+10`: Eating food
- `-10`: Hitting wall or self
- `0`: Moving without eating

### Example Scripts

Run the example script to test the environment:

```bash
python example.py
```

This will give you options to:
1. Test with a random agent
2. Test with manual keyboard control

## Training with RL Algorithms

The environment is compatible with popular RL libraries like Stable Baselines3, RLlib, and others.

### Example with Stable Baselines3

```python
from stable_baselines3 import PPO
from snake_env import SnakeEnv

env = SnakeEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Training with Q-Learning

```bash
python train.py
```

This will train a simple Q-learning agent and show the results.

## Environment Parameters

- `render_mode`: "human" for pygame visualization, "rgb_array" for array output, None for no rendering
- `grid_size`: Size of the game grid (default: 20)

## Game Rules

1. The snake starts at the center of the grid
2. Control the snake to eat red food squares
3. The snake grows longer when it eats food
4. Game ends if the snake hits the wall or itself
5. Maximum episode length is 1000 steps

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or contributions, please contact:
- **Hasnain Fareed**
- Email: hasnainfarid7@yahoo.com 