# Snake Environment Classic

A Gymnasium environment for the classic Snake game, designed for reinforcement learning experiments with customizable grid sizes and multiple render modes.

## Features

- **Gymnasium Interface**: Fully compatible with gymnasium environments
- **Pygame Visualization**: Real-time rendering with pygame
- **Customizable Grid Size**: Configurable game board size
- **Multiple Render Modes**: Human visualization and RGB array output
- **RL-Ready**: Proper observation and action spaces for RL algorithms

## Environment Specifications

- **Action Space**: 4 discrete actions (Up, Right, Down, Left)
- **Observation Space**: 2D grid (0=empty, 1=snake, 2=food)
- **Reward System**: +10 for food, -10 for collision, 0 for movement
- **Grid Size**: Configurable (default: 20x20)

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
import gymnasium as gym
from snake_env import SnakeEnvClassic

# Create environment
env = SnakeEnvClassic(render_mode="human", grid_size=20)

# Reset and run
observation, info = env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.render()
env.close()
```

### Environment Registration
```python
import gymnasium as gym
env = gym.make('snake_env_classic-v0', render_mode="human")
```

### Action Space

- `0`: Move Up
- `1`: Move Right  
- `2`: Move Down
- `3`: Move Left

### Rewards

- **+10**: Eating food
- **-10**: Hitting wall or self
- **0**: Moving without eating

### Testing
```bash
python example.py
```

### Training
```bash
python train.py
```

## Project Structure

- `snake_env.py` - Core environment implementation
- `example.py` - Usage examples and testing
- `train.py` - Training script
- `test_visualization.py` - Visualization testing

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- pygame>=2.0.0
- numpy>=1.19.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025 
