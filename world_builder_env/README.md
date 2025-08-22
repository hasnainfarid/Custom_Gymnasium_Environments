[Random Actions on the left, Trained Agent on the right]

![FinalVideo_WorldBuilder](https://github.com/user-attachments/assets/f171acb6-8b54-4739-9049-e8ba847ede3a)

# World Builder Environment

A Gymnasium environment for city-building simulation where an AI agent learns resource management and construction strategies using reinforcement learning.

## Features

- **Resource Management**: Food, Wood, Stone, and Population balancing
- **Building System**: 4 building types with strategic placement
- **Grid-based World**: 10x10 tile-based construction environment
- **Population Dynamics**: Growth, consumption, and survival mechanics
- **PPO Integration**: Ready for reinforcement learning training

## Game Mechanics

### Core Resources
- **Food**: Consumed by population, produced by farms
- **Wood**: Used for building, produced by lumberyards
- **Stone**: Used for building, produced by quarries
- **Population**: Grows with excess food and available capacity

### Buildings

| Building | Cost | Production | Effect |
|----------|------|------------|--------|
| Farm | 5 wood | 2 food per step | Produces food |
| Lumberyard | 3 stone | 3 wood per step | Produces wood |
| Quarry | 5 wood | 2 stone per step | Produces stone |
| House | 10 wood + 5 stone | None | +5 population capacity |

### Game Rules

- **Grid Size**: 10x10 world with tile-based building placement
- **Population**: Consumes 1 food per step, grows with excess food
- **Actions**: 5 discrete actions (4 building types + pass)
- **Win Condition**: Reach 20 population and survive 50 steps
- **Lose Condition**: Population drops to 0

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from world_builder_env import WorldBuilderEnv

# Create environment
env = WorldBuilderEnv(render_mode="human")

# Reset and run
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Manual Control
```bash
python main.py
# Choose option 1 for manual control
```

### Random Agent
```bash
python main.py
# Choose option 2 for random agent
```

## Environment Interface

- **Observation Space**: Grid state, resource levels, population info
- **Action Space**: 5 discrete actions (0-4)
- **Reward Function**: Population growth + resource efficiency
- **Render Modes**: Human viewing and RGB array output

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.19.0
- pygame>=2.0.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025 
