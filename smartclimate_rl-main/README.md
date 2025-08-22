![ezgif-347a404b25dd75](https://github.com/user-attachments/assets/665bb9c4-a11b-44d7-ac1a-a9bdc57e646b)

# SmartClimate Environment

A Gymnasium environment for smart HVAC and lighting control optimization with RLlib integration and PyGame visualization.

## Features

- **HVAC Control**: Temperature management with comfort optimization
- **Lighting System**: Multi-zone lighting control
- **Time-based Simulation**: 24-hour cycles with 1-minute steps
- **Real-time Visualization**: PyGame-based monitoring interface
- **RLlib Integration**: Ready for reinforcement learning training

## Environment Specifications

- **Time Steps**: 1440 steps per day (1-minute intervals)
- **Action Space**: 
  - AC temperature: Continuous [16°C, 32°C]
  - Lights: 4 binary switches
- **Observation Space**: 9-dimensional vector
  - Room temperature, occupancy, time, outside temperature
  - AC setting, light states
- **Reward Function**: Comfort score minus energy penalties

## Quick Start

### Installation
```bash
pip install -e .
```

### Basic Usage
```python
import gymnasium as gym
env = gym.make('SmartClimateEnv-v0')
obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Training
```bash
cd training
python train.py
```

### Visualization
```python
env.render()  # PyGame-based real-time display
```

## Project Structure

- `smartclimate/` - Core environment and utilities
- `training/` - RLlib training and evaluation scripts
- `examples/` - Usage demonstrations and examples

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- rllib>=2.0.0
- pygame>=2.0.0
- numpy>=1.19.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025
