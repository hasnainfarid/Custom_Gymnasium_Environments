
<img width="1487" height="1017" alt="Space_00" src="https://github.com/user-attachments/assets/7b338024-69b7-48e8-b359-904b3f0071df" />

# Space Station Environment

A Gymnasium environment for space station life support management simulation with realistic orbital mechanics and system interdependencies.

## Features

- **8 Interconnected Modules**: Command, Living Quarters, Laboratory, Engineering, Hydroponics, Storage, Docking, Solar Array Control
- **6 Crew Members**: Commander, Engineer, Scientist, Medical Officer, Pilot, Mission Specialist
- **12 Critical Systems**: Complete life support with realistic interdependencies
- **Orbital Mechanics**: 90-minute orbits with day/night cycles affecting solar power
- **Resource Management**: Water, food, oxygen, power, medical supplies, and more
- **Real-time Visualization**: 1200x900 Pygame display with station cross-section

## Environment Specifications

- **Observation Space**: 110-dimensional state representation
- **Action Space**: 40 discrete actions for system control
- **Reward Structure**: Mission outcome and efficiency rewards

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Basic Usage
```python
import gymnasium as gym
from space_station_env import SpaceStationEnv

# Create environment
env = SpaceStationEnv(render_mode="human")

# Reset and run
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

### Testing
```bash
python test_station.py
```

Tests 6 emergency scenarios: Normal Operations, Power Failure, Atmospheric Leak, Medical Emergency, Solar Storm, and Cascade Failure.

## Requirements

- Python 3.7+
- gymnasium>=0.29.0
- numpy>=1.24.0
- pygame>=2.5.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025
