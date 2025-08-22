
![BusEnvGif](https://github.com/user-attachments/assets/735de23e-43e5-4eb4-bb08-f1965dd209f7)

# Bus System Environment

A Gymnasium environment for urban bus system simulation with multiple buses operating on a circular route and dynamic passenger management.

## Features

- **4 buses** operating on a circular route with **4 stops**
- **Dynamic passenger generation** (50-150 passengers per episode)
- **Centralized dwell time control** - agent manages bus waiting times
- **Realistic passenger boarding/alighting** with destination tracking
- **Pygame visualization** for demo purposes
- **Gymnasium-compatible** interface

## Environment Specifications

- **Action Space**: 4 discrete actions (dwell time 0-10 timesteps per bus)
- **Observation Space**: Bus positions, states, capacities, passenger data
- **Reward System**: Passenger delivery + waiting penalties + onboard costs
- **Episode Length**: 500 timesteps (configurable)

## Quick Start

### Installation
```bash
pip install gymnasium numpy pygame
```

### Basic Usage
```python
from bus_system_env import BusSystemEnv

# Create environment
env = BusSystemEnv(enable_visualization=True, demo_speed=0.5)
obs, info = env.reset()

# Run simulation
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

### Testing
```bash
python test_env.py
```

## Reward System

- **+5.0**: Per passenger delivered
- **-1.0**: Per waiting passenger per timestep
- **-0.5**: Per onboard passenger per timestep

## Visualization

Enable Pygame visualization for demo mode:
```python
env = BusSystemEnv(enable_visualization=True, demo_speed=0.5)
```

**Controls:**
- **SPACE**: Pause/Resume
- **ESC**: Exit

## Project Structure

- `environment.py` - Core environment implementation
- `pygame_visualizer.py` - Real-time visualization
- `utils.py` - Utility functions
- `config.py` - Configuration parameters
- `test_env.py` - Environment testing

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.19.0
- pygame>=2.0.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025 
