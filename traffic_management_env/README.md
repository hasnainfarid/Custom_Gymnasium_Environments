# Traffic Management Environment

A Gymnasium environment for traffic light optimization using reinforcement learning in grid-based traffic networks.

## Features

- **Grid-based Network**: Configurable intersections with traffic light control
- **Dynamic Traffic**: Random vehicle generation with realistic routing
- **Multi-objective Optimization**: Balance throughput, waiting time, and flow smoothness
- **Real-time Visualization**: Traffic simulation rendering
- **Flexible Configuration**: Adjustable grid size and traffic parameters

## Environment Specifications

- **Action Space**: MultiDiscrete control for each intersection (3 phases)
- **Observation Space**: Traffic light states, queue lengths, waiting times, flow metrics
- **Reward Function**: Vehicle throughput + waiting time penalties + flow smoothness
- **Network**: Configurable grid layout with multiple intersections

## Quick Start

### Installation
```bash
pip install -e .
```

### Basic Usage
```python
import gymnasium as gym
import traffic_management_env

# Create environment
env = gym.make('TrafficManagement-v0')

# Reset and run
observation, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

### Configuration
```python
env = gym.make('TrafficManagement-v0', 
               grid_size=(4, 4),           # 4x4 grid
               num_intersections=12,       # Controlled intersections
               max_vehicles=30,            # Max vehicles
               spawn_rate=0.2)             # Spawn probability
```

## Action Space

Each intersection can be controlled with:
- `0`: Maintain current phase
- `1`: North-South green light
- `2`: East-West green light

## Reward System

- **+1.0**: Vehicle passes through intersection
- **-0.1**: Per timestep per waiting vehicle
- **-0.05**: Per vehicle in queue
- **+0.5**: Smooth traffic flow bonus
- **-5.0**: Emergency stop penalty

## Project Structure

- `environment.py` - Core environment implementation
- `demo.py` - Interactive demonstration
- `test_env.py` - Environment testing
- `utils.py` - Utility functions
- `config.py` - Configuration parameters

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.19.0
- pygame>=2.0.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025