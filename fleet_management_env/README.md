

<img width="1360" height="901" alt="Fleet_00" src="https://github.com/user-attachments/assets/68ddc164-7eab-4ca5-8229-3ba54b7e29d8" />

# Fleet Management Environment

A Gymnasium environment for urban delivery logistics simulation with multi-vehicle coordination, dynamic traffic patterns, and fuel management.

## Features

- **Multi-Vehicle Fleet**: 3 vehicle types (delivery van, motorcycle, cargo truck)
- **Dynamic City Grid**: 25x25 urban environment with customer zones
- **Traffic System**: Dynamic congestion patterns and fuel management
- **Real-time Visualization**: Pygame-based city grid rendering
- **Multi-objective Optimization**: Balance time, fuel, and customer satisfaction

## Environment Specifications

- **Grid Size**: 25x25 city representation
- **Vehicles**: 3 specialized vehicles with unique capabilities
- **Customer Zones**: 4 distinct areas (Residential, Commercial, Industrial, Hospital)
- **Action Space**: Multi-discrete [8, 8, 8] for each vehicle
- **Observation Space**: 87-dimensional state representation

## Vehicle Types

| Vehicle | Speed | Capacity | Fuel Consumption | Range | Special Abilities |
|---------|-------|----------|------------------|-------|-------------------|
| Delivery Van | 1 | 3 packages | 1.0/move | 80 steps | General purpose |
| Motorcycle | 1 | 1 package | 0.5/move | 120 steps | Fast, hospital deliveries |
| Cargo Truck | 1 | 5 packages | 2.0/move | 60 steps | Heavy cargo, industrial |

## Quick Start

### Installation
```bash
pip install -e .
```

### Basic Usage
```python
import gymnasium as gym
import fleet_management_env

# Create environment
env = gym.make('FleetManagement-v0')

# Reset and run
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### Testing
```bash
python test_fleet.py
```

## Action Space

Each vehicle has 8 discrete actions:
- `0`: Stay in place
- `1-4`: Move (Up/Down/Left/Right)
- `5`: Pick up delivery
- `6`: Drop off delivery
- `7`: Refuel at station

## Reward Structure

- **+200**: Urgent delivery completed on time
- **+100**: Normal delivery completed
- **+50**: Standard delivery completed
- **+15**: Efficient routing bonus
- **+10**: Successful refueling
- **-2**: Timestep cost per active vehicle
- **-5**: Moving through heavy traffic
- **-10**: Invalid actions
- **-20**: Late delivery penalty
- **-50**: Vehicle runs out of fuel

## Requirements

- Python 3.7+
- gymnasium>=0.26.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025
