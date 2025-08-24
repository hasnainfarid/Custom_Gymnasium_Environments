
![Airline](https://github.com/user-attachments/assets/08460220-b0d9-462b-86bf-2653a850ec34)




# Airline Operations Environment

A comprehensive Gymnasium environment for simulating airline operations management with a hub-and-spoke network. This environment provides a realistic simulation of airline operations including flight scheduling, crew management, weather disruptions, and financial optimization.

## Features

- **15 airports** with 1 major hub, 4 regional hubs, and 10 spoke airports
- **25 aircraft fleet** including wide-body, narrow-body, and regional jets
- **150 daily flights** scheduled across various route types with 40 crew sets
- **Dynamic challenges** including weather systems, mechanical issues, crew management, and passenger connections
- **Complex state space** (160 dimensions) covering aircraft, flights, airports, crew, and financial metrics

## Installation

```bash
cd airline_operations_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from airline_operations_env import AirlineOperationsEnv

# Create environment
env = AirlineOperationsEnv(render_mode="human")

# Reset environment
observation, info = env.reset()

# Run simulation
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Testing

```bash
python test_airline.py
```

Run comprehensive test suite with 6 scenarios including normal operations, severe weather, mechanical crisis, and system disruptions.

## Requirements

- gymnasium>=0.28.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.3.0

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025




