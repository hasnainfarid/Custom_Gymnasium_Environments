# Agricultural Farm Management Environment

A comprehensive agricultural farm management environment for reinforcement learning, built on OpenAI Gymnasium. This environment simulates realistic farming operations including crop management, weather systems, market dynamics, and sustainability tracking.

## Features

- **25x25 farm grid** with 4 field sections and 5 crop types (wheat, corn, tomatoes, apples, soybeans)
- **Dynamic weather system** with 4 seasons, temperature, humidity, rainfall, and climate change effects
- **Farm equipment management** with 8 types of machinery and maintenance tracking
- **Complex state space** (620 dimensions) including crop growth, soil conditions, weather, equipment, and financial metrics
- **Rich action space** (45 actions) covering planting, harvesting, fertilization, irrigation, and sustainability practices

## Installation

```bash
cd agricultural_farm_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from agricultural_farm_env import AgriculturalFarmEnv

# Create environment
env = AgriculturalFarmEnv(render_mode="human")

# Reset environment
observation, info = env.reset()

# Run simulation
for _ in range(365):  # One year
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Testing

```bash
python test_farm.py
```

Choose option 1 for visualization demo or option 2 for full testing with multiple scenarios.

## Requirements

- gymnasium>=0.28.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.3.0

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025
