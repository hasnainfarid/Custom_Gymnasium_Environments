
<img width="1008" height="677" alt="image" src="https://github.com/user-attachments/assets/fb1ccd32-68f8-40cb-b84d-fba0fc3641c5" />


# Smart Parking Environment

A Gymnasium environment for smart parking lot management optimization using reinforcement learning, featuring dynamic pricing and intelligent spot assignment.

## Features

- **Multi-zone Parking**: Premium, Standard, and Economy zones with 50 total spots
- **Dynamic Pricing**: Agent-controlled pricing based on demand and occupancy
- **Customer Simulation**: Realistic behavior with time preferences and urgency levels
- **Real-time Visualization**: Pygame-based parking lot display with animations
- **RLlib Integration**: Ready for reinforcement learning training

## Environment Specifications

- **Time Scale**: 1 minute per timestep, 24-hour episodes (1440 timesteps)
- **Action Space**: 9 continuous actions (3 price levels per zone)
- **Observation Space**: Parking occupancy, customer queue, revenue metrics
- **Reward Function**: Multi-objective optimization (revenue, satisfaction, utilization)

## Zone Configuration

- **Zone A (Premium)**: 15 spots, $8 base price, closest to entrance
- **Zone B (Standard)**: 20 spots, $5 base price, middle area
- **Zone C (Economy)**: 15 spots, $3 base price, farthest area

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Basic Usage
```python
import gymnasium as gym
from smart_parking_env import SmartParkingEnv

# Create environment
env = SmartParkingEnv()

# Reset and run
observation, info = env.reset()
for step in range(1440):  # 24-hour episode
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### Training with RLlib
```bash
cd examples
python training.py
```

### Testing
```bash
cd examples
python test_env.py
```

## Project Structure

- `core/` - Core environment components
- `utils/` - Utility modules and visualization
- `examples/` - Training and testing scripts
- `Trained_PPO_Policy/` - Pre-trained models

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.19.0
- pygame>=2.0.0
- rllib>=2.0.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025 
