
<img width="1488" height="783" alt="Manufacturing_00" src="https://github.com/user-attachments/assets/b271604c-971e-41ab-bb47-3c6f95ba8bde" />

# Smart Manufacturing Environment

A Gymnasium environment for smart manufacturing facility simulation with production stations, quality control, and predictive maintenance capabilities.

## Features

- **5 Production Stations**: Cutting, Assembly, Testing, and Packaging
- **6 Product Types**: Varying complexity and quality requirements  
- **Real-time Visualization**: Pygame-based factory floor display
- **Machine Degradation**: Realistic performance modeling
- **Predictive Maintenance**: Early warning systems
- **Quality Control**: Multi-checkpoint quality assurance
- **OEE Metrics**: Real-time tracking of Availability, Performance, and Quality

## Environment Specifications

- **Observation Space**: 73-dimensional state representation
- **Action Space**: 25 discrete actions for production control
- **Reward Structure**: Production optimization rewards

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Basic Usage
```python
import gymnasium as gym
from smart_manufacturing_env import SmartManufacturingEnv

# Create environment
env = SmartManufacturingEnv(render_mode="human")

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
python test_manufacturing.py
```

Tests 4 scenarios: Normal Production, High Demand, Frequent Breakdowns, and Quality Crisis.

## Requirements

- Python 3.7+
- gymnasium>=0.29.0
- numpy>=1.19.0
- pygame>=2.0.0
- matplotlib>=3.3.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025
