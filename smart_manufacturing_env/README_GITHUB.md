# Smart Manufacturing Environment

A comprehensive reinforcement learning environment that simulates a realistic smart manufacturing facility with production stations, quality control, and predictive maintenance capabilities.

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
- **Reward Structure**: Comprehensive production optimization rewards

## Installation

```bash
# Clone the repository
git clone https://github.com/hasnainfareed/smart_manufacturing_env.git
cd smart_manufacturing_env

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from smart_manufacturing_env import SmartManufacturingEnv

# Create environment
env = SmartManufacturingEnv(render_mode="human")

# Reset environment
obs, info = env.reset()

# Run simulation
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Visualize
    
    if terminated or truncated:
        break

env.close()
```

## Testing

Run the comprehensive test suite with visualization:

```bash
python test_manufacturing.py
```

This will test 4 scenarios:
- Normal Production
- High Demand  
- Frequent Breakdowns
- Quality Crisis

## Dependencies

- gymnasium>=0.29.0
- numpy>=1.19.0
- pygame>=2.0.0
- matplotlib>=3.3.0

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Hasnain Fareed** - 2025
