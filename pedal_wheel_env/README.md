# Pedal Wheel Environment

A Gymnasium environment for unicycle simulation with pedal-controlled movement and physics-based balance mechanics.

## Features

- **Physics Simulation**: Realistic wheel dynamics and balance mechanics
- **Dual Pedal Control**: Continuous control over left and right pedal forces
- **Real-time Visualization**: Pygame-based 2D side-view rendering
- **Balance Challenge**: Maintain upright position while moving forward
- **Energy Management**: Efficient pedal usage optimization

## Environment Specifications

- **Action Space**: 2 continuous actions (left_pedal_force, right_pedal_force) ∈ [-1.0, 1.0]
- **Observation Space**: 5-dimensional state (position, velocity, tilt, angular velocities)
- **Reward Function**: Survival bonus + forward movement + energy efficiency
- **Episode Length**: 1000 timesteps or until fall

## Quick Start

### Basic Usage
```python
import gymnasium as gym
from pedal_wheel_env import PedalWheelEnv

# Create environment
env = PedalWheelEnv(render_mode="human")

# Reset and run
observation, info = env.reset()
for step in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### Testing
```bash
python test_env.py
```

### Manual Control
```bash
python pygame_visualizer.py
```

## Physics Mechanics

- **Pedal Forces**: Left and right pedal forces create torque
- **Balance System**: Tilt angle determines stability
- **Movement**: Forward velocity based on pedal force difference
- **Termination**: Episode ends if tilt exceeds threshold (fall)

## Project Structure

- `environment.py` - Main Gymnasium environment class
- `physics.py` - Wheel dynamics and physics calculations
- `config.py` - Environment constants and parameters
- `pygame_visualizer.py` - Real-time visualization
- `test_env.py` - Environment testing script

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.19.0
- pygame>=2.0.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025 