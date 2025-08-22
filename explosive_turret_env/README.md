


![Explosive_Turret_GIF](https://github.com/user-attachments/assets/cceadaad-50df-4a2b-87ef-af660912b84c)

# Explosive Turret Environment

A Gymnasium environment for turret simulation with Box2D physics, explosive shells, and destructible targets in a castle setting.

## Features

- **Realistic Physics**: Box2D-based projectile physics simulation
- **Destructible Targets**: Castle walls that break into pieces on impact
- **Explosive Effects**: Particle effects and screen shake on impact
- **Military Theme**: Castle background with military turret
- **Real-time Visualization**: Pygame-based rendering with performance metrics

## Environment Specifications

- **Action Space**: 2 continuous actions (angle [0, π/2], force [0, 100])
- **Observation Space**: Shell position, velocity, target position, destruction status
- **Reward System**: Impact velocity + destruction + bonus rewards
- **Physics Engine**: Box2D for accurate projectile simulation

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from explosive_turret_env import TurretEnv

# Create environment
env = TurretEnv(render_mode="human")

# Reset and run
observation, info = env.reset()
action = [0.5, 50.0]  # 45 degrees, 50N force
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Gymnasium Integration
```python
import gymnasium as gym

# Create environment
env = gym.make("ExplosiveTurret-v1", render_mode="human")

# Reset and run
observation, info = env.reset()
action = [0.5, 50.0]  # 45 degrees, 50N force
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Testing
```bash
python test_env.py
```

## Action Space

- **angle**: Firing angle in radians [0, π/2]
- **force**: Force applied to shell in Newtons [0, 100]

## Reward System

1. **Impact Velocity Reward**: Base reward based on shell velocity at impact
2. **Destruction Reward**: Continuous reward for 2 seconds after explosion
3. **Bonus Rewards**: One-time bonuses for massive destruction

## Project Structure

- `explosive_turret_env/` - Core environment package
- `test_env.py` - Environment testing script
- `example.py` - Usage examples
- `requirements.txt` - Dependencies

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- pygame>=2.0.0
- box2d-py>=2.3.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025 
