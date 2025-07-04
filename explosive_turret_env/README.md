# Explosive Turret Environment

A realistic Box2D + Pygame environment where a turret shoots explosive shells at destructible targets with castle/fortress background.

## Features

- **Realistic Physics**: Uses Box2D for accurate projectile physics
- **Destructible Targets**: Castle walls that break into pieces on impact
- **Explosive Effects**: Particle effects and screen shake on impact
- **Military Theme**: Castle background with military turret
- **Advanced Rewards**: Multi-component reward system based on impact velocity and destruction
- **Visual Feedback**: Real-time UI showing impact data and performance metrics

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Method 1: Direct Import
```python
from explosive_turret_env import TurretEnv

# Create environment
env = TurretEnv(render_mode="human")

# Reset environment
observation, info = env.reset()

# Take action (angle, force)
action = [0.5, 50.0]  # 45 degrees, 50N force
observation, reward, done, truncated, info = env.step(action)

env.close()
```

### Method 2: Using Gymnasium
```python
import gymnasium as gym

# Create environment using Gymnasium
env = gym.make("ExplosiveTurret-v1", render_mode="human")

# Reset environment
observation, info = env.reset()

# Take action (angle, force)
action = [0.5, 50.0]  # 45 degrees, 50N force
observation, reward, done, truncated, info = env.step(action)

env.close()
```

## Environment Details

### Action Space
- `angle`: float [0, π/2] - firing angle in radians
- `force`: float [0, 100] - force applied to shell in Newtons

### Observation Space
- Shell position (x, y)
- Shell velocity (vx, vy)
- Target position (x, y)
- Target destroyed flag

### Reward System
1. **Impact Velocity Reward**: Base reward based on shell velocity at impact
2. **Destruction Reward**: Continuous reward for 2 seconds after explosion
3. **Bonus Rewards**: One-time bonuses for massive destruction

## Testing

Run the test script to see the environment in action:

```bash
python test_env.py
```

## Gymnasium Integration

This environment is automatically registered with Gymnasium as `"ExplosiveTurret-v1"`. You can use it with any Gymnasium-compatible RL library:

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("ExplosiveTurret-v1")

# Train with Stable-Baselines3
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Requirements

- Python 3.8+
- Gymnasium
- Pygame
- Box2D
- NumPy

## License

MIT License 