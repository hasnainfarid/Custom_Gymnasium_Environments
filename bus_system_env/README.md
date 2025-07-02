# Bus System Environment

A custom OpenAI Gymnasium environment simulating an urban bus system with multiple buses operating on a circular route.

## Features

- **4 buses** operating on a circular route with **4 stops**
- **Dynamic passenger generation** (50-150 passengers per episode)
- **Centralized dwell time control** - agent manages bus waiting times
- **Realistic passenger boarding/alighting** with destination tracking
- **Pygame visualization** for demo purposes
- **Gymnasium-compatible** interface

## Installation

```bash
pip install gymnasium numpy pygame
```

## Quick Start

```python
from bus_system_env import BusSystemEnv

# Create environment
env = BusSystemEnv(enable_visualization=True, demo_speed=0.5)
obs, info = env.reset()

# Run simulation
for step in range(100):
    action = env.action_space.sample()  # Random agent
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Visualize if enabled
    
    if terminated or truncated:
        break

env.close()
```

## Environment Details

### Action Space
- **4 actions** (one per bus): dwell time at current stop (0-10 timesteps)

### Observation Space
- Bus positions, states, capacities, passenger destinations
- Stop waiting counts and destination distributions
- Global timestep and passenger statistics

### Reward System
- **+5.0** per passenger delivered
- **-1.0** per waiting passenger per timestep
- **-0.5** per onboard passenger per timestep

### Episode Termination
- Ends after 500 timesteps (configurable)

## Visualization

Enable Pygame visualization for demo mode:
```python
env = BusSystemEnv(enable_visualization=True, demo_speed=0.5)
```

**Controls:**
- **SPACE**: Pause/Resume
- **ESC**: Exit

## Testing

```bash
python bus_system_env/test_env.py
```

## Author

**Hasnain Fareed**  
Email: Hasnainfarid7@yahoo.com  
GitHub: [@hasnainfarid](https://github.com/hasnainfarid)

## License

MIT License - See [LICENSE](LICENSE) file for details. 