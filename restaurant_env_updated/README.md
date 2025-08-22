

![RestaurantEnv](https://github.com/user-attachments/assets/7372a071-0fc2-45ab-b44b-636304d4380e)

# Restaurant Management Environment

A Gymnasium environment for restaurant operations simulation with multi-agent management, customer service, and kitchen operations optimization.

## Features

- **Multi-Agent System**: 10 tables, 10 waiters, dynamic customer flow
- **Realistic Operations**: Customer arrivals, table management, kitchen processing
- **Resource Optimization**: Efficient waiter allocation and task management
- **Real-time Visualization**: Pygame-based restaurant layout display
- **Performance Analytics**: Customer satisfaction, wait times, efficiency metrics

## Environment Specifications

- **Action Space**: 4 discrete actions (seat customer, serve food, clean table, do nothing)
- **Observation Space**: Multi-dimensional state including customer queues, waiter status, table states
- **Reward System**: Task completion rewards with efficiency bonuses
- **Time Scale**: Configurable timesteps with realistic task durations

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from restaurant_env import RestaurantEnv

# Create environment
env = RestaurantEnv()

# Reset and run
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### Training
```bash
python training.py
```

### Testing
```bash
python test_env.py
```

### Visualization
```bash
python visualization.py
```

## Action Types

- **0 - Seat Customer**: Assign waiter to seat customer (2 timesteps, +2.0 reward)
- **1 - Serve Food**: Assign waiter to serve ready order (1 timestep, +1.5 reward)
- **2 - Clean Table**: Assign waiter to clean dirty table (3 timesteps, +1.0 reward)
- **3 - Do Nothing**: Strategic waiting (0 timesteps, 0.0 reward)

## Project Structure

- `restaurant_env.py` - Core environment implementation
- `entities.py` - Customer, waiter, and table classes
- `utils.py` - Utility functions and helpers
- `visualization.py` - Pygame-based rendering
- `training.py` - Training script
- `test_env.py` - Environment testing

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.19.0
- pygame>=2.0.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025 
