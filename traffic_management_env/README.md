# Traffic Management Environment

A custom Gymnasium environment for traffic management simulation using reinforcement learning.

## Overview

This environment simulates a traffic network with multiple intersections where an RL agent can control traffic lights to optimize traffic flow. The environment features:

- **Grid-based traffic network**: Configurable grid of intersections with traffic lights
- **Dynamic vehicle generation**: Vehicles spawn randomly and follow routes through the network
- **Realistic traffic simulation**: Vehicles queue at intersections and respond to traffic light states
- **Rich observation space**: Includes traffic light states, queue lengths, waiting times, and flow metrics
- **Flexible action space**: Control traffic light phases at each intersection
- **Comprehensive reward system**: Balances throughput, waiting time, and traffic flow smoothness
- **Visualization**: Real-time rendering of the traffic simulation

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import gymnasium as gym
import traffic_management_env

# Create the environment
env = gym.make('TrafficManagement-v0')

# Reset the environment
observation, info = env.reset()

# Run a simple episode
for step in range(1000):
    # Random action for demonstration
    action = env.action_space.sample()
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment (optional)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

## Environment Details

### Action Space

The action space is `MultiDiscrete([3] * num_intersections)`, where each intersection can be controlled with:
- `0`: Maintain current traffic light phase
- `1`: Switch to North-South green light
- `2`: Switch to East-West green light

### Observation Space

The observation is a flattened array containing:
- Traffic light states for each intersection (one-hot encoded)
- Queue lengths in each direction for each intersection
- Average waiting times for each intersection
- Number of vehicles passed and total waiting time per intersection
- Global metrics: total vehicles, average waiting time, average queue length, throughput

### Reward Function

The reward function balances multiple objectives:
- **+1.0** for each vehicle that passes through an intersection
- **-0.1** penalty per timestep per waiting vehicle
- **-0.05** penalty per vehicle in queue
- **+0.5** bonus for maintaining smooth traffic flow (low queue variance)
- **-5.0** penalty for emergency stops (future enhancement)

### Configuration

Key environment parameters can be configured:

```python
env = gym.make('TrafficManagement-v0', 
               grid_size=(4, 4),           # 4x4 grid of intersections
               num_intersections=12,       # Number of controlled intersections
               max_vehicles=30,            # Maximum vehicles in simulation
               spawn_rate=0.2)             # Vehicle spawn probability per timestep
```

## Environment Components

### Traffic Network
- **Grid Layout**: Intersections arranged in a configurable grid
- **Realistic Spacing**: Intersections spaced 100 meters apart
- **Route Generation**: Vehicles follow random multi-intersection routes

### Vehicle Simulation
- **Dynamic Spawning**: Vehicles spawn at random intersections with configurable probability
- **Realistic Movement**: Vehicles travel between intersections at realistic speeds
- **Queue Management**: Vehicles queue at intersections when lights are red
- **Route Following**: Vehicles follow generated routes through the network

### Traffic Light Control
- **Four-Phase System**: North-South Green/Yellow, East-West Green/Yellow
- **Configurable Timing**: Minimum and maximum phase durations
- **RL Control**: Agent can switch between green phases for optimal flow

### Metrics and Monitoring
- **Real-time Metrics**: Track throughput, waiting times, queue lengths
- **Performance Analysis**: Calculate traffic efficiency metrics
- **Visualization**: Real-time rendering of traffic simulation

## Advanced Usage

### Custom Environment Configuration

```python
from traffic_management_env.environment import TrafficManagementEnv

# Create environment with custom parameters
env = TrafficManagementEnv(
    grid_size=(6, 6),
    num_intersections=20,
    max_vehicles=100,
    spawn_rate=0.4,
    render_mode="human"
)
```

### Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
import gymnasium as gym
import traffic_management_env

# Create environment
env = gym.make('TrafficManagement-v0')

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test trained agent
obs, _ = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
```

## Visualization

The environment supports real-time visualization with:
- **Intersection Display**: Traffic lights with color-coded phases
- **Vehicle Rendering**: Moving vehicles with different types
- **Queue Visualization**: Queue lengths displayed at intersections
- **Metrics Display**: Real-time performance metrics
- **Configurable Rendering**: Human viewing or RGB array output

## Performance Metrics

The environment tracks comprehensive performance metrics:
- **Throughput**: Vehicles passed per intersection
- **Waiting Time**: Average and total vehicle waiting times
- **Queue Lengths**: Current and average queue lengths
- **Flow Efficiency**: Traffic flow smoothness measures
- **Episode Statistics**: Cumulative performance over episodes

## Contributing

This environment is designed to be extensible. Key areas for enhancement:
- Additional vehicle types (emergency vehicles, public transport)
- More complex intersection layouts
- Dynamic traffic patterns
- Weather and time-of-day effects
- Integration with real traffic data

## License

MIT License - see LICENSE file for details.