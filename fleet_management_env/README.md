# Fleet Management Environment

A comprehensive multi-agent reinforcement learning environment for urban delivery logistics simulation. This environment provides realistic fleet management challenges including multi-vehicle coordination, dynamic traffic patterns, fuel management, and time-sensitive deliveries.

![Fleet Management Environment](https://img.shields.io/badge/Environment-Fleet%20Management-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-orange)

## 🚚 Overview

The Fleet Management Environment simulates urban delivery logistics with:

- **Multi-Vehicle Fleet**: 3 different vehicle types (delivery van, motorcycle, cargo truck)
- **Dynamic City Grid**: 25x25 urban environment with distinct customer zones
- **Realistic Constraints**: Fuel management, traffic congestion, delivery time windows
- **Complex Objectives**: Multi-objective optimization balancing time, fuel, and customer satisfaction
- **Rich Visualization**: Real-time pygame rendering with comprehensive metrics

## 🎯 Key Features

### Environment Specifications
- **Grid Size**: 25x25 city representation
- **Vehicles**: 3 specialized vehicles with unique capabilities
- **Customer Zones**: 4 distinct areas (Residential, Commercial, Industrial, Hospital)
- **Delivery Requests**: 8-12 dynamic requests with varying urgency levels
- **Traffic System**: Dynamic congestion patterns changing every 50 timesteps
- **Fuel Stations**: 3 strategically placed refueling points

### Vehicle Types
| Vehicle | Speed | Capacity | Fuel Consumption | Range | Special Abilities |
|---------|-------|----------|------------------|-------|-------------------|
| Delivery Van | 1 | 3 packages | 1.0/move | 80 steps | General purpose |
| Motorcycle | 1 | 1 package | 0.5/move | 120 steps | Fast, hospital deliveries |
| Cargo Truck | 1 | 5 packages | 2.0/move | 60 steps | Heavy cargo, industrial |

### State Space (87 dimensions)
- Vehicle positions (3 × 2 = 6)
- Vehicle fuel levels (3)
- Vehicle cargo usage (3)
- Assigned deliveries (3)
- Delivery locations (12 × 2 = 24)
- Delivery urgency levels (12)
- Traffic congestion map (5 × 5 = 25)

### Action Space
Multi-discrete action space `[8, 8, 8]` for each vehicle:
- `0`: Stay in place
- `1-4`: Move (Up/Down/Left/Right)
- `5`: Pick up delivery
- `6`: Drop off delivery
- `7`: Refuel at station

### Reward Structure
- **+200**: Urgent delivery completed on time
- **+100**: Normal delivery completed
- **+50**: Standard delivery completed
- **+15**: Efficient routing bonus
- **+10**: Successful refueling
- **-2**: Timestep cost per active vehicle
- **-5**: Moving through heavy traffic
- **-10**: Invalid actions
- **-20**: Late delivery penalty (urgency multiplied)
- **-50**: Vehicle runs out of fuel

## 🚀 Installation

### From Source
```bash
git clone https://github.com/example/fleet_management_env.git
cd fleet_management_env
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

### Requirements
- Python 3.8+
- gymnasium>=0.26.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- pandas>=1.3.0

## 💻 Quick Start

### Basic Usage
```python
import gymnasium as gym
import fleet_management_env

# Create environment
env = gym.make('FleetManagement-v0', render_mode='human')

# Reset environment
observation, info = env.reset()

# Run episode
for step in range(1000):
    # Random actions for demonstration
    actions = env.action_space.sample()
    
    observation, reward, terminated, truncated, info = env.step(actions)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Custom Environment Creation
```python
from fleet_management_env import FleetManagementEnv

# Create environment with custom parameters
env = FleetManagementEnv(render_mode='human')

# Access environment properties
print(f"Grid size: {env.grid_size}")
print(f"Number of vehicles: {len(env.vehicles)}")
print(f"Fuel stations: {env.fuel_stations}")
```

## 🧪 Testing and Evaluation

### Run Comprehensive Test Suite
```bash
# Run all test scenarios
python -m fleet_management_env.test_fleet

# Or use console command
fleet-test
```

### Test Scenarios
1. **Light Traffic**: Standard delivery mix with minimal congestion
2. **Heavy Traffic**: High congestion testing fuel management
3. **Mixed Urgency**: Varied urgency levels testing prioritization

### Performance Metrics
- Delivery completion rate
- Fuel efficiency (deliveries per fuel unit)
- Customer satisfaction score
- Average delivery time
- Route optimization efficiency

### Example Test Results
```python
from fleet_management_env.test_fleet import FleetTestSuite

# Initialize test suite
test_suite = FleetTestSuite()

# Run specific scenario
results = test_suite.run_scenario_tests('heavy_traffic', num_episodes=10)

# Analyze results
df = test_suite.analyze_results(results)
print(df.groupby('agent')['delivery_rate'].mean())
```

## 🎮 Visualization

The environment provides rich pygame visualization including:

- **Vehicle Representation**: 
  - Van: Blue square
  - Motorcycle: Green circle
  - Truck: Red rectangle
- **Customer Zones**: Color-coded backgrounds
- **Delivery Points**: Diamond shapes with urgency color coding
- **Traffic Zones**: Semi-transparent overlays
- **Fuel Stations**: Gas pump icons
- **Real-time Metrics**: Fuel levels, deliveries, time remaining

### Visualization Controls
```python
# Enable visualization
env = FleetManagementEnv(render_mode='human')

# Disable visualization for training
env = FleetManagementEnv(render_mode=None)

# Get RGB array for custom processing
env = FleetManagementEnv(render_mode='rgb_array')
rgb_array = env.render()
```

## 🤖 Agent Development

### Random Agent Example
```python
import numpy as np

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        return self.action_space.sample()

# Use agent
agent = RandomAgent(env.action_space)
observation, _ = env.reset()

for step in range(100):
    action = agent.get_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Greedy Nearest-First Agent
```python
class GreedyAgent:
    def get_action(self, observation):
        # Parse observation
        vehicle_positions = self._extract_positions(observation)
        delivery_locations = self._extract_deliveries(observation)
        
        actions = []
        for i, pos in enumerate(vehicle_positions):
            # Find nearest delivery
            nearest_delivery = self._find_nearest_delivery(pos, delivery_locations)
            action = self._move_towards(pos, nearest_delivery)
            actions.append(action)
        
        return actions
```

## 📊 Advanced Features

### Dynamic Traffic Patterns
- Traffic intensity changes every 50 timesteps
- Commercial zones have higher congestion during business hours
- Weather effects modify fuel consumption every 100 timesteps

### Vehicle-Specific Constraints
- Hospital deliveries require motorcycles for speed
- Industrial deliveries often need trucks for capacity
- Time windows restrict delivery availability

### Multi-Objective Optimization
- Balance delivery speed vs fuel efficiency
- Optimize customer satisfaction vs operational costs
- Coordinate multiple vehicles to avoid conflicts

### Performance Tracking
```python
# Access episode statistics
env.step(actions)
stats = env.episode_stats

print(f"Deliveries completed: {stats['deliveries_completed']}")
print(f"Fuel consumed: {stats['fuel_consumed']}")
print(f"Total distance: {stats['total_distance']}")
print(f"Customer satisfaction: {stats['customer_satisfaction']}")
```

## 🔬 Research Applications

This environment is ideal for research in:

- **Multi-Agent Reinforcement Learning**: Coordinate multiple vehicles
- **Logistics Optimization**: Real-world delivery constraints
- **Resource Management**: Fuel and capacity limitations
- **Dynamic Environment Adaptation**: Changing traffic and weather
- **Hierarchical Planning**: Route planning with operational constraints

### Benchmark Comparisons
The environment includes baseline agents for comparison:
- Random policy
- Greedy nearest-first
- Custom heuristic strategies

## 📈 Performance Optimization

### Training Tips
1. **Curriculum Learning**: Start with light traffic, progress to complex scenarios
2. **Reward Shaping**: Balance immediate vs long-term objectives
3. **Multi-Objective**: Use Pareto optimization for competing goals
4. **Experience Replay**: Store successful delivery sequences

### Hyperparameter Suggestions
```python
# Recommended training configuration
config = {
    'learning_rate': 3e-4,
    'batch_size': 256,
    'gamma': 0.99,
    'epsilon_decay': 0.995,
    'target_update_freq': 1000,
    'replay_buffer_size': 100000
}
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Clone repository
git clone https://github.com/example/fleet_management_env.git
cd fleet_management_env

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black fleet_management_env/
flake8 fleet_management_env/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this environment in your research, please cite:

```bibtex
@misc{fleet_management_env,
  title={Fleet Management Environment: A Multi-Agent Reinforcement Learning Environment for Urban Logistics},
  author={Fleet Management Environment Team},
  year={2024},
  url={https://github.com/example/fleet_management_env}
}
```

## 🆘 Support

- **Documentation**: [Read the Docs](https://fleet-management-env.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/example/fleet_management_env/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/fleet_management_env/discussions)

## 🗺️ Roadmap

### Upcoming Features
- [ ] Multi-day scenarios with vehicle maintenance
- [ ] Real-world map integration (OpenStreetMap)
- [ ] Advanced weather effects and seasonal patterns
- [ ] Customer preference learning
- [ ] Fleet expansion/contraction dynamics
- [ ] Integration with popular RL libraries (Stable-Baselines3, Ray RLlib)

### Version History
- **v1.0.0**: Initial release with core functionality
- **v0.9.0**: Beta release with visualization
- **v0.8.0**: Alpha release with basic environment

---

**Happy Fleet Managing! 🚛📦**