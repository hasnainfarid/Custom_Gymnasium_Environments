# Smart Manufacturing Environment

A comprehensive reinforcement learning environment that simulates a realistic smart manufacturing facility with production stations, quality control, and predictive maintenance capabilities. All components are self-contained within the package directory.

## Features

### 🏭 Factory Floor Simulation
- **30x15 grid factory layout** with 5 production stations in sequence
- **6 product types** (A-F) with varying complexity and quality requirements
- **4 machine types**: Cutting, Assembly, Testing, and Packaging
- **Real-time production flow** from raw materials to finished goods

### 🎮 Environment Specifications

#### State Space (73 elements)
- Products in system by type and station (30 elements)
- Machine status indicators (15 elements)
- Queue lengths at each station (5 elements)
- Product quality scores (6 elements)
- Raw material inventory level (1 element)
- Production targets remaining (6 elements)
- Machine utilization rates (5 elements)
- Maintenance schedule countdowns (5 elements)

#### Action Space (25 discrete actions)
- **0-5**: Start production of products A-F
- **6-10**: Prioritize stations 1-5 (speed boost)
- **11-15**: Schedule maintenance for stations 1-5
- **16-20**: Adjust quality control thresholds
- **21**: Emergency stop all production
- **22**: Rush order mode (↑speed, ↓quality)
- **23**: Quality focus mode (↓speed, ↑quality)
- **24**: Balanced production mode

#### Reward Structure
- **+500**: Product completed with quality >90%
- **+300**: Product completed with quality 70-90%
- **+100**: Product completed with quality 50-70%
- **-100**: Product failed quality control
- **-200**: Machine breakdown during production
- **-10**: Each timestep with idle machines
- **-50**: Raw material stockout
- **+50**: Preventive maintenance completed
- **-500**: Missing production target deadline
- **+100**: Achieving daily production quota

### 🔧 Advanced Features
- **Machine Degradation**: Performance decreases with usage
- **Predictive Maintenance**: Early warning system for failures
- **Supply Chain Disruptions**: Random material delays
- **Quality Control**: 3 checkpoints with adjustable thresholds
- **Energy Tracking**: Monitor and optimize energy consumption
- **Worker Shifts**: Efficiency varies by shift period
- **OEE Metrics**: Real-time tracking of Availability, Performance, and Quality

### 📊 Visualization
- **1200x600 pygame window** with real-time factory floor view
- Color-coded machine status indicators
- Product flow animation
- Live production dashboard
- Quality checkpoint meters
- Alert system for critical events

## Installation

### From Source
```bash
# Clone or download the package
cd smart_manufacturing_env

# Install the package
pip install -e .
```

### Dependencies
```bash
pip install gym numpy pygame matplotlib
```

## Quick Start

### Basic Usage
```python
import gym
from smart_manufacturing_env import SmartManufacturingEnv

# Create environment
env = SmartManufacturingEnv()

# Reset environment
obs = env.reset()

# Run simulation
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    env.render()  # Visualize
    
    if done:
        break

env.close()
```

### Using with Gym
```python
import gym

# Register and create environment
gym.register(
    id='SmartManufacturing-v0',
    entry_point='smart_manufacturing_env:SmartManufacturingEnv',
)

env = gym.make('SmartManufacturing-v0')
```

### Running Test Suite
```bash
# Run comprehensive test suite
python test_manufacturing.py

# Or use the console script
test-manufacturing
```

## Testing Scenarios

The package includes 4 pre-configured test scenarios:

### 1. Normal Production
Standard factory operations with balanced demand and regular maintenance requirements.

### 2. High Demand
Doubled production targets testing the system's capacity and efficiency under pressure.

### 3. Frequent Breakdowns
Increased machine failure rates testing maintenance strategies and resilience.

### 4. Quality Crisis
Stricter quality requirements with degraded machine performance testing quality control systems.

## Maintenance Strategies

The test suite compares two maintenance strategies:

### Reactive Maintenance
- Fix machines only when they break
- Lower upfront costs but higher downtime
- Suitable for non-critical operations

### Predictive Maintenance
- Schedule maintenance before failures
- Higher upfront costs but better availability
- Optimal for critical production lines

## Product Specifications

| Product | Stations Required | Timesteps | Quality Target |
|---------|------------------|-----------|----------------|
| A       | 1                | 10        | 80%            |
| B       | 2                | 15        | 85%            |
| C       | 3                | 20        | 90%            |
| D       | 4                | 25        | 85%            |
| E       | 5                | 30        | 95%            |
| F       | 3                | 18        | 75%            |

## API Reference

### Environment Methods

#### `reset() -> observation`
Reset the environment to initial state.

#### `step(action) -> (observation, reward, done, info)`
Execute one timestep with the given action.

#### `render(mode='human')`
Render the environment. Modes: 'human', 'rgb_array'.

#### `close()`
Clean up resources and close the environment.

### Info Dictionary
The `info` dictionary returned by `step()` contains:
- `timestep`: Current simulation timestep
- `total_reward`: Cumulative reward
- `products_completed`: Dictionary of completed products by type
- `oee`: OEE metrics (availability, performance, quality)
- `energy_consumption`: Total energy consumed

## Example: Custom Agent

```python
import numpy as np
from smart_manufacturing_env import SmartManufacturingEnv

class SmartAgent:
    def __init__(self):
        self.maintenance_threshold = 15
    
    def get_action(self, observation):
        # Parse observation
        maintenance_countdown = observation[60:65]
        targets_remaining = observation[56:62]
        
        # Predictive maintenance
        for i, countdown in enumerate(maintenance_countdown):
            if countdown < self.maintenance_threshold:
                return 11 + i  # Schedule maintenance
        
        # Production planning
        max_target_idx = np.argmax(targets_remaining)
        if targets_remaining[max_target_idx] > 0:
            return max_target_idx  # Start production
        
        return 24  # Balanced mode

# Use the agent
env = SmartManufacturingEnv()
agent = SmartAgent()

obs = env.reset()
for _ in range(1500):
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
    
    if done:
        print(f"Episode complete! Total reward: {info['total_reward']}")
        break
```

## Training RL Agents

The environment is compatible with popular RL libraries:

### Stable Baselines3
```python
from stable_baselines3 import PPO
from smart_manufacturing_env import SmartManufacturingEnv

env = SmartManufacturingEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Ray RLlib
```python
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

ray.init()
tune.run(
    PPOTrainer,
    config={
        "env": SmartManufacturingEnv,
        "num_workers": 4,
        "framework": "torch",
    }
)
```

## Performance Metrics

The environment tracks several key performance indicators:

### Overall Equipment Effectiveness (OEE)
- **Availability**: Percentage of time machines are operational
- **Performance**: Actual vs. theoretical production rate
- **Quality**: Percentage of products meeting quality standards

### Production Metrics
- Products completed per timestep
- Quality rate (% of products passing QC)
- Energy consumption per product
- Average machine utilization

### Maintenance Metrics
- Mean time between failures (MTBF)
- Mean time to repair (MTTR)
- Breakdown frequency
- Preventive maintenance effectiveness

## Troubleshooting

### Common Issues

**ImportError**: Make sure all dependencies are installed:
```bash
pip install gym numpy pygame matplotlib
```

**Pygame window not responding**: The render loop needs regular updates:
```python
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        break
```

**Low FPS in visualization**: Reduce render frequency:
```python
if step % 10 == 0:  # Render every 10 steps
    env.render()
```

## Contributing

This is a self-contained package designed for educational and research purposes. Feel free to modify and extend it for your needs. All components are contained within the package directory for easy customization.

## License

MIT License - See LICENSE file for details.

## Citation

If you use this environment in your research, please cite:
```bibtex
@software{smart_manufacturing_env,
  title = {Smart Manufacturing Environment},
  author = {Manufacturing AI Lab},
  year = {2024},
  version = {1.0.0},
  description = {A comprehensive RL environment for smart manufacturing}
}
```

## Acknowledgments

This environment was designed to provide a realistic yet accessible simulation of modern manufacturing challenges, incorporating industry best practices for production optimization, quality control, and predictive maintenance.

---

**Note**: All components, data, and resources are self-contained within the package directory. No external files or dependencies are required beyond the standard Python packages listed in requirements.