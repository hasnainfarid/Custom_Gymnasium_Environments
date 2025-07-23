

![RestaurantEnv](https://github.com/user-attachments/assets/7372a071-0fc2-45ab-b44b-636304d4380e)





# Restaurant Management Environment for Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Year](https://img.shields.io/badge/Year-2025-orange.svg)](https://github.com/hasnainfareed/restaurant_env_updated)

A comprehensive and sophisticated Gymnasium-based restaurant management simulation environment designed for reinforcement learning research, development, and education. This environment simulates a realistic restaurant operation with multiple tables, waiters, customers, and kitchen management systems.

## 🍽️ Overview

The Restaurant Management Environment provides a rich, multi-agent simulation of restaurant operations where an AI agent must efficiently manage:
- **10 Tables** with dynamic occupancy and cleanliness states
- **10 Waiters** with task assignment and completion tracking
- **Customer Flow** with arrival patterns, patience limits, and lifecycle management
- **Kitchen Operations** with order processing and food preparation
- **Resource Optimization** for maximum efficiency and customer satisfaction

This environment is perfect for research in:
- Multi-agent reinforcement learning
- Resource allocation and optimization
- Customer service simulation
- Operational efficiency studies
- Decision-making under uncertainty

## ✨ Key Features

### 🏪 Realistic Restaurant Simulation
- **Dynamic Customer Arrivals**: Three-tier arrival system with realistic timing patterns
- **Customer Lifecycle Management**: Complete journey from arrival to departure
- **Patience System**: Customers leave if not served within 20 timesteps
- **Table State Management**: Four distinct states (Free+Clean, Free+Dirty, Occupied+Clean, Occupied+Dirty)

### 👥 Multi-Agent System
- **Waiter Management**: 10 waiters with idle/busy states and task tracking
- **Task Assignment**: Three task types with different durations and requirements
- **Resource Coordination**: Intelligent allocation of waiters to tasks
- **Performance Tracking**: Real-time monitoring of waiter efficiency

### 🍳 Kitchen Operations
- **FIFO Order Processing**: Realistic kitchen queue management
- **Cooking Simulation**: 4-timestep cooking process for each order
- **Order Tracking**: Complete order lifecycle from placement to delivery
- **Ready Order Management**: Efficient pickup system for completed orders

### 📊 Advanced Analytics
- **Comprehensive Metrics**: Customer satisfaction, wait times, efficiency scores
- **Performance Visualization**: Real-time and post-episode analysis tools
- **Statistical Reporting**: Detailed episode statistics and performance reports
- **Debugging Tools**: Extensive logging and validation systems

## 🎯 Action Space

The environment supports 4 discrete actions for comprehensive restaurant management:

| Action | Description | Duration | Reward |
|--------|-------------|----------|---------|
| **0 - Seat Customer** | Assign waiter to seat customer at table | 2 timesteps | +2.0 |
| **1 - Serve Food** | Assign waiter to serve ready order | 1 timestep | +1.5 |
| **2 - Clean Table** | Assign waiter to clean dirty table | 3 timesteps | +1.0 |
| **3 - Do Nothing** | No operation (strategic waiting) | 0 timesteps | 0.0 |

### Action Dictionary Format
```python
{
    'type': int,        # Action type (0-3)
    'waiter_id': int,   # Waiter ID (0-9)
    'customer_id': int, # Customer index (0-49)
    'table_id': int     # Table ID (0-9)
}
```

## 📋 Observation Space

Rich observation dictionary providing complete restaurant state:

```python
{
    'waiting_customers': np.ndarray,  # Shape: (50, 2) - (customer_id_hash, wait_time)
    'waiter_status': np.ndarray,      # Shape: (10, 3) - (state, task_type, remaining_time)
    'table_occupancy': np.ndarray,    # Shape: (10,) - boolean occupancy
    'table_cleanliness': np.ndarray,  # Shape: (10,) - boolean cleanliness
    'kitchen_queue': np.ndarray,      # Shape: (50, 3) - (order_id_hash, table_id, progress)
    'ready_orders': np.ndarray,       # Shape: (20, 2) - (order_id_hash, table_id)
    'current_timestep': np.ndarray    # Shape: (1,) - current timestep
}
```

## 🏆 Reward System

### Positive Rewards
- **Successfully seat customer**: +2.0 points
- **Successfully serve food**: +1.5 points
- **Successfully clean table**: +1.0 points
- **Quick service bonus**: +0.5 points (if food served within 5 timesteps)

### Efficiency Bonuses
- **All tables clean**: +0.5 points per timestep
- **No customers waiting**: +0.3 points per timestep
- **Kitchen queue ≤ 2 orders**: +0.2 points per timestep

### Penalties
- **Customer leaves due to patience timeout**: -5.0 points
- **Attempt to seat at dirty table**: -1.5 points
- **Per timestep penalty**: -0.1 points

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hasnainfareed/restaurant_env_updated.git
cd restaurant_env_updated

# Install required dependencies
pip install gymnasium numpy pygame matplotlib pandas seaborn

# Verify installation
python test_env.py
```

### Basic Usage

```python
from restaurant_env import RestaurantEnv

# Create environment
env = RestaurantEnv()

# Reset environment
obs, info = env.reset()

# Run episode
for step in range(500):
    # Choose action (implement your agent logic here)
    action = {
        'type': 0,        # Seat customer
        'waiter_id': 0,   # Use waiter 0
        'customer_id': 0, # First waiting customer
        'table_id': 0     # Use table 0
    }
    
    # Execute step
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check if episode is done
    if terminated or truncated:
        break

env.close()
```

### Training Example

```python
import gymnasium as gym
from restaurant_env import RestaurantEnv

# Register environment
gym.register(
    id='Restaurant-v0',
    entry_point='restaurant_env:RestaurantEnv',
    max_episode_steps=500
)

# Create environment
env = gym.make('Restaurant-v0')

# Training loop
for episode in range(100):
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(500):
        # Your agent's action selection logic here
        action = env.action_space.sample()  # Random action for example
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

env.close()
```

## 🎨 Visualization

### Real-time Visualization

```python
from restaurant_env import RestaurantEnv
from visualization import RestaurantVisualization

# Create environment with visualization
env = RestaurantEnv(render_mode='human')
vis = RestaurantVisualization(env)

# Run episode with visualization
obs, info = env.reset()
vis.render(obs, info)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render current state
    vis.render(obs, info)
    
    if terminated or truncated:
        break

vis.close()
```

### Performance Analysis

```python
from utils import calculate_episode_statistics, create_performance_report

# Collect episode data
episode_data = []
obs, info = env.reset()

for step in range(500):
    action = your_agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_data.append({
        'timestep': step,
        'action': action,
        'reward': reward,
        'info': info
    })
    
    if terminated or truncated:
        break

# Analyze performance
stats = calculate_episode_statistics(episode_data)
report = create_performance_report(episode_data)
print(report)
```

## 📈 Performance Metrics

The environment tracks comprehensive performance indicators:

| Metric | Description | Target |
|--------|-------------|---------|
| **Customer Satisfaction Rate** | Ratio of served customers to total customers | > 0.85 |
| **Average Wait Time** | Mean time customers wait before being seated | < 10 timesteps |
| **Efficiency Score** | Overall restaurant efficiency (0-1 scale) | > 0.75 |
| **Table Utilization** | Percentage of time tables are occupied | > 0.60 |
| **Waiter Utilization** | Percentage of time waiters are busy | > 0.70 |
| **Kitchen Efficiency** | Average time orders spend in kitchen queue | < 6 timesteps |

## 🔧 Environment Parameters

### Customer Arrival Probabilities
- **Timesteps 1-150**: 0.12 probability per step (lunch rush)
- **Timesteps 151-350**: 0.20 probability per step (peak hours)
- **Timesteps 351-500**: 0.08 probability per step (dinner service)

### Task Durations
- **Seat Customer**: 2 timesteps
- **Serve Food**: 1 timestep
- **Clean Table**: 3 timesteps
- **Cook Order**: 4 timesteps
- **Customer Eating**: 10 timesteps

### Customer Patience
- **Maximum wait time**: 20 timesteps
- **Customers leave** if not seated within patience limit

## 🛠️ Customization

### Modifying Parameters

```python
class CustomRestaurantEnv(RestaurantEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        
        # Customize parameters
        self.num_tables = 15
        self.num_waiters = 12
        self.customer_patience = 25
        
        # Customize rewards
        self.rewards['seat_customer'] = 3.0
        self.rewards['serve_food'] = 2.0
```

### Adding New Actions

```python
# In __init__
self.action_space = spaces.Dict({
    'type': spaces.Discrete(5),  # Add one more action
    'waiter_id': spaces.Discrete(self.num_waiters),
    'customer_id': spaces.Discrete(50),
    'table_id': spaces.Discrete(self.num_tables)
})

# In step method
def _execute_action(self, action: dict):
    action_type = action.get('type')
    if action_type == 4:  # New action
        return self._action_new_task()
    # ... existing actions
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_env.py
```

The test suite covers:
- Environment initialization and reset
- All agent actions and validations
- Customer lifecycle and patience system
- Waiter task assignment and completion
- Kitchen order processing
- Reward calculation accuracy
- Observation and info structure validation

## 📊 Analysis Tools

### Episode Statistics

```python
from utils import calculate_episode_statistics

# Calculate comprehensive statistics
stats = calculate_episode_statistics(episode_data)
print(f"Customers Served: {stats['customers_served']}")
print(f"Average Wait Time: {stats['average_wait_time']:.2f}")
print(f"Efficiency Score: {stats['efficiency_score']:.3f}")
```

### Performance Visualization

```python
from utils import plot_episode_metrics

# Create performance plots
plot_episode_metrics(episode_data, save_path='episode_analysis.png')
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install gymnasium numpy pygame matplotlib pandas seaborn
   ```

2. **Visualization Issues**: Make sure pygame is properly installed
   ```bash
   pip install pygame
   ```

3. **Performance Issues**: The environment is optimized for efficiency, but complex agent policies may slow execution

### Debug Mode

Enable debug output for detailed logging:

```python
import os
os.environ['RESTAURANT_DEBUG'] = '1'
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow existing code structure** and naming conventions
3. **Add comprehensive tests** for new features
4. **Update documentation** for any API changes
5. **Ensure all tests pass** before submitting changes

### Development Setup

```bash
# Clone repository
git clone https://github.com/hasnainfareed/restaurant_env_updated.git
cd restaurant_env_updated

# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_env.py

# Run example
python example.py
```

## 📚 API Reference

### RestaurantEnv Class

#### Constructor
```python
RestaurantEnv(render_mode=None)
```

**Parameters:**
- `render_mode`: Optional rendering mode ('human' or 'rgb_array')

#### Methods

**reset(seed=None, options=None)**
- Resets environment to initial state
- Returns: (observation, info)

**step(action)**
- Executes one step in the environment
- Parameters: action (dict with type, waiter_id, customer_id, table_id)
- Returns: (observation, reward, terminated, truncated, info)

**render()**
- Renders current environment state
- Returns: rendering output based on render_mode

**close()**
- Cleans up environment resources

### Info Format

```python
{
    'total_reward': float,
    'current_timestep': int,
    'waiting_customers': int,
    'idle_waiters': int,
    'kitchen_queue_length': int,
    'ready_orders': int,
    'dirty_tables': int,
    'episode_stats': {
        'customers_served': int,
        'customers_left': int,
        'tables_cleaned': int,
        'orders_served': int,
        'total_wait_time': int,
        'average_wait_time': float
    },
    'average_wait_time': float
}
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**MIT License Features:**
- ✅ **Free to use** for commercial and non-commercial purposes
- ✅ **Free to modify** and distribute
- ✅ **Free to sell** and incorporate into proprietary software
- ✅ **No warranty** provided
- ✅ **Attribution** required

## 👨‍💻 Author

**Hasnain Fareed**
- **Email**: Hasnainfarid7@yahoo.com
- **GitHub**: [@hasnainfareed](https://github.com/hasnainfareed)
- **Year**: 2025

## 📖 Citation

If you use this environment in your research, please cite:

```bibtex
@misc{restaurant_env_2025,
  title={Restaurant Management Environment for Reinforcement Learning},
  author={Hasnain Fareed},
  year={2025},
  url={https://github.com/hasnainfareed/restaurant_env_updated},
  note={A comprehensive restaurant simulation environment for RL research}
}
```

## 🌟 Acknowledgments

- **Gymnasium Team** for the excellent RL environment framework
- **OpenAI Gym** community for inspiration and best practices
- **Reinforcement Learning Research Community** for continuous feedback and improvements

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/hasnainfareed/restaurant_env_updated/issues)
- **Email**: Hasnainfarid7@yahoo.com
- **Documentation**: [Wiki](https://github.com/hasnainfareed/restaurant_env_updated/wiki)

---

**⭐ Star this repository if you find it useful!**

**🔄 Fork and contribute to make it even better!** 
