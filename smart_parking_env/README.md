# Smart Parking Lot RL Environment

A sophisticated reinforcement learning environment for smart parking lot management optimization. This environment simulates real-world parking challenges where an RL agent makes dynamic pricing and assignment decisions with advanced features.

## Project Structure

```
smart_parking_env/
├── core/                    # Core environment components
│   ├── __init__.py
│   ├── config.py           # Environment configuration
│   ├── parking_env.py      # Main Gymnasium environment
│   ├── customer.py         # Customer management system
│   ├── parking_lot.py      # Parking lot state management
│   └── pricing.py          # Dynamic pricing system
├── utils/                   # Utility modules
│   ├── __init__.py
│   └── visualization.py    # Real-time pygame visualization
├── examples/                # Example scripts
│   ├── __init__.py
│   ├── test_env.py         # Environment testing script
│   ├── training.py         # Ray RLlib training script
│   └── inference.py        # Policy inference script
├── __init__.py             # Main package initialization
├── setup.py                # Package installation
├── requirements.txt         # Dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
└── .gitignore             # Git ignore rules
```

## Project Overview

The environment simulates a 50-spot parking lot divided into three zones:
- **Zone A (Premium)**: 15 spots, closest to entrance, highest pricing ($8 base)
- **Zone B (Standard)**: 20 spots, middle area, moderate pricing ($5 base)  
- **Zone C (Economy)**: 15 spots, farthest area, budget pricing ($3 base)

### Key Features

- **Minute-based simulation**: 1 minute per timestep, 24 hours per episode (1440 timesteps)
- **Dynamic pricing**: Agent sets 3 price levels per zone based on demand
- **Realistic customer behavior**: Time-based preferences, urgency levels, impatience
- **Duration-based pricing**: Discounts for longer stays (15% for 3-5h, 25% for 6+h)
- **Queue management**: Limited queueing with customer impatience
- **Multi-objective optimization**: Balance revenue, satisfaction, utilization, urgency
- **Advanced visualization**: Real-time pygame graphics with animations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd smart_parking_env

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
from smart_parking_env import SmartParkingEnv

# Create environment
env = SmartParkingEnv()

# Reset environment
observation, info = env.reset()

# Take actions
for step in range(1440):  # 24-hour episode (1440 minutes)
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step}: Reward={reward:.2f}, Revenue=${info['total_revenue']:.2f}")
    
    if terminated or truncated:
        break

env.close()
```

### Training with Ray RLlib

```python
from ray import tune, train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from smart_parking_env import SmartParkingEnv

def env_creator(config):
    return SmartParkingEnv()

register_env("SmartParkingEnv-v0", env_creator)

config = (
    PPOConfig()
    .environment("SmartParkingEnv-v0")
    .training(lr=0.0003, train_batch_size=4000)
    .env_runners(num_env_runners=2, num_envs_per_env_runner=4)
)

results = tune.Tuner(
    config.algo_class,
    param_space=config,
    run_config=train.RunConfig(stop={"training_iteration": 50})
).fit()

print(results.get_best_result(metric="env_runners/episode_return_mean", mode="max"))
```

### Running Tests

```bash
# Run all tests (including visualization)
python test_env.py
```

## Environment Details

### Action Space

The agent controls:
- **Zone assignment**: 0=A, 1=B, 2=C, 3=reject
- **Price levels**: 0=low, 1=medium, 2=high for each zone

Action format: `[zone_assignment, price_A, price_B, price_C]`

### Observation Space

13-dimensional observation vector:
- Zone occupancy rates (3): [zone_A_occupancy, zone_B_occupancy, zone_C_occupancy]
- Zone prices (3): [zone_A_price, zone_B_price, zone_C_price] (normalized)
- Current time (2): [hour, minute] (normalized)
- Queue information (3): [queue_length, first_wait_time, total_wait_time] (normalized)
- Price management (2): [price_changes_this_hour, time_since_last_change] (normalized)

### Reward Function

Reward = Revenue + Satisfaction Bonus + Urgency Bonus - Rejection Penalty - Empty Spot Penalty - Impatience Penalty

- **Revenue**: Based on dynamic pricing and duration discounts
- **Satisfaction Bonus**: Customer satisfaction with assignment and pricing
- **Urgency Bonus**: Bonus for serving high-urgency customers
- **Rejection Penalty**: Penalty for rejecting customers
- **Empty Spot Penalty**: Penalty for unused spots
- **Impatience Penalty**: Penalty for customers leaving due to impatience

## Customer Behavior

**Customer Arrival Pattern:**
- **Morning Rush (6-9 AM)**: High arrival rates (8-15 customers/hour)
- **Lunch Time (12-2 PM)**: Medium arrival (12 customers/hour)
- **Evening Rush (5-7 PM)**: Peak arrival (15-18 customers/hour)
- **Night (10 PM-6 AM)**: Low arrival (2-6 customers/hour)

**Customer Types:**
- **Zone Preference**: 30% prefer Premium, 35% Standard, 15% Economy, 20% flexible
- **Duration Types**: Short (1-2h), Medium (3-5h), Long (6-12h)
- **Satisfaction System**: Based on zone preference match, pricing, wait time

## Dynamic Pricing System

**Price Levels:**
- **Low**: 30% discount (70% of base price)
- **Medium**: Base price (100%)
- **High**: 30% premium (130% of base price)

**Smart Pricing Rules:**
- Max 2 price changes per hour
- Minimum 15 minutes between changes
- Duration discounts: 15% for 3-5h, 25% for 6+h stays

## Visualization

The environment includes a real-time pygame visualization showing:
- **Live parking lot layout** with zones
- **Customer movements** (cars entering/exiting)
- **Current pricing** for each zone
- **Queue status** and wait times
- **Revenue tracking** and statistics
- **Time progression** (hour:minute)

**Controls:**
- **SPACE**: Pause/Resume
- **1,2,3,4**: Speed control (1x,2x,3x,5x)
- **ESC**: Quit

## Training Process

**RL Agent Training:**
- **Thousands of episodes** (24-hour cycles)
- **Trial and error** learning
- **Reward optimization** over time
- **Policy improvement** through experience

**Expected Learning Curve:**
- **Early episodes**: Random actions, low revenue
- **Mid training**: Basic patterns, improved revenue
- **Advanced training**: Sophisticated strategies, high revenue

## Running Trained Policies

After training, you can run inference with your trained policy using the included inference script.

### Using the Inference Script

The inference script loads your trained PPO policy and runs it with real-time pygame visualization:

```bash
# Run inference with pygame visualization
python examples/inference.py
```

**Features:**
- **Real-time Visualization**: Pygame window showing live parking lot
- **Policy Execution**: Your trained agent making decisions in real-time
- **Complete Episode**: Full 24-hour simulation (1440 timesteps)
- **Interactive Controls**: SPACE=pause, 1-4=speed, ESC=quit
- **Live Statistics**: Revenue, occupancy, customer flow tracking

### What You'll See

The visualization shows:
- **Live Parking Layout**: Three zones (A, B, C) with real car movements
- **Customer Flow**: Cars arriving, parking, and leaving based on policy decisions
- **Queue Management**: Customers waiting when lot is full
- **Dynamic Pricing**: Current prices for each zone
- **Statistics Panel**: Live revenue, occupancy, time progression
- **Policy Actions**: Your trained agent's actual decisions visualized

### Expected Performance

A well-trained policy typically achieves:
- **Revenue**: $3000-4000 per 24-hour episode
- **Occupancy Rate**: 80-100% utilization
- **Rejections**: 0-5 customers (excellent service)
- **Return**: 3000-4000 cumulative reward

### Policy Folder Structure

The inference script expects your trained policy in:
```
Trained_PPO_Policy/
└── PPO_SmartParkingEnv-v0_[experiment_id]_00000_0_[timestamp]/
    └── checkpoint_000000/
        ├── learner_group/
        │   └── learner/
        │       └── rl_module/
        │           └── default_policy/
        └── [other checkpoint files]
```

### Customization

To use a different policy, update the `checkpoint_path` in `examples/inference.py`:
```python
checkpoint_path = r"path/to/your/policy/checkpoint_000000"
```

## Success Metrics

**Environment tracks:**
- **Total Revenue**: Money earned
- **Customer Satisfaction**: Average satisfaction score
- **Utilization Rate**: How many spots were used
- **Rejection Rate**: How many customers turned away
- **Wait Times**: Average customer wait time

## Real-world Applications

**This environment simulates:**
- **Airport Parking**: High-value, time-sensitive customers
- **Shopping Mall Parking**: Mixed customer types
- **Office Building Parking**: Regular commuters
- **Event Venue Parking**: Peak demand management

**Key Challenges:**
- **Demand Forecasting**: Predicting customer arrivals
- **Revenue Optimization**: Balancing price vs occupancy
- **Customer Experience**: Minimizing wait times
- **Resource Allocation**: Efficient zone management

## Requirements

- Python 3.8+
- Ray RLlib
- Pygame (for visualization)
- NumPy
- Gymnasium

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Hasnain Fareed**  
Email: hasnainfarid7@yahoo.com

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project demonstrates advanced reinforcement learning concepts for real-world optimization problems in smart city applications. 