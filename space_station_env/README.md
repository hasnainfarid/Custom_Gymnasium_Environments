# Space Station Life Support Management Environment

A comprehensive Gymnasium environment for simulating space station life support management with realistic orbital mechanics, system interdependencies, and emergency scenarios.

## Features

### 🚀 Realistic Space Station Simulation
- **8 Interconnected Modules**: Command, Living Quarters, Laboratory, Engineering, Hydroponics, Storage, Docking, Solar Array Control
- **6 Crew Members**: Commander, Engineer, Scientist, Medical Officer, Pilot, Mission Specialist
- **12 Critical Systems**: Complete life support with realistic interdependencies
- **Orbital Mechanics**: 90-minute orbits with day/night cycles affecting solar power
- **Resource Management**: Water, food, oxygen, power, medical supplies, and more

### 🎮 Rich Visualization
- **1200x900 Pygame Display** with real-time station cross-section
- **System Status Dashboard**: Live efficiency meters and alerts
- **Orbital Display**: Earth, sun position, and station orbit visualization
- **Resource Gauges**: Circular meters for critical supplies
- **Emergency Alerts**: Visual warnings for critical situations

### 🤖 Gymnasium Integration
- **State Space**: 110-element observation vector
- **Action Space**: 40 discrete actions for system control
- **Reward Structure**: Detailed rewards/penalties for mission outcomes
- **Episode Termination**: Multiple success/failure conditions

## Installation

```bash
# Clone or download the package
cd space_station_env

# Install the package
pip install -e .

# Or install dependencies directly
pip install gymnasium numpy pygame
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
from space_station_env import SpaceStationEnv

# Create environment
env = SpaceStationEnv(render_mode="human")

# Reset environment
observation, info = env.reset()

# Run simulation loop
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
    
    env.render()

env.close()
```

### Using Gymnasium Registration

```python
import gymnasium as gym

# Environment is automatically registered on import
env = gym.make('SpaceStation-v0', render_mode="human")

observation, info = env.reset()
# ... continue as above
```

## Testing Scenarios

The package includes comprehensive testing with 6 emergency scenarios:

### Run All Scenarios with Automated Controller
```bash
python test_station.py --scenario all --controller auto
```

### Test Specific Scenario with Manual Control
```bash
python test_station.py --scenario power_failure --controller manual
```

### Available Scenarios
1. **normal**: Baseline operations
2. **power_failure**: Major power grid failure
3. **atmospheric_leak**: Hull breach and depressurization
4. **medical_emergency**: Crew member injury
5. **solar_storm**: Intense radiation event
6. **cascade_failure**: Multiple system failures

### Manual Control Keys
- **1-9, 0**: Prioritize systems 1-10
- **F1-F12**: Schedule maintenance for systems
- **E**: Activate emergency protocols
- **B**: Switch to backup power
- **V**: Initiate evacuation
- **N**: Normal operations

### Command Line Options
```bash
python test_station.py [options]

Options:
  --scenario {all,normal,power_failure,...}  Scenario to test (default: all)
  --controller {auto,manual,both}            Controller type (default: auto)
  --steps INT                                Max steps per scenario (default: 500)
  --no-render                                Disable visualization
  --save-report FILE                         Save report to file (default: mission_report.txt)
```

## State Space (110 elements)

1. **Crew States** (24): Position, health, oxygen for 6 crew members
2. **System Status** (36): Efficiency, power draw, maintenance for 12 systems
3. **Module Atmospherics** (32): O2, CO2, pressure, temperature for 8 modules
4. **Power Systems** (10): Solar efficiency, battery, consumption metrics
5. **Resources** (8): Water, food, oxygen tanks, nitrogen, supplies, etc.
6. **Orbital Mechanics** (4): Phase, solar angle, shadow, radiation
7. **Emergency Alerts** (6): Fire, depressurization, radiation, etc.

## Action Space (40 discrete actions)

- **0-11**: Prioritize power to systems
- **12-23**: Schedule maintenance for systems
- **24-29**: Assign crew to emergency tasks
- **30-35**: Adjust module environmental controls
- **36**: Activate emergency protocols
- **37**: Switch to backup power
- **38**: Initiate evacuation procedures
- **39**: Normal operations mode

## Reward Structure

### Positive Rewards
- **+2000**: Successfully handle life-threatening emergency
- **+1000**: Prevent system cascade failure
- **+500**: Maintain optimal life support for 24 hours
- **+200**: Efficient resource management (>90% efficiency)
- **+100**: Successful crew task completion

### Penalties
- **-5000**: Crew member death
- **-2000**: Critical system failure (oxygen, pressure)
- **-1000**: Emergency evacuation required
- **-500**: Crew member injured
- **-100**: Resource shortage warning

## System Interdependencies

The environment simulates realistic system dependencies:

```
Power Grid → All Systems
Water Recycling → Oxygen Generation
Waste Processing → Water Recycling
Nitrogen Supply → Atmospheric Pressure
Atmospheric Pressure → Fire Suppression
```

## Advanced Features

### Orbital Mechanics
- Solar panel efficiency varies with orbital position
- Earth shadow periods every 45 minutes
- Radiation exposure varies with orbital location
- Communication blackouts during certain positions

### System Degradation
- Components wear out over time
- Maintenance requirements increase
- Random failure probability
- Cascade failures for interdependent systems

### Crew Dynamics
- Fatigue accumulation affects performance
- Role-specific skills and efficiency bonuses
- Health affected by environmental conditions
- Task assignment and emergency response

### Resource Management
- Water recycling with 95% efficiency
- CO2 scrubbing and oxygen generation
- Power consumption and generation balance
- Emergency resource reserves

## Custom Controllers

### Implementing Your Own Controller

```python
class MyController:
    def get_action(self, observation, info):
        # Parse observation
        crew_health = observation[2::4][:6]
        system_efficiency = observation[24::3][:12]
        battery_level = observation[89]
        
        # Your control logic here
        if battery_level < 30:
            return 37  # Backup power
        elif min(crew_health) < 50:
            return 27  # Medical emergency
        else:
            return 39  # Normal operations
```

### Using with Training

```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('SpaceStation-v0')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Mission Reports

The test script generates comprehensive mission reports:

```
SPACE STATION MISSION REPORT
Generated: 2024-01-01 12:00:00
================================================================================

Scenario: POWER_FAILURE
----------------------------------------
Controller: auto
Total Reward: 45230.50
Steps Survived: 500
Days Survived: 1
Crew Casualties: 0
System Failures: 1
  Failed Systems: Power Grid
Min Crew Health: 78.3%
Avg Crew Health: 92.1%
Avg Battery Level: 35.2%
Emergency Resolved: True
Mission Success: False
Performance Rating: EXCELLENT (90/100)
```

## Package Structure

```
space_station_env/
├── __init__.py          # Package initialization and registration
├── station_env.py       # Main environment class
├── test_station.py      # Testing and visualization script
├── setup.py            # Package installation
├── README.md           # This file
└── systems/
    └── __init__.py     # Internal system data and configurations
```

## Requirements

- Python >= 3.8
- gymnasium >= 0.29.0
- numpy >= 1.24.0
- pygame >= 2.5.0

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional emergency scenarios
- More sophisticated AI controllers
- Enhanced visualization features
- Multiplayer support
- VR integration
- Real mission data integration

## Acknowledgments

Inspired by real space station operations and life support systems. All system parameters and interdependencies are based on publicly available information about orbital habitats and life support engineering.

---

**Note**: This is a simulation for research and educational purposes. While based on realistic principles, it simplifies many aspects of actual space station operations.