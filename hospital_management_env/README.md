# Hospital Management Environment

A comprehensive reinforcement learning environment that simulates realistic hospital operations with patient management, resource allocation, and emergency response capabilities.

## Features

- **6 Specialized Departments**: Emergency, ICU, Surgery, Pediatrics, Cardiology, and General Medicine
- **Dynamic Patient Flow**: 5 severity levels with realistic medical scenarios
- **Resource Management**: Staff, beds, equipment, and medicine allocation
- **Real-time Visualization**: Pygame-based hospital layout display
- **Emergency Events**: Outbreaks, mass casualties, and critical situations
- **Staff Management**: Doctor and nurse scheduling with fatigue modeling
- **Patient Outcomes**: Life-critical decision making with realistic consequences

## Environment Specifications

- **Observation Space**: 295-dimensional state representation
- **Action Space**: 35 discrete actions for hospital management
- **Reward Structure**: Comprehensive patient outcome and efficiency rewards

## Installation

```bash
# Clone the repository
git clone https://github.com/hasnainfareed/hospital-management-env.git
cd hospital-management-env

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from hospital_management_env import HospitalManagementEnv

# Create environment
env = HospitalManagementEnv(render_mode="human")

# Reset environment
obs, info = env.reset()

# Run simulation
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Visualize
    
    if terminated or truncated:
        break

env.close()
```

## Testing

Run the comprehensive test suite with visualization:

```bash
python test_hospital.py
```

This will test various hospital scenarios including:
- Normal Operations
- Emergency Response
- Resource Shortages
- Mass Casualty Events

## Dependencies

- gymnasium>=0.28.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Hasnain Fareed** - 2025