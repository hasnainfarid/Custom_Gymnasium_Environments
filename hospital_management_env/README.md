
<img width="1240" height="1002" alt="Hospital_00" src="https://github.com/user-attachments/assets/afd48e39-05b8-4e94-9364-a437201ce20c" />

# Hospital Management Environment

A Gymnasium environment for hospital operations simulation with patient management, resource allocation, and emergency response capabilities.

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
- **Reward Structure**: Patient outcome and efficiency rewards

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Basic Usage
```python
import gymnasium as gym
from hospital_management_env import HospitalManagementEnv

# Create environment
env = HospitalManagementEnv(render_mode="human")

# Reset and run
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

### Testing
```bash
python test_hospital.py
```

Tests various hospital scenarios: Normal Operations, Emergency Response, Resource Shortages, and Mass Casualty Events.

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025
