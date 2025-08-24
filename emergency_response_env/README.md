

<img width="970" height="675" alt="image" src="https://github.com/user-attachments/assets/8d6d4af2-33c0-44ec-97ce-34d5d71f9fc9" />


# Emergency Response Environment 🚨

A comprehensive emergency response coordination environment for reinforcement learning that simulates realistic disaster scenarios with multiple emergency services, complex resource allocation, and time-critical decision making.

## Features

- **30x30 metropolitan grid** with residential, commercial, and industrial zones
- **40+ emergency response units** including fire trucks, police cars, ambulances, search & rescue teams, and hazmat units
- **8 emergency types** covering building fires, accidents, chemical spills, floods, earthquakes, gas leaks, medical emergencies, and civil unrest
- **Complex state space** (276 dimensions) including unit positions, emergency status, weather, traffic, and hospital capacity
- **50 discrete actions** for comprehensive emergency management from unit dispatch to declaring states of emergency

## 🎨 Enhanced Visualization Features

- **Advanced Pygame Interface** with resizable window and fullscreen support
- **Enhanced City Grid** with visual patterns for different zone types
- **Animated Emergency Indicators** with pulsing effects and particle systems
- **Improved Response Unit Graphics** with status indicators and readiness levels
- **Professional Information Panels** with icons, color coding, and real-time updates
- **Mini-Map Overview** showing city-wide emergency distribution
- **Interactive Legend** explaining all symbols and color codes
- **Weather Effects** including fog and rain visualization
- **Connection Lines** between units and their assigned incidents
- **Responsive Design** that adapts to different window sizes

## Installation

```bash
cd emergency_response_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
import emergency_response_env

# Create environment with pygame visualization  
env = gym.make('EmergencyResponse-v1')

# Reset environment
observation, info = env.reset()

# Take actions
for step in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## 🎮 Visualization Controls

- **ESC**: Exit the application
- **F11**: Toggle fullscreen mode
- **Mouse**: Resize window to see responsive design
- **Window**: Drag to reposition, resize to see adaptive layout

## Testing

### Comprehensive Test Suite
```bash
python test_emergency.py
```

Run comprehensive test suite with 5 scenarios including single incident, multiple incidents, natural disaster, technological disaster, and mass casualty events.

### Visualization Test
```bash
python test_visualization.py
```

Test the enhanced pygame visualization with improved graphics, animations, and UI panels.

## Requirements

- gymnasium>=0.28.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025




