# Pedal Wheel Environment

A custom OpenAI Gymnasium environment that simulates a pedal-controlled unicycle moving on a flat 2D road. The agent controls two pedals to move forward and maintain balance.

## 🎯 Environment Concept

The environment features a side-view 2D world with:
- A single wheel-based vehicle with pedals on both sides
- Continuous control over left and right pedal forces
- Physics-based movement and balance mechanics
- Real-time Pygame visualization

## 🕹️ Controls / Action Space

The agent has continuous control over two pedal forces:
- `left_pedal_force` ∈ [-1.0, 1.0]
- `right_pedal_force` ∈ [-1.0, 1.0]

The difference between the two forces creates torque and affects:
- Forward movement
- Tilt/balance of the unicycle
- Energy dynamics

**Action Space:**
```python
spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

## 🧠 Observation Space

The observation provides normalized physical state information:
- Wheel position on the X-axis (normalized)
- Velocity of the wheel (normalized)
- Tilt angle (normalized)
- Angular velocity (normalized)
- Wheel angular velocity (normalized)

**Observation Space:**
```python
spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
```

## 🎁 Reward Function

The reward encourages efficient movement and balance:
- **+1** per timestep survived while upright and moving forward
- **-10** if the unicycle tilts beyond threshold (episode ends = fall)
- **Small bonus** for higher forward velocity
- **Small penalty** for energy overuse (to encourage control)

## 🔁 Episode Termination

Episodes end when:
- Tilt angle exceeds threshold (falls over)
- Maximum time limit is reached (1000 steps)

## 📦 Package Structure

```
pedal_wheel_env/
│
├── __init__.py              # Package initialization
├── environment.py           # Main Gymnasium Env class
├── physics.py              # Handles wheel dynamics, pedal forces, tilt, etc.
├── config.py               # Constants like gravity, max speed, etc.
├── pygame_visualizer.py    # Pygame code to render the environment
├── test_env.py             # Sample script to test the environment
└── README.md               # This file
```

## 🚀 Quick Start

### Basic Usage

```python
import gymnasium as gym
from pedal_wheel_env import PedalWheelEnv

# Create environment
env = PedalWheelEnv(render_mode="human")

# Reset environment
observation, info = env.reset()

# Run a simple episode
for step in range(100):
    # Random action
    action = env.action_space.sample()
    
    # Take step
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### Manual Control

```python
from pedal_wheel_env.test_env import test_manual_control

# Run manual control test
test_manual_control()
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd pedal_wheel_env
python test_env.py
```

The test suite includes:
- **Random Actions Test**: Tests with random pedal forces
- **Fixed Patterns Test**: Tests with predefined action patterns
- **Manual Control Test**: Interactive keyboard control
- **Physics Consistency Test**: Validates physics bounds
- **Observation Space Test**: Validates observation and action spaces

### Manual Controls (when running manual test):
- **W/S**: Left pedal up/down
- **Up/Down arrows**: Right pedal up/down
- **Q**: Quit
- **R**: Reset environment

## 🔧 Physics Model

The environment uses a simplified but realistic physics model:

### Wheel Dynamics
- Torque from pedal forces drives wheel rotation
- Wheel rotation creates forward motion
- Friction opposes motion

### Balance Mechanics
- Gravity creates restoring torque when tilted
- Forward motion creates gyroscopic effects
- Pedal forces can be used for balance correction

### Energy Tracking
- Total energy used is tracked
- Energy usage affects reward (encourages efficiency)

## 🎨 Visualization Features

The Pygame visualizer provides:
- **Real-time rendering** of the unicycle
- **Side-view perspective** showing wheel, frame, and pedals
- **Dynamic camera** that follows the wheel
- **UI elements** showing position, velocity, tilt, and energy
- **Tilt indicator** with color coding (green=stable, red=falling)
- **Ground texture** for visual reference

## 🔮 Future Extensions

The environment is designed to be extensible for:
- **Terrain variations** (hills, bumps, obstacles)
- **Weather effects** (wind, rain, friction changes)
- **Multiple objectives** (speed targets, energy efficiency)
- **Multi-agent scenarios** (racing, cooperative tasks)
- **Advanced physics** (suspension, tire dynamics)

## 📋 Requirements

- Python 3.7+
- Gymnasium
- NumPy
- Pygame (for visualization)

## 📄 License

This environment is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to extend the environment with additional features, improved physics, or new visualization options! 