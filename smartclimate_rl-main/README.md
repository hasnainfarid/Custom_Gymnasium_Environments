![Gif](https://github.com/user-attachments/assets/9f426fad-823e-45b7-b589-aaa5a63ed0cb)


# SmartClimate RL

A production-ready Gymnasium environment for smart HVAC and lighting control, with RLlib integration and PyGame visualization.

## Install
```bash
pip install -e .
```

## Usage
```python
import gymnasium as gym
env = gym.make('SmartClimateEnv-v0')
obs, info = env.reset()
```

## Training
See `training/train.py` for RLlib PPO training example.

## Visualization
PyGame-based real-time visualization: `env.render()`

## Structure
- `smartclimate/` - core environment, visualizer, utils
- `training/` - RLlib training/evaluation
- `examples/` - usage demos

## Environment
- 24h, 1-min steps (1440 total)
- Action: Dict(ac_temp: Box(16,32), lights: MultiBinary(4))
- Observation: Box(9,) [room_temp, num_people, time_hour, outside_temp, ac_setting, light1-4]
- Reward: Comfort, AC penalty, light penalty

## License
Personal
