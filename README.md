# Custom Gymnasium Environments

A comprehensive collection of reinforcement learning environments built with Gymnasium, designed for research, education, and practical applications across diverse domains.

## Overview

This repository contains a curated selection of custom reinforcement learning environments that demonstrate various real-world scenarios and specialized applications. Each environment is built following Gymnasium standards and includes comprehensive documentation, testing suites, and visualization capabilities.

## Environment Categories

The collection spans multiple domains including:

- **Manufacturing & Logistics**: Production optimization, fleet management, traffic control
- **Healthcare & Emergency**: Hospital operations, emergency response simulation
- **Finance & Trading**: Cryptocurrency trading, market dynamics
- **Infrastructure**: Smart cities, climate control, parking management
- **Gaming & Entertainment**: Classic games, physics simulations
- **Space & Engineering**: Life support systems, orbital mechanics

## Key Features

- **Gymnasium Compatibility**: All environments follow the latest Gymnasium API standards
- **Real-time Visualization**: Pygame-based rendering for interactive development
- **Comprehensive Testing**: Built-in test suites and example scripts
- **Modular Design**: Self-contained environments with minimal dependencies
- **Research Ready**: Proper observation/action spaces and reward structures
- **Documentation**: Detailed READMEs and usage examples for each environment

## Quick Start

### Installation
```bash
# Navigate to any environment directory
cd environment_name

# Install dependencies
pip install -r requirements.txt

# Run the environment
python test_environment.py
```

### Basic Usage
```python
import gymnasium as gym
from environment_name import EnvironmentName

# Create and use environment
env = EnvironmentName(render_mode="human")
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

## Requirements

All environments require:
- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.19.0
- pygame>=2.0.0

Additional dependencies vary by environment (see individual requirements.txt files).

## Contributing

Each environment is designed to be extensible and customizable. Feel free to:
- Modify parameters and configurations
- Add new features and capabilities
- Integrate with additional RL libraries
- Create new environments following the established patterns

## License

MIT License - see individual environment LICENSE files for details.

## Author

**Hasnain Fareed** - 2025

---

**Note:** Each environment is self-contained and can be used independently for reinforcement learning research and training. Browse the individual environment directories for specific features and detailed documentation.
