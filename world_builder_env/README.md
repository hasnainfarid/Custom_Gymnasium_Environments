[Random Actions on the left, Trained Agent on the right]

![FinalVideo_WorldBuilder](https://github.com/user-attachments/assets/f171acb6-8b54-4739-9049-e8ba847ede3a)



# AI City Builder - Reinforcement Learning Project

A reinforcement learning project where an AI agent learns to play a city-building game using PPO (Proximal Policy Optimization).

**Author:** Hasnain Fareed  
**Email:** Hasnainfarid7@yahoo.com  
**Year:** 2025  
**License:** MIT License

## Game Overview

The agent manages resources (Food, Wood, Stone, Population) and builds structures to grow their population while maintaining resource balance.

### Core Resources
- **Food**: Consumed by population, produced by farms
- **Wood**: Used for building farms and houses, produced by lumberyards
- **Stone**: Used for building lumberyards and houses, produced by quarries
- **Population**: Grows when excess food exists and capacity allows

### Buildings

| Building | Cost | Production | Effect |
|----------|------|------------|--------|
| Farm | 5 wood | 2 food per step | Produces food |
| Lumberyard | 3 stone | 3 wood per step | Produces wood |
| Quarry | 5 wood | 2 stone per step | Produces stone |
| House | 10 wood + 5 stone | None | +5 population capacity |

### Game Mechanics

- **10x10 grid world**: Agent can place buildings on empty tiles
- **Population management**: 
  - Each person consumes 1 food per step
  - Population grows by 1 when excess food > 2 and capacity allows
  - Population dies if food goes negative
- **Actions**: 4 building types + pass action (5 total discrete actions)
- **Win Condition**: Reach 20 population and survive for 50 steps
- **Lose Condition**: Population drops to 0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Manual Control
Run the game with manual control:

```bash
python main.py
```

Then choose option 1 for manual control.

**Controls:**
- `0`: Pass (do nothing)
- `1`: Build Farm
- `2`: Build Lumberyard
- `3`: Build Quarry
- `4`: Build House
- `q`: Quit

### Random Agent
Choose option 2 to watch a random agent play the game.

### Testing
Test the environment:

```bash
python test_env.py
```

## Environment Interface

The environment follows the standard Gymnasium interface:

```python
from world_builder_env import WorldBuilderEnv

# Create environment
env = WorldBuilderEnv(render_mode="human")

# Reset environment
obs, info = env.reset()

# Take action
obs, reward, done, truncated, info = env.step(action)

# Close environment
env.close()
```

### Action Space
- Discrete(5): 0=Pass, 1=Farm, 2=Lumberyard, 3=Quarry, 4=House

### Observation Space
- `grid`: (10, 10) array of building types
- `resources`: (4,) array of [food, wood, stone, population]
- `population_capacity`: (1,) array of max population
- `win_steps`: (1,) array of steps since reaching 20 population

### Rewards
- +1 for successful building placement
- -1 for failed building placement
- +100 for winning (reach 20 population and survive 50 steps)
- -100 for losing (population drops to 0)
- 0 for passing

## File Structure

```
world_builder_env/
├── __init__.py              # Package initialization
├── world_builder_env.py     # Main environment class
├── game_logic.py           # Resource management logic
├── renderer.py             # Pygame rendering
├── main.py                 # Demo script
├── test_env.py            # Test script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Strategy Tips

1. **Start with farms**: You need food to grow your population
2. **Build lumberyards early**: Wood is needed for farms and houses
3. **Balance resources**: Don't over-invest in one resource type
4. **Plan for houses**: You need population capacity to reach 20 people
5. **Monitor food**: Always ensure you have enough food for your population 
