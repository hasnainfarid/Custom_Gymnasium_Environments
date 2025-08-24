# Agricultural Farm Management Environment

A comprehensive agricultural farm management environment for reinforcement learning, built on OpenAI Gymnasium. This environment simulates realistic farming operations including crop management, weather systems, market dynamics, and sustainability tracking.

## Features

### 🌾 Comprehensive Farming Simulation
- **25x25 farm grid** divided into 4 field sections (North, South, East, West)
- **5 crop types** with realistic growth cycles:
  - Wheat (90-day cycle)
  - Corn (120-day cycle)
  - Tomatoes (60-day cycle)
  - Apples (365-day cycle, perennial)
  - Soybeans (100-day cycle, nitrogen-fixing)

### 🌤️ Dynamic Weather System
- 4 seasons with realistic weather patterns
- Temperature, humidity, rainfall, and wind tracking
- Frost risk and drought/flood conditions
- Climate change effects over time

### 🚜 Farm Equipment Management
- 8 types of farming equipment:
  - Tractor, Harvester, Irrigation System
  - Fertilizer Spreader, Pesticide Sprayer
  - Seed Planter, Soil Tester, Weather Station
- Equipment fuel and maintenance tracking

### 📊 Complex State Space
- 620-dimensional observation space including:
  - Crop growth stages and health
  - Soil conditions (pH, NPK, moisture, compaction)
  - Weather data and forecasts
  - Equipment status
  - Market prices
  - Financial metrics
  - Sustainability indicators

### 🎮 Rich Action Space
- 45 discrete actions covering:
  - Planting and harvesting
  - Fertilization and irrigation
  - Pest and disease management
  - Equipment operation
  - Market timing decisions
  - Sustainability practices

### 💰 Realistic Reward Structure
- Yield-based rewards
- Market timing bonuses
- Sustainability incentives
- Environmental penalties
- Equipment and weather damage costs

### 🎨 Enhanced Pygame Visualization (1400x900)

#### Visual Features:
- **Dynamic Backgrounds**: Gradient sky that changes with seasons and weather
- **Advanced Farm View**:
  - Detailed crop visualization with individual plant rendering
  - Growth stage animations (seeds → mature plants)
  - Soil quality gradients and texture patterns
  - Moisture level overlays
  - Field health border indicators (green=healthy, red=pest issues, orange=disease)
  
#### Dashboard Components:
- **Weather Station**: 
  - Circular temperature and humidity gauges
  - Animated rainfall display
  - 7-day forecast preview
  - Wind speed indicator
  
- **Financial Dashboard**:
  - Real-time cash flow with color coding
  - Revenue/expense tracking
  - Profit margin gauge
  - Historical revenue graph
  
- **Market Panel**:
  - Live price displays with trend arrows (↑↓→)
  - Price ratio bars
  - Seasonal price predictions
  
- **Resource Meters**:
  - Animated liquid tanks for water resources
  - Wave effects in reservoirs
  - Water quality circular meter
  - Irrigation capacity bar
  
- **Sustainability Dashboard**:
  - Leaf-shaped eco-meters for carbon, water, biodiversity
  - Overall sustainability score display
  - Soil health trend indicator
  
- **Equipment Status**:
  - Equipment icons with operational status
  - Fuel level bars
  - Maintenance warning indicators
  - Pulsing animation for active equipment

#### Advanced Visualizations:
- **Real-time Performance Graphs**: 
  - Revenue trend with grid lines
  - Yield performance tracking
  - Sustainability score evolution
  
- **Seasonal Calendar**:
  - 12-month overview with season colors
  - Planting schedule visualization
  - Current month highlighting
  - Crop timeline display
  
- **Weather Effects**:
  - Animated rain particles
  - Frost overlay with transparency
  - Seasonal atmosphere changes
  
- **Notification System**:
  - Smart alerts for critical events
  - Harvest ready notifications
  - Pest/disease warnings
  - Low resource alerts
  
- **Field Detail Popup**:
  - Comprehensive field information
  - NPK nutrient levels
  - Growth stage details
  - Health and yield metrics

## Installation

### From Source
```bash
# Clone or navigate to the package directory
cd agricultural_farm_env

# Install the package
pip install -e .
```

### Dependencies
- gymnasium>=0.28.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.3.0

## Quick Start

### Basic Usage
```python
import gymnasium as gym
from agricultural_farm_env import AgriculturalFarmEnv

# Create environment
env = AgriculturalFarmEnv(render_mode="human")

# Reset environment
observation, info = env.reset()

# Run simulation
for _ in range(365):  # One year
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Using with Gymnasium
```python
import gymnasium as gym

# Register and create environment
gym.register(
    id='AgriculturalFarm-v0',
    entry_point='agricultural_farm_env:AgriculturalFarmEnv',
    max_episode_steps=365
)

env = gym.make('AgriculturalFarm-v0', render_mode="human")
```

## Testing

### Running Tests

Run the test script which offers two modes:

```bash
python test_farm.py
```

Choose option 1 for a quick visualization demo or option 2 for full testing.

The full test will:
- Test 4 scenarios (Normal Year, Drought, Pest Outbreak, Market Crash)
- Compare 3 strategies (Conventional, Organic, Smart Farming)
- Generate comparison plots
- Save detailed reports

## Environment Details

### State Space
The observation is a 620-dimensional numpy array containing:
- **Field conditions** (60 elements): crop type, growth stage, health, yield potential, soil metrics
- **Weather data** (10 elements): temperature, humidity, rainfall, wind, forecasts
- **Equipment status** (40 elements): location, fuel, maintenance for 8 machines
- **Market prices** (10 elements): current and predicted prices for 5 crops
- **Water resources** (5 elements): reservoir level, quality, costs
- **Financial status** (5 elements): cash flow, costs, debt, profit
- **Sustainability metrics** (4 elements): carbon, water conservation, biodiversity, soil health

### Action Space
45 discrete actions:
- **0-4**: Plant crop in field sections
- **5-9**: Harvest from field sections
- **10-14**: Apply fertilizer
- **15-19**: Irrigate fields
- **20-24**: Apply pesticides
- **25-29**: Operate equipment
- **30-34**: Soil testing
- **35-39**: Market timing
- **40**: Emergency weather protection
- **41**: Sustainable farming mode
- **42**: Intensive production mode
- **43**: Crop rotation planning
- **44**: Equipment maintenance

### Reward Structure
- **+5000**: Exceptional harvest (>120% expected)
- **+2000**: Good harvest (90-120%)
- **+1000**: Average harvest (70-90%)
- **+500**: Sustainability bonus
- **+300**: Water conservation
- **-1000**: Crop failure
- **-2000**: Severe soil degradation
- **-500**: Equipment breakdown
- **-300**: Weather damage

## Advanced Features

### Crop Dynamics
- Growth stages: Planting → Germination → Vegetative → Reproductive → Maturation → Harvest
- Yield affected by: soil conditions, weather, pests, diseases, management practices
- Crop rotation benefits and nitrogen fixation for legumes

### Soil Management
- NPK nutrient tracking
- pH level management
- Organic matter content
- Compaction from equipment use
- Microbiome health

### Market System
- Seasonal price variations
- Market volatility
- Storage and timing decisions
- Commodity price predictions

### Sustainability
- Carbon sequestration tracking
- Water conservation scoring
- Biodiversity index
- Soil health trends
- Environmental penalties for chemical overuse

## Strategies

The test suite includes three pre-built strategies:

### Conventional Farming
- High input, high output approach
- Frequent fertilization and irrigation
- Regular pesticide use
- Focus on maximizing yield

### Organic Farming
- Sustainable, low-input approach
- Crop rotation emphasis
- Minimal chemical use
- Focus on soil health and biodiversity

### Smart Farming
- Data-driven precision agriculture
- Adaptive decision making
- Optimized resource use
- Balance of yield and sustainability

## Customization

You can easily customize the environment by modifying:
- Crop types and characteristics in `crops/crop_data.py`
- Weather patterns and climate parameters
- Market price dynamics
- Reward structure
- Field layout and size

## Performance Metrics

The environment tracks:
- **Financial**: Revenue, expenses, profit margin, cash flow
- **Production**: Yield per crop, harvest timing, crop health
- **Sustainability**: Carbon sequestration, water usage, biodiversity, soil health
- **Efficiency**: Equipment utilization, resource optimization

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

See LICENSE file for full details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this environment in your research, please cite:
```bibtex
@software{agricultural_farm_env,
  author = {Fareed, Hasnain},
  title = {Agricultural Farm Management Environment},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/agricultural-farm-env}
}
```

## Acknowledgments

This environment was designed to provide a realistic and challenging agricultural simulation for reinforcement learning research, with a focus on sustainable farming practices and long-term resource management.
