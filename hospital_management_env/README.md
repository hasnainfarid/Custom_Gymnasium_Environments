# Hospital Management Environment

A realistic hospital operations simulation environment for reinforcement learning, featuring complex patient dynamics, resource management, and life-critical decision making.

## 🏥 Overview

The Hospital Management Environment simulates a complete hospital ecosystem with:
- **20x20 hospital layout** with 6 specialized departments
- **Dynamic patient flow** with 5 severity levels
- **Resource management** for staff, beds, equipment, and medicine
- **Real-time visualization** using Pygame
- **Complex reward structure** based on patient outcomes
- **Special events** including outbreaks and mass casualties

## 🚀 Installation

```bash
# Clone or download the package
cd hospital_management_env

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements
- Python 3.8+
- gymnasium >= 0.28.0
- numpy >= 1.21.0
- pygame >= 2.1.0
- matplotlib >= 3.5.0

## 📊 Environment Specifications

### State Space (295 dimensions)
- **Doctor Information** (45): Location and availability for 15 doctors
- **Nurse Distribution** (6): Nurses per department
- **Bed Occupancy** (80): Status and severity for 40 beds
- **Patient Queues** (30): Queue sizes by severity per department
- **Equipment Status** (10): Availability of critical machines
- **Medicine Inventory** (15): Stock levels for common medications
- **Department Metrics** (12): Utilization and wait times
- **Staff Fatigue** (40): Fatigue levels for all staff
- **Additional Metrics** (5): Deaths, treatments, time, special events

### Action Space (35 discrete actions)
- **0-5**: Assign nurse to department
- **6-11**: Reassign doctor between departments
- **12-17**: Adjust department priority levels
- **18-23**: Schedule equipment maintenance
- **24-29**: Order emergency supplies
- **30**: Call additional staff (overtime)
- **31**: Discharge stable patients
- **32**: Transfer patients to other hospitals
- **33**: Activate emergency protocol
- **34**: Normal operation mode

### Reward Structure
```python
+1000: Life saved (severity 5 patient)
+500:  Critical patient (severity 4) treated
+200:  Emergency patient (severity 3) treated
+100:  Standard patient treated
-2000: Patient death due to delays
-500:  Critical patient waiting >60 minutes
-100:  Equipment breakdown
-50:   Staff overtime costs
-200:  Patient transfer costs
+300:  Department efficiency bonus
```

## 🎮 Basic Usage

### Quick Start
```python
import gymnasium as gym
from hospital_management_env import HospitalManagementEnv

# Create environment
env = HospitalManagementEnv(render_mode="human")

# Run a simple episode
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

### Using with Gymnasium
```python
import gymnasium as gym

# Register and create environment
env = gym.make('HospitalManagement-v0', render_mode="human")

# Training loop
for episode in range(10):
    obs, info = env.reset()
    total_reward = 0
    
    while True:
        # Your policy here
        action = your_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode {episode}: Reward = {total_reward}")
            break
```

## 🧪 Testing Scenarios

Run the comprehensive test suite:
```bash
python test_hospital.py
```

This runs 5 different scenarios:
1. **Normal Day**: Standard hospital operations
2. **Mass Casualty**: Sudden influx of trauma patients
3. **Flu Outbreak**: Contagious disease spreading
4. **Staff Shortage**: 30% reduction in available staff
5. **Equipment Failure**: Critical equipment breakdowns

### Custom Scenarios
```python
from test_hospital import HospitalTestRunner

runner = HospitalTestRunner(render=True)

# Define custom policy
def my_policy(obs, env):
    # Your decision logic here
    return action

# Run custom scenario
results = runner.run_scenario(
    "Custom Scenario",
    my_policy,
    episode_length=500,
    special_conditions={
        'outbreak': 'covid',
        'staff_shortage': True,
        'staff_reduction': 0.4
    }
)
```

## 🏗️ Environment Architecture

### Departments
- **Emergency** (4x4): Handles urgent and critical cases
- **ICU** (3x3): Intensive care for severe patients
- **Surgery** (4x2): Surgical procedures
- **General Ward** (6x4): Standard patient care
- **Pharmacy** (2x2): Medicine distribution
- **Labs** (3x3): Diagnostic services

### Staff Distribution
- **15 Doctors**: 3 Emergency, 2 ICU, 4 General, 3 Surgeons, 2 Specialists, 1 Chief
- **25 Nurses**: Distributed across departments
- **Specializations**: Each type has efficiency bonuses for specific conditions

### Patient Dynamics
```python
Severity Levels:
- Level 5 (Critical): 5% arrival, 2-hour treatment, ICU required
- Level 4 (Urgent): 10% arrival, 1-hour treatment, Emergency
- Level 3 (Emergency): 20% arrival, 45-min treatment
- Level 2 (Standard): 35% arrival, 30-min treatment
- Level 1 (Minor): 30% arrival, 15-min treatment
```

## 📈 Performance Metrics

The environment tracks:
- **Mortality Rate**: Deaths / Total Patients
- **Average Wait Time**: Mean patient waiting time
- **Treatment Efficiency**: Patients treated per hour
- **Resource Utilization**: Bed and staff usage
- **Cost Efficiency**: Reward per resource spent

## 🎨 Visualization Features

The Pygame visualization includes:
- **Hospital Floor Plan**: Color-coded departments
- **Staff Indicators**: Doctors (blue) and nurses (green)
- **Patient Queues**: Severity-based color coding
- **Real-time Metrics**: Deaths, treatments, wait times
- **Alert System**: Critical warnings and emergencies
- **Resource Bars**: Equipment status and medicine levels

### Color Coding
- 🔴 Red: Critical/Emergency
- 🟠 Orange: Urgent
- 🟡 Yellow: Warning/Standard
- 🟢 Green: Stable/Available
- 🔵 Blue: Medical Staff
- 🟣 Purple: Exhausted Staff

## 🔧 Advanced Configuration

### Custom Medical Data
The environment uses internal medical data from `data/medical_data.json`:
```python
# Load and modify medical data
import json

with open('hospital_management_env/data/medical_data.json', 'r') as f:
    medical_data = json.load(f)

# Modify disease parameters
medical_data['diseases']['covid']['mortality_risk'] = 0.08
```

### Environment Parameters
```python
env = HospitalManagementEnv(
    render_mode="human",  # or "rgb_array" for training
)

# Access internal parameters
env.severity_arrival_rates  # Modify arrival rates
env.treatment_times         # Adjust treatment durations
env.bed_distribution        # Change bed allocations
```

## 📝 Policy Development Tips

### Reactive Strategy
```python
def reactive_policy(obs, env):
    # Respond to immediate crises
    if critical_patients > threshold:
        return 33  # Emergency protocol
    elif high_fatigue_staff > limit:
        return 30  # Call additional staff
    # ...
```

### Predictive Strategy
```python
def predictive_policy(obs, env):
    # Anticipate future needs
    if time_of_day == "morning":
        return prepare_for_rush_hour()
    elif equipment_degradation > 0.7:
        return schedule_maintenance()
    # ...
```

### Optimal Heuristic
```python
def optimal_policy(obs, env):
    # Prioritize by impact
    priorities = calculate_priorities()
    return execute_highest_priority(priorities)
```

## 🚨 Special Events

### Disease Outbreaks
- Increased patient arrival rates
- Specific disease prevalence
- Contagion spread dynamics

### Mass Casualty Events
- Sudden influx of trauma patients
- Resource strain simulation
- Emergency protocol activation

### Equipment Failures
- Random breakdowns
- Maintenance requirements
- Critical equipment prioritization

## 📊 Data Logging

The environment generates:
- **Episode logs**: Complete state-action-reward sequences
- **Critical incidents**: Deaths and emergency events
- **Performance reports**: JSON-formatted metrics
- **Visualization plots**: Comparative analysis graphs

## 🤝 Contributing

This is a self-contained package. To extend functionality:

1. Modify `hospital_env.py` for core mechanics
2. Update `medical_data.json` for medical parameters
3. Enhance `test_hospital.py` for new scenarios
4. Adjust visualization in the render methods

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

This environment simulates realistic hospital operations for research and educational purposes. All medical data and scenarios are generated internally without external dependencies.

## 📧 Support

For issues or questions, please refer to the documentation in the source code or create an issue in the repository.

---

**Note**: This environment is designed for reinforcement learning research and does not constitute medical advice or real hospital management guidance.