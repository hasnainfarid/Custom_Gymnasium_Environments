import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math

@dataclass
class Module:
    """Represents a station module"""
    name: str
    width: int
    height: int
    x: int = 0
    y: int = 0
    oxygen_percent: float = 21.0
    co2_ppm: float = 400.0
    pressure: float = 101.3  # kPa
    temperature: float = 22.0  # Celsius
    
@dataclass
class CrewMember:
    """Represents a crew member"""
    role: str
    x: float = 0.0
    y: float = 0.0
    health: float = 100.0
    oxygen_level: float = 100.0
    fatigue: float = 0.0
    current_module: str = "Command"
    task: Optional[str] = None
    
@dataclass
class System:
    """Represents a critical system"""
    name: str
    efficiency: float = 100.0
    power_draw: float = 0.0  # Watts
    maintenance_needed: float = 0.0
    is_critical: bool = False
    dependencies: List[str] = field(default_factory=list)
    last_maintenance: int = 0
    failure_probability: float = 0.001

class SpaceStationEnv(gym.Env):
    """
    Space Station Life Support Management Environment
    
    Manage a space station with 6 crew members, 12 critical systems,
    and 8 modules in a realistic orbital environment.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        # Time tracking
        self.time_step = 0
        self.minutes_per_step = 5  # Each step is 5 minutes
        self.orbital_period = 90  # minutes
        self.mission_duration_days = 30
        self.max_steps = (self.mission_duration_days * 24 * 60) // self.minutes_per_step
        
        # Initialize modules
        self.modules = self._initialize_modules()
        
        # Initialize crew
        self.crew = self._initialize_crew()
        
        # Initialize systems
        self.systems = self._initialize_systems()
        
        # Resources
        self.resources = {
            'water': 1000.0,  # liters
            'food': 500.0,  # kg
            'oxygen_tanks': 20.0,  # units
            'nitrogen': 100.0,  # kg
            'medical_supplies': 100.0,  # units
            'spare_parts': 50.0,  # units
            'fuel': 200.0,  # kg
            'waste_capacity': 100.0  # percent remaining
        }
        
        # Power systems
        self.solar_efficiency = 1.0
        self.battery_level = 100.0  # percent
        self.total_power_consumption = 0.0
        self.backup_power_available = True
        self.power_generation = 5000.0  # Watts
        
        # Orbital mechanics
        self.orbital_phase = 0.0  # 0 to 360 degrees
        self.solar_angle = 0.0
        self.in_earth_shadow = False
        self.radiation_level = 0.2  # normalized
        
        # Emergency systems
        self.emergency_alerts = {
            'fire': False,
            'depressurization': False,
            'radiation_storm': False,
            'power_failure': False,
            'life_support_critical': False,
            'medical_emergency': False
        }
        
        # Mission tracking
        self.mission_success = False
        self.crew_casualties = 0
        self.system_failures = []
        self.emergency_count = 0
        self.resource_efficiency = 100.0
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(40)
        
        # State space: 110 elements as specified
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(110,), 
            dtype=np.float32
        )
        
        # Reward tracking
        self.episode_reward = 0.0
        self.last_emergency_time = 0
        
    def _initialize_modules(self) -> Dict[str, Module]:
        """Initialize station modules"""
        modules = {
            'Command': Module('Command', 3, 3, 0, 0),
            'Living Quarters': Module('Living Quarters', 4, 6, 3, 0),
            'Laboratory': Module('Laboratory', 5, 4, 7, 0),
            'Engineering': Module('Engineering', 3, 5, 0, 3),
            'Hydroponics': Module('Hydroponics', 4, 4, 3, 6),
            'Storage': Module('Storage', 3, 3, 7, 4),
            'Docking': Module('Docking', 2, 4, 10, 4),
            'Solar Array Control': Module('Solar Array Control', 2, 2, 0, 8)
        }
        return modules
    
    def _initialize_crew(self) -> List[CrewMember]:
        """Initialize crew members"""
        roles = [
            'Commander',
            'Engineer', 
            'Scientist',
            'Medical Officer',
            'Pilot',
            'Mission Specialist'
        ]
        crew = []
        for role in roles:
            member = CrewMember(role=role)
            # Start crew in different modules
            module_name = list(self.modules.keys())[len(crew) % len(self.modules)]
            module = self.modules[module_name]
            member.current_module = module_name
            member.x = module.x + module.width / 2
            member.y = module.y + module.height / 2
            crew.append(member)
        return crew
    
    def _initialize_systems(self) -> Dict[str, System]:
        """Initialize critical systems"""
        systems = {
            'Oxygen Generation': System(
                'Oxygen Generation', 
                power_draw=500.0, 
                is_critical=True,
                dependencies=['Power Grid', 'Water Recycling']
            ),
            'CO2 Scrubbing': System(
                'CO2 Scrubbing',
                power_draw=300.0,
                is_critical=True,
                dependencies=['Power Grid']
            ),
            'Water Recycling': System(
                'Water Recycling',
                power_draw=400.0,
                is_critical=True,
                dependencies=['Power Grid', 'Waste Processing']
            ),
            'Power Grid': System(
                'Power Grid',
                power_draw=0.0,
                is_critical=True,
                dependencies=[]
            ),
            'Thermal Control': System(
                'Thermal Control',
                power_draw=600.0,
                is_critical=True,
                dependencies=['Power Grid']
            ),
            'Atmospheric Pressure': System(
                'Atmospheric Pressure',
                power_draw=200.0,
                is_critical=True,
                dependencies=['Power Grid', 'Nitrogen Supply']
            ),
            'Nitrogen Supply': System(
                'Nitrogen Supply',
                power_draw=100.0,
                dependencies=['Power Grid']
            ),
            'Waste Processing': System(
                'Waste Processing',
                power_draw=350.0,
                dependencies=['Power Grid']
            ),
            'Fire Suppression': System(
                'Fire Suppression',
                power_draw=50.0,
                is_critical=True,
                dependencies=['Power Grid', 'Atmospheric Pressure']
            ),
            'Radiation Shielding': System(
                'Radiation Shielding',
                power_draw=800.0,
                is_critical=True,
                dependencies=['Power Grid']
            ),
            'Communications': System(
                'Communications',
                power_draw=250.0,
                dependencies=['Power Grid']
            ),
            'Artificial Gravity': System(
                'Artificial Gravity',
                power_draw=1000.0,
                dependencies=['Power Grid']
            )
        }
        return systems
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset time
        self.time_step = 0
        
        # Reset modules
        for module in self.modules.values():
            module.oxygen_percent = 21.0 + np.random.normal(0, 0.5)
            module.co2_ppm = 400.0 + np.random.normal(0, 50)
            module.pressure = 101.3 + np.random.normal(0, 1)
            module.temperature = 22.0 + np.random.normal(0, 1)
        
        # Reset crew
        for member in self.crew:
            member.health = 100.0
            member.oxygen_level = 100.0
            member.fatigue = 0.0
            member.task = None
        
        # Reset systems
        for system in self.systems.values():
            system.efficiency = 100.0 - np.random.uniform(0, 5)
            system.maintenance_needed = np.random.uniform(0, 10)
            system.last_maintenance = 0
        
        # Reset resources
        self.resources = {
            'water': 1000.0,
            'food': 500.0,
            'oxygen_tanks': 20.0,
            'nitrogen': 100.0,
            'medical_supplies': 100.0,
            'spare_parts': 50.0,
            'fuel': 200.0,
            'waste_capacity': 100.0
        }
        
        # Reset power
        self.solar_efficiency = 1.0
        self.battery_level = 100.0
        self.backup_power_available = True
        
        # Reset orbital position
        self.orbital_phase = 0.0
        self.in_earth_shadow = False
        
        # Reset emergencies
        for key in self.emergency_alerts:
            self.emergency_alerts[key] = False
        
        # Reset mission tracking
        self.mission_success = False
        self.crew_casualties = 0
        self.system_failures = []
        self.emergency_count = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step"""
        self.time_step += 1
        
        # Process action
        self._process_action(action)
        
        # Update orbital mechanics
        self._update_orbital_position()
        
        # Update power systems
        self._update_power_systems()
        
        # Update life support systems
        self._update_life_support()
        
        # Update crew status
        self._update_crew()
        
        # Update resources
        self._update_resources()
        
        # Check for emergencies
        self._check_emergencies()
        
        # System degradation
        self._update_system_degradation()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.time_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action: int):
        """Process the selected action"""
        if action < 12:
            # Prioritize power to system
            system_name = list(self.systems.keys())[action]
            self.systems[system_name].efficiency = min(100.0, 
                self.systems[system_name].efficiency + 10.0)
            
        elif action < 24:
            # Schedule maintenance
            system_idx = action - 12
            system_name = list(self.systems.keys())[system_idx]
            if self.resources['spare_parts'] > 0:
                self.systems[system_name].maintenance_needed = max(0, 
                    self.systems[system_name].maintenance_needed - 20.0)
                self.systems[system_name].last_maintenance = self.time_step
                self.resources['spare_parts'] -= 0.5
                
        elif action < 30:
            # Assign crew to emergency task
            crew_idx = action - 24
            if crew_idx < len(self.crew):
                self.crew[crew_idx].task = "emergency_response"
                
        elif action < 36:
            # Adjust environmental controls
            module_idx = action - 30
            if module_idx < len(self.modules):
                module = list(self.modules.values())[module_idx]
                # Adjust temperature and pressure
                module.temperature = np.clip(module.temperature + np.random.uniform(-1, 1), 18, 26)
                module.pressure = np.clip(module.pressure + np.random.uniform(-0.5, 0.5), 95, 108)
                
        elif action == 36:
            # Activate emergency protocols
            self._activate_emergency_protocols()
            
        elif action == 37:
            # Switch to backup power
            if self.backup_power_available:
                self.battery_level = min(100.0, self.battery_level + 20.0)
                self.backup_power_available = False
                
        elif action == 38:
            # Initiate evacuation procedures
            self._initiate_evacuation()
            
        # Action 39 is normal operations (no special action)
    
    def _update_orbital_position(self):
        """Update orbital mechanics"""
        # Update orbital phase (degrees per time step)
        degrees_per_minute = 360.0 / self.orbital_period
        self.orbital_phase = (self.orbital_phase + degrees_per_minute * self.minutes_per_step) % 360
        
        # Calculate solar angle and shadow
        self.solar_angle = math.radians(self.orbital_phase)
        self.in_earth_shadow = 180 < self.orbital_phase < 270
        
        # Update solar efficiency based on angle and shadow
        if self.in_earth_shadow:
            self.solar_efficiency = 0.0
        else:
            self.solar_efficiency = max(0, math.cos(self.solar_angle))
        
        # Update radiation level
        if 90 < self.orbital_phase < 270:
            self.radiation_level = 0.3 + 0.2 * math.sin(math.radians(self.orbital_phase - 90))
        else:
            self.radiation_level = 0.2
    
    def _update_power_systems(self):
        """Update power generation and consumption"""
        # Calculate total power consumption
        self.total_power_consumption = sum(
            system.power_draw * (system.efficiency / 100.0) 
            for system in self.systems.values()
        )
        
        # Generate power from solar panels
        solar_generation = self.power_generation * self.solar_efficiency
        
        # Update battery
        power_balance = solar_generation - self.total_power_consumption
        battery_change = (power_balance / self.power_generation) * 10  # Scale to battery percentage
        self.battery_level = np.clip(self.battery_level + battery_change, 0, 100)
        
        # Check for power failure
        if self.battery_level < 10:
            self.emergency_alerts['power_failure'] = True
            # Reduce system efficiency
            for system in self.systems.values():
                system.efficiency *= 0.8
    
    def _update_life_support(self):
        """Update life support systems"""
        o2_gen = self.systems['Oxygen Generation']
        co2_scrub = self.systems['CO2 Scrubbing']
        
        for module in self.modules.values():
            # Crew oxygen consumption
            crew_in_module = sum(1 for c in self.crew if c.current_module == module.name)
            o2_consumption = crew_in_module * 0.01  # percent per step
            co2_production = crew_in_module * 5  # ppm per step
            
            # Apply life support effects
            if o2_gen.efficiency > 50:
                module.oxygen_percent += (o2_gen.efficiency / 100.0) * 0.05
            module.oxygen_percent -= o2_consumption
            
            if co2_scrub.efficiency > 50:
                module.co2_ppm -= (co2_scrub.efficiency / 100.0) * 10
            module.co2_ppm += co2_production
            
            # Clamp values
            module.oxygen_percent = np.clip(module.oxygen_percent, 0, 30)
            module.co2_ppm = np.clip(module.co2_ppm, 0, 10000)
            
            # Temperature control
            thermal = self.systems['Thermal Control']
            if thermal.efficiency > 50:
                # Move temperature toward optimal
                target_temp = 22.0
                module.temperature += (target_temp - module.temperature) * 0.1 * (thermal.efficiency / 100.0)
            else:
                # Temperature drifts
                module.temperature += np.random.uniform(-0.5, 0.5)
            
            module.temperature = np.clip(module.temperature, 10, 35)
    
    def _update_crew(self):
        """Update crew status"""
        for member in self.crew:
            module = self.modules[member.current_module]
            
            # Update oxygen level based on module atmosphere
            if module.oxygen_percent < 19:
                member.oxygen_level -= 2.0
            elif module.oxygen_percent > 19:
                member.oxygen_level = min(100, member.oxygen_level + 1.0)
            
            # CO2 effects
            if module.co2_ppm > 1000:
                member.health -= 0.5
            if module.co2_ppm > 5000:
                member.health -= 2.0
            
            # Temperature effects
            if module.temperature < 15 or module.temperature > 30:
                member.health -= 0.3
            
            # Fatigue
            member.fatigue += 0.1
            if member.fatigue > 80:
                member.health -= 0.2
            
            # Task completion reduces fatigue slightly
            if member.task == "emergency_response":
                member.fatigue = max(0, member.fatigue - 5)
                member.task = None
            
            # Radiation effects
            if self.radiation_level > 0.5:
                member.health -= self.radiation_level * 0.1
            
            # Clamp values
            member.health = np.clip(member.health, 0, 100)
            member.oxygen_level = np.clip(member.oxygen_level, 0, 100)
            member.fatigue = np.clip(member.fatigue, 0, 100)
            
            # Check for death
            if member.health <= 0 or member.oxygen_level <= 0:
                self.crew_casualties += 1
    
    def _update_resources(self):
        """Update resource consumption"""
        crew_alive = sum(1 for c in self.crew if c.health > 0)
        
        # Water consumption and recycling
        water_consumed = crew_alive * 0.02  # liters per step
        water_recycling = self.systems['Water Recycling']
        if water_recycling.efficiency > 50:
            water_recovered = water_consumed * (water_recycling.efficiency / 100.0) * 0.9
            self.resources['water'] -= (water_consumed - water_recovered)
        else:
            self.resources['water'] -= water_consumed
        
        # Food consumption
        self.resources['food'] -= crew_alive * 0.01
        
        # Oxygen tank usage (emergency only)
        if self.emergency_alerts['life_support_critical']:
            self.resources['oxygen_tanks'] -= 0.1
        
        # Waste accumulation
        waste_processing = self.systems['Waste Processing']
        waste_generated = crew_alive * 0.05
        if waste_processing.efficiency > 50:
            waste_processed = waste_generated * (waste_processing.efficiency / 100.0)
            self.resources['waste_capacity'] -= (waste_generated - waste_processed) * 0.1
        else:
            self.resources['waste_capacity'] -= waste_generated * 0.2
        
        # Clamp resources
        for key in self.resources:
            self.resources[key] = max(0, self.resources[key])
    
    def _check_emergencies(self):
        """Check for and trigger emergency events"""
        # Random emergency probability
        if np.random.random() < 0.001:
            emergency_type = np.random.choice([
                'micrometeorite', 'system_failure', 'medical', 'solar_storm'
            ])
            
            if emergency_type == 'micrometeorite':
                # Damage to random module
                module = np.random.choice(list(self.modules.values()))
                module.pressure -= 10
                self.emergency_alerts['depressurization'] = True
                self.emergency_count += 1
                
            elif emergency_type == 'system_failure':
                # Random system fails
                system = np.random.choice(list(self.systems.values()))
                system.efficiency = 0
                self.system_failures.append(system.name)
                if system.is_critical:
                    self.emergency_alerts['life_support_critical'] = True
                self.emergency_count += 1
                
            elif emergency_type == 'medical':
                # Random crew member injured
                member = np.random.choice(self.crew)
                member.health -= 30
                self.emergency_alerts['medical_emergency'] = True
                self.emergency_count += 1
                
            elif emergency_type == 'solar_storm':
                # Radiation event
                self.radiation_level = 0.8
                self.emergency_alerts['radiation_storm'] = True
                self.emergency_count += 1
        
        # Check for critical conditions
        avg_o2 = np.mean([m.oxygen_percent for m in self.modules.values()])
        if avg_o2 < 18:
            self.emergency_alerts['life_support_critical'] = True
        
        if self.battery_level < 5:
            self.emergency_alerts['power_failure'] = True
    
    def _update_system_degradation(self):
        """Update system wear and degradation"""
        for system in self.systems.values():
            # Natural degradation
            system.efficiency -= 0.01
            system.maintenance_needed += 0.05
            
            # Accelerated degradation if not maintained
            if self.time_step - system.last_maintenance > 200:
                system.efficiency -= 0.1
                system.maintenance_needed += 0.2
            
            # Random failure chance
            if np.random.random() < system.failure_probability:
                system.efficiency = max(0, system.efficiency - 20)
            
            # Clamp values
            system.efficiency = np.clip(system.efficiency, 0, 100)
            system.maintenance_needed = np.clip(system.maintenance_needed, 0, 100)
    
    def _activate_emergency_protocols(self):
        """Activate emergency response procedures"""
        # Boost critical systems
        for system in self.systems.values():
            if system.is_critical:
                system.efficiency = min(100, system.efficiency + 20)
        
        # Use emergency oxygen
        if self.resources['oxygen_tanks'] > 0:
            for module in self.modules.values():
                module.oxygen_percent = min(25, module.oxygen_percent + 2)
            self.resources['oxygen_tanks'] -= 0.5
        
        # Alert crew
        for member in self.crew:
            member.task = "emergency_response"
    
    def _initiate_evacuation(self):
        """Initiate evacuation procedures"""
        # Move all crew to docking module
        for member in self.crew:
            member.current_module = 'Docking'
            member.task = "evacuation"
        
        # Prepare docking module
        docking = self.modules['Docking']
        docking.pressure = 101.3
        docking.oxygen_percent = 21.0
        docking.temperature = 22.0
    
    def _calculate_reward(self) -> float:
        """Calculate step reward"""
        reward = 0.0
        
        # Survival bonus
        crew_alive = sum(1 for c in self.crew if c.health > 0)
        reward += crew_alive * 10
        
        # Life support quality
        avg_o2 = np.mean([m.oxygen_percent for m in self.modules.values()])
        avg_co2 = np.mean([m.co2_ppm for m in self.modules.values()])
        
        if 19 < avg_o2 < 23:
            reward += 50
        if avg_co2 < 1000:
            reward += 30
        
        # System efficiency
        avg_efficiency = np.mean([s.efficiency for s in self.systems.values()])
        if avg_efficiency > 90:
            reward += 200
        elif avg_efficiency > 70:
            reward += 50
        
        # Resource management
        if all(self.resources[r] > 20 for r in ['water', 'food', 'oxygen_tanks']):
            reward += 100
        
        # Emergency handling
        if any(self.emergency_alerts.values()):
            if self.time_step - self.last_emergency_time > 10:
                # New emergency
                self.last_emergency_time = self.time_step
            else:
                # Handling ongoing emergency
                if crew_alive == len(self.crew):
                    reward += 500  # Successfully handling emergency
        
        # Penalties
        if self.crew_casualties > 0:
            reward -= 5000 * self.crew_casualties
        
        for alert_type, is_active in self.emergency_alerts.items():
            if is_active:
                if alert_type in ['life_support_critical', 'depressurization']:
                    reward -= 2000
                else:
                    reward -= 500
        
        # Resource shortage penalties
        for resource, amount in self.resources.items():
            if amount < 10:
                reward -= 100
        
        # Optimal conditions bonus
        if (self.battery_level > 50 and avg_efficiency > 80 and 
            crew_alive == len(self.crew) and not any(self.emergency_alerts.values())):
            reward += 500
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Mission success - survived 30 days
        if self.time_step >= self.max_steps:
            self.mission_success = True
            return True
        
        # All crew dead
        if all(c.health <= 0 for c in self.crew):
            return True
        
        # Critical system cascade failure
        critical_failed = sum(1 for s in self.systems.values() 
                             if s.is_critical and s.efficiency < 10)
        if critical_failed >= 3:
            return True
        
        # Station uninhabitable
        avg_o2 = np.mean([m.oxygen_percent for m in self.modules.values()])
        avg_pressure = np.mean([m.pressure for m in self.modules.values()])
        if avg_o2 < 15 or avg_pressure < 80:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector (110 elements)"""
        obs = []
        
        # Crew member states (24 values)
        for member in self.crew:
            obs.extend([member.x, member.y, member.health, member.oxygen_level])
        
        # System states (36 values)
        for system in self.systems.values():
            obs.extend([system.efficiency, system.power_draw, system.maintenance_needed])
        
        # Module atmospherics (32 values)
        for module in self.modules.values():
            obs.extend([module.oxygen_percent, module.co2_ppm, module.pressure, module.temperature])
        
        # Power systems (10 values)
        obs.extend([
            self.solar_efficiency,
            self.battery_level,
            self.total_power_consumption,
            float(self.backup_power_available),
            self.power_generation,
            0.0, 0.0, 0.0, 0.0, 0.0  # Reserved for future power metrics
        ])
        
        # Resources (8 values)
        for resource in ['water', 'food', 'oxygen_tanks', 'nitrogen', 
                        'medical_supplies', 'spare_parts', 'fuel', 'waste_capacity']:
            obs.append(self.resources[resource])
        
        # Orbital mechanics (4 values)
        obs.extend([
            self.orbital_phase / 360.0,
            math.cos(self.solar_angle),
            float(self.in_earth_shadow),
            self.radiation_level
        ])
        
        # Emergency alerts (6 values)
        for alert in ['fire', 'depressurization', 'radiation_storm', 
                     'power_failure', 'life_support_critical', 'medical_emergency']:
            obs.append(float(self.emergency_alerts[alert]))
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        return {
            'time_step': self.time_step,
            'day': (self.time_step * self.minutes_per_step) // (24 * 60),
            'crew_alive': sum(1 for c in self.crew if c.health > 0),
            'system_failures': self.system_failures,
            'emergency_count': self.emergency_count,
            'mission_success': self.mission_success,
            'episode_reward': self.episode_reward,
            'battery_level': self.battery_level,
            'avg_crew_health': np.mean([c.health for c in self.crew])
        }
    
    def render(self):
        """Render the environment using pygame"""
        if self.render_mode == "human":
            return self._render_pygame()
        return None
    
    def _render_pygame(self):
        """Detailed pygame visualization"""
        import pygame
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((1200, 900))
            pygame.display.set_caption("Space Station Life Support Management")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
        
        # Clear screen
        self.screen.fill((10, 10, 30))  # Dark space background
        
        # Draw station modules
        self._draw_modules()
        
        # Draw crew members
        self._draw_crew()
        
        # Draw system status panel
        self._draw_systems_panel()
        
        # Draw resource gauges
        self._draw_resources()
        
        # Draw orbital display
        self._draw_orbital_display()
        
        # Draw alerts
        self._draw_alerts()
        
        # Draw mission info
        self._draw_mission_info()
        
        pygame.display.flip()
        self.clock.tick(30)
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), 
            axes=(1, 0, 2)
        )
    
    def _draw_modules(self):
        """Draw station modules"""
        import pygame
        
        scale = 40  # pixels per unit
        offset_x, offset_y = 50, 50
        
        for name, module in self.modules.items():
            x = offset_x + module.x * scale
            y = offset_y + module.y * scale
            w = module.width * scale
            h = module.height * scale
            
            # Determine module color based on atmosphere
            if module.oxygen_percent < 19:
                color = (200, 100, 100)  # Red tint for low O2
            elif module.co2_ppm > 2000:
                color = (200, 200, 100)  # Yellow for high CO2
            else:
                color = (100, 200, 100)  # Green for good atmosphere
            
            # Draw module
            pygame.draw.rect(self.screen, color, (x, y, w, h), 2)
            
            # Draw module name
            text = self.small_font.render(name, True, (255, 255, 255))
            text_rect = text.get_rect(center=(x + w//2, y + h//2))
            self.screen.blit(text, text_rect)
            
            # Draw atmosphere info
            o2_text = self.small_font.render(f"O2: {module.oxygen_percent:.1f}%", True, (200, 200, 255))
            self.screen.blit(o2_text, (x + 5, y + 5))
            
            temp_text = self.small_font.render(f"T: {module.temperature:.1f}°C", True, (255, 200, 200))
            self.screen.blit(temp_text, (x + 5, y + 25))
    
    def _draw_crew(self):
        """Draw crew members"""
        import pygame
        
        scale = 40
        offset_x, offset_y = 50, 50
        
        for member in self.crew:
            if member.health <= 0:
                continue
                
            x = offset_x + member.x * scale
            y = offset_y + member.y * scale
            
            # Crew color based on health
            if member.health > 70:
                color = (100, 255, 100)
            elif member.health > 40:
                color = (255, 255, 100)
            else:
                color = (255, 100, 100)
            
            # Draw crew member
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
            
            # Draw role label
            role_text = self.small_font.render(member.role[:3], True, (255, 255, 255))
            self.screen.blit(role_text, (x - 10, y + 10))
            
            # Draw health bar
            bar_width = 30
            bar_height = 4
            pygame.draw.rect(self.screen, (100, 100, 100), 
                           (x - bar_width//2, y - 20, bar_width, bar_height))
            pygame.draw.rect(self.screen, color,
                           (x - bar_width//2, y - 20, 
                            int(bar_width * member.health / 100), bar_height))
    
    def _draw_systems_panel(self):
        """Draw system status panel"""
        import pygame
        
        panel_x = 600
        panel_y = 50
        panel_width = 550
        panel_height = 400
        
        # Panel background
        pygame.draw.rect(self.screen, (30, 30, 50), 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, (100, 100, 150), 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font.render("System Status", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Draw each system
        y_offset = 40
        for i, (name, system) in enumerate(self.systems.items()):
            y = panel_y + y_offset + i * 30
            
            # System name
            name_text = self.small_font.render(name, True, (200, 200, 255))
            self.screen.blit(name_text, (panel_x + 10, y))
            
            # Efficiency bar
            bar_x = panel_x + 200
            bar_width = 150
            bar_height = 20
            
            # Background
            pygame.draw.rect(self.screen, (50, 50, 50), 
                           (bar_x, y, bar_width, bar_height))
            
            # Efficiency level
            if system.efficiency > 70:
                bar_color = (100, 255, 100)
            elif system.efficiency > 40:
                bar_color = (255, 255, 100)
            else:
                bar_color = (255, 100, 100)
            
            pygame.draw.rect(self.screen, bar_color,
                           (bar_x, y, int(bar_width * system.efficiency / 100), bar_height))
            
            # Efficiency text
            eff_text = self.small_font.render(f"{system.efficiency:.1f}%", True, (255, 255, 255))
            self.screen.blit(eff_text, (bar_x + bar_width + 10, y))
            
            # Maintenance indicator
            if system.maintenance_needed > 50:
                maint_text = self.small_font.render("MAINT", True, (255, 150, 0))
                self.screen.blit(maint_text, (bar_x + bar_width + 70, y))
    
    def _draw_resources(self):
        """Draw resource gauges"""
        import pygame
        
        gauge_x = 50
        gauge_y = 500
        gauge_size = 80
        spacing = 100
        
        resources_to_show = ['water', 'food', 'oxygen_tanks', 'battery_level']
        max_values = {'water': 1000, 'food': 500, 'oxygen_tanks': 20, 'battery_level': 100}
        
        for i, resource in enumerate(resources_to_show):
            x = gauge_x + i * spacing
            y = gauge_y
            
            # Get value
            if resource == 'battery_level':
                value = self.battery_level
                max_val = 100
            else:
                value = self.resources[resource]
                max_val = max_values[resource]
            
            # Draw gauge background
            pygame.draw.circle(self.screen, (50, 50, 50), (x + gauge_size//2, y + gauge_size//2), gauge_size//2, 2)
            
            # Draw gauge fill
            if value / max_val > 0.5:
                color = (100, 255, 100)
            elif value / max_val > 0.25:
                color = (255, 255, 100)
            else:
                color = (255, 100, 100)
            
            # Draw arc for gauge
            import math
            angle = -90 - (270 * (value / max_val))
            for a in range(-90, int(angle), -1):
                end_x = x + gauge_size//2 + int(gauge_size//2 * 0.8 * math.cos(math.radians(a)))
                end_y = y + gauge_size//2 + int(gauge_size//2 * 0.8 * math.sin(math.radians(a)))
                pygame.draw.line(self.screen, color, 
                               (x + gauge_size//2, y + gauge_size//2), 
                               (end_x, end_y), 3)
            
            # Label
            label = self.small_font.render(resource.replace('_', ' ').title(), True, (255, 255, 255))
            label_rect = label.get_rect(center=(x + gauge_size//2, y + gauge_size + 10))
            self.screen.blit(label, label_rect)
            
            # Value
            value_text = self.small_font.render(f"{value:.1f}", True, (255, 255, 255))
            value_rect = value_text.get_rect(center=(x + gauge_size//2, y + gauge_size//2))
            self.screen.blit(value_text, value_rect)
    
    def _draw_orbital_display(self):
        """Draw orbital position display"""
        import pygame
        import math
        
        center_x = 950
        center_y = 550
        earth_radius = 30
        orbit_radius = 80
        
        # Draw Earth
        pygame.draw.circle(self.screen, (100, 150, 255), (center_x, center_y), earth_radius)
        pygame.draw.circle(self.screen, (50, 100, 200), (center_x, center_y), earth_radius, 2)
        
        # Draw orbit path
        pygame.draw.circle(self.screen, (100, 100, 100), (center_x, center_y), orbit_radius, 1)
        
        # Draw sun direction
        sun_x = center_x + int(150 * math.cos(math.radians(45)))
        sun_y = center_y - int(150 * math.sin(math.radians(45)))
        pygame.draw.circle(self.screen, (255, 255, 100), (sun_x, sun_y), 20)
        
        # Draw station position
        station_x = center_x + int(orbit_radius * math.cos(math.radians(self.orbital_phase)))
        station_y = center_y - int(orbit_radius * math.sin(math.radians(self.orbital_phase)))
        pygame.draw.circle(self.screen, (255, 255, 255), (station_x, station_y), 5)
        
        # Draw shadow region
        if self.in_earth_shadow:
            shadow_color = (50, 50, 100, 128)
            # Simple shadow indication
            pygame.draw.arc(self.screen, (50, 50, 100), 
                          (center_x - orbit_radius, center_y - orbit_radius, 
                           orbit_radius * 2, orbit_radius * 2),
                          math.radians(180), math.radians(270), 3)
        
        # Orbital info
        orbit_text = self.small_font.render(f"Orbit: {self.orbital_phase:.1f}°", True, (255, 255, 255))
        self.screen.blit(orbit_text, (center_x - 50, center_y + orbit_radius + 20))
        
        solar_text = self.small_font.render(f"Solar: {self.solar_efficiency*100:.1f}%", True, (255, 255, 100))
        self.screen.blit(solar_text, (center_x - 50, center_y + orbit_radius + 40))
    
    def _draw_alerts(self):
        """Draw emergency alerts"""
        import pygame
        
        alert_x = 600
        alert_y = 700
        
        active_alerts = [name for name, active in self.emergency_alerts.items() if active]
        
        if active_alerts:
            # Draw alert box
            pygame.draw.rect(self.screen, (150, 50, 50), 
                           (alert_x, alert_y, 550, 100))
            pygame.draw.rect(self.screen, (255, 100, 100), 
                           (alert_x, alert_y, 550, 100), 3)
            
            # Alert title
            alert_title = self.font.render("⚠ EMERGENCY ALERTS ⚠", True, (255, 255, 100))
            self.screen.blit(alert_title, (alert_x + 10, alert_y + 10))
            
            # List alerts
            for i, alert in enumerate(active_alerts):
                alert_text = self.small_font.render(f"• {alert.replace('_', ' ').upper()}", 
                                                   True, (255, 200, 200))
                self.screen.blit(alert_text, (alert_x + 20, alert_y + 40 + i * 20))
    
    def _draw_mission_info(self):
        """Draw mission information"""
        import pygame
        
        info_x = 50
        info_y = 700
        
        # Mission day
        day = (self.time_step * self.minutes_per_step) // (24 * 60)
        day_text = self.font.render(f"Mission Day: {day}/{self.mission_duration_days}", 
                                   True, (255, 255, 255))
        self.screen.blit(day_text, (info_x, info_y))
        
        # Crew status
        crew_alive = sum(1 for c in self.crew if c.health > 0)
        crew_text = self.font.render(f"Crew: {crew_alive}/{len(self.crew)}", 
                                    True, (100, 255, 100) if crew_alive == len(self.crew) else (255, 100, 100))
        self.screen.blit(crew_text, (info_x, info_y + 30))
        
        # Power status
        power_text = self.font.render(f"Power: {self.battery_level:.1f}% | Load: {self.total_power_consumption:.0f}W", 
                                     True, (255, 255, 100))
        self.screen.blit(power_text, (info_x, info_y + 60))
        
        # Score
        score_text = self.font.render(f"Score: {self.episode_reward:.0f}", 
                                     True, (255, 255, 255))
        self.screen.blit(score_text, (info_x, info_y + 90))
    
    def close(self):
        """Close the environment"""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()