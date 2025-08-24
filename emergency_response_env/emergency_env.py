"""
Emergency Response Environment for Reinforcement Learning

A comprehensive emergency management simulation where an AI agent coordinates
disaster response across a metropolitan area with realistic constraints,
multiple emergency types, and complex resource allocation challenges.

Author: Hasnain Fareed
License: MIT (2025)
"""

import gymnasium as gym
import numpy as np
import pygame
import random
import math
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time


class EmergencyType(Enum):
    """Types of emergencies that can occur in the simulation"""
    BUILDING_FIRE = 0
    MULTI_VEHICLE_ACCIDENT = 1
    CHEMICAL_SPILL = 2
    FLOOD = 3
    EARTHQUAKE = 4
    GAS_LEAK = 5
    MEDICAL_EMERGENCY = 6
    CIVIL_UNREST = 7


class UnitType(Enum):
    """Types of emergency response units"""
    FIRE_TRUCK = 0
    POLICE_CAR = 1
    AMBULANCE = 2
    SEARCH_RESCUE = 3
    HAZMAT = 4
    COMMAND = 5


class ZoneType(Enum):
    """City zone classifications"""
    RESIDENTIAL = 0
    COMMERCIAL = 1
    INDUSTRIAL = 2


@dataclass
class EmergencyIncident:
    """Represents an active emergency incident"""
    x: int
    y: int
    emergency_type: EmergencyType
    severity: int  # 1-10 scale
    start_time: int
    casualties: int = 0
    contained: bool = False
    escalation_risk: float = 0.0
    required_units: Dict[UnitType, int] = field(default_factory=dict)
    active_units: List[int] = field(default_factory=list)


@dataclass
class ResponseUnit:
    """Represents an emergency response unit"""
    unit_id: int
    unit_type: UnitType
    x: int
    y: int
    home_x: int
    home_y: int
    busy: bool = False
    readiness: float = 1.0  # 0.0 to 1.0
    fatigue: float = 0.0   # 0.0 to 1.0
    assigned_incident: Optional[int] = None
    travel_time: int = 0


@dataclass
class Hospital:
    """Hospital facility information"""
    x: int
    y: int
    beds_available: int
    trauma_capacity: int
    max_beds: int
    max_trauma: int


class WeatherCondition:
    """Weather conditions affecting emergency response"""
    def __init__(self):
        self.visibility = 1.0      # 0.0 to 1.0
        self.road_conditions = 1.0  # 0.0 to 1.0
        self.wind_speed = 0.0      # 0.0 to 1.0
        self.precipitation = 0.0   # 0.0 to 1.0
        self.temperature = 0.5     # 0.0 to 1.0 (normalized)


class EmergencyResponseEnv(gym.Env):
    """
    Emergency Response Coordination Environment
    
    The agent acts as an Emergency Response Commander coordinating
    disaster response across a 30x30 metropolitan area.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Environment parameters
        self.grid_size = 30
        self.max_emergencies = 20
        self.num_units = 40
        self.num_hospitals = 8
        self.max_timesteps = 1440  # 24 hours
        
        # Pygame setup
        self.render_mode = render_mode
        self.window_size = (1200, 900)
        self.window = None
        self.clock = None
        
        # State and action spaces
        self._setup_spaces()
        
        # Initialize city infrastructure
        self._initialize_city()
        self._initialize_units()
        self._initialize_hospitals()
        
        # Game state
        self.current_timestep = 0
        self.active_emergencies: Dict[int, EmergencyIncident] = {}
        self.emergency_counter = 0
        self.weather = WeatherCondition()
        self.state_of_emergency = False
        self.mutual_aid_requested = False
        self.national_guard_active = False
        
        # Statistics tracking
        self.total_casualties = 0
        self.emergencies_resolved = 0
        self.lives_saved = 0
        self.response_times = []
        
        # Emergency generation parameters
        self.base_emergency_rate = 0.02  # Base probability per timestep
        self.cascade_multiplier = 1.0
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # State space: 135 elements as specified
        # - Active emergencies: 20 * 4 = 80
        # - Unit positions/status: 40 * 3 = 120  
        # - Unit specialization: 40 * 2 = 80
        # - Traffic congestion: 25 major roads
        # - Hospital capacity: 8 * 3 = 24
        # - Weather: 5 values
        # - Public safety: 6 zones * 2 = 12
        # - Communication: 10 values
        # Total: 80 + 120 + 80 + 25 + 24 + 5 + 12 + 10 = 356 (simplified to key elements)
        
        state_size = (
            self.max_emergencies * 4 +  # Emergency data: 80
            self.num_units * 3 +         # Unit positions/status: 120
            25 +                         # Traffic segments: 25
            self.num_hospitals * 3 +     # Hospital capacity: 24
            5 +                          # Weather: 5
            12 +                         # Public safety zones: 12
            10                           # Communication status: 10
        )  # Total: 276
        
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=10.0,
            shape=(276,),
            dtype=np.float32
        )
        
        # Action space: 50 discrete actions as specified
        self.action_space = gym.spaces.Discrete(50)
        
    def _initialize_city(self):
        """Initialize city grid with zones and population density"""
        self.city_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.population_density = np.zeros((self.grid_size, self.grid_size))
        self.traffic_congestion = np.zeros(25)  # 25 major road segments
        
        # Create zones (simplified pattern)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i < 10 or i > 20:  # Residential areas
                    self.city_grid[i, j] = ZoneType.RESIDENTIAL.value
                    self.population_density[i, j] = random.randint(100, 300)
                elif 10 <= i <= 15:  # Commercial district
                    self.city_grid[i, j] = ZoneType.COMMERCIAL.value
                    self.population_density[i, j] = random.randint(50, 500)
                else:  # Industrial area
                    self.city_grid[i, j] = ZoneType.INDUSTRIAL.value
                    self.population_density[i, j] = random.randint(20, 100)
        
        # Initialize evacuation zones
        self.evacuation_zones = np.zeros(6, dtype=bool)
        self.zone_casualties = np.zeros(6)
        
        # Communication network
        self.radio_towers = 8
        self.cellular_coverage = 0.95
        self.emergency_broadcast_reach = 0.90
        
    def _initialize_units(self):
        """Initialize emergency response units"""
        self.response_units: List[ResponseUnit] = []
        
        # Unit distribution as specified:
        # 12 Fire trucks, 8 Police cars, 10 Ambulances, 
        # 6 Search & Rescue, 2 Hazmat, 2 Command
        unit_configs = [
            (UnitType.FIRE_TRUCK, 12),
            (UnitType.POLICE_CAR, 8),
            (UnitType.AMBULANCE, 10),
            (UnitType.SEARCH_RESCUE, 6),
            (UnitType.HAZMAT, 2),
            (UnitType.COMMAND, 2)
        ]
        
        unit_id = 0
        for unit_type, count in unit_configs:
            for _ in range(count):
                # Distribute units across appropriate stations
                if unit_type == UnitType.FIRE_TRUCK:
                    # 4 fire stations
                    stations = [(5, 5), (25, 5), (5, 25), (25, 25)]
                    station = stations[unit_id % 4]
                elif unit_type == UnitType.POLICE_CAR:
                    # 3 police precincts
                    stations = [(10, 10), (20, 10), (15, 20)]
                    station = stations[unit_id % 3]
                elif unit_type == UnitType.AMBULANCE:
                    # 5 ambulance stations
                    stations = [(8, 8), (22, 8), (8, 22), (22, 22), (15, 15)]
                    station = stations[unit_id % 5]
                else:
                    # Other units distributed randomly
                    station = (random.randint(5, 25), random.randint(5, 25))
                
                unit = ResponseUnit(
                    unit_id=unit_id,
                    unit_type=unit_type,
                    x=station[0],
                    y=station[1],
                    home_x=station[0],
                    home_y=station[1]
                )
                self.response_units.append(unit)
                unit_id += 1
                
    def _initialize_hospitals(self):
        """Initialize hospital facilities"""
        self.hospitals: List[Hospital] = []
        hospital_locations = [
            (7, 7), (23, 7), (7, 23), (23, 23),
            (15, 10), (10, 15), (20, 20), (15, 25)
        ]
        
        for i, (x, y) in enumerate(hospital_locations):
            hospital = Hospital(
                x=x, y=y,
                beds_available=random.randint(50, 150),
                trauma_capacity=random.randint(10, 30),
                max_beds=random.randint(80, 200),
                max_trauma=random.randint(15, 40)
            )
            self.hospitals.append(hospital)
    
    def _generate_emergencies(self):
        """Generate new emergency incidents based on current conditions"""
        # Adjust emergency rate based on conditions
        emergency_rate = self.base_emergency_rate * self.cascade_multiplier
        
        # Weather effects
        if self.weather.precipitation > 0.7:
            emergency_rate *= 1.5  # More accidents in heavy rain
        if self.weather.wind_speed > 0.8:
            emergency_rate *= 1.3  # Wind increases fire spread risk
            
        # Generate new emergencies
        if random.random() < emergency_rate and len(self.active_emergencies) < self.max_emergencies:
            self._create_emergency()
            
        # Update cascade effects
        self._update_cascade_effects()
    
    def _create_emergency(self):
        """Create a new emergency incident"""
        # Choose emergency type based on zone and conditions
        x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
        zone_type = ZoneType(self.city_grid[x, y])
        
        # Emergency type probabilities by zone
        if zone_type == ZoneType.RESIDENTIAL:
            emergency_types = [
                (EmergencyType.BUILDING_FIRE, 0.3),
                (EmergencyType.MEDICAL_EMERGENCY, 0.4),
                (EmergencyType.GAS_LEAK, 0.2),
                (EmergencyType.CIVIL_UNREST, 0.1)
            ]
        elif zone_type == ZoneType.COMMERCIAL:
            emergency_types = [
                (EmergencyType.BUILDING_FIRE, 0.25),
                (EmergencyType.MULTI_VEHICLE_ACCIDENT, 0.3),
                (EmergencyType.MEDICAL_EMERGENCY, 0.25),
                (EmergencyType.CIVIL_UNREST, 0.2)
            ]
        else:  # Industrial
            emergency_types = [
                (EmergencyType.CHEMICAL_SPILL, 0.3),
                (EmergencyType.BUILDING_FIRE, 0.2),
                (EmergencyType.GAS_LEAK, 0.25),
                (EmergencyType.MULTI_VEHICLE_ACCIDENT, 0.25)
            ]
        
        # Select emergency type
        rand_val = random.random()
        cumulative = 0
        selected_type = emergency_types[0][0]
        for emergency_type, prob in emergency_types:
            cumulative += prob
            if rand_val <= cumulative:
                selected_type = emergency_type
                break
        
        # Create incident
        severity = random.randint(1, 10)
        incident = EmergencyIncident(
            x=x, y=y,
            emergency_type=selected_type,
            severity=severity,
            start_time=self.current_timestep,
            casualties=max(0, int(severity * self.population_density[x, y] / 1000))
        )
        
        # Set required units based on emergency type
        incident.required_units = self._get_required_units(selected_type, severity)
        
        self.active_emergencies[self.emergency_counter] = incident
        self.emergency_counter += 1
        
    def _get_required_units(self, emergency_type: EmergencyType, severity: int) -> Dict[UnitType, int]:
        """Determine required units for an emergency type"""
        requirements = {}
        
        if emergency_type == EmergencyType.BUILDING_FIRE:
            requirements[UnitType.FIRE_TRUCK] = min(4, max(2, severity // 3))
            requirements[UnitType.AMBULANCE] = min(2, max(1, severity // 5))
            if severity > 7:
                requirements[UnitType.SEARCH_RESCUE] = 1
                
        elif emergency_type == EmergencyType.MULTI_VEHICLE_ACCIDENT:
            requirements[UnitType.POLICE_CAR] = min(3, max(1, severity // 4))
            requirements[UnitType.AMBULANCE] = min(4, max(1, severity // 3))
            if severity > 6:
                requirements[UnitType.FIRE_TRUCK] = 1
                
        elif emergency_type == EmergencyType.CHEMICAL_SPILL:
            requirements[UnitType.HAZMAT] = min(2, max(1, severity // 5))
            requirements[UnitType.FIRE_TRUCK] = 1
            requirements[UnitType.AMBULANCE] = 1
            
        elif emergency_type == EmergencyType.FLOOD:
            requirements[UnitType.SEARCH_RESCUE] = min(3, max(1, severity // 3))
            requirements[UnitType.AMBULANCE] = min(2, max(1, severity // 5))
            
        elif emergency_type == EmergencyType.EARTHQUAKE:
            requirements[UnitType.SEARCH_RESCUE] = min(4, max(2, severity // 2))
            requirements[UnitType.AMBULANCE] = min(5, max(2, severity // 2))
            requirements[UnitType.FIRE_TRUCK] = min(3, max(1, severity // 3))
            
        elif emergency_type == EmergencyType.GAS_LEAK:
            requirements[UnitType.FIRE_TRUCK] = min(2, max(1, severity // 4))
            requirements[UnitType.HAZMAT] = 1
            
        elif emergency_type == EmergencyType.MEDICAL_EMERGENCY:
            requirements[UnitType.AMBULANCE] = min(3, max(1, severity // 4))
            
        elif emergency_type == EmergencyType.CIVIL_UNREST:
            requirements[UnitType.POLICE_CAR] = min(5, max(2, severity // 2))
            requirements[UnitType.AMBULANCE] = 1
            
        return requirements
    
    def _update_cascade_effects(self):
        """Update cascade effects and emergency escalation"""
        # Create a copy of items to avoid dictionary size change during iteration
        active_incidents = list(self.active_emergencies.items())
        for incident_id, incident in active_incidents:
            # Calculate escalation risk
            time_factor = (self.current_timestep - incident.start_time) / 60  # Hours
            severity_factor = incident.severity / 10.0
            response_factor = len(incident.active_units) / max(1, sum(incident.required_units.values()))
            
            incident.escalation_risk = min(1.0, time_factor * severity_factor * (1.0 - response_factor))
            
            # Check for escalation
            if incident.escalation_risk > 0.8 and random.random() < 0.1:
                self._escalate_emergency(incident)
                
            # Check for cascade events
            if incident.escalation_risk > 0.6 and random.random() < 0.05:
                self._trigger_cascade_event(incident)
    
    def _escalate_emergency(self, incident: EmergencyIncident):
        """Escalate an emergency incident"""
        incident.severity = min(10, incident.severity + 1)
        incident.casualties += random.randint(1, 5)
        self.total_casualties += random.randint(1, 5)
        
        # Update required units
        incident.required_units = self._get_required_units(incident.emergency_type, incident.severity)
    
    def _trigger_cascade_event(self, incident: EmergencyIncident):
        """Trigger a cascade emergency near an existing one"""
        if len(self.active_emergencies) >= self.max_emergencies:
            return
            
        # Create nearby emergency
        new_x = max(0, min(self.grid_size-1, incident.x + random.randint(-3, 3)))
        new_y = max(0, min(self.grid_size-1, incident.y + random.randint(-3, 3)))
        
        # Cascade emergency types
        cascade_types = {
            EmergencyType.BUILDING_FIRE: EmergencyType.BUILDING_FIRE,
            EmergencyType.MULTI_VEHICLE_ACCIDENT: EmergencyType.MULTI_VEHICLE_ACCIDENT,
            EmergencyType.CHEMICAL_SPILL: EmergencyType.GAS_LEAK,
            EmergencyType.EARTHQUAKE: EmergencyType.BUILDING_FIRE,
            EmergencyType.GAS_LEAK: EmergencyType.BUILDING_FIRE
        }
        
        cascade_type = cascade_types.get(incident.emergency_type, EmergencyType.MEDICAL_EMERGENCY)
        
        cascade_incident = EmergencyIncident(
            x=new_x, y=new_y,
            emergency_type=cascade_type,
            severity=random.randint(3, 7),
            start_time=self.current_timestep,
            casualties=random.randint(0, 3)
        )
        
        cascade_incident.required_units = self._get_required_units(cascade_type, cascade_incident.severity)
        self.active_emergencies[self.emergency_counter] = cascade_incident
        self.emergency_counter += 1
        
        # Increase cascade multiplier
        self.cascade_multiplier = min(3.0, self.cascade_multiplier + 0.1)
    
    def _update_units(self):
        """Update response unit status and positions"""
        for unit in self.response_units:
            if unit.busy and unit.assigned_incident is not None:
                incident = self.active_emergencies.get(unit.assigned_incident)
                if incident:
                    # Move towards incident
                    if unit.travel_time > 0:
                        unit.travel_time -= 1
                    else:
                        # Arrived at incident
                        unit.x = incident.x
                        unit.y = incident.y
                        
                        # Work on incident
                        self._work_on_incident(unit, incident)
                else:
                    # Incident resolved, return to base
                    unit.busy = False
                    unit.assigned_incident = None
                    unit.x = unit.home_x
                    unit.y = unit.home_y
            
            # Update fatigue and readiness
            if unit.busy:
                unit.fatigue = min(1.0, unit.fatigue + 0.01)
                unit.readiness = max(0.3, 1.0 - unit.fatigue)
            else:
                unit.fatigue = max(0.0, unit.fatigue - 0.005)
                unit.readiness = min(1.0, 1.0 - unit.fatigue * 0.5)
    
    def _work_on_incident(self, unit: ResponseUnit, incident: EmergencyIncident):
        """Unit works on resolving an incident"""
        # Effectiveness based on unit type, readiness, and weather
        effectiveness = unit.readiness * self.weather.visibility * self.weather.road_conditions
        
        # Reduce incident severity
        if random.random() < effectiveness * 0.1:
            incident.severity = max(0, incident.severity - 1)
            
            # Chance to save lives
            if random.random() < 0.3:
                self.lives_saved += 1
                
        # Check if incident is resolved
        if incident.severity <= 0:
            incident.contained = True
            self.emergencies_resolved += 1
            
            # Calculate response time
            response_time = self.current_timestep - incident.start_time
            self.response_times.append(response_time)
            
            # Free up assigned units
            for unit_id in incident.active_units:
                if unit_id < len(self.response_units):
                    self.response_units[unit_id].busy = False
                    self.response_units[unit_id].assigned_incident = None
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        obs = np.zeros(276, dtype=np.float32)
        idx = 0
        
        # Active emergencies (20 * 4 = 80)
        for i in range(self.max_emergencies):
            if i < len(list(self.active_emergencies.values())):
                incident = list(self.active_emergencies.values())[i]
                obs[idx] = incident.x / self.grid_size
                obs[idx+1] = incident.y / self.grid_size
                obs[idx+2] = incident.emergency_type.value / 7.0
                obs[idx+3] = incident.severity / 10.0
            idx += 4
            
        # Response units (40 * 3 = 120) - limit to original units to avoid index overflow
        for i in range(min(self.num_units, len(self.response_units))):
            unit = self.response_units[i]
            obs[idx] = unit.x / self.grid_size
            obs[idx+1] = unit.y / self.grid_size
            obs[idx+2] = 1.0 if unit.busy else 0.0
            idx += 3
            
        # Traffic congestion (25)
        for i in range(25):
            obs[idx] = self.traffic_congestion[i]
            idx += 1
            
        # Hospital capacity (8 * 3 = 24)
        for i in range(min(self.num_hospitals, len(self.hospitals))):
            hospital = self.hospitals[i]
            obs[idx] = hospital.beds_available / max(1, hospital.max_beds)
            obs[idx+1] = hospital.trauma_capacity / max(1, hospital.max_trauma)
            obs[idx+2] = min(1.0, sum(math.sqrt((inc.x - hospital.x)**2 + (inc.y - hospital.y)**2) 
                                      for inc in self.active_emergencies.values()) / 100.0)
            idx += 3
            
        # Weather conditions (5)
        obs[idx] = self.weather.visibility
        obs[idx+1] = self.weather.road_conditions
        obs[idx+2] = self.weather.wind_speed
        obs[idx+3] = self.weather.precipitation
        obs[idx+4] = self.weather.temperature
        idx += 5
        
        # Public safety zones (6 * 2 = 12)
        for i in range(6):
            obs[idx] = 1.0 if self.evacuation_zones[i] else 0.0
            obs[idx+1] = self.zone_casualties[i] / 100.0
            idx += 2
            
        # Communication status (10)
        obs[idx] = self.radio_towers / 10.0
        obs[idx+1] = self.cellular_coverage
        obs[idx+2] = self.emergency_broadcast_reach
        obs[idx+3] = 1.0 if self.state_of_emergency else 0.0
        obs[idx+4] = 1.0 if self.mutual_aid_requested else 0.0
        obs[idx+5] = 1.0 if self.national_guard_active else 0.0
        obs[idx+6] = len(self.active_emergencies) / self.max_emergencies
        obs[idx+7] = sum(u.busy for u in self.response_units[:self.num_units]) / self.num_units
        obs[idx+8] = self.total_casualties / 100.0
        obs[idx+9] = self.current_timestep / self.max_timesteps
        
        return obs
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on current state and action"""
        reward = 0.0
        
        # Positive rewards
        new_resolved = self.emergencies_resolved - getattr(self, '_prev_resolved', 0)
        if new_resolved > 0:
            reward += 2000 * new_resolved  # Emergency resolved
            
        new_lives_saved = self.lives_saved - getattr(self, '_prev_lives_saved', 0)
        if new_lives_saved > 0:
            reward += 5000 * new_lives_saved  # Lives saved
            
        # Quick response bonus
        if self.response_times and self.response_times[-1] <= 15:  # 15 minutes
            reward += 1000
            
        # Resource coordination bonus
        active_unit_types = set()
        for unit in self.response_units:
            if unit.busy:
                active_unit_types.add(unit.unit_type)
        if len(active_unit_types) >= 3:
            reward += 500  # Multi-agency coordination
            
        # Prevention bonus
        if self.cascade_multiplier < 1.2:  # Low cascade effect
            reward += 300
            
        # Negative rewards
        new_casualties = self.total_casualties - getattr(self, '_prev_casualties', 0)
        if new_casualties > 0:
            reward -= 10000 * new_casualties  # Casualties
            
        # Escalation penalty
        escalated_incidents = sum(1 for inc in self.active_emergencies.values() 
                                if inc.escalation_risk > 0.8)
        reward -= 2000 * escalated_incidents
        
        # Resource misallocation penalty
        overallocated = sum(1 for inc in self.active_emergencies.values()
                          if len(inc.active_units) > sum(inc.required_units.values()) * 1.5)
        reward -= 1000 * overallocated
        
        # Traffic gridlock penalty
        if np.mean(self.traffic_congestion) > 0.8:
            reward -= 500
            
        # Communication breakdown penalty
        if self.cellular_coverage < 0.7:
            reward -= 300
            
        # Update previous values for next reward calculation
        self._prev_resolved = self.emergencies_resolved
        self._prev_lives_saved = self.lives_saved
        self._prev_casualties = self.total_casualties
        
        return reward
    
    def _execute_action(self, action: int):
        """Execute the selected action"""
        if 0 <= action <= 19:  # Dispatch specific unit
            self._dispatch_unit(action)
        elif 20 <= action <= 25:  # Establish evacuation zones
            zone_id = action - 20
            if zone_id < 6:
                self.evacuation_zones[zone_id] = True
        elif 26 <= action <= 31:  # Request mutual aid
            if not self.mutual_aid_requested:
                self.mutual_aid_requested = True
                # Add temporary units
                self._add_mutual_aid_units()
        elif 32 <= action <= 37:  # Activate emergency shelters
            # Increase hospital capacity temporarily
            for hospital in self.hospitals:
                hospital.beds_available = min(hospital.max_beds, 
                                            hospital.beds_available + 10)
        elif 38 <= action <= 43:  # Redirect traffic
            segment_id = action - 38
            if segment_id < 25:
                self.traffic_congestion[segment_id] *= 0.7  # Reduce congestion
        elif action == 44:  # Declare state of emergency
            if not self.state_of_emergency:
                self.state_of_emergency = True
                # Unlock additional resources
                for unit in self.response_units:
                    unit.readiness = min(1.0, unit.readiness + 0.2)
        elif action == 45:  # Activate emergency broadcast
            self.emergency_broadcast_reach = min(1.0, self.emergency_broadcast_reach + 0.1)
        elif action == 46:  # Request National Guard
            if not self.national_guard_active and self.state_of_emergency:
                self.national_guard_active = True
                self._add_national_guard_units()
        elif action == 47:  # Coordinate with utilities
            # Improve response effectiveness
            for unit in self.response_units:
                if unit.unit_type in [UnitType.FIRE_TRUCK, UnitType.HAZMAT]:
                    unit.readiness = min(1.0, unit.readiness + 0.1)
        elif action == 48:  # Establish incident command post
            # Improve coordination
            self.cellular_coverage = min(1.0, self.cellular_coverage + 0.05)
        # action == 49 is normal operations (no action)
    
    def _dispatch_unit(self, unit_index: int):
        """Dispatch a specific unit to the nearest appropriate emergency"""
        if unit_index >= len(self.response_units):
            return
            
        unit = self.response_units[unit_index]
        if unit.busy:
            return
            
        # Find nearest appropriate emergency
        best_incident = None
        best_distance = float('inf')
        
        # Create a copy of items to avoid dictionary size change during iteration
        active_incidents = list(self.active_emergencies.items())
        for incident_id, incident in active_incidents:
            if incident.contained:
                continue
                
            # Check if unit type is needed
            if unit.unit_type not in incident.required_units:
                continue
                
            # Check if more units of this type are needed
            current_units_of_type = sum(1 for uid in incident.active_units
                                      if uid < len(self.response_units) and 
                                      self.response_units[uid].unit_type == unit.unit_type)
            
            if current_units_of_type >= incident.required_units[unit.unit_type]:
                continue
                
            # Calculate distance
            distance = math.sqrt((unit.x - incident.x)**2 + (unit.y - incident.y)**2)
            
            if distance < best_distance:
                best_distance = distance
                best_incident = (incident_id, incident)
        
        # Dispatch unit
        if best_incident:
            incident_id, incident = best_incident
            unit.busy = True
            unit.assigned_incident = incident_id
            unit.travel_time = max(1, int(best_distance * (2.0 - self.weather.road_conditions)))
            incident.active_units.append(unit_index)
    
    def _add_mutual_aid_units(self):
        """Add temporary mutual aid units"""
        # Add 5 temporary units
        for i in range(5):
            unit = ResponseUnit(
                unit_id=len(self.response_units),
                unit_type=UnitType.FIRE_TRUCK if i < 3 else UnitType.AMBULANCE,
                x=random.randint(0, self.grid_size-1),
                y=random.randint(0, self.grid_size-1),
                home_x=random.randint(0, self.grid_size-1),
                home_y=random.randint(0, self.grid_size-1),
                readiness=0.8  # Slightly lower readiness
            )
            self.response_units.append(unit)
    
    def _add_national_guard_units(self):
        """Add National Guard units"""
        # Add 10 National Guard units
        for i in range(10):
            unit = ResponseUnit(
                unit_id=len(self.response_units),
                unit_type=UnitType.SEARCH_RESCUE if i < 5 else UnitType.COMMAND,
                x=random.randint(0, self.grid_size-1),
                y=random.randint(0, self.grid_size-1),
                home_x=random.randint(0, self.grid_size-1),
                home_y=random.randint(0, self.grid_size-1),
                readiness=0.9
            )
            self.response_units.append(unit)
    
    def _update_weather(self):
        """Update weather conditions"""
        # Simple weather simulation
        self.weather.visibility += random.uniform(-0.1, 0.1)
        self.weather.visibility = max(0.3, min(1.0, self.weather.visibility))
        
        self.weather.precipitation += random.uniform(-0.05, 0.05)
        self.weather.precipitation = max(0.0, min(1.0, self.weather.precipitation))
        
        self.weather.road_conditions = max(0.5, 1.0 - self.weather.precipitation * 0.5)
        
        self.weather.wind_speed += random.uniform(-0.1, 0.1)
        self.weather.wind_speed = max(0.0, min(1.0, self.weather.wind_speed))
    
    def _update_traffic(self):
        """Update traffic congestion"""
        base_congestion = 0.3  # Base traffic level
        
        # Time of day effects
        hour = (self.current_timestep // 60) % 24
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_congestion = 0.7
        elif 22 <= hour or hour <= 5:  # Night
            base_congestion = 0.1
            
        # Emergency effects
        emergency_congestion = len(self.active_emergencies) * 0.05
        
        # Weather effects
        weather_congestion = self.weather.precipitation * 0.3
        
        for i in range(25):
            target_congestion = base_congestion + emergency_congestion + weather_congestion
            target_congestion += random.uniform(-0.1, 0.1)
            target_congestion = max(0.0, min(1.0, target_congestion))
            
            # Smooth transition
            self.traffic_congestion[i] = (self.traffic_congestion[i] * 0.8 + 
                                        target_congestion * 0.2)
    
    def _check_termination(self) -> Tuple[bool, str]:
        """Check if episode should terminate"""
        # Success: All emergencies resolved
        if len(self.active_emergencies) == 0 and self.current_timestep > 60:
            return True, "success"
            
        # Failure: Too many casualties
        if self.total_casualties > 50:
            return True, "casualties"
            
        # Failure: Emergency services overwhelmed
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        if avg_response_time > 45:  # 45 minutes
            return True, "overwhelmed"
            
        # Time limit
        if self.current_timestep >= self.max_timesteps:
            return True, "timeout"
            
        return False, ""
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_timestep = 0
        self.active_emergencies.clear()
        self.emergency_counter = 0
        self.total_casualties = 0
        self.emergencies_resolved = 0
        self.lives_saved = 0
        self.response_times.clear()
        
        # Reset flags
        self.state_of_emergency = False
        self.mutual_aid_requested = False
        self.national_guard_active = False
        self.cascade_multiplier = 1.0
        
        # Reset units
        for unit in self.response_units[:self.num_units]:  # Keep original units only
            unit.x = unit.home_x
            unit.y = unit.home_y
            unit.busy = False
            unit.readiness = 1.0
            unit.fatigue = 0.0
            unit.assigned_incident = None
            unit.travel_time = 0
        
        # Remove temporary units
        self.response_units = self.response_units[:self.num_units]
        
        # Reset infrastructure
        for hospital in self.hospitals:
            hospital.beds_available = hospital.max_beds
            hospital.trauma_capacity = hospital.max_trauma
            
        self.evacuation_zones.fill(False)
        self.zone_casualties.fill(0)
        self.traffic_congestion.fill(0.3)
        
        # Reset weather
        self.weather = WeatherCondition()
        
        # Reset communication
        self.cellular_coverage = 0.95
        self.emergency_broadcast_reach = 0.90
        
        # Generate initial emergencies
        for _ in range(random.randint(1, 3)):
            self._create_emergency()
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """Execute one time step"""
        # Execute action
        self._execute_action(action)
        
        # Update environment
        self._generate_emergencies()
        self._update_units()
        self._update_weather()
        self._update_traffic()
        
        # Remove resolved emergencies
        resolved_incidents = [iid for iid, inc in self.active_emergencies.items() 
                            if inc.contained]
        for iid in resolved_incidents:
            del self.active_emergencies[iid]
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination
        terminated, reason = self._check_termination()
        
        # Update timestep
        self.current_timestep += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Info
        info = {
            "active_emergencies": len(self.active_emergencies),
            "total_casualties": self.total_casualties,
            "lives_saved": self.lives_saved,
            "emergencies_resolved": self.emergencies_resolved,
            "avg_response_time": np.mean(self.response_times) if self.response_times else 0,
            "termination_reason": reason if terminated else None
        }
        
        return observation, reward, terminated, False, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            return self._render_human()
    
    def _render_human(self):
        """Render for human viewing"""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
            pygame.display.set_caption("🚨 Emergency Response Environment - Advanced Visualization")
            # Set window icon (if available)
            try:
                # Create a simple icon
                icon_surface = pygame.Surface((32, 32))
                icon_surface.fill((220, 20, 60))  # Emergency red
                pygame.draw.circle(icon_surface, (255, 255, 255), (16, 16), 12)
                pygame.draw.circle(icon_surface, (220, 20, 60), (16, 16), 10)
                pygame.display.set_icon(icon_surface)
            except:
                pass
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # Handle window events
        self._handle_window_events()
            
        return self._render_rgb_array()
    
    def _render_rgb_array(self):
        """Render and return RGB array"""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
            
        # Enhanced color scheme with better contrast
        colors = {
            'background': (245, 245, 250),
            'grid_lines': (220, 220, 230),
            'residential': (152, 251, 152),
            'commercial': (135, 206, 250),
            'industrial': (176, 196, 222),
            'fire': (220, 20, 60),
            'police': (30, 144, 255),
            'ambulance': (255, 255, 255),
            'search_rescue': (255, 140, 0),
            'hazmat': (128, 0, 128),
            'command': (255, 215, 0),
            'emergency_critical': (220, 20, 60),
            'emergency_serious': (255, 69, 0),
            'emergency_moderate': (255, 215, 0),
            'emergency_minor': (50, 205, 50),
            'hospital': (34, 139, 34),
            'text': (25, 25, 112),
            'text_light': (70, 70, 70),
            'panel_bg': (255, 255, 255),
            'panel_border': (200, 200, 200),
            'status_good': (34, 139, 34),
            'status_warning': (255, 165, 0),
            'status_critical': (220, 20, 60)
        }
        
        # Clear screen with gradient effect
        self.window.fill(colors['background'])
        
        # Draw title bar
        title_bar_rect = pygame.Rect(0, 0, self.window_size[0], 40)
        pygame.draw.rect(self.window, (70, 130, 180), title_bar_rect)
        pygame.draw.rect(self.window, (200, 200, 200), title_bar_rect, 2)
        
        # Title text
        title_font = pygame.font.Font(None, 32)
        title_text = f"🚨 Emergency Response Environment - Timestep: {self.current_timestep}"
        title_surface = title_font.render(title_text, True, (255, 255, 255))
        self.window.blit(title_surface, (20, 10))
        
        # Status indicator in title bar
        if self.state_of_emergency:
            status_text = "🚨 STATE OF EMERGENCY DECLARED 🚨"
            status_surface = title_font.render(status_text, True, (255, 255, 0))
            self.window.blit(status_surface, (self.window_size[0] - 400, 10))
        
        # Calculate grid cell size for main map
        map_width = 800
        map_height = 600
        cell_size = min(map_width // self.grid_size, map_height // self.grid_size)
        
        # Draw main map background
        map_rect = pygame.Rect(50, 50, map_width, map_height)
        pygame.draw.rect(self.window, (255, 255, 255), map_rect)
        pygame.draw.rect(self.window, colors['panel_border'], map_rect, 3)
        
        # Draw city grid with enhanced visuals
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * cell_size + 50
                y = i * cell_size + 50
                
                zone_type = ZoneType(self.city_grid[i, j])
                if zone_type == ZoneType.RESIDENTIAL:
                    color = colors['residential']
                    # Add subtle pattern for residential areas
                    pygame.draw.rect(self.window, color, (x, y, cell_size, cell_size))
                    pygame.draw.circle(self.window, (255, 255, 255, 50), (x + cell_size//4, y + cell_size//4), 2)
                    pygame.draw.circle(self.window, (255, 255, 255, 50), (x + 3*cell_size//4, y + 3*cell_size//4), 2)
                elif zone_type == ZoneType.COMMERCIAL:
                    color = colors['commercial']
                    # Add building pattern for commercial areas
                    pygame.draw.rect(self.window, color, (x, y, cell_size, cell_size))
                    pygame.draw.rect(self.window, (255, 255, 255, 100), (x + 2, y + 2, cell_size - 4, 3))
                    pygame.draw.rect(self.window, (255, 255, 255, 100), (x + 2, y + cell_size - 5, cell_size - 4, 3))
                else:
                    color = colors['industrial']
                    # Add industrial pattern
                    pygame.draw.rect(self.window, color, (x, y, cell_size, cell_size))
                    pygame.draw.line(self.window, (255, 255, 255, 100), (x, y + cell_size//2), (x + cell_size, y + cell_size//2), 2)
                
                # Draw grid lines with better visibility
                pygame.draw.rect(self.window, colors['grid_lines'], (x, y, cell_size, cell_size), 1)
        
        # Draw hospitals with enhanced graphics
        for hospital in self.hospitals:
            x = hospital.y * cell_size + 50 + cell_size // 2
            y = hospital.x * cell_size + 50 + cell_size // 2
            
            # Draw hospital building
            hospital_rect = pygame.Rect(x - cell_size//3, y - cell_size//3, 2*cell_size//3, 2*cell_size//3)
            pygame.draw.rect(self.window, colors['hospital'], hospital_rect)
            pygame.draw.rect(self.window, colors['text'], hospital_rect, 2)
            
            # Draw cross symbol
            cross_color = (255, 255, 255)
            cross_size = cell_size // 6
            pygame.draw.line(self.window, cross_color, (x - cross_size, y), (x + cross_size, y), 3)
            pygame.draw.line(self.window, cross_color, (x, y - cross_size), (x, y + cross_size), 3)
        
        # Draw emergencies with enhanced effects
        for incident in self.active_emergencies.values():
            x = incident.y * cell_size + 50 + cell_size // 2
            y = incident.x * cell_size + 50 + cell_size // 2
            
            # Color and size based on severity
            if incident.severity >= 8:
                color = colors['emergency_critical']
                size = cell_size // 2
                pulse_factor = 1.2
            elif incident.severity >= 5:
                color = colors['emergency_serious']
                size = cell_size // 2
                pulse_factor = 1.1
            else:
                color = colors['emergency_moderate']
                size = cell_size // 3
                pulse_factor = 1.0
            
            # Enhanced flashing and pulsing effect
            pulse = abs(math.sin(self.current_timestep * 0.3)) * 0.3 + 0.7
            current_size = int(size * pulse * pulse_factor)
            
            # Draw emergency indicator with glow effect
            pygame.draw.circle(self.window, color, (x, y), current_size)
            pygame.draw.circle(self.window, (255, 255, 255), (x, y), current_size, 2)
            
            # Add emergency type indicator
            if incident.emergency_type == EmergencyType.BUILDING_FIRE:
                # Draw fire symbol
                fire_points = [(x, y - current_size//2), (x - current_size//4, y), (x + current_size//4, y)]
                pygame.draw.polygon(self.window, (255, 69, 0), fire_points)
                
                # Add fire particles
                for _ in range(3):
                    particle_x = x + random.randint(-current_size, current_size)
                    particle_y = y + random.randint(-current_size, current_size)
                    particle_size = random.randint(2, 4)
                    particle_alpha = random.randint(100, 200)
                    particle_color = (255, random.randint(100, 200), 0, particle_alpha)
                    pygame.draw.circle(self.window, particle_color, (particle_x, particle_y), particle_size)
                    
            elif incident.emergency_type == EmergencyType.CHEMICAL_SPILL:
                # Draw hazard symbol
                pygame.draw.circle(self.window, (255, 255, 0), (x, y), current_size//3)
                pygame.draw.circle(self.window, (255, 0, 0), (x, y), current_size//3, 2)
                
                # Add chemical spill particles
                for _ in range(5):
                    particle_x = x + random.randint(-current_size, current_size)
                    particle_y = y + random.randint(-current_size, current_size)
                    particle_size = random.randint(1, 3)
                    particle_alpha = random.randint(50, 150)
                    particle_color = (random.randint(200, 255), random.randint(200, 255), 0, particle_alpha)
                    pygame.draw.circle(self.window, particle_color, (particle_x, particle_y), particle_size)
        
        # Draw response units with enhanced graphics
        for unit in self.response_units[:self.num_units]:  # Only original units
            x = unit.y * cell_size + 50 + cell_size // 2
            y = unit.x * cell_size + 50 + cell_size // 2
            
            # Unit type colors and symbols
            if unit.unit_type == UnitType.FIRE_TRUCK:
                color = colors['fire']
                symbol = "🔥"
            elif unit.unit_type == UnitType.POLICE_CAR:
                color = colors['police']
                symbol = "🚔"
            elif unit.unit_type == UnitType.AMBULANCE:
                color = colors['ambulance']
                symbol = "🚑"
            elif unit.unit_type == UnitType.SEARCH_RESCUE:
                color = colors['search_rescue']
                symbol = "🔍"
            elif unit.unit_type == UnitType.HAZMAT:
                color = colors['hazmat']
                symbol = "☣️"
            elif unit.unit_type == UnitType.COMMAND:
                color = colors['command']
                symbol = "⭐"
            else:
                color = colors['text']
                symbol = "🚨"
            
            # Size based on status
            base_size = 8 if unit.busy else 10
            size = int(base_size * (0.8 + 0.4 * unit.readiness))
            
            # Draw unit with status indicators
            pygame.draw.circle(self.window, color, (x, y), size)
            pygame.draw.circle(self.window, (255, 255, 255), (x, y), size, 2)
            
            # Draw readiness indicator
            if unit.readiness < 0.5:
                pygame.draw.circle(self.window, colors['status_warning'], (x, y), size//2)
            
            # Draw fatigue indicator
            if unit.fatigue > 0.7:
                pygame.draw.circle(self.window, colors['status_critical'], (x, y), size//3)
            
            # Draw unit ID
            font = pygame.font.Font(None, 16)
            unit_text = font.render(str(unit.unit_id), True, (0, 0, 0))
            self.window.blit(unit_text, (x - 8, y - 8))
            
            # Draw connection line to assigned incident
            if unit.assigned_incident is not None and unit.assigned_incident in self.active_emergencies:
                incident = self.active_emergencies[unit.assigned_incident]
                incident_x = incident.y * cell_size + 50 + cell_size // 2
                incident_y = incident.x * cell_size + 50 + cell_size // 2
                
                # Draw animated connection line
                line_color = (100, 100, 100, 150)
                line_width = 2
                
                # Add some animation to the line
                offset = int(math.sin(self.current_timestep * 0.2) * 3)
                mid_x = (x + incident_x) // 2 + offset
                mid_y = (y + incident_y) // 2 + offset
                
                # Draw curved line
                pygame.draw.line(self.window, line_color, (x, y), (mid_x, mid_y), line_width)
                pygame.draw.line(self.window, line_color, (mid_x, mid_y), (incident_x, incident_y), line_width)
        
        # Draw information panels
        self._draw_info_panels()
        
        # Draw mini-map
        self._draw_mini_map(50, 700, 200, 150)
        
        # Draw legend
        self._draw_legend(270, 700, 250, 150)
        
        # Draw weather effects
        self._draw_weather_effects()
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
    
    def _draw_info_panels(self):
        """Draw enhanced information panels with better visual design"""
        title_font = pygame.font.Font(None, 28)
        font = pygame.font.Font(None, 22)
        small_font = pygame.font.Font(None, 18)
        
        # Panel dimensions and positioning
        panel_width = 300
        panel_x = 870
        panel_spacing = 20
        
        # Colors for panels
        panel_bg = (255, 255, 255)
        panel_border = (200, 200, 200)
        title_bg = (70, 130, 180)
        title_text = (255, 255, 255)
        
        # 1. Emergency Status Panel
        panel_y = 50
        self._draw_panel(panel_x, panel_y, panel_width, 200, "🚨 Active Emergencies", title_font, font, small_font)
        
        y_offset = panel_y + 60
        for i, (incident_id, incident) in enumerate(list(self.active_emergencies.items())[:8]):
            # Color code by severity
            if incident.severity >= 8:
                severity_color = (220, 20, 60)
            elif incident.severity >= 5:
                severity_color = (255, 69, 0)
            else:
                severity_color = (255, 215, 0)
            
            # Emergency type icon
            if incident.emergency_type == EmergencyType.BUILDING_FIRE:
                icon = "🔥"
            elif incident.emergency_type == EmergencyType.CHEMICAL_SPILL:
                icon = "☣️"
            elif incident.emergency_type == EmergencyType.MULTI_VEHICLE_ACCIDENT:
                icon = "🚗"
            else:
                icon = "🚨"
            
            emergency_text = f"{icon} {incident.emergency_type.name[:12]}: Sev {incident.severity}"
            text = small_font.render(emergency_text, True, severity_color)
            self.window.blit(text, (panel_x + 15, y_offset))
            
            # Show casualties if any
            if incident.casualties > 0:
                casualty_text = f"   💀 {incident.casualties} casualties"
                casualty_surface = small_font.render(casualty_text, True, (100, 100, 100))
                self.window.blit(casualty_surface, (panel_x + 15, y_offset + 15))
                y_offset += 35
            else:
                y_offset += 25
        
        # 2. Statistics Panel
        panel_y = 270
        self._draw_panel(panel_x, panel_y, panel_width, 180, "📊 Statistics", title_font, font, small_font)
        
        y_offset = panel_y + 60
        stats_data = [
            ("⏰", f"Time: {self.current_timestep // 60:02d}:{self.current_timestep % 60:02d}"),
            ("💀", f"Casualties: {self.total_casualties}"),
            ("💚", f"Lives Saved: {self.lives_saved}"),
            ("✅", f"Resolved: {self.emergencies_resolved}"),
            ("🚨", f"Active Units: {sum(1 for u in self.response_units if u.busy)}"),
            ("⚡", f"Avg Response: {np.mean(self.response_times):.1f}m" if self.response_times else "Avg Response: N/A")
        ]
        
        for icon, stat in stats_data:
            icon_surface = small_font.render(icon, True, (70, 130, 180))
            stat_surface = small_font.render(stat, True, (50, 50, 50))
            self.window.blit(icon_surface, (panel_x + 15, y_offset))
            self.window.blit(stat_surface, (panel_x + 35, y_offset))
            y_offset += 25
        
        # 3. System Status Panel
        panel_y = 470
        self._draw_panel(panel_x, panel_y, panel_width, 200, "⚙️ System Status", title_font, font, small_font)
        
        y_offset = panel_y + 60
        status_data = [
            ("🚨", "Emergency Declared", self.state_of_emergency, "status_critical" if self.state_of_emergency else "status_good"),
            ("🤝", "Mutual Aid", self.mutual_aid_requested, "status_warning" if self.mutual_aid_requested else "status_good"),
            ("🛡️", "National Guard", self.national_guard_active, "status_warning" if self.national_guard_active else "status_good"),
            ("🌤️", f"Weather: {self.weather.visibility:.1f} vis", None, "status_good" if self.weather.visibility > 0.7 else "status_warning"),
            ("🚦", f"Traffic: {np.mean(self.traffic_congestion):.1f}", None, "status_good" if np.mean(self.traffic_congestion) < 0.7 else "status_warning")
        ]
        
        for icon, text, boolean_value, status_type in status_data:
            # Draw icon
            icon_surface = small_font.render(icon, True, (70, 130, 180))
            self.window.blit(icon_surface, (panel_x + 15, y_offset))
            
            # Draw text
            text_surface = small_font.render(text, True, (50, 50, 50))
            self.window.blit(text_surface, (panel_x + 35, y_offset))
            
            # Draw status indicator
            if boolean_value is not None:
                status_color = self._get_status_color(status_type)
                status_text = "YES" if boolean_value else "NO"
                status_surface = small_font.render(status_text, True, status_color)
                self.window.blit(status_surface, (panel_x + panel_width - 50, y_offset))
            else:
                # For non-boolean values, show status based on threshold
                if "warning" in status_type:
                    status_color = (255, 165, 0)
                else:
                    status_color = (34, 139, 34)
                status_surface = small_font.render("●", True, status_color)
                self.window.blit(status_surface, (panel_x + panel_width - 20, y_offset))
            
            y_offset += 25
        
        # 4. Quick Actions Panel
        panel_y = 690
        self._draw_panel(panel_x, panel_y, panel_width, 120, "⚡ Quick Actions", title_font, font, small_font)
        
        y_offset = panel_y + 60
        actions = [
            ("🚨", "Declare Emergency"),
            ("🤝", "Request Aid"),
            ("🛡️", "Call Guard"),
            ("🚦", "Manage Traffic")
        ]
        
        for icon, action in actions:
            icon_surface = small_font.render(icon, True, (70, 130, 180))
            action_surface = small_font.render(action, True, (50, 50, 50))
            self.window.blit(icon_surface, (panel_x + 15, y_offset))
            self.window.blit(action_surface, (panel_x + 35, y_offset))
            y_offset += 25
    
    def _draw_panel(self, x, y, width, height, title, title_font, font, small_font):
        """Draw a styled panel with title"""
        # Panel background
        panel_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.window, (255, 255, 255), panel_rect)
        pygame.draw.rect(self.window, (200, 200, 200), panel_rect, 2)
        
        # Title bar
        title_rect = pygame.Rect(x, y, width, 35)
        pygame.draw.rect(self.window, (70, 130, 180), title_rect)
        pygame.draw.rect(self.window, (200, 200, 200), title_rect, 2)
        
        # Title text
        title_surface = title_font.render(title, True, (255, 255, 255))
        self.window.blit(title_surface, (x + 10, y + 8))
    
    def _get_status_color(self, status_type):
        """Get color for status indicators"""
        colors = {
            'status_good': (34, 139, 34),
            'status_warning': (255, 165, 0),
            'status_critical': (220, 20, 60)
        }
        return colors.get(status_type, (100, 100, 100))
    
    def _draw_mini_map(self, x, y, width, height):
        """Draw a mini-map showing the overall city state"""
        # Mini-map background
        mini_map_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.window, (255, 255, 255), mini_map_rect)
        pygame.draw.rect(self.window, (100, 100, 100), mini_map_rect, 2)
        
        # Title
        font = pygame.font.Font(None, 20)
        title = font.render("🗺️ City Overview", True, (50, 50, 50))
        self.window.blit(title, (x + 5, y + 5))
        
        # Calculate mini-map cell size
        cell_size = min(width // self.grid_size, height // self.grid_size) - 2
        
        # Draw mini-map grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                map_x = x + 10 + j * (cell_size + 1)
                map_y = y + 30 + i * (cell_size + 1)
                
                # Zone colors (lighter for mini-map)
                zone_type = ZoneType(self.city_grid[i, j])
                if zone_type == ZoneType.RESIDENTIAL:
                    color = (200, 255, 200)
                elif zone_type == ZoneType.COMMERCIAL:
                    color = (200, 220, 255)
                else:
                    color = (220, 220, 240)
                
                pygame.draw.rect(self.window, color, (map_x, map_y, cell_size, cell_size))
        
        # Draw incidents on mini-map
        for incident in self.active_emergencies.values():
            map_x = x + 10 + incident.y * (cell_size + 1) + cell_size // 2
            map_y = y + 30 + incident.x * (cell_size + 1) + cell_size // 2
            
            if incident.severity >= 8:
                color = (220, 20, 60)
            elif incident.severity >= 5:
                color = (255, 69, 0)
            else:
                color = (255, 215, 0)
            
            pygame.draw.circle(self.window, color, (map_x, map_y), 3)
        
        # Draw response units on mini-map
        for unit in self.response_units[:self.num_units]:
            map_x = x + 10 + unit.y * (cell_size + 1) + cell_size // 2
            map_y = y + 30 + unit.x * (cell_size + 1) + cell_size // 2
            
            if unit.unit_type == UnitType.FIRE_TRUCK:
                color = (220, 20, 60)
            elif unit.unit_type == UnitType.POLICE_CAR:
                color = (30, 144, 255)
            else:
                color = (255, 255, 255)
            
            pygame.draw.circle(self.window, color, (map_x, map_y), 2)
            pygame.draw.circle(self.window, (0, 0, 0), (map_x, map_y), 2, 1)
    
    def _draw_weather_effects(self):
        """Draw weather effects on the main map"""
        if self.weather.visibility < 0.5:
            # Fog effect
            fog_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
            fog_alpha = int((0.5 - self.weather.visibility) * 100)
            fog_surface.fill((200, 200, 200, fog_alpha))
            self.window.blit(fog_surface, (50, 50))
        
        if self.weather.road_conditions < 0.7:
            # Rain effect
            rain_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
            for _ in range(50):
                rain_x = random.randint(50, 850)
                rain_y = random.randint(50, 650)
                rain_length = random.randint(5, 15)
                rain_alpha = random.randint(30, 80)
                rain_color = (100, 150, 255, rain_alpha)
                pygame.draw.line(rain_surface, rain_color, 
                               (rain_x, rain_y), (rain_x, rain_y + rain_length), 2)
            self.window.blit(rain_surface, (50, 50))
    
    def _handle_window_events(self):
        """Handle pygame window events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resizing
                new_size = (max(1200, event.w), max(900, event.h))
                self.window_size = new_size
                self.window = pygame.display.set_mode(new_size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_F11:
                    # Toggle fullscreen
                    pygame.display.toggle_fullscreen()
    
    def _draw_legend(self, x, y, width, height):
        """Draw a legend explaining the symbols and colors"""
        # Legend background
        legend_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.window, (255, 255, 255), legend_rect)
        pygame.draw.rect(self.window, (100, 100, 100), legend_rect, 2)
        
        # Title
        font = pygame.font.Font(None, 20)
        title = font.render("🔑 Legend", True, (50, 50, 50))
        self.window.blit(title, (x + 5, y + 5))
        
        # Legend items
        legend_items = [
            ("🔥", "Fire Truck", (220, 20, 60)),
            ("🚔", "Police Car", (30, 144, 255)),
            ("🚑", "Ambulance", (255, 255, 255)),
            ("🔍", "Search & Rescue", (255, 140, 0)),
            ("☣️", "Hazmat", (128, 0, 128)),
            ("⭐", "Command", (255, 215, 0)),
            ("🏥", "Hospital", (34, 139, 34)),
            ("🚨", "Emergency", (220, 20, 60))
        ]
        
        y_offset = y + 30
        for icon, label, color in legend_items:
            # Icon
            icon_surface = font.render(icon, True, (70, 130, 180))
            self.window.blit(icon_surface, (x + 10, y_offset))
            
            # Label
            label_surface = font.render(label, True, (50, 50, 50))
            self.window.blit(label_surface, (x + 35, y_offset))
            
            # Color indicator
            pygame.draw.circle(self.window, color, (x + width - 20, y_offset + 8), 6)
            pygame.draw.circle(self.window, (0, 0, 0), (x + width - 20, y_offset + 8), 6, 1)
            
            y_offset += 25
        
        # Emergency severity legend
        y_offset += 10
        severity_title = font.render("Emergency Severity:", True, (50, 50, 50))
        self.window.blit(severity_title, (x + 10, y_offset))
        y_offset += 20
        
        severity_items = [
            ("🔴", "Critical (8-10)", (220, 20, 60)),
            ("🟠", "Serious (5-7)", (255, 69, 0)),
            ("🟡", "Moderate (3-4)", (255, 215, 0)),
            ("🟢", "Minor (1-2)", (50, 205, 50))
        ]
        
        for icon, label, color in severity_items:
            icon_surface = font.render(icon, True, (70, 130, 180))
            self.window.blit(icon_surface, (x + 10, y_offset))
            
            label_surface = font.render(label, True, (50, 50, 50))
            self.window.blit(label_surface, (x + 35, y_offset))
            
            pygame.draw.circle(self.window, color, (x + width - 20, y_offset + 8), 6)
            pygame.draw.circle(self.window, (0, 0, 0), (x + width - 20, y_offset + 8), 6, 1)
            
            y_offset += 20
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
