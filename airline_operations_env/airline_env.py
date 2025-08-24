"""
Airline Operations Environment
A comprehensive simulation of airline operations management with hub-and-spoke network
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import colorsys


class AircraftType(Enum):
    WIDEBODY = "widebody"
    NARROWBODY = "narrowbody"
    REGIONAL = "regional"


class AirportType(Enum):
    MAJOR_HUB = "major_hub"
    REGIONAL_HUB = "regional_hub"
    SPOKE = "spoke"


class CrewType(Enum):
    CAPTAIN = "captain"
    FIRST_OFFICER = "first_officer"
    FLIGHT_ATTENDANT = "flight_attendant"


class WeatherCondition(Enum):
    CLEAR = 0
    LIGHT_RAIN = 1
    HEAVY_RAIN = 2
    THUNDERSTORM = 3
    SNOW = 4
    FOG = 5
    SEVERE = 6


@dataclass
class Aircraft:
    """Represents an aircraft in the fleet"""
    id: int
    type: AircraftType
    current_airport: int
    destination: int
    passengers_onboard: int
    fuel_level: float
    maintenance_status: float  # Hours until maintenance required
    delay_minutes: int
    in_flight: bool = False
    flight_progress: float = 0.0  # 0 to 1 for flight completion
    speed: float = 500.0  # km/h
    max_passengers: int = 150
    range_hours: float = 6.0
    fuel_consumption: float = 1.0
    
    def __post_init__(self):
        if self.type == AircraftType.WIDEBODY:
            self.max_passengers = 300
            self.range_hours = 12.0
            self.fuel_consumption = 2.0
            self.speed = 850.0
        elif self.type == AircraftType.NARROWBODY:
            self.max_passengers = 150
            self.range_hours = 6.0
            self.fuel_consumption = 1.0
            self.speed = 750.0
        elif self.type == AircraftType.REGIONAL:
            self.max_passengers = 70
            self.range_hours = 3.0
            self.fuel_consumption = 0.5
            self.speed = 550.0


@dataclass
class Airport:
    """Represents an airport in the network"""
    id: int
    name: str
    type: AirportType
    x: float  # Position for visualization
    y: float
    weather: WeatherCondition = WeatherCondition.CLEAR
    runway_availability: float = 1.0
    gate_availability: float = 1.0
    passenger_volume: int = 0
    fuel_price: float = 2.5  # Per unit
    weather_severity: float = 0.0


@dataclass
class Flight:
    """Represents a scheduled flight"""
    id: int
    origin: int
    destination: int
    scheduled_time: float  # Hour of day (0-24)
    actual_time: float
    aircraft_id: Optional[int] = None
    passengers: int = 0
    status: str = "scheduled"  # scheduled, boarding, departed, arrived, cancelled, delayed
    priority: float = 1.0


@dataclass
class CrewSet:
    """Represents a crew set (pilots and flight attendants)"""
    id: int
    location: int
    duty_time_remaining: float  # Hours
    qualification_type: CrewType
    assigned_flight: Optional[int] = None
    fatigue_level: float = 0.0


class AirlineOperationsEnv(gym.Env):
    """
    Airline Operations Management Environment
    
    Manages a hub-and-spoke airline network with complex operational challenges
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        
        # Environment dimensions
        self.n_airports = 15
        self.n_aircraft = 25
        self.n_flights = 150
        self.n_crew = 40
        self.n_hubs = 5
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(45)
        
        # State space: 160 elements as specified
        # Simplified to key metrics for computational efficiency
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(160,), dtype=np.float32
        )
        
        # Initialize components
        self.airports: List[Airport] = []
        self.aircraft: List[Aircraft] = []
        self.flights: List[Flight] = []
        self.crew: List[CrewSet] = []
        
        # Performance metrics
        self.on_time_performance = 0.95
        self.passenger_satisfaction = 1.0
        self.daily_revenue = 0.0
        self.daily_costs = 0.0
        self.fuel_costs = 0.0
        self.delay_compensation = 0.0
        self.crew_overtime = 0.0
        
        # Operational state
        self.current_hour = 6.0  # Start at 6 AM
        self.disruption_active = False
        self.weather_systems: List[Dict] = []
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        if self.render_mode == "human":
            pygame.init()
            self.screen_width = 1400
            self.screen_height = 1000
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Airline Operations Control Center")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
        
        # Initialize environment
        self._initialize_network()
        self.reset()
    
    def _initialize_network(self):
        """Initialize the airline network with airports, aircraft, and flights"""
        # Create airports in hub-and-spoke configuration
        # 1 major hub
        self.airports.append(Airport(
            0, "MAIN_HUB", AirportType.MAJOR_HUB,
            self.screen_width // 2 if self.screen else 700,
            self.screen_height // 2 if self.screen else 500
        ))
        
        # 4 regional hubs
        hub_positions = [
            (300, 300), (1100, 300), (300, 700), (1100, 700)
        ]
        for i in range(4):
            self.airports.append(Airport(
                i + 1, f"REGIONAL_{i+1}", AirportType.REGIONAL_HUB,
                hub_positions[i][0], hub_positions[i][1]
            ))
        
        # 10 spoke airports
        for i in range(10):
            angle = (i / 10) * 2 * math.pi
            x = 700 + 400 * math.cos(angle)
            y = 500 + 300 * math.sin(angle)
            self.airports.append(Airport(
                i + 5, f"SPOKE_{i+1}", AirportType.SPOKE, x, y
            ))
        
        # Create aircraft fleet
        aircraft_id = 0
        # 8 Wide-body aircraft
        for i in range(8):
            self.aircraft.append(Aircraft(
                aircraft_id, AircraftType.WIDEBODY,
                random.randint(0, 4),  # Start at hubs
                random.randint(0, self.n_airports - 1),
                0, 1.0, random.uniform(50, 200), 0
            ))
            aircraft_id += 1
        
        # 12 Narrow-body aircraft
        for i in range(12):
            self.aircraft.append(Aircraft(
                aircraft_id, AircraftType.NARROWBODY,
                random.randint(0, self.n_airports - 1),
                random.randint(0, self.n_airports - 1),
                0, 1.0, random.uniform(30, 150), 0
            ))
            aircraft_id += 1
        
        # 5 Regional jets
        for i in range(5):
            self.aircraft.append(Aircraft(
                aircraft_id, AircraftType.REGIONAL,
                random.randint(5, self.n_airports - 1),  # Start at spokes
                random.randint(5, self.n_airports - 1),
                0, 1.0, random.uniform(20, 100), 0
            ))
            aircraft_id += 1
        
        # Create flight schedule
        for i in range(self.n_flights):
            origin = random.randint(0, self.n_airports - 1)
            destination = random.randint(0, self.n_airports - 1)
            while destination == origin:
                destination = random.randint(0, self.n_airports - 1)
            
            scheduled_time = random.uniform(6, 22)  # Flights between 6 AM and 10 PM
            self.flights.append(Flight(
                i, origin, destination, scheduled_time, scheduled_time,
                passengers=random.randint(50, 250)
            ))
        
        # Create crew sets
        for i in range(self.n_crew):
            crew_type = random.choice(list(CrewType))
            self.crew.append(CrewSet(
                i, random.randint(0, self.n_airports - 1),
                random.uniform(8, 12), crew_type
            ))
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset time
        self.current_hour = 6.0
        
        # Reset metrics
        self.on_time_performance = 0.95
        self.passenger_satisfaction = 1.0
        self.daily_revenue = 0.0
        self.daily_costs = 0.0
        self.fuel_costs = 0.0
        self.delay_compensation = 0.0
        self.crew_overtime = 0.0
        
        # Reset operational state
        self.disruption_active = False
        self.weather_systems = []
        
        # Reset aircraft positions and status
        for aircraft in self.aircraft:
            aircraft.delay_minutes = 0
            aircraft.in_flight = False
            aircraft.flight_progress = 0.0
            aircraft.fuel_level = random.uniform(0.7, 1.0)
            aircraft.maintenance_status = random.uniform(50, 200)
            aircraft.passengers_onboard = 0
        
        # Reset airport conditions
        for airport in self.airports:
            airport.weather = WeatherCondition.CLEAR
            airport.runway_availability = 1.0
            airport.gate_availability = random.uniform(0.7, 1.0)
            airport.passenger_volume = random.randint(100, 1000)
            airport.fuel_price = random.uniform(2.0, 3.0)
            airport.weather_severity = 0.0
        
        # Reset flights
        for flight in self.flights:
            flight.status = "scheduled"
            flight.actual_time = flight.scheduled_time
            flight.aircraft_id = None
        
        # Reset crew
        for crew in self.crew:
            crew.duty_time_remaining = random.uniform(8, 12)
            crew.assigned_flight = None
            crew.fatigue_level = 0.0
        
        # Generate initial weather systems
        self._generate_weather_systems()
        
        # Create initial info dictionary
        info = {
            'on_time_performance': self.on_time_performance,
            'passenger_satisfaction': self.passenger_satisfaction,
            'daily_revenue': self.daily_revenue,
            'daily_costs': self.daily_costs,
            'current_hour': self.current_hour
        }
        
        return self._get_observation(), info
    
    def _generate_weather_systems(self):
        """Generate random weather systems affecting regions"""
        self.weather_systems = []
        n_systems = random.randint(0, 3)
        for _ in range(n_systems):
            system = {
                'x': random.uniform(100, self.screen_width - 100) if self.screen else random.uniform(100, 1300),
                'y': random.uniform(100, self.screen_height - 100) if self.screen else random.uniform(100, 900),
                'radius': random.uniform(100, 300),
                'severity': random.choice([WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN, 
                                         WeatherCondition.THUNDERSTORM, WeatherCondition.FOG]),
                'movement_x': random.uniform(-20, 20),
                'movement_y': random.uniform(-20, 20)
            }
            self.weather_systems.append(system)
    
    def _get_observation(self) -> np.ndarray:
        """Generate the current state observation"""
        obs = np.zeros(160, dtype=np.float32)
        idx = 0
        
        # Aircraft states (simplified: top 25 aircraft, 4 features each = 100)
        for i, aircraft in enumerate(self.aircraft[:25]):
            if idx + 4 <= 160:
                obs[idx] = aircraft.current_airport / self.n_airports
                obs[idx + 1] = aircraft.fuel_level
                obs[idx + 2] = aircraft.delay_minutes / 240.0  # Normalize to 4 hours
                obs[idx + 3] = aircraft.maintenance_status / 200.0
                idx += 4
        
        # Airport conditions (15 airports, 2 features each = 30)
        for i, airport in enumerate(self.airports):
            if idx + 2 <= 160:
                obs[idx] = airport.weather_severity
                obs[idx + 1] = airport.gate_availability
                idx += 2
        
        # Flight schedule adherence (top 10 priority flights, 2 features = 20)
        priority_flights = sorted(self.flights, key=lambda f: f.priority, reverse=True)[:10]
        for flight in priority_flights:
            if idx + 2 <= 160:
                obs[idx] = (flight.actual_time - flight.scheduled_time) / 4.0  # Normalize delay
                obs[idx + 1] = flight.passengers / 300.0
                idx += 2
        
        # Global metrics (10 features)
        if idx + 10 <= 160:
            obs[idx] = self.on_time_performance
            obs[idx + 1] = self.passenger_satisfaction
            obs[idx + 2] = self.current_hour / 24.0
            obs[idx + 3] = len([a for a in self.aircraft if a.in_flight]) / len(self.aircraft)
            obs[idx + 4] = self.daily_revenue / 1000000.0  # Normalize to millions
            obs[idx + 5] = self.daily_costs / 1000000.0
            obs[idx + 6] = self.fuel_costs / 100000.0
            obs[idx + 7] = self.delay_compensation / 100000.0
            obs[idx + 8] = 1.0 if self.disruption_active else 0.0
            obs[idx + 9] = len(self.weather_systems) / 5.0
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        reward = 0.0
        
        # Process action
        reward += self._process_action(action)
        
        # Update time
        self.current_hour += 0.25  # 15-minute intervals
        if self.current_hour >= 24:
            self.current_hour -= 24
        
        # Update weather systems
        self._update_weather()
        
        # Update aircraft positions
        self._update_aircraft()
        
        # Update flight status
        self._update_flights()
        
        # Update crew status
        self._update_crew()
        
        # Calculate operational metrics
        self._calculate_metrics()
        
        # Calculate reward based on performance
        reward += self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = False
        
        # Generate observation
        observation = self._get_observation()
        
        info = {
            'on_time_performance': self.on_time_performance,
            'passenger_satisfaction': self.passenger_satisfaction,
            'daily_revenue': self.daily_revenue,
            'daily_costs': self.daily_costs,
            'current_hour': self.current_hour
        }
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action: int) -> float:
        """Process the selected action and return immediate reward"""
        reward = 0.0
        
        if action < 25:  # Reassign aircraft
            aircraft_id = action
            if aircraft_id < len(self.aircraft):
                aircraft = self.aircraft[aircraft_id]
                if not aircraft.in_flight:
                    # Find a new route for the aircraft
                    new_dest = random.randint(0, self.n_airports - 1)
                    if new_dest != aircraft.current_airport:
                        aircraft.destination = new_dest
                        aircraft.in_flight = True
                        aircraft.flight_progress = 0.0
                        aircraft.passengers_onboard = random.randint(50, aircraft.max_passengers)
                        reward += 100  # Successful reassignment
        
        elif action < 30:  # Cancel flight
            flight_idx = random.randint(0, min(len(self.flights) - 1, 50))
            flight = self.flights[flight_idx]
            if flight.status == "scheduled":
                flight.status = "cancelled"
                self.passenger_satisfaction *= 0.9
                self.delay_compensation += flight.passengers * 200
                reward -= 5000  # High penalty for cancellation
        
        elif action < 35:  # Delay departure
            flight_idx = random.randint(0, min(len(self.flights) - 1, 50))
            flight = self.flights[flight_idx]
            if flight.status == "scheduled":
                flight.actual_time += 0.5  # 30-minute delay
                flight.status = "delayed"
                reward -= 100  # Small penalty for controlled delay
        
        elif action < 40:  # Request priority handling
            airport_idx = action - 35
            if airport_idx < len(self.airports):
                self.airports[airport_idx].gate_availability = min(1.0, 
                    self.airports[airport_idx].gate_availability + 0.2)
                reward += 50
        
        elif action == 40:  # Irregular operations mode
            self.disruption_active = True
            reward -= 200  # Cost of activating special procedures
        
        elif action == 41:  # Deploy spare aircraft
            if self.daily_costs < 500000:  # Budget constraint
                self.daily_costs += 50000
                reward += 200  # Helps maintain schedule
        
        elif action == 42:  # Hotel accommodations
            self.delay_compensation += 10000
            self.passenger_satisfaction = min(1.0, self.passenger_satisfaction + 0.05)
            reward -= 100
        
        elif action == 43:  # Ground stop
            for aircraft in self.aircraft:
                if not aircraft.in_flight:
                    aircraft.delay_minutes += 30
            reward -= 500
        
        elif action == 44:  # Normal operations
            self.disruption_active = False
            reward += 50
        
        return reward
    
    def _update_weather(self):
        """Update weather systems and their effects on airports"""
        # Move weather systems
        for system in self.weather_systems:
            system['x'] += system['movement_x'] * 0.1
            system['y'] += system['movement_y'] * 0.1
            
            # Wrap around screen
            if self.screen:
                if system['x'] < 0:
                    system['x'] = self.screen_width
                elif system['x'] > self.screen_width:
                    system['x'] = 0
                if system['y'] < 0:
                    system['y'] = self.screen_height
                elif system['y'] > self.screen_height:
                    system['y'] = 0
        
        # Update airport weather based on systems
        for airport in self.airports:
            airport.weather = WeatherCondition.CLEAR
            airport.weather_severity = 0.0
            
            for system in self.weather_systems:
                dist = math.sqrt((airport.x - system['x'])**2 + (airport.y - system['y'])**2)
                if dist < system['radius']:
                    airport.weather = system['severity']
                    airport.weather_severity = 1.0 - (dist / system['radius'])
                    
                    # Weather affects operations
                    if airport.weather == WeatherCondition.THUNDERSTORM:
                        airport.runway_availability *= 0.5
                    elif airport.weather == WeatherCondition.FOG:
                        airport.runway_availability *= 0.7
                    elif airport.weather == WeatherCondition.SNOW:
                        airport.runway_availability *= 0.6
    
    def _update_aircraft(self):
        """Update aircraft positions and status"""
        for aircraft in self.aircraft:
            if aircraft.in_flight:
                # Progress flight
                aircraft.flight_progress += 0.05  # 5% per time step
                
                if aircraft.flight_progress >= 1.0:
                    # Aircraft has arrived
                    aircraft.in_flight = False
                    aircraft.current_airport = aircraft.destination
                    aircraft.flight_progress = 0.0
                    aircraft.passengers_onboard = 0
                    
                    # Fuel consumption
                    aircraft.fuel_level -= 0.1 * aircraft.fuel_consumption
                    aircraft.fuel_level = max(0.1, aircraft.fuel_level)
                    
                    # Update revenue
                    self.daily_revenue += aircraft.passengers_onboard * 200
            
            # Maintenance degradation
            aircraft.maintenance_status -= 0.5
            if aircraft.maintenance_status <= 0:
                aircraft.delay_minutes += 120  # 2-hour maintenance delay
                aircraft.maintenance_status = 100
                self.daily_costs += 10000  # Maintenance cost
    
    def _update_flights(self):
        """Update flight statuses"""
        for flight in self.flights:
            if flight.status == "scheduled" and self.current_hour >= flight.scheduled_time:
                # Try to assign aircraft
                available_aircraft = [a for a in self.aircraft 
                                    if not a.in_flight and a.current_airport == flight.origin]
                if available_aircraft:
                    aircraft = available_aircraft[0]
                    flight.aircraft_id = aircraft.id
                    flight.status = "boarding"
                    aircraft.destination = flight.destination
                    aircraft.passengers_onboard = min(flight.passengers, aircraft.max_passengers)
                else:
                    flight.status = "delayed"
                    flight.actual_time += 0.25
            
            elif flight.status == "boarding":
                # Complete boarding and depart
                flight.status = "departed"
                if flight.aircraft_id is not None:
                    self.aircraft[flight.aircraft_id].in_flight = True
    
    def _update_crew(self):
        """Update crew duty times and fatigue"""
        for crew in self.crew:
            if crew.assigned_flight is not None:
                crew.duty_time_remaining -= 0.25
                crew.fatigue_level += 0.01
                
                if crew.duty_time_remaining <= 0:
                    # Crew timeout violation
                    self.crew_overtime += 5000
                    crew.duty_time_remaining = 0
                    crew.assigned_flight = None
            else:
                # Rest period
                crew.duty_time_remaining = min(12, crew.duty_time_remaining + 0.1)
                crew.fatigue_level = max(0, crew.fatigue_level - 0.02)
    
    def _calculate_metrics(self):
        """Calculate operational metrics"""
        # On-time performance
        total_flights = len([f for f in self.flights if f.status != "scheduled"])
        if total_flights > 0:
            on_time_flights = len([f for f in self.flights 
                                 if f.status != "scheduled" and abs(f.actual_time - f.scheduled_time) < 0.25])
            self.on_time_performance = on_time_flights / total_flights
        
        # Calculate costs
        self.fuel_costs = sum(a.fuel_consumption * self.airports[a.current_airport].fuel_price 
                             for a in self.aircraft if not a.in_flight) * 100
        self.daily_costs += self.fuel_costs + self.delay_compensation + self.crew_overtime
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current performance"""
        reward = 0.0
        
        # On-time performance bonus
        if self.on_time_performance > 0.95:
            reward += 500
        elif self.on_time_performance > 0.90:
            reward += 200
        elif self.on_time_performance < 0.70:
            reward -= 500
        
        # Passenger satisfaction
        if self.passenger_satisfaction > 0.9:
            reward += 300
        elif self.passenger_satisfaction < 0.6:
            reward -= 1000
        
        # Financial performance
        profit = self.daily_revenue - self.daily_costs
        if profit > 100000:
            reward += 500
        elif profit < -100000:
            reward -= 500
        
        # Crew management
        fatigued_crew = len([c for c in self.crew if c.fatigue_level > 0.8])
        if fatigued_crew > 5:
            reward -= 200
        
        # Safety bonus
        if all(a.fuel_level > 0.2 for a in self.aircraft):
            reward += 100
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Complete 24-hour cycle
        if self.current_hour >= 23.75:
            return True
        
        # Safety violation
        if any(c.duty_time_remaining <= 0 and c.assigned_flight is not None for c in self.crew):
            return True
        
        # Customer satisfaction too low
        if self.passenger_satisfaction < 0.6:
            return True
        
        # Financial crisis
        if self.daily_costs > 2 * self.daily_revenue and self.daily_revenue > 0:
            return True
        
        return False
    
    def render(self):
        """Render the environment using Pygame"""
        if self.render_mode is None:
            return None
        
        if self.screen is None:
            pygame.init()
            self.screen_width = 1400
            self.screen_height = 1000
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Airline Operations Control Center")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
        
        # Clear screen
        self.screen.fill((20, 20, 40))  # Dark blue background
        
        # Draw weather systems
        for system in self.weather_systems:
            color = (100, 100, 100, 50)  # Gray with transparency
            if system['severity'] == WeatherCondition.THUNDERSTORM:
                color = (150, 50, 150, 50)  # Purple
            elif system['severity'] == WeatherCondition.FOG:
                color = (200, 200, 200, 50)  # Light gray
            
            # Draw weather circle
            s = pygame.Surface((int(system['radius']*2), int(system['radius']*2)), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (int(system['radius']), int(system['radius'])), 
                             int(system['radius']))
            self.screen.blit(s, (system['x'] - system['radius'], system['y'] - system['radius']))
        
        # Draw route network
        for i, origin in enumerate(self.airports):
            for j, dest in enumerate(self.airports):
                if i < j and (origin.type == AirportType.MAJOR_HUB or 
                            dest.type == AirportType.MAJOR_HUB or
                            (origin.type == AirportType.REGIONAL_HUB and 
                             dest.type == AirportType.REGIONAL_HUB)):
                    pygame.draw.line(self.screen, (40, 40, 60), 
                                   (origin.x, origin.y), (dest.x, dest.y), 1)
        
        # Draw airports
        for airport in self.airports:
            # Determine airport color based on status
            if airport.weather_severity > 0.5:
                color = (255, 100, 100)  # Red for severe weather
            elif airport.gate_availability < 0.3:
                color = (255, 200, 100)  # Orange for congestion
            else:
                color = (100, 255, 100)  # Green for normal
            
            # Draw airport circle (size based on type)
            size = 20 if airport.type == AirportType.MAJOR_HUB else (
                   15 if airport.type == AirportType.REGIONAL_HUB else 10)
            pygame.draw.circle(self.screen, color, (int(airport.x), int(airport.y)), size)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(airport.x), int(airport.y)), size, 2)
            
            # Draw airport label
            label = self.small_font.render(airport.name, True, (255, 255, 255))
            self.screen.blit(label, (airport.x - 30, airport.y + size + 5))
        
        # Draw aircraft
        for aircraft in self.aircraft:
            if aircraft.in_flight:
                # Calculate position between airports
                origin = self.airports[aircraft.current_airport]
                dest = self.airports[aircraft.destination]
                x = origin.x + (dest.x - origin.x) * aircraft.flight_progress
                y = origin.y + (dest.y - origin.y) * aircraft.flight_progress
                
                # Aircraft color based on type
                if aircraft.type == AircraftType.WIDEBODY:
                    color = (255, 255, 100)  # Yellow
                elif aircraft.type == AircraftType.NARROWBODY:
                    color = (100, 200, 255)  # Light blue
                else:
                    color = (200, 150, 255)  # Light purple
                
                # Draw aircraft triangle
                pygame.draw.polygon(self.screen, color, [
                    (x, y - 8), (x - 6, y + 6), (x + 6, y + 6)
                ])
                
                # Draw delay indicator
                if aircraft.delay_minutes > 30:
                    pygame.draw.circle(self.screen, (255, 0, 0), (int(x), int(y)), 12, 2)
        
        # Draw dashboard
        dashboard_y = 20
        
        # Title
        title = self.font.render("AIRLINE OPERATIONS CONTROL CENTER", True, (255, 255, 255))
        self.screen.blit(title, (self.screen_width // 2 - 150, dashboard_y))
        
        # Current time
        time_str = f"Time: {int(self.current_hour):02d}:{int((self.current_hour % 1) * 60):02d}"
        time_text = self.font.render(time_str, True, (255, 255, 255))
        self.screen.blit(time_text, (50, dashboard_y))
        
        # Metrics panel
        metrics_x = 50
        metrics_y = self.screen_height - 150
        
        # Background for metrics
        pygame.draw.rect(self.screen, (30, 30, 50), 
                        (metrics_x - 10, metrics_y - 10, 400, 140))
        pygame.draw.rect(self.screen, (100, 100, 150), 
                        (metrics_x - 10, metrics_y - 10, 400, 140), 2)
        
        # Display metrics
        metrics = [
            f"On-Time Performance: {self.on_time_performance:.1%}",
            f"Passenger Satisfaction: {self.passenger_satisfaction:.1%}",
            f"Revenue: ${self.daily_revenue:,.0f}",
            f"Costs: ${self.daily_costs:,.0f}",
            f"Active Flights: {len([a for a in self.aircraft if a.in_flight])}",
            f"Weather Systems: {len(self.weather_systems)}"
        ]
        
        for i, metric in enumerate(metrics):
            text = self.small_font.render(metric, True, (255, 255, 255))
            self.screen.blit(text, (metrics_x, metrics_y + i * 22))
        
        # Alert panel
        alert_x = self.screen_width - 350
        alert_y = self.screen_height - 150
        
        pygame.draw.rect(self.screen, (50, 30, 30), 
                        (alert_x - 10, alert_y - 10, 340, 140))
        pygame.draw.rect(self.screen, (150, 100, 100), 
                        (alert_x - 10, alert_y - 10, 340, 140), 2)
        
        alert_title = self.small_font.render("ALERTS", True, (255, 200, 200))
        self.screen.blit(alert_title, (alert_x, alert_y))
        
        # Generate alerts
        alerts = []
        if self.on_time_performance < 0.80:
            alerts.append("WARNING: Low on-time performance")
        if any(a.fuel_level < 0.3 for a in self.aircraft):
            alerts.append("CRITICAL: Low fuel aircraft")
        if any(c.duty_time_remaining < 2 for c in self.crew):
            alerts.append("ALERT: Crew timeout approaching")
        if self.passenger_satisfaction < 0.7:
            alerts.append("WARNING: Low passenger satisfaction")
        if len(self.weather_systems) > 2:
            alerts.append("ALERT: Multiple weather systems active")
        
        for i, alert in enumerate(alerts[:5]):
            text = self.small_font.render(alert, True, (255, 150, 150))
            self.screen.blit(text, (alert_x, alert_y + 25 + i * 22))
        
        # Status bar
        status_text = "IRREGULAR OPS" if self.disruption_active else "NORMAL OPERATIONS"
        status_color = (255, 100, 100) if self.disruption_active else (100, 255, 100)
        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (self.screen_width - 200, dashboard_y))
        
        # Update display
        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.metadata["render_fps"])
        
        return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2)) if self.render_mode == "rgb_array" else None
    
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
