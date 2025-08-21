"""Smart Manufacturing Environment

A realistic factory floor simulation with production stations, quality control,
and machine maintenance. All components are self-contained within the package.
"""

import gym
from gym import spaces
import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum
import math


class MachineType(Enum):
    """Types of machines in the factory"""
    CUTTING = "Cutting"
    ASSEMBLY = "Assembly"
    TESTING = "Testing"
    PACKAGING = "Packaging"


class MachineStatus(Enum):
    """Machine operational status"""
    OPERATIONAL = 0
    BROKEN = 1
    MAINTENANCE = 2


class ProductType(Enum):
    """Product types with their specifications"""
    A = ("A", 1, 10, 0.80)  # (name, stations_required, timesteps, quality_target)
    B = ("B", 2, 15, 0.85)
    C = ("C", 3, 20, 0.90)
    D = ("D", 4, 25, 0.85)
    E = ("E", 5, 30, 0.95)
    F = ("F", 3, 18, 0.75)


@dataclass
class Product:
    """Represents a product in the production system"""
    product_type: ProductType
    current_station: int
    quality_score: float
    timesteps_remaining: int
    position: Tuple[float, float]
    id: int


@dataclass
class Station:
    """Represents a production station"""
    station_id: int
    machine_type: MachineType
    status: MachineStatus
    position: Tuple[int, int]
    utilization_rate: float
    operations_count: int
    maintenance_countdown: int
    current_product: Optional[Product]
    queue: List[Product]
    performance_degradation: float


class SmartManufacturingEnv(gym.Env):
    """Smart Manufacturing Environment for RL training"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        super(SmartManufacturingEnv, self).__init__()
        
        # Environment dimensions
        self.factory_width = 30
        self.factory_height = 15
        self.num_stations = 5
        self.num_product_types = 6
        self.num_checkpoints = 3
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(25)
        
        # State space: 73 elements total
        # Products (30) + Machine status (15) + Queues (5) + Quality (6) + 
        # Raw materials (1) + Targets (6) + Utilization (5) + Maintenance (5)
        self.observation_space = spaces.Box(
            low=0, high=500, shape=(73,), dtype=np.float32
        )
        
        # Initialize pygame for rendering
        self.screen = None
        self.clock = None
        self.font = None
        self.render_mode = None
        
        # Production specifications
        self.product_specs = {
            'A': {'stations': 1, 'timesteps': 10, 'quality_target': 0.80},
            'B': {'stations': 2, 'timesteps': 15, 'quality_target': 0.85},
            'C': {'stations': 3, 'timesteps': 20, 'quality_target': 0.90},
            'D': {'stations': 4, 'timesteps': 25, 'quality_target': 0.85},
            'E': {'stations': 5, 'timesteps': 30, 'quality_target': 0.95},
            'F': {'stations': 3, 'timesteps': 18, 'quality_target': 0.75},
        }
        
        # Initialize environment state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        # Initialize stations
        self.stations = []
        machine_types = [MachineType.CUTTING, MachineType.ASSEMBLY, 
                        MachineType.ASSEMBLY, MachineType.TESTING, MachineType.PACKAGING]
        
        for i in range(self.num_stations):
            station = Station(
                station_id=i,
                machine_type=machine_types[i],
                status=MachineStatus.OPERATIONAL,
                position=(6 * (i + 1), 7),
                utilization_rate=0.0,
                operations_count=0,
                maintenance_countdown=random.randint(100, 200),
                current_product=None,
                queue=[],
                performance_degradation=0.0
            )
            self.stations.append(station)
        
        # Initialize production state
        self.products_in_system = []
        self.completed_products = []
        self.scrapped_products = []
        self.product_id_counter = 0
        
        # Quality control checkpoints at stations 1, 3, 4 (0-indexed)
        self.quality_checkpoints = [1, 3, 4]
        self.quality_thresholds = [0.70, 0.80, 0.85]
        
        # Inventory and targets
        self.raw_material_inventory = 250
        self.production_targets = {
            'A': random.randint(5, 10),
            'B': random.randint(4, 8),
            'C': random.randint(3, 6),
            'D': random.randint(2, 5),
            'E': random.randint(2, 4),
            'F': random.randint(3, 7)
        }
        self.products_completed = {k: 0 for k in self.production_targets.keys()}
        
        # Production modes
        self.production_mode = "balanced"  # balanced, rush, quality
        self.emergency_stop = False
        
        # Metrics
        self.timestep = 0
        self.total_reward = 0
        self.quality_rate_history = []
        self.downtime_counter = 0
        self.oee_metrics = {
            'availability': 1.0,
            'performance': 1.0,
            'quality': 1.0
        }
        
        # Supply chain
        self.supply_disruption = False
        self.supply_disruption_countdown = 0
        
        # Energy tracking
        self.energy_consumption = 0
        
        # Worker shifts (efficiency modifier)
        self.current_shift_efficiency = 1.0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Generate the observation vector (73 elements)"""
        obs = np.zeros(73, dtype=np.float32)
        idx = 0
        
        # Products in system (30 elements: 6 types * 5 stations)
        for station_id in range(self.num_stations):
            for product_type in ['A', 'B', 'C', 'D', 'E', 'F']:
                count = sum(1 for p in self.products_in_system 
                          if p.current_station == station_id and 
                          p.product_type.value[0] == product_type)
                obs[idx] = count
                idx += 1
        
        # Machine status (15 elements: 5 stations * 3 attributes)
        for station in self.stations:
            obs[idx] = float(station.status == MachineStatus.OPERATIONAL)
            obs[idx + 1] = float(station.status == MachineStatus.BROKEN)
            obs[idx + 2] = float(station.status == MachineStatus.MAINTENANCE)
            idx += 3
        
        # Queue lengths (5 elements)
        for station in self.stations:
            obs[idx] = len(station.queue)
            idx += 1
        
        # Product quality scores (6 elements)
        for product_type in ['A', 'B', 'C', 'D', 'E', 'F']:
            products = [p for p in self.products_in_system 
                       if p.product_type.value[0] == product_type]
            if products:
                obs[idx] = np.mean([p.quality_score for p in products]) * 100
            else:
                obs[idx] = 85.0  # Default quality
            idx += 1
        
        # Raw material inventory (1 element)
        obs[idx] = self.raw_material_inventory
        idx += 1
        
        # Production targets remaining (6 elements)
        for product_type in ['A', 'B', 'C', 'D', 'E', 'F']:
            obs[idx] = max(0, self.production_targets[product_type] - 
                          self.products_completed[product_type])
            idx += 1
        
        # Machine utilization rates (5 elements)
        for station in self.stations:
            obs[idx] = station.utilization_rate * 100
            idx += 1
        
        # Maintenance schedule countdown (5 elements)
        for station in self.stations:
            obs[idx] = station.maintenance_countdown
            idx += 1
        
        return obs
    
    def step(self, action):
        """Execute one timestep of the environment"""
        self.timestep += 1
        reward = 0
        done = False
        info = {}
        
        # Process action
        reward += self._process_action(action)
        
        # Update production system
        self._update_production()
        
        # Check machine degradation and breakdowns
        self._update_machine_status()
        
        # Process quality control
        reward += self._quality_control()
        
        # Update metrics
        self._update_metrics()
        
        # Calculate timestep penalties/rewards
        reward += self._calculate_timestep_rewards()
        
        # Check termination conditions
        done = self._check_termination()
        
        # Update supply chain
        self._update_supply_chain()
        
        # Update worker shifts
        self._update_shifts()
        
        self.total_reward += reward
        
        info = {
            'timestep': self.timestep,
            'total_reward': self.total_reward,
            'products_completed': self.products_completed.copy(),
            'oee': self.oee_metrics.copy(),
            'energy_consumption': self.energy_consumption
        }
        
        return self._get_observation(), reward, done, info
    
    def _process_action(self, action):
        """Process the selected action"""
        reward = 0
        
        if action <= 5:
            # Start production of product type A-F
            product_type = ['A', 'B', 'C', 'D', 'E', 'F'][action]
            if self.raw_material_inventory >= 10:
                self._start_production(product_type)
                self.raw_material_inventory -= 10
            else:
                reward -= 50  # Penalty for stockout
        
        elif action <= 10:
            # Prioritize station (speed up processing)
            station_id = action - 6
            if station_id < self.num_stations:
                self.stations[station_id].utilization_rate = min(1.0, 
                    self.stations[station_id].utilization_rate + 0.2)
                self.energy_consumption += 5
        
        elif action <= 15:
            # Schedule maintenance
            station_id = action - 11
            if station_id < self.num_stations:
                if self.stations[station_id].status == MachineStatus.OPERATIONAL:
                    self.stations[station_id].status = MachineStatus.MAINTENANCE
                    self.stations[station_id].maintenance_countdown = 20
                    reward += 50  # Preventive maintenance reward
        
        elif action <= 20:
            # Adjust quality control threshold
            checkpoint_id = action - 16
            if checkpoint_id < len(self.quality_checkpoints):
                self.quality_thresholds[checkpoint_id] = min(0.95, 
                    self.quality_thresholds[checkpoint_id] + 0.05)
        
        elif action == 21:
            # Emergency stop
            self.emergency_stop = not self.emergency_stop
            if self.emergency_stop:
                reward -= 100
        
        elif action == 22:
            # Rush order mode
            self.production_mode = "rush"
            self.energy_consumption += 10
        
        elif action == 23:
            # Quality focus mode
            self.production_mode = "quality"
        
        elif action == 24:
            # Balanced production mode
            self.production_mode = "balanced"
        
        return reward
    
    def _start_production(self, product_type):
        """Start production of a new product"""
        product_enum = ProductType[product_type]
        
        new_product = Product(
            product_type=product_enum,
            current_station=-1,  # Not yet at any station
            quality_score=0.85 + random.uniform(-0.1, 0.1),
            timesteps_remaining=product_enum.value[2],
            position=(0, 7),
            id=self.product_id_counter
        )
        
        self.product_id_counter += 1
        self.products_in_system.append(new_product)
        
        # Add to first station queue
        if self.stations[0].status == MachineStatus.OPERATIONAL:
            self.stations[0].queue.append(new_product)
    
    def _update_production(self):
        """Update production at each station"""
        if self.emergency_stop:
            return
        
        for station in self.stations:
            if station.status != MachineStatus.OPERATIONAL:
                continue
            
            # Process current product
            if station.current_product:
                product = station.current_product
                product.timesteps_remaining -= 1
                
                # Apply production mode modifiers
                if self.production_mode == "rush":
                    product.timesteps_remaining -= 0.5
                    product.quality_score *= 0.98
                elif self.production_mode == "quality":
                    product.quality_score *= 1.02
                
                # Apply performance degradation
                product.quality_score *= (1 - station.performance_degradation)
                
                if product.timesteps_remaining <= 0:
                    # Move to next station or complete
                    station.current_product = None
                    station.operations_count += 1
                    
                    next_station = product.current_station + 1
                    if next_station < product.product_type.value[1]:
                        # Move to next required station
                        if next_station < self.num_stations:
                            self.stations[next_station].queue.append(product)
                            product.current_station = next_station
                    else:
                        # Product completed
                        self._complete_product(product)
            
            # Load next product from queue
            if not station.current_product and station.queue:
                station.current_product = station.queue.pop(0)
                station.utilization_rate = 0.8
            else:
                station.utilization_rate *= 0.95  # Decay when idle
    
    def _update_machine_status(self):
        """Update machine status and handle breakdowns"""
        for station in self.stations:
            # Handle maintenance
            if station.status == MachineStatus.MAINTENANCE:
                station.maintenance_countdown -= 1
                if station.maintenance_countdown <= 0:
                    station.status = MachineStatus.OPERATIONAL
                    station.performance_degradation = 0
                    station.operations_count = 0
                    station.maintenance_countdown = random.randint(100, 200)
            
            # Check for breakdowns
            elif station.status == MachineStatus.OPERATIONAL:
                # Breakdown probability increases with usage
                breakdown_prob = 0.001 * (1 + station.operations_count / 100)
                if random.random() < breakdown_prob:
                    station.status = MachineStatus.BROKEN
                    station.maintenance_countdown = 30
                
                # Performance degradation
                if station.operations_count % 100 == 0:
                    station.performance_degradation += 0.005
                
                # Maintenance countdown
                station.maintenance_countdown -= 1
                if station.maintenance_countdown <= 10:
                    # Warning: maintenance needed soon
                    pass
            
            # Handle broken machines
            elif station.status == MachineStatus.BROKEN:
                station.maintenance_countdown -= 1
                if station.maintenance_countdown <= 0:
                    station.status = MachineStatus.OPERATIONAL
                    station.maintenance_countdown = random.randint(100, 200)
    
    def _quality_control(self):
        """Perform quality control checks"""
        reward = 0
        
        for i, checkpoint_station in enumerate(self.quality_checkpoints):
            if checkpoint_station < len(self.stations):
                station = self.stations[checkpoint_station]
                if station.current_product:
                    product = station.current_product
                    if product.quality_score < self.quality_thresholds[i]:
                        # Product failed quality control
                        self.scrapped_products.append(product)
                        self.products_in_system.remove(product)
                        station.current_product = None
                        reward -= 100
        
        return reward
    
    def _complete_product(self, product):
        """Handle product completion"""
        product_type = product.product_type.value[0]
        self.products_completed[product_type] += 1
        self.completed_products.append(product)
        
        if product in self.products_in_system:
            self.products_in_system.remove(product)
        
        # Calculate reward based on quality
        quality_percent = product.quality_score * 100
        if quality_percent > 90:
            return 500
        elif quality_percent > 70:
            return 300
        elif quality_percent > 50:
            return 100
        else:
            return -100
    
    def _calculate_timestep_rewards(self):
        """Calculate rewards/penalties for current timestep"""
        reward = 0
        
        # Penalty for idle machines
        idle_count = sum(1 for s in self.stations 
                        if s.status == MachineStatus.OPERATIONAL and 
                        s.current_product is None and len(s.queue) == 0)
        reward -= idle_count * 10
        
        # Check for broken machines
        broken_count = sum(1 for s in self.stations 
                          if s.status == MachineStatus.BROKEN)
        if broken_count > 0:
            reward -= broken_count * 50
        
        # Check production targets
        for product_type, target in self.production_targets.items():
            if self.timestep > 1000 and self.products_completed[product_type] < target:
                reward -= 50  # Penalty for missing targets
        
        # Quality rate bonus/penalty
        if len(self.completed_products) > 0:
            recent_quality = np.mean([p.quality_score for p in self.completed_products[-10:]])
            if recent_quality > 0.9:
                reward += 20
            elif recent_quality < 0.6:
                reward -= 30
        
        return reward
    
    def _update_metrics(self):
        """Update OEE and other metrics"""
        # Calculate availability
        operational_stations = sum(1 for s in self.stations 
                                 if s.status == MachineStatus.OPERATIONAL)
        self.oee_metrics['availability'] = operational_stations / self.num_stations
        
        # Calculate performance
        avg_utilization = np.mean([s.utilization_rate for s in self.stations])
        self.oee_metrics['performance'] = avg_utilization
        
        # Calculate quality
        if len(self.completed_products) > 0:
            good_products = len([p for p in self.completed_products 
                               if p.quality_score > 0.7])
            self.oee_metrics['quality'] = good_products / len(self.completed_products)
        
        # Track quality rate history
        if len(self.completed_products) > 0:
            recent_quality = np.mean([p.quality_score for p in self.completed_products[-20:]])
            self.quality_rate_history.append(recent_quality)
    
    def _check_termination(self):
        """Check if episode should terminate"""
        # Check if all production targets achieved
        all_targets_met = all(self.products_completed[p] >= t 
                             for p, t in self.production_targets.items())
        if all_targets_met:
            return True
        
        # Check for too many broken machines
        broken_count = sum(1 for s in self.stations if s.status == MachineStatus.BROKEN)
        if broken_count >= 3:
            return True
        
        # Check timestep limit
        if self.timestep >= 1500:
            return True
        
        # Check quality rate
        if len(self.quality_rate_history) >= 100:
            recent_quality = np.mean(self.quality_rate_history[-100:])
            if recent_quality < 0.6:
                return True
        
        return False
    
    def _update_supply_chain(self):
        """Update supply chain and raw materials"""
        # Random supply disruptions
        if not self.supply_disruption and random.random() < 0.01:
            self.supply_disruption = True
            self.supply_disruption_countdown = random.randint(20, 50)
        
        if self.supply_disruption:
            self.supply_disruption_countdown -= 1
            if self.supply_disruption_countdown <= 0:
                self.supply_disruption = False
        else:
            # Regular material delivery
            if self.timestep % 50 == 0:
                self.raw_material_inventory = min(500, 
                    self.raw_material_inventory + random.randint(50, 100))
    
    def _update_shifts(self):
        """Update worker shift efficiency"""
        # Simulate 3 shifts with different efficiency levels
        shift_hour = (self.timestep // 100) % 3
        if shift_hour == 0:
            self.current_shift_efficiency = 1.0  # Day shift
        elif shift_hour == 1:
            self.current_shift_efficiency = 0.95  # Evening shift
        else:
            self.current_shift_efficiency = 0.9  # Night shift
    
    def render(self, mode='human'):
        """Render the environment using pygame"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 600))
            pygame.display.set_caption("Smart Manufacturing Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        # Clear screen
        self.screen.fill((240, 240, 240))
        
        # Draw factory floor
        self._draw_factory_floor()
        
        # Draw stations
        self._draw_stations()
        
        # Draw products
        self._draw_products()
        
        # Draw dashboard
        self._draw_dashboard()
        
        # Draw alerts
        self._draw_alerts()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)
        
        if mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), 
                axes=(1, 0, 2)
            )
    
    def _draw_factory_floor(self):
        """Draw the factory floor layout"""
        # Draw floor grid
        for x in range(0, 1200, 40):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, 400), 1)
        for y in range(0, 400, 40):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (1200, y), 1)
        
        # Draw conveyor belt path
        pygame.draw.line(self.screen, (100, 100, 100), (20, 200), (1180, 200), 3)
        
        # Draw input/output areas
        pygame.draw.circle(self.screen, (0, 255, 0), (20, 200), 15)  # Input
        text = self.font.render("RAW", True, (255, 255, 255))
        self.screen.blit(text, (5, 185))
        
        pygame.draw.circle(self.screen, (0, 0, 255), (1180, 200), 15)  # Output
        text = self.font.render("OUT", True, (255, 255, 255))
        self.screen.blit(text, (1165, 185))
    
    def _draw_stations(self):
        """Draw production stations"""
        colors = {
            MachineStatus.OPERATIONAL: (0, 200, 0),
            MachineStatus.BROKEN: (200, 0, 0),
            MachineStatus.MAINTENANCE: (200, 200, 0)
        }
        
        for i, station in enumerate(self.stations):
            x = 200 + i * 200
            y = 150
            
            # Draw station rectangle
            color = colors[station.status]
            pygame.draw.rect(self.screen, color, (x, y, 80, 100), 0)
            pygame.draw.rect(self.screen, (0, 0, 0), (x, y, 80, 100), 2)
            
            # Draw station label
            text = self.font.render(f"S{i+1}", True, (255, 255, 255))
            self.screen.blit(text, (x + 30, y + 10))
            
            # Draw machine type
            text = self.font.render(station.machine_type.value[:3], True, (255, 255, 255))
            self.screen.blit(text, (x + 25, y + 35))
            
            # Draw utilization bar
            util_width = int(70 * station.utilization_rate)
            pygame.draw.rect(self.screen, (0, 100, 200), (x + 5, y + 60, util_width, 10))
            pygame.draw.rect(self.screen, (0, 0, 0), (x + 5, y + 60, 70, 10), 1)
            
            # Draw health/degradation bar
            health = 1.0 - station.performance_degradation
            health_width = int(70 * health)
            color = (0, 255, 0) if health > 0.7 else (255, 255, 0) if health > 0.3 else (255, 0, 0)
            pygame.draw.rect(self.screen, color, (x + 5, y + 75, health_width, 10))
            pygame.draw.rect(self.screen, (0, 0, 0), (x + 5, y + 75, 70, 10), 1)
            
            # Draw queue length
            text = self.font.render(f"Q:{len(station.queue)}", True, (0, 0, 0))
            self.screen.blit(text, (x + 20, y + 105))
            
            # Draw maintenance countdown if low
            if station.maintenance_countdown < 20:
                text = self.font.render(f"M:{station.maintenance_countdown}", True, (255, 0, 0))
                self.screen.blit(text, (x + 20, y - 25))
    
    def _draw_products(self):
        """Draw products in the system"""
        product_colors = {
            'A': (255, 100, 100),
            'B': (100, 255, 100),
            'C': (100, 100, 255),
            'D': (255, 255, 100),
            'E': (255, 100, 255),
            'F': (100, 255, 255)
        }
        
        for product in self.products_in_system:
            # Calculate position based on station
            if product.current_station >= 0 and product.current_station < self.num_stations:
                x = 240 + product.current_station * 200
                y = 200
            else:
                x, y = 50, 200
            
            color = product_colors.get(product.product_type.value[0], (150, 150, 150))
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
            
            # Draw product type label
            text = self.font.render(product.product_type.value[0], True, (0, 0, 0))
            self.screen.blit(text, (int(x) - 5, int(y) - 5))
    
    def _draw_dashboard(self):
        """Draw production dashboard"""
        dashboard_y = 420
        
        # Background
        pygame.draw.rect(self.screen, (50, 50, 50), (0, dashboard_y, 1200, 180))
        
        # Production metrics
        metrics_text = [
            f"Timestep: {self.timestep}",
            f"Mode: {self.production_mode.upper()}",
            f"Raw Materials: {self.raw_material_inventory}/500",
            f"Energy: {self.energy_consumption:.0f}",
            f"Shift Efficiency: {self.current_shift_efficiency:.0%}"
        ]
        
        for i, text_str in enumerate(metrics_text):
            text = self.font.render(text_str, True, (255, 255, 255))
            self.screen.blit(text, (20, dashboard_y + 10 + i * 30))
        
        # OEE Metrics
        oee_text = [
            f"OEE Availability: {self.oee_metrics['availability']:.1%}",
            f"OEE Performance: {self.oee_metrics['performance']:.1%}",
            f"OEE Quality: {self.oee_metrics['quality']:.1%}",
            f"Overall OEE: {(self.oee_metrics['availability'] * self.oee_metrics['performance'] * self.oee_metrics['quality']):.1%}"
        ]
        
        for i, text_str in enumerate(oee_text):
            text = self.font.render(text_str, True, (255, 255, 255))
            self.screen.blit(text, (300, dashboard_y + 10 + i * 30))
        
        # Production targets
        target_x = 600
        text = self.font.render("Production Status:", True, (255, 255, 255))
        self.screen.blit(text, (target_x, dashboard_y + 10))
        
        for i, (product_type, target) in enumerate(self.production_targets.items()):
            completed = self.products_completed[product_type]
            color = (0, 255, 0) if completed >= target else (255, 255, 0) if completed > 0 else (255, 0, 0)
            text = self.font.render(f"{product_type}: {completed}/{target}", True, color)
            self.screen.blit(text, (target_x + (i % 3) * 100, dashboard_y + 40 + (i // 3) * 30))
        
        # Quality meters for checkpoints
        quality_x = 900
        text = self.font.render("Quality Checkpoints:", True, (255, 255, 255))
        self.screen.blit(text, (quality_x, dashboard_y + 10))
        
        for i, threshold in enumerate(self.quality_thresholds):
            # Draw quality bar
            bar_height = int(threshold * 50)
            color = (0, 255, 0) if threshold > 0.8 else (255, 255, 0) if threshold > 0.6 else (255, 0, 0)
            pygame.draw.rect(self.screen, color, 
                            (quality_x + i * 40, dashboard_y + 90 - bar_height, 30, bar_height))
            pygame.draw.rect(self.screen, (255, 255, 255), 
                            (quality_x + i * 40, dashboard_y + 40, 30, 50), 1)
            
            text = self.font.render(f"C{i+1}", True, (255, 255, 255))
            self.screen.blit(text, (quality_x + i * 40 + 5, dashboard_y + 95))
    
    def _draw_alerts(self):
        """Draw alerts and warnings"""
        alerts = []
        
        # Check for broken machines
        for station in self.stations:
            if station.status == MachineStatus.BROKEN:
                alerts.append(f"⚠ Station {station.station_id + 1} BROKEN!")
            elif station.maintenance_countdown < 10:
                alerts.append(f"⚠ Station {station.station_id + 1} needs maintenance")
        
        # Check for low materials
        if self.raw_material_inventory < 50:
            alerts.append("⚠ Low raw materials!")
        
        # Check for supply disruption
        if self.supply_disruption:
            alerts.append(f"⚠ Supply disruption! ({self.supply_disruption_countdown} steps)")
        
        # Check quality issues
        if len(self.quality_rate_history) > 10:
            recent_quality = np.mean(self.quality_rate_history[-10:])
            if recent_quality < 0.7:
                alerts.append(f"⚠ Quality issues! ({recent_quality:.1%})")
        
        # Draw alerts
        alert_y = 300
        for i, alert in enumerate(alerts[:5]):  # Show max 5 alerts
            color = (255, 0, 0) if "BROKEN" in alert else (255, 200, 0)
            text = self.font.render(alert, True, color)
            self.screen.blit(text, (20, alert_y + i * 25))
    
    def close(self):
        """Close the environment and clean up pygame"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None