"""
Agricultural Farm Management Environment for Gymnasium.
A comprehensive farming simulation with crop management, weather systems, and market dynamics.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import pygame
import random
from dataclasses import dataclass, field
from enum import IntEnum
import math
import os

try:
    from .crops.crop_data import (
        CROP_DATA, CropType, CropStage, 
        get_crop_stage_duration, calculate_yield_modifier,
        get_market_price_multiplier
    )
except ImportError:
    from crops.crop_data import (
        CROP_DATA, CropType, CropStage, 
        get_crop_stage_duration, calculate_yield_modifier,
        get_market_price_multiplier
    )


class Season(IntEnum):
    """Seasons of the year."""
    SPRING = 0
    SUMMER = 1
    FALL = 2
    WINTER = 3


class EquipmentType(IntEnum):
    """Types of farming equipment."""
    TRACTOR = 0
    HARVESTER = 1
    IRRIGATION_SYSTEM = 2
    FERTILIZER_SPREADER = 3
    PESTICIDE_SPRAYER = 4
    SEED_PLANTER = 5
    SOIL_TESTER = 6
    WEATHER_STATION = 7


@dataclass
class FieldSection:
    """Represents a section of the farm field."""
    id: int
    name: str
    size: int  # Number of grid cells
    crop_type: Optional[CropType] = None
    growth_stage: CropStage = CropStage.DORMANT
    health: float = 1.0
    yield_potential: float = 1.0
    days_to_harvest: int = 0
    days_since_planting: int = 0
    last_crop: Optional[CropType] = None
    
    # Soil conditions
    soil_moisture: float = 0.5
    soil_ph: float = 6.5
    soil_nitrogen: float = 100.0
    soil_phosphorus: float = 50.0
    soil_potassium: float = 50.0
    soil_compaction: float = 0.3
    soil_organic_matter: float = 0.04
    
    # Pest and disease
    pest_level: float = 0.0
    disease_risk: float = 0.0
    
    # Management history
    last_fertilized_day: int = -100
    last_irrigated_day: int = -10
    last_pesticide_day: int = -100


@dataclass
class Equipment:
    """Represents a piece of farming equipment."""
    type: EquipmentType
    location: Tuple[int, int] = (0, 0)
    fuel_level: float = 1.0
    maintenance_needed: float = 0.0
    is_operating: bool = False
    operation_field: Optional[int] = None


@dataclass
class Weather:
    """Weather conditions."""
    temperature: float = 20.0
    humidity: float = 0.5
    rainfall: float = 0.0
    wind_speed: float = 10.0
    solar_radiation: float = 15.0
    frost_risk: float = 0.0
    drought_days: int = 0
    excessive_rain_days: int = 0


@dataclass
class MarketPrices:
    """Market prices for crops."""
    current_prices: Dict[CropType, float] = field(default_factory=dict)
    predicted_prices: Dict[CropType, float] = field(default_factory=dict)
    price_volatility: float = 0.1


class AgriculturalFarmEnv(gym.Env):
    """
    Agricultural Farm Management Environment.
    
    The agent manages a 25x25 farm grid divided into 4 field sections,
    making decisions about planting, harvesting, irrigation, fertilization,
    and pest management while dealing with weather uncertainty and market dynamics.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        
        # Farm dimensions
        self.grid_size = 25
        self.num_fields = 4
        
        # Time tracking
        self.current_day = 0
        self.current_season = Season.SPRING
        self.year = 1
        self.max_years = 1
        
        # Initialize field sections
        self.field_sections: List[FieldSection] = []
        self._initialize_fields()
        
        # Initialize equipment
        self.equipment: List[Equipment] = []
        self._initialize_equipment()
        
        # Weather system
        self.weather = Weather()
        self.climate_change_factor = 0.0
        
        # Market system
        self.market = MarketPrices()
        self._initialize_market()
        
        # Water resources
        self.reservoir_level = 0.8
        self.irrigation_capacity = 1.0
        self.water_quality = 0.9
        self.groundwater_level = 0.7
        self.water_costs = 10.0
        
        # Farm finances
        self.cash_flow = 50000.0
        self.operating_costs = 0.0
        self.debt_level = 0.0
        self.profit_margin = 0.0
        self.total_revenue = 0.0
        self.total_expenses = 0.0
        
        # Sustainability metrics
        self.carbon_sequestered = 0.0
        self.water_conservation_score = 0.0
        self.biodiversity_index = 0.5
        self.soil_health_trend = 0.0
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(45)
        
        # State space: Complex observation with multiple components
        # Total: ~620 elements to capture full farm state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(620,), dtype=np.float32
        )
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.font = None
        if self.render_mode == "human":
            self._init_pygame()
    
    def _initialize_fields(self):
        """Initialize the 4 field sections of the farm."""
        field_configs = [
            {"id": 0, "name": "North Field", "size": 156, "initial_crop": CropType.WHEAT},
            {"id": 1, "name": "South Field", "size": 156, "initial_crop": CropType.CORN},
            {"id": 2, "name": "East Field", "size": 156, "initial_crop": CropType.TOMATOES},
            {"id": 3, "name": "West Field", "size": 157, "initial_crop": None}  # Fruit trees area
        ]
        
        for config in field_configs:
            field = FieldSection(
                id=config["id"],
                name=config["name"],
                size=config["size"],
                crop_type=config["initial_crop"]
            )
            
            # Randomize initial soil conditions
            field.soil_moisture = np.random.uniform(0.3, 0.7)
            field.soil_ph = np.random.uniform(6.0, 7.0)
            field.soil_nitrogen = np.random.uniform(80, 120)
            field.soil_phosphorus = np.random.uniform(40, 60)
            field.soil_potassium = np.random.uniform(40, 60)
            field.soil_compaction = np.random.uniform(0.2, 0.4)
            field.soil_organic_matter = np.random.uniform(0.03, 0.05)
            
            self.field_sections.append(field)
    
    def _initialize_equipment(self):
        """Initialize farming equipment."""
        equipment_types = list(EquipmentType)
        for eq_type in equipment_types:
            self.equipment.append(Equipment(
                type=eq_type,
                location=(np.random.randint(0, self.grid_size), 
                         np.random.randint(0, self.grid_size)),
                fuel_level=np.random.uniform(0.7, 1.0),
                maintenance_needed=np.random.uniform(0.0, 0.2)
            ))
    
    def _initialize_market(self):
        """Initialize market prices for crops."""
        for crop_type in CropType:
            base_price = CROP_DATA[crop_type].market_base_price
            variation = np.random.uniform(0.9, 1.1)
            self.market.current_prices[crop_type] = base_price * variation
            self.market.predicted_prices[crop_type] = base_price * np.random.uniform(0.85, 1.15)
    
    def _get_season_weather_params(self, season: Season) -> Dict[str, Tuple[float, float]]:
        """Get weather parameter ranges for each season."""
        weather_params = {
            Season.SPRING: {
                "temperature": (10.0, 22.0),
                "humidity": (0.4, 0.7),
                "rainfall": (0.0, 15.0),
                "wind_speed": (5.0, 20.0),
                "frost_risk": (0.0, 0.2)
            },
            Season.SUMMER: {
                "temperature": (20.0, 35.0),
                "humidity": (0.3, 0.6),
                "rainfall": (0.0, 10.0),
                "wind_speed": (3.0, 15.0),
                "frost_risk": (0.0, 0.0)
            },
            Season.FALL: {
                "temperature": (8.0, 20.0),
                "humidity": (0.5, 0.8),
                "rainfall": (0.0, 20.0),
                "wind_speed": (8.0, 25.0),
                "frost_risk": (0.0, 0.3)
            },
            Season.WINTER: {
                "temperature": (-5.0, 10.0),
                "humidity": (0.6, 0.9),
                "rainfall": (0.0, 25.0),
                "wind_speed": (10.0, 30.0),
                "frost_risk": (0.2, 0.7)
            }
        }
        return weather_params[season]
    
    def _update_weather(self):
        """Update daily weather conditions."""
        params = self._get_season_weather_params(self.current_season)
        
        # Add climate change effects
        temp_shift = self.climate_change_factor * 2.0
        
        # Update weather with some continuity from previous day
        alpha = 0.7  # Continuity factor
        
        temp_range = params["temperature"]
        new_temp = np.random.uniform(temp_range[0] + temp_shift, temp_range[1] + temp_shift)
        self.weather.temperature = alpha * self.weather.temperature + (1 - alpha) * new_temp
        
        humidity_range = params["humidity"]
        new_humidity = np.random.uniform(*humidity_range)
        self.weather.humidity = alpha * self.weather.humidity + (1 - alpha) * new_humidity
        
        # Rainfall with probability
        if np.random.random() < 0.3:  # 30% chance of rain
            rainfall_range = params["rainfall"]
            self.weather.rainfall = np.random.uniform(*rainfall_range)
        else:
            self.weather.rainfall = 0.0
        
        wind_range = params["wind_speed"]
        self.weather.wind_speed = np.random.uniform(*wind_range)
        
        frost_range = params["frost_risk"]
        self.weather.frost_risk = np.random.uniform(*frost_range) if self.weather.temperature < 5 else 0.0
        
        # Track drought and excessive rain
        if self.weather.rainfall < 2.0:
            self.weather.drought_days += 1
            self.weather.excessive_rain_days = 0
        elif self.weather.rainfall > 15.0:
            self.weather.excessive_rain_days += 1
            self.weather.drought_days = 0
        else:
            self.weather.drought_days = max(0, self.weather.drought_days - 1)
            self.weather.excessive_rain_days = max(0, self.weather.excessive_rain_days - 1)
    
    def _update_crop_growth(self):
        """Update crop growth for all fields."""
        for field in self.field_sections:
            if field.crop_type is not None and field.growth_stage != CropStage.HARVESTED:
                field.days_since_planting += 1
                
                # Check growth stage progression
                crop_data = CROP_DATA[field.crop_type]
                stage_duration = get_crop_stage_duration(field.crop_type, field.growth_stage)
                
                # Progress to next stage if time elapsed
                if field.days_since_planting % stage_duration == 0 and field.growth_stage < CropStage.HARVEST_READY:
                    field.growth_stage = CropStage(field.growth_stage + 1)
                
                # Update days to harvest
                field.days_to_harvest = max(0, crop_data.growth_cycle_days - field.days_since_planting)
                
                # Calculate yield potential based on conditions
                yield_modifier = calculate_yield_modifier(
                    field.crop_type,
                    field.soil_moisture,
                    field.soil_ph,
                    {"nitrogen": field.soil_nitrogen, 
                     "phosphorus": field.soil_phosphorus,
                     "potassium": field.soil_potassium},
                    self.weather.temperature,
                    field.pest_level,
                    field.disease_risk
                )
                
                field.yield_potential = yield_modifier
                field.health = max(0.0, min(1.0, field.health - 0.01 * (field.pest_level + field.disease_risk)))
                
                # Natural pest and disease progression
                if np.random.random() < 0.05:  # 5% chance of pest increase
                    field.pest_level = min(1.0, field.pest_level + np.random.uniform(0.01, 0.05))
                
                if self.weather.humidity > 0.7 and self.weather.temperature > 15:
                    field.disease_risk = min(1.0, field.disease_risk + 0.02)
    
    def _update_soil_conditions(self):
        """Update soil conditions based on weather and management."""
        for field in self.field_sections:
            # Moisture changes from rainfall and evaporation
            field.soil_moisture += self.weather.rainfall * 0.01
            field.soil_moisture -= (self.weather.temperature / 100.0) * (1.0 - self.weather.humidity)
            field.soil_moisture = np.clip(field.soil_moisture, 0.0, 1.0)
            
            # Nutrient depletion from crop growth
            if field.crop_type is not None:
                crop_data = CROP_DATA[field.crop_type]
                daily_depletion = 0.001
                field.soil_nitrogen -= crop_data.nutrient_needs["nitrogen"] * daily_depletion
                field.soil_phosphorus -= crop_data.nutrient_needs["phosphorus"] * daily_depletion
                field.soil_potassium -= crop_data.nutrient_needs["potassium"] * daily_depletion
                
                # Nitrogen fixation for soybeans
                if field.crop_type == CropType.SOYBEANS:
                    field.soil_nitrogen += 0.5
            
            # Soil compaction from equipment
            if any(eq.operation_field == field.id for eq in self.equipment if eq.is_operating):
                field.soil_compaction = min(1.0, field.soil_compaction + 0.01)
            else:
                # Natural decompaction
                field.soil_compaction = max(0.0, field.soil_compaction - 0.001)
            
            # Organic matter changes
            if field.last_crop is not None:
                field.soil_organic_matter += 0.0001  # Residue decomposition
            field.soil_organic_matter = np.clip(field.soil_organic_matter, 0.01, 0.1)
    
    def _update_market_prices(self):
        """Update market prices with volatility."""
        for crop_type in CropType:
            base_price = CROP_DATA[crop_type].market_base_price
            seasonal_mult = get_market_price_multiplier(
                ["spring", "summer", "fall", "winter"][self.current_season],
                crop_type
            )
            
            # Random walk with mean reversion
            current = self.market.current_prices[crop_type]
            change = np.random.normal(0, self.market.price_volatility * base_price)
            new_price = current + change
            
            # Mean reversion
            target = base_price * seasonal_mult
            new_price = 0.9 * new_price + 0.1 * target
            
            self.market.current_prices[crop_type] = max(base_price * 0.5, min(base_price * 2.0, new_price))
            
            # Update predictions (less accurate)
            self.market.predicted_prices[crop_type] = new_price * np.random.uniform(0.9, 1.1)
    
    def _calculate_operating_costs(self) -> float:
        """Calculate daily operating costs."""
        costs = 0.0
        
        # Equipment fuel costs
        for eq in self.equipment:
            if eq.is_operating:
                costs += 50.0  # Daily fuel cost when operating
        
        # Labor costs
        costs += 200.0  # Base daily labor
        
        # Water costs
        costs += self.water_costs * (1.0 - self.reservoir_level) * 10
        
        # Maintenance costs
        for eq in self.equipment:
            if eq.maintenance_needed > 0.5:
                costs += 100.0
        
        return costs
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation of the environment."""
        obs = []
        
        # Field conditions (4 fields * 15 features = 60)
        for field in self.field_sections:
            obs.extend([
                field.crop_type.value if field.crop_type else -1,
                field.growth_stage.value,
                field.health,
                field.yield_potential,
                field.days_to_harvest,
                field.soil_moisture,
                field.soil_ph,
                field.soil_nitrogen / 200.0,  # Normalize
                field.soil_phosphorus / 100.0,
                field.soil_potassium / 100.0,
                field.soil_compaction,
                field.soil_organic_matter * 10,
                field.pest_level,
                field.disease_risk,
                field.days_since_planting / 365.0
            ])
        
        # Weather data (10)
        obs.extend([
            self.weather.temperature / 40.0,
            self.weather.humidity,
            self.weather.rainfall / 30.0,
            self.weather.wind_speed / 40.0,
            self.weather.solar_radiation / 30.0,
            self.weather.frost_risk,
            self.weather.drought_days / 30.0,
            self.weather.excessive_rain_days / 10.0,
            self.current_season.value / 3.0,
            self.current_day / 365.0
        ])
        
        # Equipment status (8 * 5 = 40)
        for eq in self.equipment:
            obs.extend([
                eq.location[0] / self.grid_size,
                eq.location[1] / self.grid_size,
                eq.fuel_level,
                eq.maintenance_needed,
                float(eq.is_operating)
            ])
        
        # Market prices (5 * 2 = 10)
        for crop_type in CropType:
            obs.append(self.market.current_prices[crop_type] / 1000.0)
            obs.append(self.market.predicted_prices[crop_type] / 1000.0)
        
        # Water resources (5)
        obs.extend([
            self.reservoir_level,
            self.irrigation_capacity,
            self.water_quality,
            self.groundwater_level,
            self.water_costs / 50.0
        ])
        
        # Financial status (5)
        obs.extend([
            self.cash_flow / 100000.0,
            self.operating_costs / 1000.0,
            self.debt_level / 100000.0,
            self.profit_margin,
            (self.total_revenue - self.total_expenses) / 100000.0
        ])
        
        # Sustainability metrics (4)
        obs.extend([
            self.carbon_sequestered / 100.0,
            self.water_conservation_score,
            self.biodiversity_index,
            self.soil_health_trend
        ])
        
        # Pad to match observation space
        while len(obs) < 620:
            obs.append(0.0)
        
        return np.array(obs[:620], dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset time
        self.current_day = 0
        self.current_season = Season.SPRING
        self.year = 1
        
        # Reset fields
        self.field_sections = []
        self._initialize_fields()
        
        # Reset equipment
        self.equipment = []
        self._initialize_equipment()
        
        # Reset weather
        self.weather = Weather()
        self.climate_change_factor = 0.0
        
        # Reset market
        self.market = MarketPrices()
        self._initialize_market()
        
        # Reset water
        self.reservoir_level = 0.8
        self.groundwater_level = 0.7
        
        # Reset finances
        self.cash_flow = 50000.0
        self.operating_costs = 0.0
        self.debt_level = 0.0
        self.profit_margin = 0.0
        self.total_revenue = 0.0
        self.total_expenses = 0.0
        
        # Reset sustainability
        self.carbon_sequestered = 0.0
        self.water_conservation_score = 0.0
        self.biodiversity_index = 0.5
        self.soil_health_trend = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        reward = 0.0
        
        # Process action
        reward += self._process_action(action)
        
        # Daily updates
        self._update_weather()
        self._update_crop_growth()
        self._update_soil_conditions()
        self._update_market_prices()
        
        # Update water resources
        self.reservoir_level += self.weather.rainfall * 0.001
        self.reservoir_level -= 0.01  # Daily evaporation
        self.reservoir_level = np.clip(self.reservoir_level, 0.0, 1.0)
        
        # Calculate daily costs
        self.operating_costs = self._calculate_operating_costs()
        self.total_expenses += self.operating_costs
        self.cash_flow -= self.operating_costs
        
        # Check for harvest opportunities
        for field in self.field_sections:
            if field.growth_stage == CropStage.HARVEST_READY and field.days_to_harvest <= 0:
                # Auto-harvest if overdue
                harvest_reward = self._harvest_crop(field)
                reward += harvest_reward
        
        # Environmental rewards/penalties
        reward += self._calculate_environmental_rewards()
        
        # Update sustainability metrics
        self._update_sustainability_metrics()
        
        # Advance time
        self.current_day += 1
        if self.current_day % 90 == 0:  # Season change every 90 days
            self.current_season = Season((self.current_season + 1) % 4)
            if self.current_season == Season.SPRING:
                self.year += 1
                self.climate_change_factor += 0.05  # Gradual climate change
        
        # Check termination conditions
        terminated = False
        if self.year > self.max_years:
            terminated = True
        elif self.cash_flow < -50000:  # Bankruptcy
            terminated = True
            reward -= 5000
        elif all(field.soil_organic_matter < 0.01 for field in self.field_sections):
            terminated = True  # Severe soil degradation
            reward -= 3000
        
        truncated = False
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action: int) -> float:
        """Process the selected action and return immediate reward."""
        reward = 0.0
        
        if 0 <= action <= 4:
            # Plant crop in field
            field_id = action
            if field_id < len(self.field_sections):
                field = self.field_sections[field_id]
                if field.crop_type is None or field.growth_stage == CropStage.HARVESTED:
                    # Randomly select a crop type suitable for the season
                    suitable_crops = []
                    season_name = ["spring", "summer", "fall", "winter"][self.current_season]
                    for crop_type in CropType:
                        if season_name in CROP_DATA[crop_type].planting_seasons:
                            suitable_crops.append(crop_type)
                    
                    if suitable_crops:
                        field.crop_type = random.choice(suitable_crops)
                        field.growth_stage = CropStage.PLANTING
                        field.days_since_planting = 0
                        field.health = 1.0
                        field.yield_potential = 1.0
                        
                        # Planting costs
                        self.cash_flow -= 500
                        reward += 50  # Small reward for planting
                        
                        # Rotation bonus
                        if field.last_crop and field.last_crop != field.crop_type:
                            reward += 100
        
        elif 5 <= action <= 9:
            # Harvest crop from field
            field_id = action - 5
            if field_id < len(self.field_sections):
                field = self.field_sections[field_id]
                if field.growth_stage == CropStage.HARVEST_READY:
                    reward += self._harvest_crop(field)
        
        elif 10 <= action <= 14:
            # Apply fertilizer
            field_id = action - 10
            if field_id < len(self.field_sections):
                field = self.field_sections[field_id]
                if self.current_day - field.last_fertilized_day > 14:  # Min 2 weeks between applications
                    field.soil_nitrogen += 30
                    field.soil_phosphorus += 20
                    field.soil_potassium += 20
                    field.last_fertilized_day = self.current_day
                    
                    self.cash_flow -= 300
                    reward += 20
                    
                    # Over-fertilization penalty
                    if field.soil_nitrogen > 150:
                        reward -= 100
                        self.water_quality -= 0.02
        
        elif 15 <= action <= 19:
            # Irrigate field
            field_id = action - 15
            if field_id < len(self.field_sections):
                field = self.field_sections[field_id]
                if self.reservoir_level > 0.1:
                    irrigation_amount = min(0.05, self.reservoir_level)
                    field.soil_moisture += irrigation_amount * 2
                    field.soil_moisture = min(1.0, field.soil_moisture)
                    self.reservoir_level -= irrigation_amount
                    field.last_irrigated_day = self.current_day
                    
                    self.cash_flow -= self.water_costs * irrigation_amount * 100
                    
                    if field.soil_moisture < 0.3:  # Was critically dry
                        reward += 50
        
        elif 20 <= action <= 24:
            # Apply pesticide/fungicide
            field_id = action - 20
            if field_id < len(self.field_sections):
                field = self.field_sections[field_id]
                if self.current_day - field.last_pesticide_day > 21:  # Min 3 weeks between applications
                    field.pest_level = max(0, field.pest_level - 0.5)
                    field.disease_risk = max(0, field.disease_risk - 0.3)
                    field.last_pesticide_day = self.current_day
                    
                    self.cash_flow -= 400
                    
                    # Environmental penalty for chemical use
                    self.biodiversity_index -= 0.02
                    self.water_quality -= 0.01
                    reward -= 50
                    
                    if field.pest_level > 0.5:  # Was severe infestation
                        reward += 200
        
        elif 25 <= action <= 29:
            # Operate specific equipment
            eq_id = action - 25
            if eq_id < len(self.equipment):
                eq = self.equipment[eq_id]
                if eq.fuel_level > 0.1 and eq.maintenance_needed < 0.8:
                    eq.is_operating = True
                    eq.fuel_level -= 0.1
                    eq.maintenance_needed += 0.02
                    reward += 10
                else:
                    reward -= 20  # Penalty for trying to use broken equipment
        
        elif 30 <= action <= 34:
            # Soil testing and preparation
            field_id = action - 30
            if field_id < len(self.field_sections):
                field = self.field_sections[field_id]
                # Soil testing provides information bonus
                reward += 15
                self.cash_flow -= 100
                
                # Soil preparation reduces compaction
                field.soil_compaction = max(0, field.soil_compaction - 0.1)
        
        elif 35 <= action <= 39:
            # Market timing decisions
            crop_type = CropType(action - 35)
            # Sell stored crops at current market price
            # (Simplified - assumes some crops in storage)
            if self.cash_flow < 10000:  # Need money
                revenue = self.market.current_prices[crop_type] * 10  # Sell 10 tons
                self.cash_flow += revenue
                self.total_revenue += revenue
                reward += 100
        
        elif action == 40:
            # Emergency weather protection
            for field in self.field_sections:
                if self.weather.frost_risk > 0.5 or self.weather.excessive_rain_days > 3:
                    field.health = max(field.health - 0.05, 0.5)  # Reduce damage
                    reward += 50
            self.cash_flow -= 1000
        
        elif action == 41:
            # Sustainable farming mode
            self.biodiversity_index += 0.01
            self.soil_health_trend += 0.02
            self.carbon_sequestered += 1.0
            reward += 500
        
        elif action == 42:
            # Intensive production mode
            for field in self.field_sections:
                if field.crop_type:
                    field.yield_potential *= 1.1
                    field.soil_organic_matter -= 0.001
            reward += 100
        
        elif action == 43:
            # Crop rotation planning
            # Bonus for diverse crops
            crop_types = set(f.crop_type for f in self.field_sections if f.crop_type)
            if len(crop_types) >= 3:
                reward += 200
        
        elif action == 44:
            # Equipment maintenance
            for eq in self.equipment:
                if eq.maintenance_needed > 0.5:
                    eq.maintenance_needed = 0.0
                    eq.fuel_level = 1.0
                    self.cash_flow -= 500
                    reward += 30
        
        return reward
    
    def _harvest_crop(self, field: FieldSection) -> float:
        """Harvest a crop from a field and calculate reward."""
        if field.crop_type is None:
            return 0.0
        
        crop_data = CROP_DATA[field.crop_type]
        
        # Calculate yield
        base_yield = crop_data.yield_per_hectare * (field.size / 625.0)  # Convert to hectares
        actual_yield = base_yield * field.yield_potential * field.health
        
        # Calculate revenue
        market_price = self.market.current_prices[field.crop_type]
        revenue = actual_yield * market_price
        self.cash_flow += revenue
        self.total_revenue += revenue
        
        # Calculate reward based on yield
        yield_ratio = actual_yield / base_yield
        if yield_ratio > 1.2:
            reward = 5000  # Exceptional harvest
        elif yield_ratio > 0.9:
            reward = 2000  # Good harvest
        elif yield_ratio > 0.7:
            reward = 1000  # Average harvest
        else:
            reward = -1000  # Poor harvest
        
        # Market timing bonus
        avg_price = crop_data.market_base_price
        if market_price > avg_price * 1.2:
            reward += 800  # Good market timing
        elif market_price < avg_price * 0.8:
            reward -= 400  # Poor market timing
        
        # Reset field
        field.last_crop = field.crop_type
        field.crop_type = None
        field.growth_stage = CropStage.DORMANT
        field.days_since_planting = 0
        field.health = 1.0
        field.yield_potential = 1.0
        
        return reward
    
    def _calculate_environmental_rewards(self) -> float:
        """Calculate rewards based on environmental factors."""
        reward = 0.0
        
        # Soil health improvement
        avg_organic_matter = np.mean([f.soil_organic_matter for f in self.field_sections])
        if avg_organic_matter > 0.045:
            reward += 500  # Sustainable farming bonus
        elif avg_organic_matter < 0.02:
            reward -= 2000  # Severe soil degradation
        
        # Water conservation
        if self.reservoir_level > 0.5 and self.weather.drought_days > 10:
            reward += 300  # Water conservation achievement
        
        # Biodiversity
        if self.biodiversity_index > 0.7:
            reward += 200
        elif self.biodiversity_index < 0.3:
            reward -= 300
        
        # Weather damage
        for field in self.field_sections:
            if field.crop_type:
                if self.weather.frost_risk > 0.5 and field.growth_stage in [CropStage.GERMINATION, CropStage.REPRODUCTIVE]:
                    field.health -= 0.2
                    reward -= 300
                
                if self.weather.drought_days > 20 and field.soil_moisture < 0.2:
                    field.health -= 0.15
                    reward -= 300
                
                if self.weather.excessive_rain_days > 5:
                    field.disease_risk += 0.1
                    reward -= 200
        
        return reward
    
    def _update_sustainability_metrics(self):
        """Update sustainability tracking metrics."""
        # Carbon sequestration from crops
        for field in self.field_sections:
            if field.crop_type:
                crop_data = CROP_DATA[field.crop_type]
                self.carbon_sequestered += crop_data.carbon_sequestration * 0.01
        
        # Water conservation score
        water_used = 1.0 - self.reservoir_level
        if water_used < 0.3:
            self.water_conservation_score = min(1.0, self.water_conservation_score + 0.01)
        else:
            self.water_conservation_score = max(0.0, self.water_conservation_score - 0.01)
        
        # Soil health trend
        avg_compaction = np.mean([f.soil_compaction for f in self.field_sections])
        avg_organic = np.mean([f.soil_organic_matter for f in self.field_sections])
        self.soil_health_trend = (avg_organic * 10 - avg_compaction) / 2
        
        # Update profit margin
        if self.total_revenue > 0:
            self.profit_margin = (self.total_revenue - self.total_expenses) / self.total_revenue
        else:
            self.profit_margin = -1.0
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            "day": self.current_day,
            "season": ["spring", "summer", "fall", "winter"][self.current_season],
            "year": self.year,
            "cash_flow": self.cash_flow,
            "total_revenue": self.total_revenue,
            "total_expenses": self.total_expenses,
            "profit_margin": self.profit_margin,
            "carbon_sequestered": self.carbon_sequestered,
            "water_conservation_score": self.water_conservation_score,
            "biodiversity_index": self.biodiversity_index,
            "soil_health_trend": self.soil_health_trend,
            "field_status": [
                {
                    "name": f.name,
                    "crop": CROP_DATA[f.crop_type].name if f.crop_type else "None",
                    "stage": f.growth_stage.name,
                    "health": f.health,
                    "yield_potential": f.yield_potential
                }
                for f in self.field_sections
            ]
        }
    
    def _init_pygame(self):
        """Initialize pygame for enhanced rendering."""
        pygame.init()
        
        # Handle high DPI displays on Windows
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass  # Not on Windows or DPI awareness not available
        
        # Get display info to ensure window fits on screen
        display_info = pygame.display.Info()
        screen_width = display_info.current_w
        screen_height = display_info.current_h
        
        # Calculate window size (ensure it fits on screen with margins)
        window_width = min(1400, screen_width - 100)
        window_height = min(900, screen_height - 100)
        
        # Center the window on screen
        window_x = (screen_width - window_width) // 2
        window_y = (screen_height - window_height) // 2
        
        # Set window position and size
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Agricultural Farm Management System - Advanced Visualization")
        
        # Ensure window is visible and properly positioned on Windows
        pygame.display.flip()
        pygame.time.wait(100)  # Small delay to ensure window is properly positioned
        
        # Store window dimensions for scaling
        self.window_width = window_width
        self.window_height = window_height
        
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.title_font = pygame.font.Font(None, 36)
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 14)
        
        # Animation and history tracking
        self.animation_frame = 0
        self.weather_particles = []
        self.revenue_history = []
        self.yield_history = []
        self.sustainability_history = []
        self.max_history_length = 100
        
        # Visual enhancements
        self.gradient_cache = {}
        self.icon_cache = {}
        self.last_harvest_effects = []
        self.notification_queue = []
        
        # Initialize visual components
        self._init_visual_assets()
    
    def _init_visual_assets(self):
        """Initialize visual assets and create icons."""
        # Create equipment icons
        self.equipment_icons = {
            EquipmentType.TRACTOR: self._create_tractor_icon(),
            EquipmentType.HARVESTER: self._create_harvester_icon(),
            EquipmentType.IRRIGATION_SYSTEM: self._create_irrigation_icon(),
            EquipmentType.FERTILIZER_SPREADER: self._create_fertilizer_icon(),
            EquipmentType.PESTICIDE_SPRAYER: self._create_sprayer_icon(),
            EquipmentType.SEED_PLANTER: self._create_planter_icon(),
            EquipmentType.SOIL_TESTER: self._create_tester_icon(),
            EquipmentType.WEATHER_STATION: self._create_weather_station_icon()
        }
        
        # Create crop stage icons
        self.crop_stage_icons = {}
        
        # Weather effect surfaces
        self.rain_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        self.frost_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
    
    def _create_tractor_icon(self) -> pygame.Surface:
        """Create tractor icon."""
        icon = pygame.Surface((20, 20), pygame.SRCALPHA)
        # Body
        pygame.draw.rect(icon, (50, 150, 50), (4, 6, 12, 8))
        # Wheels
        pygame.draw.circle(icon, (30, 30, 30), (6, 14), 3)
        pygame.draw.circle(icon, (30, 30, 30), (14, 14), 3)
        # Cabin
        pygame.draw.rect(icon, (80, 180, 80), (8, 2, 4, 4))
        return icon
    
    def _create_harvester_icon(self) -> pygame.Surface:
        """Create harvester icon."""
        icon = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.rect(icon, (200, 150, 50), (2, 4, 16, 10))
        pygame.draw.rect(icon, (150, 100, 30), (2, 14, 16, 3))
        return icon
    
    def _create_irrigation_icon(self) -> pygame.Surface:
        """Create irrigation system icon."""
        icon = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(icon, (100, 150, 200), (10, 10), 8)
        for angle in range(0, 360, 45):
            x = 10 + int(6 * np.cos(np.radians(angle)))
            y = 10 + int(6 * np.sin(np.radians(angle)))
            pygame.draw.line(icon, (50, 100, 150), (10, 10), (x, y), 2)
        return icon
    
    def _create_fertilizer_icon(self) -> pygame.Surface:
        """Create fertilizer spreader icon."""
        icon = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.rect(icon, (139, 69, 19), (5, 5, 10, 10))
        pygame.draw.circle(icon, (100, 50, 20), (10, 10), 3)
        return icon
    
    def _create_sprayer_icon(self) -> pygame.Surface:
        """Create pesticide sprayer icon."""
        icon = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.rect(icon, (180, 50, 50), (8, 5, 4, 10))
        pygame.draw.polygon(icon, (150, 30, 30), [(5, 8), (15, 8), (10, 2)])
        return icon
    
    def _create_planter_icon(self) -> pygame.Surface:
        """Create seed planter icon."""
        icon = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.rect(icon, (100, 70, 40), (5, 8, 10, 6))
        for x in [7, 10, 13]:
            pygame.draw.circle(icon, (50, 150, 50), (x, 15), 2)
        return icon
    
    def _create_tester_icon(self) -> pygame.Surface:
        """Create soil tester icon."""
        icon = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.rect(icon, (100, 100, 100), (8, 3, 4, 14))
        pygame.draw.circle(icon, (200, 50, 50), (10, 5), 2)
        return icon
    
    def _create_weather_station_icon(self) -> pygame.Surface:
        """Create weather station icon."""
        icon = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.line(icon, (150, 150, 150), (10, 17), (10, 3), 2)
        pygame.draw.polygon(icon, (200, 200, 200), [(6, 5), (14, 5), (10, 8)])
        pygame.draw.circle(icon, (255, 200, 50), (10, 12), 2)
        return icon
    
    def render(self):
        """Enhanced render with sophisticated visualization."""
        if self.render_mode == "human":
            if self.screen is None:
                self._init_pygame()
            
            # Handle window resize events
            self._handle_window_events()
            
            # Update animation frame
            self.animation_frame += 1
            
            # Create gradient background
            self._draw_gradient_background()
            
            # Draw main components with enhanced visuals
            self._draw_enhanced_farm_view()      # Left side - enhanced farm view
            self._draw_weather_effects()         # Weather overlay
            self._draw_equipment_overlay()       # Equipment with animations
            self._draw_advanced_info_panels()    # Right side - advanced panels
            self._draw_real_time_graphs()        # Bottom - real-time graphs
            self._draw_notification_bar()        # Top - notifications
            self._draw_seasonal_calendar()       # Calendar view
            self._draw_field_detail_popup()      # Hover information
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
    
    def _handle_window_events(self):
        """Handle window resize and other events."""
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                # Update window dimensions
                self.window_width = event.w
                self.window_height = event.h
                
                # Recreate weather effect surfaces with new dimensions
                self.rain_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                self.frost_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                
                # Clear the screen
                self.screen.fill((0, 0, 0))
            elif event.type == pygame.QUIT:
                pygame.quit()
                return
    
    def _draw_gradient_background(self):
        """Draw gradient background based on time of day."""
        # Sky gradient based on season and weather
        if self.current_season == Season.SPRING:
            top_color = (135, 206, 235)  # Sky blue
            bottom_color = (255, 248, 220)  # Cream
        elif self.current_season == Season.SUMMER:
            top_color = (87, 160, 211)  # Deeper blue
            bottom_color = (255, 239, 213)  # Papaya
        elif self.current_season == Season.FALL:
            top_color = (255, 140, 90)  # Orange sky
            bottom_color = (255, 218, 185)  # Peach
        else:  # Winter
            top_color = (176, 196, 222)  # Light steel blue
            bottom_color = (245, 245, 245)  # White smoke
        
        # Adjust for weather
        if self.weather.rainfall > 10:
            top_color = tuple(int(c * 0.7) for c in top_color)
            bottom_color = tuple(int(c * 0.8) for c in bottom_color)
        
        # Draw gradient using dynamic window dimensions
        for y in range(self.window_height):
            ratio = y / self.window_height
            r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
            g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
            b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.window_width, y))
    
    def _draw_enhanced_farm_view(self):
        """Draw enhanced farm view with detailed field visualization."""
        # Main farm area background
        farm_area = pygame.Rect(20, 80, 600, 600)
        pygame.draw.rect(self.screen, (34, 139, 34), farm_area)
        pygame.draw.rect(self.screen, (20, 80, 20), farm_area, 3)
        
        # Title
        title_text = self.title_font.render("Farm Overview", True, (255, 255, 255))
        self.screen.blit(title_text, (20, 40))
        
        # Draw field sections with enhanced visuals
        field_rects = [
            {"field": 0, "rect": pygame.Rect(30, 90, 285, 285)},
            {"field": 1, "rect": pygame.Rect(325, 90, 285, 285)},
            {"field": 2, "rect": pygame.Rect(30, 385, 285, 285)},
            {"field": 3, "rect": pygame.Rect(325, 385, 285, 285)}
        ]
        
        for field_info in field_rects:
            field = self.field_sections[field_info["field"]]
            rect = field_info["rect"]
            
            # Draw field with texture and patterns
            self._draw_field_section(field, rect)
            
            # Draw field information overlay
            self._draw_field_info_overlay(field, rect)
    
    def _draw_field_section(self, field: FieldSection, rect: pygame.Rect):
        """Draw a single field section with rich details."""
        # Base field color with soil texture
        if field.crop_type is None:
            # Empty field - show soil quality gradient
            soil_color = self._get_soil_color(field)
            pygame.draw.rect(self.screen, soil_color, rect)
            
            # Add soil texture pattern
            for y in range(rect.top, rect.bottom, 10):
                for x in range(rect.left, rect.right, 20):
                    if (x + y) % 30 == 0:
                        pygame.draw.line(self.screen, 
                                       tuple(int(c * 0.9) for c in soil_color),
                                       (x, y), (x + 15, y + 2), 1)
        else:
            # Planted field - show crop with growth visualization
            base_color = self._get_enhanced_crop_color(field)
            pygame.draw.rect(self.screen, base_color, rect)
            
            # Draw crop rows
            self._draw_crop_rows(field, rect)
        
        # Field border with health indicator
        border_color = self._get_field_border_color(field)
        pygame.draw.rect(self.screen, border_color, rect, 3)
        
        # Moisture indicator (blue gradient on edges)
        if field.soil_moisture > 0.7:
            moisture_alpha = int((field.soil_moisture - 0.7) * 500)
            moisture_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.rect(moisture_surf, (100, 150, 200, min(moisture_alpha, 50)), 
                           (0, 0, rect.width, rect.height))
            self.screen.blit(moisture_surf, rect.topleft)
    
    def _draw_crop_rows(self, field: FieldSection, rect: pygame.Rect):
        """Draw crop rows with growth stage visualization."""
        if field.crop_type is None:
            return
        
        rows = 8
        cols = 8
        cell_w = rect.width // cols
        cell_h = rect.height // rows
        
        for row in range(rows):
            for col in range(cols):
                x = rect.left + col * cell_w + cell_w // 2
                y = rect.top + row * cell_h + cell_h // 2
                
                # Vary plant size based on growth stage and health
                size = int(3 + field.growth_stage.value * 2 * field.health)
                
                if field.crop_type == CropType.WHEAT:
                    # Draw wheat stalks
                    color = (255, 215, 0) if field.growth_stage >= CropStage.MATURATION else (50, 205, 50)
                    for offset in [-2, 0, 2]:
                        pygame.draw.line(self.screen, color, 
                                       (x + offset, y + size), (x + offset, y - size), 1)
                    if field.growth_stage >= CropStage.REPRODUCTIVE:
                        pygame.draw.circle(self.screen, (255, 223, 0), (x, y - size), 2)
                
                elif field.crop_type == CropType.CORN:
                    # Draw corn stalks
                    stalk_color = (34, 139, 34) if field.growth_stage < CropStage.MATURATION else (184, 134, 11)
                    pygame.draw.line(self.screen, stalk_color, (x, y + size), (x, y - size), 2)
                    if field.growth_stage >= CropStage.REPRODUCTIVE:
                        # Draw corn ears
                        pygame.draw.ellipse(self.screen, (255, 215, 0), 
                                          (x - 3, y - 2, 6, 8))
                
                elif field.crop_type == CropType.TOMATOES:
                    # Draw tomato plants
                    pygame.draw.circle(self.screen, (0, 128, 0), (x, y), size)
                    if field.growth_stage >= CropStage.REPRODUCTIVE:
                        # Draw tomatoes
                        pygame.draw.circle(self.screen, (255, 0, 0), (x - 2, y), 2)
                        pygame.draw.circle(self.screen, (255, 99, 71), (x + 2, y - 1), 2)
                
                elif field.crop_type == CropType.APPLES:
                    # Draw apple trees
                    pygame.draw.rect(self.screen, (101, 67, 33), (x - 2, y, 4, size))
                    pygame.draw.circle(self.screen, (34, 139, 34), (x, y - size), size + 2)
                    if field.growth_stage >= CropStage.MATURATION:
                        # Draw apples
                        pygame.draw.circle(self.screen, (255, 0, 0), (x - 3, y - size), 2)
                        pygame.draw.circle(self.screen, (178, 34, 34), (x + 2, y - size + 2), 2)
                
                elif field.crop_type == CropType.SOYBEANS:
                    # Draw soybean plants
                    pygame.draw.circle(self.screen, (85, 107, 47), (x, y), size - 1)
                    if field.growth_stage >= CropStage.REPRODUCTIVE:
                        # Draw pods
                        pygame.draw.ellipse(self.screen, (144, 238, 144), 
                                          (x - 2, y - 1, 4, 6))
    
    def _get_soil_color(self, field: FieldSection) -> Tuple[int, int, int]:
        """Get soil color based on quality."""
        # Base brown color
        r, g, b = 101, 67, 33
        
        # Modify based on organic matter
        organic_factor = field.soil_organic_matter * 10
        r = int(r * (1 - organic_factor * 0.3))
        g = int(g * (1 - organic_factor * 0.2))
        b = int(b * (1 - organic_factor * 0.1))
        
        # Adjust for moisture
        if field.soil_moisture > 0.5:
            moisture_factor = field.soil_moisture - 0.5
            r = int(r * (1 - moisture_factor * 0.2))
            g = int(g * (1 - moisture_factor * 0.15))
            b = int(b * (1 - moisture_factor * 0.1))
        
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    def _get_enhanced_crop_color(self, field: FieldSection) -> Tuple[int, int, int]:
        """Get enhanced crop background color."""
        if field.crop_type == CropType.WHEAT:
            colors = [(50, 100, 50), (100, 150, 50), (150, 200, 50), (200, 200, 50), (220, 200, 100)]
        elif field.crop_type == CropType.CORN:
            colors = [(50, 120, 50), (75, 150, 75), (100, 180, 100), (150, 200, 100), (200, 220, 100)]
        elif field.crop_type == CropType.TOMATOES:
            colors = [(60, 100, 60), (80, 120, 80), (100, 140, 100), (120, 140, 100), (140, 120, 100)]
        elif field.crop_type == CropType.APPLES:
            colors = [(40, 80, 40), (60, 100, 60), (80, 120, 80), (100, 140, 100), (120, 160, 120)]
        else:  # SOYBEANS
            colors = [(60, 90, 60), (80, 110, 80), (100, 130, 100), (120, 150, 120), (140, 170, 140)]
        
        stage_index = min(field.growth_stage.value, len(colors) - 1)
        base_color = colors[stage_index]
        
        # Modify by health and ensure valid color values
        r = max(0, min(255, int(base_color[0] * field.health)))
        g = max(0, min(255, int(base_color[1] * field.health)))
        b = max(0, min(255, int(base_color[2] * field.health)))
        
        return (r, g, b)
    
    def _get_field_border_color(self, field: FieldSection) -> Tuple[int, int, int]:
        """Get field border color based on health and issues."""
        if field.pest_level > 0.5:
            return (255, 0, 0)  # Red for pest issues
        elif field.disease_risk > 0.5:
            return (255, 165, 0)  # Orange for disease risk
        elif field.soil_moisture < 0.3:
            return (139, 69, 19)  # Brown for dry
        elif field.health < 0.5:
            return (255, 255, 0)  # Yellow for poor health
        else:
            return (0, 255, 0)  # Green for healthy
    
    def _draw_field_info_overlay(self, field: FieldSection, rect: pygame.Rect):
        """Draw information overlay on field."""
        # Semi-transparent background for text
        info_surf = pygame.Surface((rect.width, 60), pygame.SRCALPHA)
        pygame.draw.rect(info_surf, (0, 0, 0, 180), (0, 0, rect.width, 60))
        self.screen.blit(info_surf, (rect.left, rect.top))
        
        # Field name
        name_text = self.font.render(field.name, True, (255, 255, 255))
        self.screen.blit(name_text, (rect.left + 5, rect.top + 5))
        
        # Crop info
        if field.crop_type is not None:
            crop_name = CROP_DATA[field.crop_type].name
            stage_name = field.growth_stage.name.replace("_", " ").title()
            crop_text = self.small_font.render(f"{crop_name} - {stage_name}", True, (200, 200, 200))
            self.screen.blit(crop_text, (rect.left + 5, rect.top + 25))
            
            # Days to harvest
            if field.days_to_harvest > 0:
                days_text = self.tiny_font.render(f"Harvest in {field.days_to_harvest} days", True, (255, 215, 0))
                self.screen.blit(days_text, (rect.left + 5, rect.top + 42))
        else:
            empty_text = self.small_font.render("Empty Field", True, (150, 150, 150))
            self.screen.blit(empty_text, (rect.left + 5, rect.top + 25))
        
        # Health and yield bars
        self._draw_mini_bar(rect.right - 85, rect.top + 8, 80, 8, 
                           field.health, (0, 255, 0), "Health")
        self._draw_mini_bar(rect.right - 85, rect.top + 20, 80, 8,
                           field.yield_potential, (255, 215, 0), "Yield")
        
        # Alert icons
        if field.pest_level > 0.3:
            self._draw_alert_icon(rect.right - 25, rect.bottom - 25, "pest")
        if field.disease_risk > 0.3:
            self._draw_alert_icon(rect.right - 50, rect.bottom - 25, "disease")
        if field.soil_moisture < 0.3:
            self._draw_alert_icon(rect.right - 75, rect.bottom - 25, "water")
    
    def _draw_mini_bar(self, x: int, y: int, width: int, height: int, 
                      value: float, color: Tuple[int, int, int], label: str):
        """Draw a small progress bar."""
        # Background
        pygame.draw.rect(self.screen, (50, 50, 50), (x, y, width, height))
        # Fill
        fill_width = int(width * value)
        pygame.draw.rect(self.screen, color, (x, y, fill_width, height))
        # Border
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height), 1)
    
    def _draw_alert_icon(self, x: int, y: int, alert_type: str):
        """Draw alert icons for field issues."""
        if alert_type == "pest":
            color = (255, 0, 0)
            symbol = "!"
        elif alert_type == "disease":
            color = (255, 165, 0)
            symbol = "D"
        elif alert_type == "water":
            color = (0, 150, 255)
            symbol = "W"
        else:
            return
        
        pygame.draw.circle(self.screen, color, (x, y), 10)
        text = self.tiny_font.render(symbol, True, (255, 255, 255))
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)
    
    def _draw_weather_effects(self):
        """Draw animated weather effects."""
        if self.weather.rainfall > 0:
            # Rain animation - focus on farm area (left side where fields are)
            farm_area_x = 20
            farm_area_y = 80
            farm_area_width = 600
            farm_area_height = 600
            
            # Draw heavy rain over the farm fields (more visible)
            rain_intensity = min(self.weather.rainfall / 10.0, 1.0)  # Normalize rainfall
            num_drops = int(rain_intensity * 15)  # More drops for heavier rain
            
            for _ in range(num_drops):
                x = np.random.randint(farm_area_x, farm_area_x + farm_area_width)
                y = np.random.randint(farm_area_y, farm_area_y + farm_area_height)
                length = np.random.randint(10, 25)
                thickness = np.random.randint(1, 3)
                
                # Rain drops with better visibility over fields
                rain_color = (80, 140, 200)  # More visible blue
                pygame.draw.line(self.screen, rain_color, 
                               (x, y), (x - 2, y + length), thickness)
            
            # Add some rain over the entire window for atmosphere
            for _ in range(int(rain_intensity * 8)):
                x = np.random.randint(0, self.window_width)
                y = np.random.randint(0, self.window_height)
                length = np.random.randint(5, 15)
                pygame.draw.line(self.screen, (100, 150, 200, 100), 
                               (x, y), (x - 1, y + length), 1)
            
            # Add rain splash effects on the ground
            if rain_intensity > 0.5:
                for _ in range(int(rain_intensity * 5)):
                    x = np.random.randint(farm_area_x, farm_area_x + farm_area_width)
                    y = farm_area_y + farm_area_height + np.random.randint(0, 20)
                    pygame.draw.circle(self.screen, (120, 160, 220), (x, y), 1)
            
            # Add wind effect to rain (rain falls at an angle based on wind)
            wind_angle = min(self.weather.wind_speed / 30.0, 1.0)  # Normalize wind
            for _ in range(int(rain_intensity * 10)):
                x = np.random.randint(farm_area_x, farm_area_x + farm_area_width)
                y = np.random.randint(farm_area_y, farm_area_y + farm_area_height)
                length = np.random.randint(8, 18)
                
                # Rain angle based on wind speed
                wind_offset = int(wind_angle * 8)
                end_x = x - wind_offset
                end_y = y + length
                
                pygame.draw.line(self.screen, (90, 150, 210), 
                               (x, y), (end_x, end_y), 1)
        
        if self.weather.frost_risk > 0.3:
            # Frost overlay - focus on farm area
            frost_surf = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
            frost_alpha = int(self.weather.frost_risk * 120)
            frost_surf.fill((200, 220, 255, frost_alpha))
            self.screen.blit(frost_surf, (0, 0))
        
        # Add subtle darkening effect over farm area when raining
        if self.weather.rainfall > 5:
            farm_area_x = 20
            farm_area_y = 80
            farm_area_width = 600
            farm_area_height = 600
            
            # Create a semi-transparent dark overlay for the farm area
            rain_overlay = pygame.Surface((farm_area_width, farm_area_height), pygame.SRCALPHA)
            overlay_alpha = min(int(self.weather.rainfall * 3), 60)  # Max 60 alpha
            rain_overlay.fill((0, 0, 0, overlay_alpha))
            self.screen.blit(rain_overlay, (farm_area_x, farm_area_y))
            
            # Add rain indicator text over the farm area
            if self.weather.rainfall > 10:
                rain_text = self.small_font.render(f"🌧️ Heavy Rain: {self.weather.rainfall:.1f}mm", True, (255, 255, 255))
                text_rect = rain_text.get_rect(center=(farm_area_x + farm_area_width//2, farm_area_y + 30))
                # Add background for better visibility
                pygame.draw.rect(self.screen, (0, 0, 0, 100), text_rect.inflate(20, 10))
                self.screen.blit(rain_text, text_rect)
    
    def _draw_equipment_overlay(self):
        """Draw equipment with animations."""
        for eq in self.equipment:
            if eq.is_operating:
                # Calculate screen position
                x = 30 + eq.location[0] * 23
                y = 90 + eq.location[1] * 23
                
                # Draw equipment icon
                if eq.type in self.equipment_icons:
                    icon = self.equipment_icons[eq.type]
                    # Add pulsing effect for operating equipment
                    scale = 1.0 + 0.2 * np.sin(self.animation_frame * 0.1)
                    scaled_icon = pygame.transform.scale(icon, 
                                                        (int(20 * scale), int(20 * scale)))
                    self.screen.blit(scaled_icon, (x - 10, y - 10))
                
                # Draw fuel indicator
                fuel_bar_width = 20
                fuel_bar_height = 3
                pygame.draw.rect(self.screen, (50, 50, 50), 
                               (x - 10, y + 12, fuel_bar_width, fuel_bar_height))
                pygame.draw.rect(self.screen, (0, 255, 0) if eq.fuel_level > 0.3 else (255, 0, 0),
                               (x - 10, y + 12, int(fuel_bar_width * eq.fuel_level), fuel_bar_height))
    
    def _draw_advanced_info_panels(self):
        """Draw advanced information panels with enhanced visuals."""
        # Weather panel with visual indicators
        self._draw_weather_panel(640, 80)
        
        # Financial dashboard with graphs
        self._draw_financial_dashboard(920, 80)
        
        # Market panel with trend indicators
        self._draw_market_panel(640, 280)
        
        # Resource meters
        self._draw_resource_meters(920, 280)
        
        # Sustainability dashboard
        self._draw_sustainability_panel(1200, 80)
        
        # Equipment status panel
        self._draw_equipment_status(1200, 280)
    
    def _draw_weather_panel(self, x: int, y: int):
        """Draw enhanced weather panel."""
        panel_width, panel_height = 260, 180
        panel = pygame.Rect(x, y, panel_width, panel_height)
        
        # Gradient background
        self._draw_gradient_panel(panel, (30, 60, 90), (20, 40, 60))
        
        # Title
        title = self.font.render("Weather Station", True, (255, 255, 255))
        self.screen.blit(title, (x + 10, y + 10))
        
        # Season indicator with icon
        season_names = ["🌸 Spring", "☀️ Summer", "🍂 Fall", "❄️ Winter"]
        season_text = self.small_font.render(season_names[self.current_season], True, (255, 255, 255))
        self.screen.blit(season_text, (x + 10, y + 40))
        
        # Temperature gauge
        self._draw_gauge(x + 10, y + 65, 60, self.weather.temperature, -10, 40, "°C", (255, 100, 100))
        
        # Humidity gauge
        self._draw_gauge(x + 80, y + 65, 60, self.weather.humidity * 100, 0, 100, "%", (100, 150, 255))
        
        # Rainfall bar
        self._draw_bar_with_icon(x + 150, y + 65, 100, 20, self.weather.rainfall / 30,
                                 (100, 150, 200), "Rain")
        
        # Wind indicator
        wind_text = self.tiny_font.render(f"Wind: {self.weather.wind_speed:.0f} km/h", True, (200, 200, 200))
        self.screen.blit(wind_text, (x + 10, y + 130))
        
        # Forecast preview
        forecast_text = self.tiny_font.render("7-Day Forecast:", True, (200, 200, 200))
        self.screen.blit(forecast_text, (x + 10, y + 145))
        
        # Mini forecast icons
        for i in range(7):
            forecast_x = x + 10 + i * 35
            forecast_y = y + 160
            # Simulate forecast
            if np.random.random() < 0.3:
                pygame.draw.circle(self.screen, (255, 200, 0), (forecast_x + 10, forecast_y), 5)  # Sun
            else:
                pygame.draw.circle(self.screen, (150, 150, 150), (forecast_x + 10, forecast_y), 5)  # Cloud
    
    def _draw_financial_dashboard(self, x: int, y: int):
        """Draw financial dashboard with graphs."""
        panel_width, panel_height = 260, 180
        panel = pygame.Rect(x, y, panel_width, panel_height)
        
        # Gradient background
        self._draw_gradient_panel(panel, (20, 60, 20), (10, 40, 10))
        
        # Title
        title = self.font.render("Financial Status", True, (255, 255, 255))
        self.screen.blit(title, (x + 10, y + 10))
        
        # Cash flow with color coding
        cash_color = (0, 255, 0) if self.cash_flow > 0 else (255, 0, 0)
        cash_text = self.small_font.render(f"Cash: ${self.cash_flow:,.0f}", True, cash_color)
        self.screen.blit(cash_text, (x + 10, y + 40))
        
        # Revenue and expenses
        rev_text = self.tiny_font.render(f"Revenue: ${self.total_revenue:,.0f}", True, (100, 255, 100))
        self.screen.blit(rev_text, (x + 10, y + 60))
        exp_text = self.tiny_font.render(f"Expenses: ${self.total_expenses:,.0f}", True, (255, 100, 100))
        self.screen.blit(exp_text, (x + 10, y + 75))
        
        # Profit margin gauge
        margin_color = (0, 255, 0) if self.profit_margin > 0 else (255, 0, 0)
        self._draw_horizontal_bar(x + 10, y + 95, 240, 15, 
                                 (self.profit_margin + 1) / 2, margin_color, "Profit Margin")
        
        # Mini revenue graph
        self._draw_mini_graph(x + 10, y + 120, 240, 50, self.revenue_history, (100, 255, 100))
    
    def _draw_market_panel(self, x: int, y: int):
        """Draw market prices with trend indicators."""
        panel_width, panel_height = 260, 180
        panel = pygame.Rect(x, y, panel_width, panel_height)
        
        # Gradient background
        self._draw_gradient_panel(panel, (60, 40, 20), (40, 20, 10))
        
        # Title
        title = self.font.render("Market Prices", True, (255, 255, 255))
        self.screen.blit(title, (x + 10, y + 10))
        
        # Crop prices with trend arrows
        y_offset = 40
        for crop_type in CropType:
            crop_data = CROP_DATA[crop_type]
            current_price = self.market.current_prices[crop_type]
            predicted_price = self.market.predicted_prices[crop_type]
            
            # Crop name
            name_text = self.tiny_font.render(crop_data.name[:8], True, (200, 200, 200))
            self.screen.blit(name_text, (x + 10, y + y_offset))
            
            # Price
            price_text = self.tiny_font.render(f"${current_price:.0f}", True, (255, 255, 255))
            self.screen.blit(price_text, (x + 80, y + y_offset))
            
            # Trend arrow
            if predicted_price > current_price * 1.05:
                arrow_color = (0, 255, 0)
                arrow = "↑"
            elif predicted_price < current_price * 0.95:
                arrow_color = (255, 0, 0)
                arrow = "↓"
            else:
                arrow_color = (255, 255, 0)
                arrow = "→"
            
            arrow_text = self.small_font.render(arrow, True, arrow_color)
            self.screen.blit(arrow_text, (x + 130, y + y_offset - 2))
            
            # Price bar
            price_ratio = current_price / (crop_data.market_base_price * 2)
            self._draw_mini_bar(x + 150, y + y_offset, 100, 10, 
                              min(1.0, price_ratio), (255, 215, 0), "")
            
            y_offset += 25
    
    def _draw_resource_meters(self, x: int, y: int):
        """Draw resource meters with visual indicators."""
        panel_width, panel_height = 260, 180
        panel = pygame.Rect(x, y, panel_width, panel_height)
        
        # Gradient background
        self._draw_gradient_panel(panel, (20, 40, 60), (10, 20, 40))
        
        # Title
        title = self.font.render("Resources", True, (255, 255, 255))
        self.screen.blit(title, (x + 10, y + 10))
        
        # Water resources with animated fill
        self._draw_liquid_tank(x + 10, y + 40, 50, 80, self.reservoir_level, 
                              (100, 150, 200), "Reservoir")
        self._draw_liquid_tank(x + 70, y + 40, 50, 80, self.groundwater_level,
                              (80, 120, 160), "Groundwater")
        
        # Water quality meter
        quality_color = (0, 255, 0) if self.water_quality > 0.7 else (255, 255, 0) if self.water_quality > 0.4 else (255, 0, 0)
        self._draw_circular_meter(x + 140, y + 60, 30, self.water_quality, quality_color, "Quality")
        
        # Irrigation capacity
        self._draw_horizontal_bar(x + 10, y + 130, 240, 15, self.irrigation_capacity,
                                 (100, 200, 255), "Irrigation Capacity")
        
        # Water cost indicator
        cost_text = self.tiny_font.render(f"Water Cost: ${self.water_costs:.0f}/unit", True, (200, 200, 200))
        self.screen.blit(cost_text, (x + 10, y + 155))
    
    def _draw_sustainability_panel(self, x: int, y: int):
        """Draw sustainability metrics with visual indicators."""
        panel_width, panel_height = 180, 180
        panel = pygame.Rect(x, y, panel_width, panel_height)
        
        # Gradient background
        self._draw_gradient_panel(panel, (20, 50, 20), (10, 30, 10))
        
        # Title
        title = self.small_font.render("Sustainability", True, (255, 255, 255))
        self.screen.blit(title, (x + 10, y + 10))
        
        # Carbon meter
        self._draw_eco_meter(x + 10, y + 35, "CO₂", self.carbon_sequestered / 100, (100, 200, 100))
        
        # Water conservation
        self._draw_eco_meter(x + 50, y + 35, "H₂O", self.water_conservation_score, (100, 150, 200))
        
        # Biodiversity
        self._draw_eco_meter(x + 90, y + 35, "BIO", self.biodiversity_index, (200, 150, 100))
        
        # Soil health
        self._draw_eco_meter(x + 130, y + 35, "SOIL", (self.soil_health_trend + 1) / 2, (139, 69, 19))
        
        # Overall sustainability score
        overall_score = (self.carbon_sequestered / 100 + self.water_conservation_score + 
                        self.biodiversity_index + (self.soil_health_trend + 1) / 2) / 4
        self._draw_circular_meter(x + 90, y + 120, 40, overall_score, (0, 255, 0), "Overall")
    
    def _draw_equipment_status(self, x: int, y: int):
        """Draw equipment status panel."""
        panel_width, panel_height = 180, 180
        panel = pygame.Rect(x, y, panel_width, panel_height)
        
        # Gradient background
        self._draw_gradient_panel(panel, (40, 40, 40), (20, 20, 20))
        
        # Title
        title = self.small_font.render("Equipment", True, (255, 255, 255))
        self.screen.blit(title, (x + 10, y + 10))
        
        # Equipment list
        y_offset = 35
        for i, eq in enumerate(self.equipment[:6]):  # Show first 6
            # Icon
            if eq.type in self.equipment_icons:
                icon = pygame.transform.scale(self.equipment_icons[eq.type], (15, 15))
                self.screen.blit(icon, (x + 10, y + y_offset))
            
            # Status indicator
            status_color = (0, 255, 0) if eq.is_operating else (100, 100, 100)
            pygame.draw.circle(self.screen, status_color, (x + 35, y + y_offset + 7), 3)
            
            # Fuel bar
            self._draw_mini_bar(x + 50, y + y_offset + 2, 40, 8, eq.fuel_level,
                              (255, 165, 0), "")
            
            # Maintenance indicator
            if eq.maintenance_needed > 0.7:
                pygame.draw.circle(self.screen, (255, 0, 0), (x + 100, y + y_offset + 7), 3)
            
            y_offset += 22
    
    def _draw_real_time_graphs(self):
        """Draw real-time performance graphs at the bottom."""
        graph_area = pygame.Rect(20, 700, 1360, 180)
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), graph_area)
        pygame.draw.rect(self.screen, (60, 60, 60), graph_area, 2)
        
        # Title
        title = self.font.render("Performance Analytics", True, (255, 255, 255))
        self.screen.blit(title, (30, 705))
        
        # Update history arrays
        if len(self.revenue_history) >= self.max_history_length:
            self.revenue_history.pop(0)
        self.revenue_history.append(self.total_revenue)
        
        if len(self.yield_history) >= self.max_history_length:
            self.yield_history.pop(0)
        avg_yield = np.mean([f.yield_potential for f in self.field_sections if f.crop_type])
        self.yield_history.append(avg_yield if not np.isnan(avg_yield) else 0)
        
        if len(self.sustainability_history) >= self.max_history_length:
            self.sustainability_history.pop(0)
        sustainability_score = (self.carbon_sequestered / 100 + self.water_conservation_score + 
                               self.biodiversity_index + (self.soil_health_trend + 1) / 2) / 4
        self.sustainability_history.append(sustainability_score)
        
        # Draw three graphs
        self._draw_graph(40, 740, 420, 120, self.revenue_history, 
                        (100, 255, 100), "Revenue Trend", "$")
        self._draw_graph(490, 740, 420, 120, self.yield_history,
                        (255, 215, 0), "Yield Performance", "%")
        self._draw_graph(940, 740, 420, 120, self.sustainability_history,
                        (100, 200, 255), "Sustainability Score", "")
    
    def _draw_notification_bar(self):
        """Draw notification bar at the top."""
        # Check for notifications
        notifications = []
        
        for field in self.field_sections:
            if field.growth_stage == CropStage.HARVEST_READY:
                notifications.append(f"🌾 {field.name} ready for harvest!")
            if field.pest_level > 0.5:
                notifications.append(f"⚠️ Pest alert in {field.name}!")
            if field.soil_moisture < 0.2:
                notifications.append(f"💧 {field.name} needs water!")
        
        if self.cash_flow < 1000:
            notifications.append("💰 Low cash warning!")
        
        if notifications:
            # Draw notification area
            notif_area = pygame.Rect(20, 10, 1360, 25)
            pygame.draw.rect(self.screen, (80, 40, 40), notif_area)
            pygame.draw.rect(self.screen, (120, 60, 60), notif_area, 2)
            
            # Display most recent notification
            if notifications:
                notif_text = self.small_font.render(notifications[0], True, (255, 255, 255))
                self.screen.blit(notif_text, (30, 15))
    
    def _draw_seasonal_calendar(self):
        """Draw seasonal calendar view."""
        calendar_x, calendar_y = 640, 480
        calendar_width, calendar_height = 540, 180
        
        # Background
        cal_rect = pygame.Rect(calendar_x, calendar_y, calendar_width, calendar_height)
        self._draw_gradient_panel(cal_rect, (40, 30, 20), (20, 15, 10))
        
        # Title
        title = self.font.render("Seasonal Calendar", True, (255, 255, 255))
        self.screen.blit(title, (calendar_x + 10, calendar_y + 10))
        
        # Draw months
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_width = calendar_width // 12
        
        current_month = (self.current_day // 30) % 12
        
        for i, month in enumerate(months):
            x = calendar_x + i * month_width
            y = calendar_y + 40
            
            # Highlight current month
            if i == current_month:
                pygame.draw.rect(self.screen, (100, 100, 50), (x, y, month_width - 2, 30))
            
            # Month name
            month_text = self.tiny_font.render(month, True, (200, 200, 200))
            self.screen.blit(month_text, (x + 5, y + 5))
            
            # Season color bar
            if i < 3 or i == 11:  # Winter
                color = (200, 200, 255)
            elif i < 6:  # Spring
                color = (100, 255, 100)
            elif i < 9:  # Summer
                color = (255, 200, 100)
            else:  # Fall
                color = (255, 150, 50)
            
            pygame.draw.rect(self.screen, color, (x + 2, y + 25, month_width - 4, 3))
        
        # Draw planting schedule
        y_offset = calendar_y + 80
        for field in self.field_sections:
            # Field name
            field_text = self.tiny_font.render(field.name, True, (200, 200, 200))
            self.screen.blit(field_text, (calendar_x + 10, y_offset))
            
            # Crop timeline
            if field.crop_type:
                crop_data = CROP_DATA[field.crop_type]
                start_month = (self.current_day - field.days_since_planting) // 30
                duration_months = crop_data.growth_cycle_days // 30
                
                for m in range(duration_months):
                    month_idx = (start_month + m) % 12
                    x = calendar_x + month_idx * month_width
                    
                    # Growth stage color
                    progress = m / max(duration_months, 1)
                    color = tuple(int(50 + progress * 150) for _ in range(3))
                    pygame.draw.rect(self.screen, color, (x + 2, y_offset, month_width - 4, 15))
            
            y_offset += 20
    
    def _draw_field_detail_popup(self):
        """Draw detailed field information popup (placeholder for hover functionality)."""
        # This would normally be triggered by mouse hover
        # For demonstration, show details for the first field with a crop
        for field in self.field_sections:
            if field.crop_type and field.growth_stage != CropStage.DORMANT:
                popup_x, popup_y = 200, 300
                popup_width, popup_height = 300, 200
                
                # Semi-transparent background
                popup_surf = pygame.Surface((popup_width, popup_height), pygame.SRCALPHA)
                pygame.draw.rect(popup_surf, (0, 0, 0, 220), (0, 0, popup_width, popup_height))
                pygame.draw.rect(popup_surf, (255, 255, 255, 100), (0, 0, popup_width, popup_height), 2)
                self.screen.blit(popup_surf, (popup_x, popup_y))
                
                # Field details
                y_offset = 10
                details = [
                    f"Field: {field.name}",
                    f"Crop: {CROP_DATA[field.crop_type].name}",
                    f"Stage: {field.growth_stage.name}",
                    f"Health: {field.health:.1%}",
                    f"Yield Potential: {field.yield_potential:.1%}",
                    f"Days to Harvest: {field.days_to_harvest}",
                    f"Soil Moisture: {field.soil_moisture:.1%}",
                    f"Soil pH: {field.soil_ph:.1f}",
                    f"NPK: {field.soil_nitrogen:.0f}/{field.soil_phosphorus:.0f}/{field.soil_potassium:.0f}",
                    f"Pest Level: {field.pest_level:.1%}",
                    f"Disease Risk: {field.disease_risk:.1%}"
                ]
                
                for detail in details:
                    text = self.tiny_font.render(detail, True, (255, 255, 255))
                    self.screen.blit(text, (popup_x + 10, popup_y + y_offset))
                    y_offset += 17
                
                break  # Only show one popup for demonstration
    
    # Helper drawing methods
    def _draw_gradient_panel(self, rect: pygame.Rect, color1: Tuple[int, int, int], color2: Tuple[int, int, int]):
        """Draw a panel with gradient background."""
        for y in range(rect.height):
            ratio = y / rect.height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), 
                           (rect.left, rect.top + y), 
                           (rect.right, rect.top + y))
        pygame.draw.rect(self.screen, (100, 100, 100), rect, 2)
    
    def _draw_gauge(self, x: int, y: int, size: int, value: float, min_val: float, 
                   max_val: float, unit: str, color: Tuple[int, int, int]):
        """Draw a circular gauge."""
        # Background circle
        pygame.draw.circle(self.screen, (50, 50, 50), (x + size // 2, y + size // 2), size // 2)
        
        # Value arc
        angle = -90 + (value - min_val) / (max_val - min_val) * 270
        for a in range(-90, int(angle), 2):
            arc_x = x + size // 2 + int((size // 2 - 3) * np.cos(np.radians(a)))
            arc_y = y + size // 2 + int((size // 2 - 3) * np.sin(np.radians(a)))
            pygame.draw.circle(self.screen, color, (arc_x, arc_y), 2)
        
        # Center text
        value_text = self.tiny_font.render(f"{value:.0f}{unit}", True, (255, 255, 255))
        text_rect = value_text.get_rect(center=(x + size // 2, y + size // 2))
        self.screen.blit(value_text, text_rect)
    
    def _draw_bar_with_icon(self, x: int, y: int, width: int, height: int, 
                           value: float, color: Tuple[int, int, int], label: str):
        """Draw a bar with an icon."""
        # Background
        pygame.draw.rect(self.screen, (50, 50, 50), (x, y, width, height))
        # Fill
        fill_width = int(width * value)
        pygame.draw.rect(self.screen, color, (x, y, fill_width, height))
        # Border
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height), 1)
        # Label
        if label:
            label_text = self.tiny_font.render(label, True, (200, 200, 200))
            self.screen.blit(label_text, (x + 2, y + 2))
    
    def _draw_horizontal_bar(self, x: int, y: int, width: int, height: int,
                            value: float, color: Tuple[int, int, int], label: str):
        """Draw a horizontal progress bar."""
        # Background
        pygame.draw.rect(self.screen, (50, 50, 50), (x, y, width, height))
        # Fill
        fill_width = int(width * min(1.0, max(0.0, value)))
        pygame.draw.rect(self.screen, color, (x, y, fill_width, height))
        # Border
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height), 1)
        # Label
        if label:
            label_text = self.tiny_font.render(label, True, (200, 200, 200))
            self.screen.blit(label_text, (x + 2, y - 12))
    
    def _draw_mini_graph(self, x: int, y: int, width: int, height: int,
                        data: List[float], color: Tuple[int, int, int]):
        """Draw a mini line graph."""
        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), (x, y, width, height))
        
        if len(data) > 1:
            # Normalize data
            min_val = min(data) if data else 0
            max_val = max(data) if data else 1
            range_val = max_val - min_val if max_val != min_val else 1
            
            # Draw line
            points = []
            for i, value in enumerate(data[-50:]):  # Show last 50 points
                px = x + int(i * width / min(len(data), 50))
                py = y + height - int((value - min_val) / range_val * height)
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
        
        # Border
        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, width, height), 1)
    
    def _draw_liquid_tank(self, x: int, y: int, width: int, height: int,
                         level: float, color: Tuple[int, int, int], label: str):
        """Draw a liquid tank with animated fill."""
        # Tank outline
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height), 2)
        
        # Liquid fill with wave effect
        fill_height = int(height * level)
        
        for ly in range(fill_height):
            ratio = ly / max(fill_height, 1)
            liquid_color = tuple(int(c * (0.7 + 0.3 * ratio)) for c in color)
            wave = int(np.sin((ly + self.animation_frame) * 0.2) * 2)
            pygame.draw.line(self.screen, liquid_color,
                           (x + 2 + wave, y + height - ly),
                           (x + width - 2 + wave, y + height - ly))
        
        # Label
        label_text = self.tiny_font.render(label, True, (200, 200, 200))
        text_rect = label_text.get_rect(center=(x + width // 2, y + height + 10))
        self.screen.blit(label_text, text_rect)
        
        # Percentage
        percent_text = self.tiny_font.render(f"{level:.0%}", True, (255, 255, 255))
        percent_rect = percent_text.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(percent_text, percent_rect)
    
    def _draw_circular_meter(self, x: int, y: int, radius: int, value: float,
                            color: Tuple[int, int, int], label: str):
        """Draw a circular meter."""
        # Background circle
        pygame.draw.circle(self.screen, (50, 50, 50), (x, y), radius, 2)
        
        # Fill arc
        angle = int(value * 360)
        for a in range(0, angle, 2):
            arc_x = x + int(radius * np.cos(np.radians(a - 90)))
            arc_y = y + int(radius * np.sin(np.radians(a - 90)))
            pygame.draw.circle(self.screen, color, (arc_x, arc_y), 3)
        
        # Center text
        percent_text = self.small_font.render(f"{value:.0%}", True, (255, 255, 255))
        text_rect = percent_text.get_rect(center=(x, y))
        self.screen.blit(percent_text, text_rect)
        
        # Label
        if label:
            label_text = self.tiny_font.render(label, True, (200, 200, 200))
            label_rect = label_text.get_rect(center=(x, y + radius + 10))
            self.screen.blit(label_text, label_rect)
    
    def _draw_eco_meter(self, x: int, y: int, label: str, value: float, color: Tuple[int, int, int]):
        """Draw an ecological indicator meter."""
        # Leaf-shaped background
        leaf_points = [
            (x + 15, y),
            (x + 25, y + 10),
            (x + 25, y + 30),
            (x + 15, y + 40),
            (x + 5, y + 30),
            (x + 5, y + 10)
        ]
        pygame.draw.polygon(self.screen, (50, 100, 50), leaf_points)
        
        # Fill based on value
        fill_height = int(40 * value)
        for fy in range(fill_height):
            fill_ratio = fy / 40
            fill_color = tuple(int(c * (0.5 + 0.5 * fill_ratio)) for c in color)
            pygame.draw.line(self.screen, fill_color,
                           (x + 5, y + 40 - fy), (x + 25, y + 40 - fy))
        
        # Outline
        pygame.draw.polygon(self.screen, (100, 150, 100), leaf_points, 2)
        
        # Label
        label_text = self.tiny_font.render(label, True, (200, 200, 200))
        text_rect = label_text.get_rect(center=(x + 15, y + 50))
        self.screen.blit(label_text, text_rect)
    
    def _draw_graph(self, x: int, y: int, width: int, height: int,
                   data: List[float], color: Tuple[int, int, int], title: str, unit: str):
        """Draw a detailed graph with axes and labels."""
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), (x, y, width, height))
        
        # Title
        title_text = self.small_font.render(title, True, (255, 255, 255))
        self.screen.blit(title_text, (x + 5, y - 20))
        
        # Grid lines
        for i in range(5):
            grid_y = y + int(i * height / 4)
            pygame.draw.line(self.screen, (60, 60, 60), (x, grid_y), (x + width, grid_y), 1)
        
        if len(data) > 1:
            # Scale data
            min_val = min(data)
            max_val = max(data)
            range_val = max_val - min_val if max_val != min_val else 1
            
            # Plot line
            points = []
            for i, value in enumerate(data):
                px = x + int(i * width / len(data))
                py = y + height - int((value - min_val) / range_val * (height - 10))
                points.append((px, py))
                
                # Draw point
                if i == len(data) - 1:  # Highlight last point
                    pygame.draw.circle(self.screen, color, (px, py), 4)
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
            
            # Value labels
            if max_val > 0:
                max_text = self.tiny_font.render(f"{max_val:.0f}{unit}", True, (200, 200, 200))
                self.screen.blit(max_text, (x + width - 40, y + 2))
            if min_val != max_val:
                min_text = self.tiny_font.render(f"{min_val:.0f}{unit}", True, (200, 200, 200))
                self.screen.blit(min_text, (x + width - 40, y + height - 12))
        
        # Border
        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, width, height), 2)
    
    def close(self):
        """Close the environment and cleanup."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
