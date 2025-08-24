"""
Self-contained crop data for the agricultural farm environment.
All agricultural parameters are generated internally without external dependencies.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List, Tuple


class CropType(IntEnum):
    """Enumeration of available crop types."""
    WHEAT = 0
    CORN = 1
    TOMATOES = 2
    APPLES = 3
    SOYBEANS = 4


class CropStage(IntEnum):
    """Growth stages for crops."""
    PLANTING = 0
    GERMINATION = 1
    VEGETATIVE = 2
    REPRODUCTIVE = 3
    MATURATION = 4
    HARVEST_READY = 5
    HARVESTED = 6
    DORMANT = 7  # For perennial crops like apples in winter


@dataclass
class CropCharacteristics:
    """Detailed characteristics for each crop type."""
    name: str
    growth_cycle_days: int
    optimal_temp_range: Tuple[float, float]  # Min, Max in Celsius
    water_needs: float  # Liters per square meter per day
    nutrient_needs: Dict[str, float]  # NPK requirements
    drought_resistance: float  # 0.0 to 1.0
    disease_susceptibility: float  # 0.0 to 1.0
    optimal_ph_range: Tuple[float, float]
    market_base_price: float  # $ per ton
    yield_per_hectare: float  # Tons per hectare
    planting_seasons: List[str]  # Suitable planting seasons
    harvest_window_days: int  # Days the crop can wait before quality loss
    storage_duration_days: int  # How long crop can be stored
    fertilizer_schedule: List[int]  # Days after planting to apply fertilizer
    pesticide_threshold: float  # Pest level requiring treatment
    rotation_benefit: float  # Yield bonus for proper rotation
    carbon_sequestration: float  # CO2 tons per hectare
    labor_intensity: float  # Hours per hectare
    equipment_requirements: List[str]  # Required equipment types


# Self-contained crop database
CROP_DATA: Dict[CropType, CropCharacteristics] = {
    CropType.WHEAT: CropCharacteristics(
        name="Wheat",
        growth_cycle_days=90,
        optimal_temp_range=(10.0, 25.0),
        water_needs=4.5,
        nutrient_needs={"nitrogen": 120, "phosphorus": 60, "potassium": 40},
        drought_resistance=0.7,
        disease_susceptibility=0.4,
        optimal_ph_range=(6.0, 7.5),
        market_base_price=250.0,
        yield_per_hectare=3.5,
        planting_seasons=["spring", "fall"],
        harvest_window_days=14,
        storage_duration_days=365,
        fertilizer_schedule=[20, 45, 70],
        pesticide_threshold=0.3,
        rotation_benefit=0.15,
        carbon_sequestration=0.8,
        labor_intensity=20,
        equipment_requirements=["tractor", "harvester", "seed_planter"]
    ),
    
    CropType.CORN: CropCharacteristics(
        name="Corn",
        growth_cycle_days=120,
        optimal_temp_range=(18.0, 32.0),
        water_needs=6.0,
        nutrient_needs={"nitrogen": 180, "phosphorus": 80, "potassium": 90},
        drought_resistance=0.5,
        disease_susceptibility=0.5,
        optimal_ph_range=(5.8, 7.0),
        market_base_price=200.0,
        yield_per_hectare=10.0,
        planting_seasons=["spring", "summer"],
        harvest_window_days=21,
        storage_duration_days=365,
        fertilizer_schedule=[15, 35, 60, 85],
        pesticide_threshold=0.25,
        rotation_benefit=0.20,
        carbon_sequestration=1.2,
        labor_intensity=25,
        equipment_requirements=["tractor", "harvester", "seed_planter", "fertilizer_spreader"]
    ),
    
    CropType.TOMATOES: CropCharacteristics(
        name="Tomatoes",
        growth_cycle_days=60,
        optimal_temp_range=(18.0, 29.0),
        water_needs=5.0,
        nutrient_needs={"nitrogen": 100, "phosphorus": 120, "potassium": 140},
        drought_resistance=0.3,
        disease_susceptibility=0.7,
        optimal_ph_range=(6.0, 6.8),
        market_base_price=800.0,
        yield_per_hectare=80.0,
        planting_seasons=["spring", "summer"],
        harvest_window_days=7,
        storage_duration_days=30,
        fertilizer_schedule=[10, 25, 40],
        pesticide_threshold=0.2,
        rotation_benefit=0.25,
        carbon_sequestration=0.5,
        labor_intensity=150,
        equipment_requirements=["tractor", "irrigation_system", "pesticide_sprayer"]
    ),
    
    CropType.APPLES: CropCharacteristics(
        name="Apples",
        growth_cycle_days=365,
        optimal_temp_range=(4.0, 24.0),
        water_needs=4.0,
        nutrient_needs={"nitrogen": 80, "phosphorus": 40, "potassium": 100},
        drought_resistance=0.6,
        disease_susceptibility=0.6,
        optimal_ph_range=(6.0, 7.0),
        market_base_price=600.0,
        yield_per_hectare=40.0,
        planting_seasons=["spring"],  # Trees planted once, produce annually
        harvest_window_days=30,
        storage_duration_days=180,
        fertilizer_schedule=[30, 90, 150, 210, 270],
        pesticide_threshold=0.15,
        rotation_benefit=0.0,  # Perennial crop, no rotation
        carbon_sequestration=2.5,
        labor_intensity=80,
        equipment_requirements=["tractor", "pesticide_sprayer", "irrigation_system"]
    ),
    
    CropType.SOYBEANS: CropCharacteristics(
        name="Soybeans",
        growth_cycle_days=100,
        optimal_temp_range=(20.0, 30.0),
        water_needs=4.5,
        nutrient_needs={"nitrogen": 40, "phosphorus": 60, "potassium": 80},  # Lower N due to nitrogen fixation
        drought_resistance=0.6,
        disease_susceptibility=0.5,
        optimal_ph_range=(6.0, 7.0),
        market_base_price=400.0,
        yield_per_hectare=3.0,
        planting_seasons=["spring", "summer"],
        harvest_window_days=14,
        storage_duration_days=365,
        fertilizer_schedule=[20, 50, 75],
        pesticide_threshold=0.3,
        rotation_benefit=0.30,  # High rotation benefit due to nitrogen fixation
        carbon_sequestration=1.0,
        labor_intensity=18,
        equipment_requirements=["tractor", "harvester", "seed_planter"]
    )
}


def get_crop_stage_duration(crop_type: CropType, stage: CropStage) -> int:
    """
    Get the duration in days for each growth stage of a crop.
    """
    crop = CROP_DATA[crop_type]
    total_days = crop.growth_cycle_days
    
    # Stage duration percentages (varies by crop)
    stage_percentages = {
        CropType.WHEAT: {
            CropStage.PLANTING: 0.05,
            CropStage.GERMINATION: 0.10,
            CropStage.VEGETATIVE: 0.35,
            CropStage.REPRODUCTIVE: 0.25,
            CropStage.MATURATION: 0.20,
            CropStage.HARVEST_READY: 0.05
        },
        CropType.CORN: {
            CropStage.PLANTING: 0.04,
            CropStage.GERMINATION: 0.08,
            CropStage.VEGETATIVE: 0.40,
            CropStage.REPRODUCTIVE: 0.25,
            CropStage.MATURATION: 0.18,
            CropStage.HARVEST_READY: 0.05
        },
        CropType.TOMATOES: {
            CropStage.PLANTING: 0.05,
            CropStage.GERMINATION: 0.12,
            CropStage.VEGETATIVE: 0.33,
            CropStage.REPRODUCTIVE: 0.30,
            CropStage.MATURATION: 0.15,
            CropStage.HARVEST_READY: 0.05
        },
        CropType.APPLES: {
            CropStage.PLANTING: 0.01,
            CropStage.GERMINATION: 0.02,
            CropStage.VEGETATIVE: 0.40,
            CropStage.REPRODUCTIVE: 0.15,
            CropStage.MATURATION: 0.20,
            CropStage.HARVEST_READY: 0.08,
            CropStage.DORMANT: 0.14
        },
        CropType.SOYBEANS: {
            CropStage.PLANTING: 0.05,
            CropStage.GERMINATION: 0.10,
            CropStage.VEGETATIVE: 0.35,
            CropStage.REPRODUCTIVE: 0.30,
            CropStage.MATURATION: 0.15,
            CropStage.HARVEST_READY: 0.05
        }
    }
    
    percentages = stage_percentages.get(crop_type, stage_percentages[CropType.WHEAT])
    return int(total_days * percentages.get(stage, 0.1))


def calculate_yield_modifier(
    crop_type: CropType,
    soil_moisture: float,
    soil_ph: float,
    nutrients: Dict[str, float],
    temperature: float,
    pest_level: float,
    disease_level: float
) -> float:
    """
    Calculate yield modifier based on environmental conditions.
    Returns a multiplier between 0.0 and 1.5.
    """
    crop = CROP_DATA[crop_type]
    modifier = 1.0
    
    # Temperature effect
    min_temp, max_temp = crop.optimal_temp_range
    if min_temp <= temperature <= max_temp:
        modifier *= 1.0
    elif temperature < min_temp:
        modifier *= max(0.3, 1.0 - (min_temp - temperature) * 0.05)
    else:
        modifier *= max(0.3, 1.0 - (temperature - max_temp) * 0.05)
    
    # Soil pH effect
    min_ph, max_ph = crop.optimal_ph_range
    if min_ph <= soil_ph <= max_ph:
        modifier *= 1.0
    else:
        ph_deviation = min(abs(soil_ph - min_ph), abs(soil_ph - max_ph))
        modifier *= max(0.5, 1.0 - ph_deviation * 0.15)
    
    # Moisture effect
    optimal_moisture = 0.6  # 60% field capacity
    if 0.4 <= soil_moisture <= 0.8:
        modifier *= 1.0
    else:
        moisture_deviation = abs(soil_moisture - optimal_moisture)
        modifier *= max(0.4, 1.0 - moisture_deviation)
    
    # Nutrient effect
    for nutrient, required in crop.nutrient_needs.items():
        if nutrient in nutrients:
            ratio = nutrients[nutrient] / required
            if 0.8 <= ratio <= 1.2:
                modifier *= 1.0
            elif ratio < 0.8:
                modifier *= max(0.5, ratio)
            else:  # Too much nutrient
                modifier *= max(0.7, 2.0 - ratio)
    
    # Pest and disease effect
    modifier *= (1.0 - pest_level * 0.5)
    modifier *= (1.0 - disease_level * 0.5)
    
    return max(0.1, min(1.5, modifier))


def get_market_price_multiplier(season: str, crop_type: CropType) -> float:
    """
    Get seasonal market price multiplier for crops.
    """
    seasonal_multipliers = {
        "spring": {
            CropType.WHEAT: 1.0,
            CropType.CORN: 0.9,
            CropType.TOMATOES: 1.2,
            CropType.APPLES: 1.1,
            CropType.SOYBEANS: 1.0
        },
        "summer": {
            CropType.WHEAT: 1.1,
            CropType.CORN: 1.0,
            CropType.TOMATOES: 0.8,
            CropType.APPLES: 0.9,
            CropType.SOYBEANS: 1.0
        },
        "fall": {
            CropType.WHEAT: 0.9,
            CropType.CORN: 1.2,
            CropType.TOMATOES: 1.3,
            CropType.APPLES: 1.2,
            CropType.SOYBEANS: 1.1
        },
        "winter": {
            CropType.WHEAT: 1.2,
            CropType.CORN: 1.3,
            CropType.TOMATOES: 1.5,
            CropType.APPLES: 1.0,
            CropType.SOYBEANS: 1.2
        }
    }
    
    return seasonal_multipliers.get(season, {}).get(crop_type, 1.0)

