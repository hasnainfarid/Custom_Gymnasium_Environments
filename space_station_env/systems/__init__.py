"""Internal system data and configurations for the space station"""

# System interdependency matrix
SYSTEM_DEPENDENCIES = {
    'Oxygen Generation': {
        'requires': ['Power Grid', 'Water Recycling'],
        'affects': ['Atmospheric Pressure', 'CO2 Scrubbing'],
        'critical_threshold': 50.0,
        'failure_cascade_probability': 0.8
    },
    'CO2 Scrubbing': {
        'requires': ['Power Grid'],
        'affects': ['Oxygen Generation'],
        'critical_threshold': 40.0,
        'failure_cascade_probability': 0.6
    },
    'Water Recycling': {
        'requires': ['Power Grid', 'Waste Processing'],
        'affects': ['Oxygen Generation'],
        'critical_threshold': 30.0,
        'failure_cascade_probability': 0.5
    },
    'Power Grid': {
        'requires': [],
        'affects': ['ALL_SYSTEMS'],
        'critical_threshold': 20.0,
        'failure_cascade_probability': 1.0
    },
    'Thermal Control': {
        'requires': ['Power Grid'],
        'affects': ['Communications', 'Artificial Gravity'],
        'critical_threshold': 35.0,
        'failure_cascade_probability': 0.4
    },
    'Atmospheric Pressure': {
        'requires': ['Power Grid', 'Nitrogen Supply'],
        'affects': ['Fire Suppression', 'Oxygen Generation'],
        'critical_threshold': 45.0,
        'failure_cascade_probability': 0.7
    },
    'Nitrogen Supply': {
        'requires': ['Power Grid'],
        'affects': ['Atmospheric Pressure'],
        'critical_threshold': 60.0,
        'failure_cascade_probability': 0.3
    },
    'Waste Processing': {
        'requires': ['Power Grid'],
        'affects': ['Water Recycling'],
        'critical_threshold': 50.0,
        'failure_cascade_probability': 0.3
    },
    'Fire Suppression': {
        'requires': ['Power Grid', 'Atmospheric Pressure'],
        'affects': [],
        'critical_threshold': 30.0,
        'failure_cascade_probability': 0.2
    },
    'Radiation Shielding': {
        'requires': ['Power Grid'],
        'affects': [],
        'critical_threshold': 40.0,
        'failure_cascade_probability': 0.1
    },
    'Communications': {
        'requires': ['Power Grid'],
        'affects': [],
        'critical_threshold': 70.0,
        'failure_cascade_probability': 0.1
    },
    'Artificial Gravity': {
        'requires': ['Power Grid'],
        'affects': [],
        'critical_threshold': 80.0,
        'failure_cascade_probability': 0.1
    }
}

# Emergency response procedures
EMERGENCY_PROCEDURES = {
    'fire': {
        'priority_systems': ['Fire Suppression', 'Atmospheric Pressure', 'Oxygen Generation'],
        'resource_usage': {'oxygen_tanks': 2.0, 'nitrogen': 5.0},
        'crew_assignments': ['Engineer', 'Mission Specialist'],
        'duration_minutes': 30
    },
    'depressurization': {
        'priority_systems': ['Atmospheric Pressure', 'Nitrogen Supply', 'Oxygen Generation'],
        'resource_usage': {'oxygen_tanks': 3.0, 'nitrogen': 10.0, 'spare_parts': 5.0},
        'crew_assignments': ['Engineer', 'Pilot'],
        'duration_minutes': 45
    },
    'radiation_storm': {
        'priority_systems': ['Radiation Shielding', 'Communications'],
        'resource_usage': {'medical_supplies': 10.0},
        'crew_assignments': ['Medical Officer', 'Commander'],
        'duration_minutes': 120
    },
    'power_failure': {
        'priority_systems': ['Power Grid'],
        'resource_usage': {'fuel': 20.0, 'spare_parts': 10.0},
        'crew_assignments': ['Engineer', 'Mission Specialist'],
        'duration_minutes': 60
    },
    'life_support_critical': {
        'priority_systems': ['Oxygen Generation', 'CO2 Scrubbing', 'Atmospheric Pressure'],
        'resource_usage': {'oxygen_tanks': 5.0, 'spare_parts': 3.0},
        'crew_assignments': ['Engineer', 'Scientist', 'Mission Specialist'],
        'duration_minutes': 90
    },
    'medical_emergency': {
        'priority_systems': ['Thermal Control'],
        'resource_usage': {'medical_supplies': 20.0, 'water': 5.0},
        'crew_assignments': ['Medical Officer', 'Commander'],
        'duration_minutes': 60
    }
}

# Crew task definitions
CREW_TASKS = {
    'Commander': {
        'primary': 'coordination',
        'emergency_skills': ['decision_making', 'resource_allocation'],
        'efficiency_bonus': 1.2,
        'fatigue_rate': 0.8
    },
    'Engineer': {
        'primary': 'maintenance',
        'emergency_skills': ['system_repair', 'power_management'],
        'efficiency_bonus': 1.5,
        'fatigue_rate': 1.2
    },
    'Scientist': {
        'primary': 'research',
        'emergency_skills': ['problem_solving', 'data_analysis'],
        'efficiency_bonus': 1.1,
        'fatigue_rate': 0.9
    },
    'Medical Officer': {
        'primary': 'health_monitoring',
        'emergency_skills': ['medical_treatment', 'life_support'],
        'efficiency_bonus': 1.3,
        'fatigue_rate': 1.0
    },
    'Pilot': {
        'primary': 'navigation',
        'emergency_skills': ['docking_procedures', 'evacuation'],
        'efficiency_bonus': 1.2,
        'fatigue_rate': 1.1
    },
    'Mission Specialist': {
        'primary': 'operations',
        'emergency_skills': ['multi_tasking', 'backup_support'],
        'efficiency_bonus': 1.4,
        'fatigue_rate': 1.0
    }
}

# Resource consumption rates (per crew member per hour)
RESOURCE_CONSUMPTION = {
    'water': {
        'nominal': 0.125,  # liters/hour
        'emergency': 0.2,
        'recycling_efficiency': 0.95
    },
    'food': {
        'nominal': 0.0625,  # kg/hour
        'emergency': 0.04,
        'recycling_efficiency': 0.0
    },
    'oxygen': {
        'nominal': 0.84,  # kg/day converted to hourly
        'emergency': 1.2,
        'recycling_efficiency': 0.0
    },
    'power': {
        'life_support_base': 2000,  # Watts
        'per_crew': 100,  # Watts per crew member
        'emergency_surge': 1.5  # multiplier
    }
}

# Orbital mechanics parameters
ORBITAL_PARAMETERS = {
    'altitude_km': 408,  # ISS altitude
    'velocity_km_s': 7.66,
    'period_minutes': 90,
    'inclination_degrees': 51.6,
    'solar_panel_efficiency': {
        'optimal': 1.0,
        'degradation_per_month': 0.02,
        'shadow_efficiency': 0.0,
        'angle_factor': 'cosine'
    },
    'radiation_zones': {
        'south_atlantic_anomaly': {
            'start_longitude': -50,
            'end_longitude': 0,
            'start_latitude': -50,
            'end_latitude': 0,
            'radiation_multiplier': 3.0
        }
    }
}

# System maintenance schedules
MAINTENANCE_SCHEDULES = {
    'Oxygen Generation': {'interval_hours': 168, 'duration_hours': 4, 'parts_required': 2},
    'CO2 Scrubbing': {'interval_hours': 336, 'duration_hours': 2, 'parts_required': 1},
    'Water Recycling': {'interval_hours': 200, 'duration_hours': 6, 'parts_required': 3},
    'Power Grid': {'interval_hours': 720, 'duration_hours': 8, 'parts_required': 5},
    'Thermal Control': {'interval_hours': 500, 'duration_hours': 3, 'parts_required': 2},
    'Atmospheric Pressure': {'interval_hours': 400, 'duration_hours': 2, 'parts_required': 1},
    'Nitrogen Supply': {'interval_hours': 600, 'duration_hours': 1, 'parts_required': 1},
    'Waste Processing': {'interval_hours': 150, 'duration_hours': 4, 'parts_required': 2},
    'Fire Suppression': {'interval_hours': 1000, 'duration_hours': 2, 'parts_required': 1},
    'Radiation Shielding': {'interval_hours': 2000, 'duration_hours': 6, 'parts_required': 4},
    'Communications': {'interval_hours': 300, 'duration_hours': 1, 'parts_required': 1},
    'Artificial Gravity': {'interval_hours': 800, 'duration_hours': 10, 'parts_required': 6}
}

# Scientific experiments and their resource requirements
EXPERIMENTS = {
    'protein_crystallization': {
        'duration_hours': 72,
        'power_watts': 200,
        'water_liters': 2,
        'crew_hours': 4,
        'reward_points': 1000
    },
    'plant_growth': {
        'duration_hours': 168,
        'power_watts': 400,
        'water_liters': 10,
        'crew_hours': 2,
        'reward_points': 1500
    },
    'material_science': {
        'duration_hours': 48,
        'power_watts': 600,
        'water_liters': 0,
        'crew_hours': 6,
        'reward_points': 800
    },
    'medical_research': {
        'duration_hours': 96,
        'power_watts': 150,
        'water_liters': 1,
        'crew_hours': 8,
        'reward_points': 1200
    }
}