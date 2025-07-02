import numpy as np
import math
from typing import Tuple

def get_outside_temp(time_of_day: float, rng: np.random.Generator) -> float:
    # 0-8h: 25±5, 8-16h: 45±5, 16-24h: 35±5
    if 0 <= time_of_day < 8:
        base = 25
    elif 8 <= time_of_day < 16:
        base = 45
    else:
        base = 35
    return float(rng.normal(base, 5))

def update_occupancy(time_of_day: float, max_occupancy: int, rng: np.random.Generator, prev_people: int) -> int:
    # 9-18h: higher chance of more people
    if 9 <= time_of_day < 18:
        change = rng.choice([-1, 0, 1, 2], p=[0.1, 0.3, 0.4, 0.2])
    else:
        change = rng.choice([-2, -1, 0, 1], p=[0.2, 0.4, 0.3, 0.1])
    num_people = int(np.clip(prev_people + change, 0, max_occupancy))
    return num_people

def room_temp_dynamics(ac_setting: float, outside_temp: float, num_people: int, prev_temp: float) -> float:
    # Each person adds +1C, AC pulls toward ac_setting, outside temp influences
    temp = prev_temp + 0.1 * (outside_temp - prev_temp) + 0.2 * (ac_setting - prev_temp) + num_people * 1.0
    # Clamp
    return float(np.clip(temp, 10, 50))

def calculate_reward(room_temp, num_people, outside_temp, ac_setting, light_states, time_of_day) -> Tuple[float, dict]:
    # Comfort
    if 20 <= room_temp <= 24:
        comfort = 10
    elif 18 <= room_temp <= 26:
        comfort = 5
    elif 16 <= room_temp <= 28:
        comfort = 0
    else:
        comfort = -15 * abs(room_temp - 22)
    # AC penalty
    ac_penalty = -0.5 * abs(ac_setting - outside_temp)
    # Lights
    required_lights = min(4, math.ceil(num_people / 2))
    lights_on = int(np.sum(light_states))
    light_penalty = -1 * max(0, lights_on - required_lights)
    total_reward = comfort + ac_penalty + light_penalty
    return total_reward, {
        'comfort': comfort,
        'ac_penalty': ac_penalty,
        'light_penalty': light_penalty
    } 