import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Tuple, Optional
import logging
from .utils import (
    get_outside_temp, update_occupancy, room_temp_dynamics, calculate_reward
)

class SmartClimateEnv(gym.Env):
    """
    SmartClimateEnv-v0: HVAC and lighting control environment for RL.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        max_occupancy: int = 8,
        comfort_temp_range: Tuple[float, float] = (20.0, 24.0),
        episode_minutes: int = 1440,
        seed: Optional[int] = None,
        log_level: int = logging.INFO,
        **kwargs
    ):
        super().__init__()
        self.max_occupancy = max_occupancy
        self.comfort_temp_range = comfort_temp_range
        self.episode_minutes = episode_minutes
        self.current_step = 0
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.logger = logging.getLogger("SmartClimateEnv")
        self.logger.setLevel(log_level)
        self._setup_spaces()
        self._init_state()

    def _setup_spaces(self):
        self.action_space = spaces.Dict({
            'ac_temp': spaces.Box(low=16.0, high=32.0, shape=(1,), dtype=np.float32),
            'lights': spaces.MultiBinary(4)
        })
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0, 0.0, 10.0, 16.0, 0, 0, 0, 0]),
            high=np.array([50.0, self.max_occupancy, 23.99, 50.0, 32.0, 1, 1, 1, 1]),
            dtype=np.float32
        )

    def _init_state(self):
        self.room_temp = self.rng.uniform(22.0, 26.0)
        self.num_people = self.rng.integers(0, self.max_occupancy + 1)
        self.time_of_day = 0.0
        self.outside_temp = get_outside_temp(self.time_of_day, self.rng)
        self.ac_setting = 24.0
        self.light_states = np.zeros(4, dtype=np.int8)
        self.done = False
        self.info = {}
        self.total_reward = 0.0
        self.comfort_time = 0
        self.energy_usage = 0.0
        self.occupancy_history = [self.num_people]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_value = seed
        self.current_step = 0
        self._init_state()
        obs = self._get_obs()
        self.logger.info("Environment reset.")
        return obs, self.info

    def _get_obs(self) -> np.ndarray:
        obs = np.array([
            self.room_temp,
            self.num_people,
            self.time_of_day,
            self.outside_temp,
            self.ac_setting,
            *self.light_states
        ], dtype=np.float32)
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        ac_temp = float(np.clip(action['ac_temp'][0], 16.0, 32.0))
        lights = np.array(action['lights'], dtype=np.int8)
        self.ac_setting = ac_temp
        self.light_states = lights
        # Advance time
        self.current_step += 1
        self.time_of_day = (self.current_step % 1440) / 60.0
        self.outside_temp = get_outside_temp(self.time_of_day, self.rng)
        self.num_people = update_occupancy(self.time_of_day, self.max_occupancy, self.rng, self.num_people)
        self.occupancy_history.append(self.num_people)
        self.room_temp = room_temp_dynamics(self.ac_setting, self.outside_temp, self.num_people, self.room_temp)
        obs = self._get_obs()
        reward, reward_info = calculate_reward(
            self.room_temp, self.num_people, self.outside_temp, self.ac_setting, self.light_states, self.time_of_day
        )
        self.total_reward += reward
        if 20 <= self.room_temp <= 24:
            self.comfort_time += 1
        self.energy_usage += abs(self.ac_setting - self.outside_temp) + np.sum(self.light_states)
        terminated = self.current_step >= self.episode_minutes
        truncated = False
        self.done = terminated
        self.info = {
            **reward_info,
            'comfort_time': self.comfort_time,
            'energy_usage': self.energy_usage,
            'step': self.current_step
        }
        return obs, reward, terminated, truncated, self.info

    def render(self, mode: str = 'human'):
        from .visualizer import SmartClimateVisualizer
        if not hasattr(self, '_visualizer'):
            self._visualizer = SmartClimateVisualizer()
        self._visualizer.render(
            room_temp=self.room_temp,
            num_people=self.num_people,
            ac_setting=self.ac_setting,
            light_states=self.light_states,
            outside_temp=self.outside_temp,
            time_of_day=self.time_of_day
        )

    def close(self):
        if hasattr(self, '_visualizer'):
            self._visualizer.close() 