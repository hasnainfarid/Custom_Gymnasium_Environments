# APPO (Asynchronous Proximal Policy Optimization) training for WorldBuilderEnv using RLlib

import sys
import os

# Import the environment
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.world_builder_env import WorldBuilderEnv

import ray
from ray import tune
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.env.env_context import EnvContext

# --- RLlib Environment Registration ---

import gymnasium as gym

def env_creator(env_config: EnvContext):
    # RLlib passes env_config as a dict, use flatten_obs=True for APPO compatibility
    return WorldBuilderEnv(render_mode=None, flatten_obs=True)

# Register the environment with RLlib
from ray.tune.registry import register_env
register_env("WorldBuilderEnv-v0", env_creator)

# --- APPO Training ---

def main():
    ray.init(ignore_reinit_error=True)

    # APPO configuration
    config = (
        APPOConfig()
        .environment(env="WorldBuilderEnv-v0")
        .framework("torch")
        .env_runners(num_env_runners=6, num_cpus_per_env_runner=1, num_envs_per_env_runner=24)
        .training(
            gamma=0.99,
            lr=1e-3,
            train_batch_size=4000,  # APPO default batch size can be adjusted
        )
        .resources(num_gpus=0)
    )

    # Run training using Tune's Tuner API
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=tune.RunConfig(
            stop={"env_runners/episode_return_mean": 450},
            verbose=2,
        ),
    )
    results = tuner.fit()

    ray.shutdown()

if __name__ == "__main__":
    main()
