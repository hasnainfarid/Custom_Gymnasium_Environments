from ray import tune, train
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Paperspace-compatible import handling
import sys
import os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from core.parking_env import SmartParkingEnv
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    raise

def env_creator(config):
    return SmartParkingEnv()

register_env("SmartParkingEnv-v0", env_creator)

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True
    )
    .environment("SmartParkingEnv-v0")
    .training(
        lr=0.00001,
        train_batch_size=10000,
        #entropy_coeff=0.01,  # Encourage exploration
        #use_gae=True,
        #vf_clip_param=1000.0,
        #normalize_rewards=True,
    )
    .env_runners(
        num_env_runners=6,
        num_envs_per_env_runner=24,
        env_to_module_connector=lambda env: FlattenObservations()
    )
)

# Train through Ray Tune with W&B
results = tune.Tuner(
    config.algo_class,
    param_space=config,
    run_config=train.RunConfig(
        #stop={"env_runners/episode_return_mean": 250.0},
        verbose=2,
        progress_reporter=tune.CLIReporter(
            metric_columns=[
                "episode_return_mean",
                "episode_len_mean",
                "training_iteration"
            ]
        ),
        #callbacks=[WandbLoggerCallback(project="restaurant-rl")]
    ),
).fit()

print(results.get_best_result(metric="episode_return_mean", mode="max"))