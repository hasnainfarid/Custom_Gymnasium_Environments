import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
import gymnasium
import smartclimate  # This ensures the env is registered!

def main():
    ray.init(ignore_reinit_error=True)
    config = (
        PPOConfig()
        .environment(env='SmartClimateEnv-v0')
        .env_runners(num_env_runners=6 ,num_envs_per_env_runner= 24)
        .training(train_batch_size=3000)
    )
    tuner = tune.Tuner(
        config.algo_class,
        run_config=air.RunConfig(
            stop={"env_runners/episode_return_mean": 100},
            #checkpoint_config=air.CheckpointConfig(num_to_keep=2, checkpoint_frequency=10),
            verbose=2
        ),
        param_space=config.to_dict(),
    )
    results = tuner.fit()
    #print("Best checkpoint:", results.get_best_result().checkpoint)

if __name__ == "__main__":
    main() 