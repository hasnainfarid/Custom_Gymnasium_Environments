from gymnasium.envs.registration import register

register(
    id='snake_env_classic-v0',
    entry_point='snake_env_classic.snake_env:SnakeEnvClassic',
    max_episode_steps=1000,
) 