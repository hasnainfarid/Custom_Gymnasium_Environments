import gymnasium as gym
from ray.rllib.algorithms.ppo import PPO
import numpy as np

def evaluate(checkpoint_path, episodes=5):
    algo = PPO.from_checkpoint(checkpoint_path)
    env = gym.make('SmartClimateEnv-v0')
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()
        print(f"Episode {ep+1}: Reward={total_reward}")
    env.close()

if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'last_checkpoint/'
    evaluate(checkpoint) 