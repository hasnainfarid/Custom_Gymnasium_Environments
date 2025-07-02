def main():
    import smartclimate
    import gymnasium as gym
    env = gym.make('SmartClimateEnv-v0')
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        done = terminated or truncated
    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main() 