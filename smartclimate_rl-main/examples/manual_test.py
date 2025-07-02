import smartclimate  # This ensures the env is registered!

def main():
    import gymnasium as gym
    env = gym.make('SmartClimateEnv-v0')
    obs, info = env.reset()
    done = False
    while not done:
        print(f"Obs: {obs}")
        ac = float(input("Set AC temp (16-32): "))
        lights = [int(input(f"Light {i+1} (0/1): ")) for i in range(4)]
        action = {'ac_temp': [ac], 'lights': lights}
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Info: {info}")
        env.render()
        done = terminated or truncated
    env.close()

if __name__ == "__main__":
    main() 