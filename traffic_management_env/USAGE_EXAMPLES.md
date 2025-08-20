# Traffic Management Environment - Usage Examples

This document provides practical examples of how to use the Traffic Management Environment for various purposes.

## Quick Start

```python
import gymnasium as gym
import traffic_management_env

# Create environment
env = gym.make('TrafficManagement-v0')

# Basic usage
observation, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Custom Configuration

```python
from traffic_management_env.environment import TrafficManagementEnv

# Create custom environment
env = TrafficManagementEnv(
    grid_size=(6, 6),        # 6x6 grid of intersections
    num_intersections=16,    # Control 16 intersections
    max_vehicles=80,         # Up to 80 vehicles
    spawn_rate=0.5,          # Higher spawn rate
    render_mode='human'      # Enable visualization
)

observation, info = env.reset()
# ... use environment
env.close()
```

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import traffic_management_env

# Create vectorized environment for training
env = make_vec_env('TrafficManagement-v0', n_envs=4)

# Initialize PPO agent
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

# Train the agent
model.learn(total_timesteps=1000000)

# Save the trained model
model.save("traffic_management_ppo")

# Test the trained agent
env = gym.make('TrafficManagement-v0', render_mode='human')
obs, _ = env.reset()

for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
```

## Custom Policy Implementation

```python
import numpy as np
import gymnasium as gym
import traffic_management_env

class TrafficLightController:
    """Custom traffic light controller using heuristics."""
    
    def __init__(self, num_intersections):
        self.num_intersections = num_intersections
        self.phase_timers = np.zeros(num_intersections)
        self.min_phase_time = 10  # Minimum time before switching
    
    def get_action(self, observation, info):
        """Get action based on current state."""
        actions = []
        intersection_states = info.get('intersection_states', [])
        
        for i, intersection_state in enumerate(intersection_states):
            if i >= self.num_intersections:
                break
            
            # Get queue information
            queue_lengths = intersection_state.get('queue_lengths', {})
            current_phase = intersection_state.get('light_phase', 'NS_GREEN')
            
            # Calculate total queue lengths for each direction
            ns_total = queue_lengths.get('NORTH', 0) + queue_lengths.get('SOUTH', 0)
            ew_total = queue_lengths.get('EAST', 0) + queue_lengths.get('WEST', 0)
            
            # Decision logic
            self.phase_timers[i] += 1
            
            if self.phase_timers[i] < self.min_phase_time:
                # Don't switch too frequently
                actions.append(0)  # Maintain
            elif current_phase in ['NS_GREEN', 'NS_YELLOW']:
                if ew_total > ns_total + 3:  # Switch if EW has significantly more traffic
                    actions.append(2)  # Switch to EW_GREEN
                    self.phase_timers[i] = 0
                else:
                    actions.append(0)  # Maintain
            else:  # EW_GREEN or EW_YELLOW
                if ns_total > ew_total + 3:  # Switch if NS has significantly more traffic
                    actions.append(1)  # Switch to NS_GREEN
                    self.phase_timers[i] = 0
                else:
                    actions.append(0)  # Maintain
        
        # Pad with maintain actions if needed
        while len(actions) < self.num_intersections:
            actions.append(0)
        
        return np.array(actions[:self.num_intersections])

# Usage
env = gym.make('TrafficManagement-v0')
controller = TrafficLightController(env.unwrapped.num_intersections)

observation, info = env.reset()
total_reward = 0

for step in range(1000):
    action = controller.get_action(observation, info)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if step % 100 == 0:
        print(f"Step {step}: Total Reward = {total_reward:.2f}")
    
    if terminated or truncated:
        print(f"Episode finished at step {step}")
        break

env.close()
print(f"Final total reward: {total_reward:.2f}")
```

## Environment Analysis and Evaluation

```python
import numpy as np
import gymnasium as gym
import traffic_management_env

def evaluate_policy(env, policy_fn, num_episodes=10):
    """Evaluate a policy over multiple episodes."""
    episode_rewards = []
    episode_lengths = []
    traffic_metrics = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = policy_fn(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        traffic_metrics.append(info.get('metrics', {}))
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'traffic_metrics': traffic_metrics
    }

# Random policy
def random_policy(obs, info):
    return env.action_space.sample()

# Evaluate
env = gym.make('TrafficManagement-v0')
results = evaluate_policy(env, random_policy, num_episodes=5)

print("Random Policy Results:")
print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"Mean Episode Length: {results['mean_length']:.1f}")

# Analyze traffic metrics
final_metrics = results['traffic_metrics'][-1]
print(f"Final Traffic Metrics:")
print(f"  Vehicles Passed: {final_metrics.get('total_vehicles_passed', 0)}")
print(f"  Average Waiting Time: {final_metrics.get('average_waiting_time', 0):.2f}")
print(f"  Average Queue Length: {final_metrics.get('average_queue_length', 0):.2f}")

env.close()
```

## Visualization and Monitoring

```python
import gymnasium as gym
import traffic_management_env
import time

# Create environment with human rendering
env = gym.make('TrafficManagement-v0', render_mode='human')

observation, info = env.reset()

print("Starting traffic simulation with visualization...")
print("Watch the traffic lights and vehicle flow!")

for step in range(500):
    # Simple policy: switch lights every 20 steps
    if step % 20 == 0:
        action = np.array([1] * env.unwrapped.num_intersections)  # All NS_GREEN
    elif step % 20 == 10:
        action = np.array([2] * env.unwrapped.num_intersections)  # All EW_GREEN
    else:
        action = np.array([0] * env.unwrapped.num_intersections)  # Maintain
    
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    # Print status every 50 steps
    if step % 50 == 0:
        metrics = info.get('metrics', {})
        print(f"Step {step}: Vehicles={info['num_vehicles']}, "
              f"Reward={reward:.2f}, "
              f"Avg Wait={metrics.get('average_waiting_time', 0):.2f}")
    
    time.sleep(0.1)  # Slow down for better visualization
    
    if terminated or truncated:
        break

env.close()
```

## Performance Benchmarking

```python
import time
import numpy as np
import gymnasium as gym
import traffic_management_env

def benchmark_environment():
    """Benchmark environment performance."""
    env = gym.make('TrafficManagement-v0')
    
    # Warm up
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)
    
    # Benchmark
    num_steps = 10000
    start_time = time.time()
    
    env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            env.reset()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Environment Performance Benchmark:")
    print(f"  Steps: {num_steps}")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print(f"  Steps per second: {num_steps / elapsed_time:.1f}")
    print(f"  Time per step: {elapsed_time / num_steps * 1000:.3f} ms")
    
    env.close()

if __name__ == "__main__":
    benchmark_environment()
```

## Integration with Ray RLlib

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
import gymnasium as gym
import traffic_management_env

# Initialize Ray
ray.init()

# Configuration for Ray RLlib
config = {
    "env": "TrafficManagement-v0",
    "framework": "torch",
    "num_workers": 4,
    "num_envs_per_worker": 2,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
}

# Train with Ray RLlib
analysis = tune.run(
    PPO,
    config=config,
    stop={"training_iteration": 100},
    checkpoint_freq=10,
)

print("Training completed!")
ray.shutdown()
```

These examples demonstrate the flexibility and power of the Traffic Management Environment for various reinforcement learning applications, from simple policy evaluation to advanced multi-agent training scenarios.