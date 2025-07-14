import time
import numpy as np
from restaurant_env import RestaurantEnv
from visualization import RestaurantVisualization
import random

if __name__ == "__main__":
    env = RestaurantEnv(render_mode='human')
    vis = RestaurantVisualization(env)
    obs, info = env.reset()
    total_reward = 0.0
    print("Starting random single-action agent episode with visualization...")
    vis.render(obs, info, last_actions=[], last_reward=0.0)
    time.sleep(0.5)

    for step in range(env.max_episode_steps):
        # Generate a single random action
        idle_waiters = [w.waiter_id for w in env.waiters if w.is_idle()]
        if idle_waiters:
            waiter_id = random.choice(idle_waiters)
        else:
            waiter_id = 0
        action_type = random.choice([0, 1, 2, 3])  # 0=seat, 1=serve, 2=clean, 3=do nothing
        action = {'type': action_type, 'waiter_id': waiter_id, 'customer_id': 0, 'table_id': 0}
        if action_type == 0 and len(env.waiting_customers) > 0:
            action['customer_id'] = random.randint(0, len(env.waiting_customers)-1)
            action['table_id'] = random.randint(0, env.num_tables-1)
        elif action_type in [1, 2]:
            action['table_id'] = random.randint(0, env.num_tables-1)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step:3d} | Reward: {reward:6.2f} | Total: {total_reward:7.2f}")
        print(f"  Action: {action}")
        vis.render(obs, info, last_actions=[action], last_reward=reward)
        time.sleep(0.1)
        if terminated or truncated:
            break
    print(f"Episode finished. Total reward: {total_reward:.2f}")
    vis.close()
    env.close() 