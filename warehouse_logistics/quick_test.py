#!/usr/bin/env python3
"""
Quick test to verify the warehouse environment works
"""

try:
    from warehouse_env import WarehouseEnv, Action, CellType, Package
    print("✓ Successfully imported warehouse environment classes")
    
    # Test basic environment creation
    env = WarehouseEnv(width=5, height=5, num_packages=1)
    print("✓ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset, observation shape: {obs.shape}")
    
    # Test step
    action = 0  # Move up
    obs, reward, done, truncated, info = env.step(action)
    print(f"✓ Step executed, reward: {reward}")
    
    # Test action space
    print(f"✓ Action space size: {env.action_space.n}")
    print(f"✓ Observation space shape: {env.observation_space.shape}")
    
    env.close()
    print("✓ Environment closed successfully")
    print("\n🎉 All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()