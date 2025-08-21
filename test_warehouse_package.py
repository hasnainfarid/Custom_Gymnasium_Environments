#!/usr/bin/env python3
"""
Test the warehouse logistics package from the parent directory
Usage: python test_warehouse_package.py
"""

import sys
import os

try:
    # Import the package
    from warehouse_logistics import WarehouseEnv, Action, CellType, Package
    print("✓ Successfully imported warehouse environment from package")
    
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
    print("\nTo run the full test suite:")
    print("cd warehouse_logistics && python test_warehouse.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure you have the required dependencies installed:")
    print("pip install -r warehouse_logistics/requirements.txt")
    import traceback
    traceback.print_exc()