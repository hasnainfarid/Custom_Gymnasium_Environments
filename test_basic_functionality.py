#!/usr/bin/env python3
"""
Basic functionality test for Fleet Management Environment
Tests core functionality without requiring package installation
"""

import sys
import os
import numpy as np

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fleet_management_env'))

try:
    from fleet_env import FleetManagementEnv, VehicleType, UrgencyLevel, CustomerZone
    print("✓ Successfully imported FleetManagementEnv")
except ImportError as e:
    print(f"✗ Failed to import FleetManagementEnv: {e}")
    sys.exit(1)

def test_environment_creation():
    """Test basic environment creation"""
    print("\n--- Testing Environment Creation ---")
    
    try:
        env = FleetManagementEnv(render_mode=None)
        print("✓ Environment created successfully")
        
        # Test basic properties
        assert env.grid_size == 25, f"Expected grid size 25, got {env.grid_size}"
        assert len(env.vehicles) == 3, f"Expected 3 vehicles, got {len(env.vehicles)}"
        assert len(env.fuel_stations) == 3, f"Expected 3 fuel stations, got {len(env.fuel_stations)}"
        print("✓ Environment properties correct")
        
        return env
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return None

def test_reset_functionality(env):
    """Test environment reset"""
    print("\n--- Testing Reset Functionality ---")
    
    try:
        observation, info = env.reset(seed=42)
        print("✓ Environment reset successfully")
        
        # Check observation shape
        expected_shape = (87,)  # As defined in the environment
        assert observation.shape == expected_shape, f"Expected observation shape {expected_shape}, got {observation.shape}"
        print(f"✓ Observation shape correct: {observation.shape}")
        
        # Check info structure
        assert isinstance(info, dict), "Info should be a dictionary"
        required_keys = ["timestep", "active_deliveries", "completed_deliveries", "vehicles_with_fuel"]
        for key in required_keys:
            assert key in info, f"Missing key in info: {key}"
        print("✓ Info structure correct")
        
        return observation, info
    except Exception as e:
        print(f"✗ Reset functionality failed: {e}")
        return None, None

def test_action_space(env):
    """Test action space functionality"""
    print("\n--- Testing Action Space ---")
    
    try:
        action_space = env.action_space
        print(f"✓ Action space: {action_space}")
        
        # Test action sampling
        action = action_space.sample()
        assert len(action) == 3, f"Expected 3 actions (one per vehicle), got {len(action)}"
        print(f"✓ Sample action: {action}")
        
        # Test action bounds
        for i, a in enumerate(action):
            assert 0 <= a <= 7, f"Action {i} out of bounds: {a}"
        print("✓ Action bounds correct")
        
        return action
    except Exception as e:
        print(f"✗ Action space test failed: {e}")
        return None

def test_step_functionality(env, action):
    """Test environment step"""
    print("\n--- Testing Step Functionality ---")
    
    try:
        observation, reward, terminated, truncated, info = env.step(action)
        print("✓ Step executed successfully")
        
        # Check return types
        assert isinstance(observation, np.ndarray), "Observation should be numpy array"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(terminated, bool), "Terminated should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        print("✓ Return types correct")
        
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Active deliveries: {info.get('active_deliveries', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"✗ Step functionality failed: {e}")
        return False

def test_vehicle_properties(env):
    """Test vehicle specifications"""
    print("\n--- Testing Vehicle Properties ---")
    
    try:
        vehicles = env.vehicles
        vehicle_types = [v.vehicle_type for v in vehicles]
        
        expected_types = [VehicleType.VAN, VehicleType.MOTORCYCLE, VehicleType.TRUCK]
        assert len(set(vehicle_types)) == 3, "Should have 3 different vehicle types"
        print("✓ Vehicle types correct")
        
        # Test vehicle specifications
        for vehicle in vehicles:
            assert hasattr(vehicle, 'position'), "Vehicle should have position"
            assert hasattr(vehicle, 'fuel'), "Vehicle should have fuel"
            assert hasattr(vehicle, 'capacity'), "Vehicle should have capacity"
            assert vehicle.fuel > 0, "Vehicle should start with fuel"
            print(f"  {vehicle.vehicle_type.value}: pos={vehicle.position}, fuel={vehicle.fuel}, capacity={vehicle.capacity}")
        
        print("✓ Vehicle properties correct")
        return True
    except Exception as e:
        print(f"✗ Vehicle properties test failed: {e}")
        return False

def test_delivery_requests(env):
    """Test delivery request generation"""
    print("\n--- Testing Delivery Requests ---")
    
    try:
        deliveries = env.delivery_requests
        assert 8 <= len(deliveries) <= 12, f"Expected 8-12 deliveries, got {len(deliveries)}"
        print(f"✓ Generated {len(deliveries)} delivery requests")
        
        # Test delivery properties
        for i, delivery in enumerate(deliveries[:3]):  # Test first 3
            assert hasattr(delivery, 'pickup_location'), "Delivery should have pickup location"
            assert hasattr(delivery, 'delivery_location'), "Delivery should have delivery location"
            assert hasattr(delivery, 'urgency'), "Delivery should have urgency"
            assert delivery.pickup_location != delivery.delivery_location, "Pickup and delivery should be different"
            print(f"  Delivery {i}: {delivery.pickup_location} → {delivery.delivery_location}, urgency={delivery.urgency.value}")
        
        print("✓ Delivery requests correct")
        return True
    except Exception as e:
        print(f"✗ Delivery requests test failed: {e}")
        return False

def test_multiple_steps(env, num_steps=10):
    """Test multiple environment steps"""
    print(f"\n--- Testing {num_steps} Steps ---")
    
    try:
        total_reward = 0
        for step in range(num_steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                print(f"  Step {step}: reward={reward:.2f}, active_deliveries={info.get('active_deliveries', 'N/A')}")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                break
        
        print(f"✓ Completed {num_steps} steps, total reward: {total_reward:.2f}")
        return True
    except Exception as e:
        print(f"✗ Multiple steps test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Fleet Management Environment - Basic Functionality Test")
    print("=" * 60)
    
    # Test environment creation
    env = test_environment_creation()
    if env is None:
        return False
    
    # Test reset
    observation, info = test_reset_functionality(env)
    if observation is None:
        return False
    
    # Test action space
    action = test_action_space(env)
    if action is None:
        return False
    
    # Test step functionality
    if not test_step_functionality(env, action):
        return False
    
    # Test vehicle properties
    if not test_vehicle_properties(env):
        return False
    
    # Test delivery requests
    if not test_delivery_requests(env):
        return False
    
    # Test multiple steps
    if not test_multiple_steps(env, 20):
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Fleet Management Environment is working correctly!")
    print("=" * 60)
    
    # Print summary information
    print(f"\nEnvironment Summary:")
    print(f"  Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"  Vehicles: {len(env.vehicles)}")
    print(f"  Fuel Stations: {len(env.fuel_stations)}")
    print(f"  Delivery Requests: {len(env.delivery_requests)}")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space: {env.action_space}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)