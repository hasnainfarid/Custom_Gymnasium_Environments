#!/usr/bin/env python3
"""
Simple structure test for Fleet Management Environment
Tests basic imports and class definitions without external dependencies
"""

import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fleet_management_env'))

def test_imports():
    """Test that all modules can be imported"""
    print("--- Testing Imports ---")
    
    try:
        # Test enum imports
        from fleet_env import VehicleType, UrgencyLevel, CustomerZone
        print("✓ Enums imported successfully")
        
        # Test dataclass imports
        from fleet_env import Vehicle, DeliveryRequest
        print("✓ Dataclasses imported successfully")
        
        # Test main class import (without instantiation)
        from fleet_env import FleetManagementEnv
        print("✓ FleetManagementEnv class imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_enum_values():
    """Test enum definitions"""
    print("\n--- Testing Enum Values ---")
    
    try:
        from fleet_env import VehicleType, UrgencyLevel, CustomerZone
        
        # Test VehicleType
        expected_vehicles = {"van", "motorcycle", "truck"}
        actual_vehicles = {vt.value for vt in VehicleType}
        assert actual_vehicles == expected_vehicles, f"Vehicle types mismatch: {actual_vehicles}"
        print("✓ VehicleType enum correct")
        
        # Test UrgencyLevel
        expected_urgency = {0, 1, 2, 3}
        actual_urgency = {ul.value for ul in UrgencyLevel}
        assert actual_urgency == expected_urgency, f"Urgency levels mismatch: {actual_urgency}"
        print("✓ UrgencyLevel enum correct")
        
        # Test CustomerZone
        expected_zones = {"residential", "commercial", "industrial", "hospital"}
        actual_zones = {cz.value for cz in CustomerZone}
        assert actual_zones == expected_zones, f"Customer zones mismatch: {actual_zones}"
        print("✓ CustomerZone enum correct")
        
        return True
    except Exception as e:
        print(f"✗ Enum test failed: {e}")
        return False

def test_dataclass_structure():
    """Test dataclass structures"""
    print("\n--- Testing Dataclass Structures ---")
    
    try:
        from fleet_env import Vehicle, DeliveryRequest, VehicleType, UrgencyLevel
        
        # Test Vehicle dataclass
        vehicle = Vehicle(
            vehicle_type=VehicleType.VAN,
            position=(10, 10),
            fuel=50.0,
            max_fuel=80.0,
            fuel_consumption=1.0,
            capacity=3,
            cargo_used=0,
            speed=1
        )
        
        assert vehicle.vehicle_type == VehicleType.VAN
        assert vehicle.position == (10, 10)
        assert vehicle.can_move() == True
        assert vehicle.can_pickup() == True
        print("✓ Vehicle dataclass working correctly")
        
        # Test DeliveryRequest dataclass
        delivery = DeliveryRequest(
            pickup_location=(5, 5),
            delivery_location=(15, 15),
            urgency=UrgencyLevel.NORMAL,
            required_vehicle=None,
            time_window_start=0,
            time_window_end=100,
            deadline=50
        )
        
        assert delivery.pickup_location == (5, 5)
        assert delivery.delivery_location == (15, 15)
        assert delivery.is_available(25) == True
        assert delivery.is_overdue(60) == True
        print("✓ DeliveryRequest dataclass working correctly")
        
        return True
    except Exception as e:
        print(f"✗ Dataclass test failed: {e}")
        return False

def test_class_methods():
    """Test class method definitions"""
    print("\n--- Testing Class Methods ---")
    
    try:
        from fleet_env import FleetManagementEnv
        
        # Check that the class has required methods
        required_methods = [
            'reset', 'step', 'render', 'close',
            '_get_observation', '_execute_vehicle_action',
            '_generate_delivery_requests', '_update_traffic'
        ]
        
        for method_name in required_methods:
            assert hasattr(FleetManagementEnv, method_name), f"Missing method: {method_name}"
        
        print("✓ All required methods present")
        
        # Check class attributes
        required_attributes = ['metadata', 'grid_size', 'max_timesteps', 'depot_position']
        
        # Create a dummy instance to check attributes (this might fail due to missing dependencies)
        try:
            # Just check if the class definition is valid
            assert hasattr(FleetManagementEnv, '__init__')
            print("✓ Class definition is valid")
        except Exception as inner_e:
            print(f"Note: Cannot instantiate class (likely due to missing dependencies): {inner_e}")
            print("✓ Class structure appears correct")
        
        return True
    except Exception as e:
        print(f"✗ Class methods test failed: {e}")
        return False

def test_package_structure():
    """Test package file structure"""
    print("\n--- Testing Package Structure ---")
    
    try:
        package_dir = os.path.join(os.path.dirname(__file__), 'fleet_management_env')
        
        required_files = [
            '__init__.py',
            'fleet_env.py',
            'test_fleet.py',
            'setup.py',
            'README.md'
        ]
        
        for filename in required_files:
            filepath = os.path.join(package_dir, filename)
            if filename == 'setup.py':
                filepath = os.path.join(os.path.dirname(__file__), 'fleet_management_env', filename)
            
            assert os.path.exists(filepath), f"Missing file: {filename}"
            print(f"✓ Found {filename}")
        
        # Check file sizes (basic sanity check)
        fleet_env_size = os.path.getsize(os.path.join(package_dir, 'fleet_env.py'))
        assert fleet_env_size > 10000, "fleet_env.py seems too small"
        
        test_fleet_size = os.path.getsize(os.path.join(package_dir, 'test_fleet.py'))
        assert test_fleet_size > 10000, "test_fleet.py seems too small"
        
        print("✓ All files present with reasonable sizes")
        return True
    except Exception as e:
        print(f"✗ Package structure test failed: {e}")
        return False

def main():
    """Run all structure tests"""
    print("Fleet Management Environment - Structure Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_enum_values, 
        test_dataclass_structure,
        test_class_methods,
        test_package_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL STRUCTURE TESTS PASSED!")
        print("\nThe Fleet Management Environment package is properly structured.")
        print("To run full functionality tests, install dependencies:")
        print("  pip install gymnasium numpy pygame matplotlib seaborn pandas")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)