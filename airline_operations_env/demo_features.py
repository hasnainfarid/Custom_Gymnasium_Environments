"""
Demonstration of Airline Operations Environment Features
Shows all major capabilities of the environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from airline_env import AirlineOperationsEnv, WeatherCondition, AircraftType
from operations import FlightDataGenerator, PerformanceMetrics, DisruptionType, SeasonalPattern, RouteType
import numpy as np
import time

def demonstrate_features():
    """Demonstrate all major features of the environment"""
    
    print("="*80)
    print("AIRLINE OPERATIONS ENVIRONMENT - FEATURE DEMONSTRATION")
    print("="*80)
    
    # 1. Environment Creation
    print("\n1. ENVIRONMENT INITIALIZATION")
    print("-"*40)
    env = AirlineOperationsEnv(render_mode="human")
    obs, info = env.reset()
    print(f"✓ Environment created with:")
    print(f"  - {env.n_airports} airports (1 major hub, 4 regional hubs, 10 spokes)")
    print(f"  - {env.n_aircraft} aircraft in fleet")
    print(f"  - {env.n_flights} daily scheduled flights")
    print(f"  - {env.n_crew} crew sets")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n} discrete actions")
    
    # 2. Aircraft Fleet Details
    print("\n2. AIRCRAFT FLEET COMPOSITION")
    print("-"*40)
    widebody_count = sum(1 for a in env.aircraft if a.type == AircraftType.WIDEBODY)
    narrowbody_count = sum(1 for a in env.aircraft if a.type == AircraftType.NARROWBODY)
    regional_count = sum(1 for a in env.aircraft if a.type == AircraftType.REGIONAL)
    print(f"  Wide-body aircraft: {widebody_count} (300 passengers, 12-hour range)")
    print(f"  Narrow-body aircraft: {narrowbody_count} (150 passengers, 6-hour range)")
    print(f"  Regional jets: {regional_count} (70 passengers, 3-hour range)")
    
    # 3. Operations Module Features
    print("\n3. OPERATIONS MODULE CAPABILITIES")
    print("-"*40)
    data_gen = FlightDataGenerator()
    
    # Generate sample route network
    routes = data_gen.generate_route_network(15)
    print(f"✓ Generated {len(routes)} routes in hub-and-spoke network")
    
    # Calculate passenger demand
    demand = data_gen.generate_passenger_demand(
        hour=14.0, 
        route_type=RouteType.DOMESTIC,
        season=SeasonalPattern.BUSINESS_PEAK
    )
    print(f"✓ Passenger demand calculation: {demand} passengers (2PM, Business Peak)")
    
    # Generate disruption scenario
    disruption = data_gen.generate_disruption_scenario()
    print(f"✓ Generated disruption: {disruption.get('type', 'unknown')}")
    
    # Calculate fuel requirement
    fuel = data_gen.calculate_fuel_requirement(1000, 'narrowbody', 1.2)
    print(f"✓ Fuel requirement for 1000km flight: {fuel:.0f} kg")
    
    # Calculate delay cost
    delay_cost = data_gen.calculate_delay_cost(120, 150, True)
    print(f"✓ Cost of 2-hour delay (150 passengers, hub): ${delay_cost:,.0f}")
    
    # 4. Weather System Simulation
    print("\n4. WEATHER SYSTEM SIMULATION")
    print("-"*40)
    env._generate_weather_systems()
    print(f"✓ Active weather systems: {len(env.weather_systems)}")
    for i, system in enumerate(env.weather_systems):
        print(f"  System {i+1}: Severity={system['severity'].name}, Radius={system['radius']:.0f}km")
    
    # 5. Action Demonstrations
    print("\n5. ACTION SPACE DEMONSTRATIONS")
    print("-"*40)
    action_descriptions = {
        5: "Reassigning aircraft #5 to new route",
        27: "Cancelling flight (passenger rebooking)",
        32: "Delaying departure for weather",
        37: "Requesting priority handling at hub",
        40: "Activating irregular operations mode",
        41: "Deploying spare aircraft from reserve",
        44: "Normal operations mode"
    }
    
    for action, description in action_descriptions.items():
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Action {action}: {description} → Reward: {reward:+.0f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # 6. Performance Metrics
    print("\n6. PERFORMANCE METRICS TRACKING")
    print("-"*40)
    metrics = PerformanceMetrics()
    
    # Simulate some metrics
    sample_data = {
        'otp': metrics.calculate_otp(92, 100),
        'completion': metrics.calculate_completion_factor(97, 100),
        'rasm': metrics.calculate_revenue_per_asm(50000, 150, 1000),
        'casm': metrics.calculate_cost_per_asm(45000, 150, 1000),
        'yield': metrics.calculate_yield(50000, 140, 1000),
        'utilization': metrics.calculate_utilization(10, 14)
    }
    
    print(f"  On-Time Performance: {sample_data['otp']:.1f}%")
    print(f"  Completion Factor: {sample_data['completion']:.1f}%")
    print(f"  Revenue per ASM: ${sample_data['rasm']:.3f}")
    print(f"  Cost per ASM: ${sample_data['casm']:.3f}")
    print(f"  Yield: ${sample_data['yield']:.3f}")
    print(f"  Aircraft Utilization: {sample_data['utilization']:.1f}%")
    
    # 7. Visualization Running
    print("\n7. REAL-TIME VISUALIZATION")
    print("-"*40)
    print("✓ Pygame visualization active showing:")
    print("  - Airport network with hub-and-spoke routes")
    print("  - Real-time aircraft positions and movements")
    print("  - Weather systems and their impacts")
    print("  - Operations dashboard with KPIs")
    print("  - Alert system for critical situations")
    print("  - Financial performance tracker")
    
    # 8. Run simulation for visual demonstration
    print("\n8. RUNNING LIVE SIMULATION")
    print("-"*40)
    print("Simulating 100 time steps...")
    
    for step in range(100):
        # Select intelligent action based on conditions
        if info['on_time_performance'] < 0.80:
            action = np.random.randint(0, 25)  # Reassign aircraft
        elif info['passenger_satisfaction'] < 0.75:
            action = 42  # Provide accommodations
        elif len(env.weather_systems) > 2:
            action = 40  # Irregular operations
        else:
            action = 44  # Normal operations
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: OTP={info['on_time_performance']:5.1%}, "
                  f"Satisfaction={info['passenger_satisfaction']:5.1%}, "
                  f"Hour={info['current_hour']:5.1f}, "
                  f"Revenue=${info['daily_revenue']:8,.0f}")
        
        if terminated or truncated:
            print(f"  Episode terminated at step {step}")
            break
        
        time.sleep(0.05)  # Small delay for visualization
    
    # 9. Episode Summary
    print("\n9. EPISODE SUMMARY")
    print("-"*40)
    print(f"  Final OTP: {info['on_time_performance']:.1%}")
    print(f"  Final Satisfaction: {info['passenger_satisfaction']:.1%}")
    print(f"  Total Revenue: ${info['daily_revenue']:,.0f}")
    print(f"  Total Costs: ${info['daily_costs']:,.0f}")
    print(f"  Net Profit/Loss: ${info['daily_revenue'] - info['daily_costs']:+,.0f}")
    
    # Clean up
    env.close()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe Airline Operations Environment provides:")
    print("• Realistic airline network simulation")
    print("• Complex operational decision-making")
    print("• Dynamic weather and disruption modeling")
    print("• Comprehensive performance metrics")
    print("• Rich visualization for monitoring")
    print("• Suitable for RL agent training")
    print("\nRun 'python test_airline.py' for comprehensive scenario testing")


if __name__ == "__main__":
    try:
        demonstrate_features()
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
