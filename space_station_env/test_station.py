#!/usr/bin/env python3
"""
Space Station Environment Testing and Visualization

Tests 6 different scenarios:
1. Normal operations
2. Power failure
3. Atmospheric leak
4. Medical emergency
5. Solar storm
6. System cascade failure

Compares manual control vs automated life support management.
"""

import gymnasium as gym
import numpy as np
import time
import pygame
from typing import Dict, List, Tuple
import argparse
import json
from datetime import datetime

# Import the environment
from station_env import SpaceStationEnv


class ManualController:
    """Manual control interface for the space station"""
    
    def __init__(self):
        self.action_descriptions = {
            # System prioritization (0-11)
            0: "Prioritize Oxygen Generation",
            1: "Prioritize CO2 Scrubbing",
            2: "Prioritize Water Recycling",
            3: "Prioritize Power Grid",
            4: "Prioritize Thermal Control",
            5: "Prioritize Atmospheric Pressure",
            6: "Prioritize Nitrogen Supply",
            7: "Prioritize Waste Processing",
            8: "Prioritize Fire Suppression",
            9: "Prioritize Radiation Shielding",
            10: "Prioritize Communications",
            11: "Prioritize Artificial Gravity",
            
            # Maintenance (12-23)
            12: "Maintain Oxygen Generation",
            13: "Maintain CO2 Scrubbing",
            14: "Maintain Water Recycling",
            15: "Maintain Power Grid",
            16: "Maintain Thermal Control",
            17: "Maintain Atmospheric Pressure",
            18: "Maintain Nitrogen Supply",
            19: "Maintain Waste Processing",
            20: "Maintain Fire Suppression",
            21: "Maintain Radiation Shielding",
            22: "Maintain Communications",
            23: "Maintain Artificial Gravity",
            
            # Crew assignments (24-29)
            24: "Assign Commander to emergency",
            25: "Assign Engineer to emergency",
            26: "Assign Scientist to emergency",
            27: "Assign Medical Officer to emergency",
            28: "Assign Pilot to emergency",
            29: "Assign Mission Specialist to emergency",
            
            # Environmental controls (30-35)
            30: "Adjust Command module environment",
            31: "Adjust Living Quarters environment",
            32: "Adjust Laboratory environment",
            33: "Adjust Engineering environment",
            34: "Adjust Hydroponics environment",
            35: "Adjust Storage environment",
            
            # Emergency actions (36-39)
            36: "Activate emergency protocols",
            37: "Switch to backup power",
            38: "Initiate evacuation",
            39: "Normal operations"
        }
        
    def get_action(self, observation, info, event=None):
        """Get manual action based on keyboard input"""
        if event and event.type == pygame.KEYDOWN:
            # Number keys for quick actions
            if pygame.K_1 <= event.key <= pygame.K_9:
                return event.key - pygame.K_1
            elif event.key == pygame.K_0:
                return 9
            
            # Function keys for maintenance
            elif pygame.K_F1 <= event.key <= pygame.K_F12:
                return 12 + (event.key - pygame.K_F1)
            
            # Letter keys for emergency actions
            elif event.key == pygame.K_e:
                return 36  # Emergency protocols
            elif event.key == pygame.K_b:
                return 37  # Backup power
            elif event.key == pygame.K_v:
                return 38  # Evacuation
            elif event.key == pygame.K_n:
                return 39  # Normal operations
        
        return 39  # Default to normal operations


class AutomatedController:
    """AI-based automated life support controller"""
    
    def __init__(self):
        self.emergency_threshold = 0.3
        self.maintenance_threshold = 50.0
        self.critical_systems = [
            'Oxygen Generation', 'CO2 Scrubbing', 'Power Grid',
            'Atmospheric Pressure', 'Thermal Control'
        ]
        
    def get_action(self, observation, info):
        """Intelligent action selection based on current state"""
        # Parse observation
        crew_health = observation[2::4][:6]  # Every 4th element starting from index 2
        system_efficiency = observation[24::3][:12]  # System efficiencies
        battery_level = observation[89]  # Battery level
        emergency_alerts = observation[104:110]  # Emergency flags
        
        # Check for emergencies
        if any(emergency_alerts > 0.5):
            if emergency_alerts[3] > 0.5:  # Power failure
                return 37  # Backup power
            elif emergency_alerts[4] > 0.5:  # Life support critical
                return 36  # Emergency protocols
            elif emergency_alerts[1] > 0.5:  # Depressurization
                return 5  # Prioritize atmospheric pressure
            elif emergency_alerts[5] > 0.5:  # Medical emergency
                return 27  # Assign medical officer
            elif emergency_alerts[2] > 0.5:  # Radiation storm
                return 9  # Prioritize radiation shielding
            else:
                return 36  # General emergency protocols
        
        # Check critical systems
        critical_indices = [0, 1, 3, 5, 4]  # Indices of critical systems
        for i, idx in enumerate(critical_indices):
            if system_efficiency[idx] < 50:
                if system_efficiency[idx] < 30:
                    return idx  # Prioritize power to critical system
                else:
                    return 12 + idx  # Schedule maintenance
        
        # Check battery level
        if battery_level < 20:
            return 37  # Switch to backup power
        elif battery_level < 40:
            return 3  # Prioritize power grid
        
        # Check crew health
        min_health = np.min(crew_health)
        if min_health < 50:
            return 27  # Assign medical officer
        
        # Routine maintenance
        for i, efficiency in enumerate(system_efficiency):
            if efficiency < self.maintenance_threshold:
                return 12 + i  # Schedule maintenance
        
        # Default to normal operations
        return 39


class ScenarioTester:
    """Test different emergency scenarios"""
    
    def __init__(self, env: SpaceStationEnv):
        self.env = env
        self.scenarios = {
            'normal': self.normal_operations,
            'power_failure': self.power_failure_scenario,
            'atmospheric_leak': self.atmospheric_leak_scenario,
            'medical_emergency': self.medical_emergency_scenario,
            'solar_storm': self.solar_storm_scenario,
            'cascade_failure': self.cascade_failure_scenario
        }
        
    def normal_operations(self):
        """Scenario 1: Normal operations for baseline"""
        print("\n=== SCENARIO 1: NORMAL OPERATIONS ===")
        print("Testing baseline performance with no emergencies...")
        # No special setup needed
        return {}
    
    def power_failure_scenario(self):
        """Scenario 2: Major power grid failure"""
        print("\n=== SCENARIO 2: POWER FAILURE ===")
        print("Simulating power grid failure at 50% efficiency...")
        
        # Reduce power grid efficiency
        self.env.systems['Power Grid'].efficiency = 20.0
        self.env.battery_level = 30.0
        self.env.emergency_alerts['power_failure'] = True
        
        return {'power_failure': True}
    
    def atmospheric_leak_scenario(self):
        """Scenario 3: Micrometeorite causes depressurization"""
        print("\n=== SCENARIO 3: ATMOSPHERIC LEAK ===")
        print("Simulating hull breach in Laboratory module...")
        
        # Create depressurization in Laboratory
        self.env.modules['Laboratory'].pressure = 85.0
        self.env.modules['Laboratory'].oxygen_percent = 17.0
        self.env.emergency_alerts['depressurization'] = True
        
        # Damage atmospheric systems
        self.env.systems['Atmospheric Pressure'].efficiency = 40.0
        
        return {'depressurization': True}
    
    def medical_emergency_scenario(self):
        """Scenario 4: Crew member medical emergency"""
        print("\n=== SCENARIO 4: MEDICAL EMERGENCY ===")
        print("Simulating crew member injury...")
        
        # Injure the Engineer
        self.env.crew[1].health = 35.0
        self.env.crew[1].oxygen_level = 60.0
        self.env.emergency_alerts['medical_emergency'] = True
        
        # Reduce medical supplies
        self.env.resources['medical_supplies'] = 30.0
        
        return {'medical_emergency': True}
    
    def solar_storm_scenario(self):
        """Scenario 5: Intense solar radiation event"""
        print("\n=== SCENARIO 5: SOLAR STORM ===")
        print("Simulating severe solar radiation storm...")
        
        # Increase radiation levels
        self.env.radiation_level = 0.85
        self.env.emergency_alerts['radiation_storm'] = True
        
        # Affect radiation shielding
        self.env.systems['Radiation Shielding'].efficiency = 60.0
        
        # Communications disrupted
        self.env.systems['Communications'].efficiency = 20.0
        
        return {'radiation_storm': True}
    
    def cascade_failure_scenario(self):
        """Scenario 6: Multiple system cascade failure"""
        print("\n=== SCENARIO 6: CASCADE FAILURE ===")
        print("Simulating multiple interdependent system failures...")
        
        # Start with water recycling failure
        self.env.systems['Water Recycling'].efficiency = 10.0
        
        # This affects oxygen generation
        self.env.systems['Oxygen Generation'].efficiency = 30.0
        
        # CO2 builds up
        self.env.systems['CO2 Scrubbing'].efficiency = 25.0
        
        # Power strain
        self.env.systems['Power Grid'].efficiency = 45.0
        self.env.battery_level = 25.0
        
        # Set multiple alerts
        self.env.emergency_alerts['life_support_critical'] = True
        self.env.emergency_alerts['power_failure'] = True
        
        # Reduce resources
        self.env.resources['water'] = 200.0
        self.env.resources['spare_parts'] = 10.0
        
        return {'cascade_failure': True}
    
    def run_scenario(self, scenario_name: str, controller, max_steps: int = 500, 
                     render: bool = True) -> Dict:
        """Run a specific scenario and collect metrics"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Reset environment
        observation, info = self.env.reset()
        
        # Apply scenario modifications
        scenario_func = self.scenarios[scenario_name]
        scenario_data = scenario_func()
        
        # Metrics collection
        metrics = {
            'scenario': scenario_name,
            'total_reward': 0,
            'steps_survived': 0,
            'crew_casualties': 0,
            'system_failures': [],
            'min_crew_health': 100,
            'avg_crew_health': [],
            'battery_levels': [],
            'emergency_resolved': False,
            'mission_success': False
        }
        
        # Run simulation
        terminated = False
        truncated = False
        step = 0
        
        clock = pygame.time.Clock() if render else None
        
        while not terminated and not truncated and step < max_steps:
            # Get action from controller
            if isinstance(controller, ManualController):
                # Handle pygame events for manual control
                action = 39  # Default
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return metrics
                    action = controller.get_action(observation, info, event)
            else:
                action = controller.get_action(observation, info)
            
            # Step environment
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Update metrics
            metrics['total_reward'] += reward
            metrics['steps_survived'] = step
            metrics['avg_crew_health'].append(info['avg_crew_health'])
            metrics['battery_levels'].append(info['battery_level'])
            metrics['min_crew_health'] = min(metrics['min_crew_health'], info['avg_crew_health'])
            
            # Render if requested
            if render:
                self.env.render()
                if clock:
                    clock.tick(30)
            
            step += 1
            
            # Check if emergency resolved
            if scenario_data and not any(self.env.emergency_alerts.values()):
                metrics['emergency_resolved'] = True
        
        # Final metrics
        metrics['crew_casualties'] = self.env.crew_casualties
        metrics['system_failures'] = self.env.system_failures
        metrics['mission_success'] = self.env.mission_success
        metrics['final_day'] = info['day']
        
        return metrics


def generate_report(results: List[Dict]) -> str:
    """Generate a comprehensive mission report"""
    report = []
    report.append("\n" + "="*80)
    report.append("SPACE STATION MISSION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80 + "\n")
    
    for result in results:
        report.append(f"\nScenario: {result['scenario'].upper()}")
        report.append("-" * 40)
        report.append(f"Controller: {result.get('controller', 'Unknown')}")
        report.append(f"Total Reward: {result['total_reward']:.2f}")
        report.append(f"Steps Survived: {result['steps_survived']}")
        report.append(f"Days Survived: {result['final_day']}")
        report.append(f"Crew Casualties: {result['crew_casualties']}")
        report.append(f"System Failures: {len(result['system_failures'])}")
        if result['system_failures']:
            report.append(f"  Failed Systems: {', '.join(result['system_failures'])}")
        report.append(f"Min Crew Health: {result['min_crew_health']:.1f}%")
        report.append(f"Avg Crew Health: {np.mean(result['avg_crew_health']):.1f}%")
        report.append(f"Avg Battery Level: {np.mean(result['battery_levels']):.1f}%")
        report.append(f"Emergency Resolved: {result['emergency_resolved']}")
        report.append(f"Mission Success: {result['mission_success']}")
        
        # Performance rating
        score = 0
        if result['crew_casualties'] == 0:
            score += 40
        if result['emergency_resolved']:
            score += 30
        if result['min_crew_health'] > 50:
            score += 20
        if np.mean(result['battery_levels']) > 40:
            score += 10
        
        rating = "EXCELLENT" if score >= 90 else "GOOD" if score >= 70 else "FAIR" if score >= 50 else "POOR"
        report.append(f"Performance Rating: {rating} ({score}/100)")
    
    report.append("\n" + "="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return "\n".join(report)


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test Space Station Environment")
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['all', 'normal', 'power_failure', 'atmospheric_leak',
                               'medical_emergency', 'solar_storm', 'cascade_failure'],
                       help='Scenario to test')
    parser.add_argument('--controller', type=str, default='auto',
                       choices=['auto', 'manual', 'both'],
                       help='Controller type to use')
    parser.add_argument('--steps', type=int, default=500,
                       help='Maximum steps per scenario')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--save-report', type=str, default='mission_report.txt',
                       help='Save report to file')
    
    args = parser.parse_args()
    
    # Initialize pygame
    if not args.no_render:
        pygame.init()
    
    # Create environment
    env = SpaceStationEnv(render_mode="human" if not args.no_render else None)
    tester = ScenarioTester(env)
    
    # Determine scenarios to run
    if args.scenario == 'all':
        scenarios = ['normal', 'power_failure', 'atmospheric_leak',
                    'medical_emergency', 'solar_storm', 'cascade_failure']
    else:
        scenarios = [args.scenario]
    
    # Determine controllers to use
    controllers = []
    if args.controller == 'both':
        controllers = [('auto', AutomatedController()), ('manual', ManualController())]
    elif args.controller == 'auto':
        controllers = [('auto', AutomatedController())]
    else:
        controllers = [('manual', ManualController())]
    
    # Run tests
    results = []
    
    for scenario in scenarios:
        for controller_name, controller in controllers:
            print(f"\n{'='*60}")
            print(f"Testing: {scenario} with {controller_name} controller")
            print('='*60)
            
            if controller_name == 'manual' and not args.no_render:
                print("\nMANUAL CONTROL KEYS:")
                print("1-9, 0: Prioritize systems 1-10")
                print("F1-F12: Schedule maintenance")
                print("E: Emergency protocols")
                print("B: Backup power")
                print("V: Evacuation")
                print("N: Normal operations")
                print("\nPress any key to start...")
                
                # Wait for user to be ready
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            waiting = False
                        elif event.type == pygame.QUIT:
                            pygame.quit()
                            return
            
            # Run scenario
            result = tester.run_scenario(
                scenario, 
                controller,
                max_steps=args.steps,
                render=not args.no_render
            )
            result['controller'] = controller_name
            results.append(result)
            
            # Short pause between scenarios
            if not args.no_render:
                time.sleep(1)
    
    # Generate and display report
    report = generate_report(results)
    print(report)
    
    # Save report if requested
    if args.save_report:
        with open(args.save_report, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {args.save_report}")
    
    # Save detailed metrics as JSON
    metrics_file = args.save_report.replace('.txt', '_metrics.json')
    with open(metrics_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for result in results:
            result['avg_crew_health'] = [float(x) for x in result['avg_crew_health']]
            result['battery_levels'] = [float(x) for x in result['battery_levels']]
        json.dump(results, f, indent=2)
    print(f"Detailed metrics saved to {metrics_file}")
    
    # Cleanup
    env.close()
    if not args.no_render:
        pygame.quit()


if __name__ == "__main__":
    main()