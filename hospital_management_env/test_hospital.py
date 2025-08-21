"""
Test Hospital Management Environment
Runs multiple scenarios to test the hospital simulation and management strategies.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from hospital_env import HospitalManagementEnv

class HospitalTestRunner:
    """Test runner for hospital management scenarios"""
    
    def __init__(self, render=True):
        self.render = render
        self.results = {}
        
    def run_scenario(self, scenario_name, policy_fn, episode_length=1440, 
                     special_conditions=None):
        """Run a single test scenario"""
        print(f"\n{'='*60}")
        print(f"Running Scenario: {scenario_name}")
        print(f"{'='*60}")
        
        # Create environment
        env = HospitalManagementEnv(render_mode="human" if self.render else None)
        
        # Apply special conditions if specified
        if special_conditions:
            self._apply_conditions(env, special_conditions)
        
        # Episode tracking
        total_reward = 0
        rewards = []
        deaths = []
        patients_treated = []
        wait_times = []
        utilizations = []
        
        # Run episode
        obs, info = env.reset()
        
        for step in range(episode_length):
            # Get action from policy
            action = policy_fn(obs, env)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track metrics
            total_reward += reward
            rewards.append(reward)
            deaths.append(info['deaths'])
            patients_treated.append(info['patients_treated'])
            wait_times.append(info['avg_wait_time'])
            
            # Calculate average utilization
            avg_util = sum(env.dept_utilization.values()) / len(env.dept_utilization)
            utilizations.append(avg_util)
            
            # Render if enabled
            if self.render:
                env.render()
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}/{episode_length} - Reward: {total_reward:.2f}, "
                      f"Deaths: {info['deaths']}, Treated: {info['patients_treated']}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        # Store results
        self.results[scenario_name] = {
            'total_reward': total_reward,
            'final_deaths': deaths[-1] if deaths else 0,
            'final_treated': patients_treated[-1] if patients_treated else 0,
            'avg_wait_time': np.mean(wait_times) if wait_times else 0,
            'avg_utilization': np.mean(utilizations) if utilizations else 0,
            'rewards': rewards,
            'deaths': deaths,
            'patients_treated': patients_treated,
            'wait_times': wait_times,
            'utilizations': utilizations
        }
        
        # Print summary
        print(f"\nScenario Summary:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Deaths: {self.results[scenario_name]['final_deaths']}")
        print(f"  Patients Treated: {self.results[scenario_name]['final_treated']}")
        print(f"  Average Wait Time: {self.results[scenario_name]['avg_wait_time']:.2f} minutes")
        print(f"  Average Utilization: {self.results[scenario_name]['avg_utilization']:.2%}")
        
        env.close()
        return self.results[scenario_name]
    
    def _apply_conditions(self, env, conditions):
        """Apply special conditions to the environment"""
        if 'outbreak' in conditions:
            env.outbreak_active = True
            env.outbreak_type = conditions['outbreak']
        
        if 'mass_casualty' in conditions:
            env.mass_casualty_event = True
            # Generate initial casualties
            for _ in range(conditions.get('casualty_count', 10)):
                from hospital_env import Patient, Department
                patient = Patient(
                    id=env.next_patient_id,
                    severity=np.random.randint(3, 6),
                    arrival_time=env.current_time,
                    treatment_time=np.random.randint(45, 120),
                    disease_type="trauma"
                )
                env.next_patient_id += 1
                env.patient_queues[Department.EMERGENCY].append(patient)
        
        if 'staff_shortage' in conditions:
            # Reduce available staff
            reduction = conditions.get('staff_reduction', 0.3)
            env.doctors = env.doctors[:int(len(env.doctors) * (1 - reduction))]
            env.nurses = env.nurses[:int(len(env.nurses) * (1 - reduction))]
        
        if 'equipment_failure' in conditions:
            # Fail some equipment
            for eq in conditions.get('failed_equipment', []):
                if eq in env.equipment_status:
                    env.equipment_status[eq] = 0.0
    
    def plot_results(self):
        """Generate plots comparing scenarios"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Hospital Management Scenario Comparison', fontsize=16)
        
        scenarios = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
        
        # Plot 1: Cumulative Rewards
        ax = axes[0, 0]
        for i, scenario in enumerate(scenarios):
            cumulative_rewards = np.cumsum(self.results[scenario]['rewards'])
            ax.plot(cumulative_rewards, label=scenario, color=colors[i], alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Rewards Over Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Deaths Over Time
        ax = axes[0, 1]
        for i, scenario in enumerate(scenarios):
            ax.plot(self.results[scenario]['deaths'], label=scenario, 
                   color=colors[i], alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Deaths')
        ax.set_title('Cumulative Deaths')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Patients Treated
        ax = axes[0, 2]
        for i, scenario in enumerate(scenarios):
            ax.plot(self.results[scenario]['patients_treated'], 
                   label=scenario, color=colors[i], alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Patients Treated')
        ax.set_title('Patients Treated Over Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Average Wait Times
        ax = axes[1, 0]
        for i, scenario in enumerate(scenarios):
            # Smooth wait times with moving average
            wait_times = self.results[scenario]['wait_times']
            if len(wait_times) > 10:
                smoothed = np.convolve(wait_times, np.ones(10)/10, mode='valid')
                ax.plot(smoothed, label=scenario, color=colors[i], alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Average Wait Time (minutes)')
        ax.set_title('Average Patient Wait Times')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Utilization
        ax = axes[1, 1]
        for i, scenario in enumerate(scenarios):
            ax.plot(self.results[scenario]['utilizations'], 
                   label=scenario, color=colors[i], alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Hospital Capacity Utilization')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Final Metrics Comparison
        ax = axes[1, 2]
        metrics = ['final_deaths', 'final_treated', 'avg_wait_time']
        metric_names = ['Deaths', 'Treated', 'Avg Wait']
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            if metric == 'avg_wait_time':
                values = [self.results[s][metric]/10 for s in scenarios]  # Scale for visibility
            else:
                values = [self.results[s][metric] for s in scenarios]
            ax.bar(x + i*width, values, width, label=metric_names[i])
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Value')
        ax.set_title('Final Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('hospital_scenario_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nPlots saved to 'hospital_scenario_comparison.png'")


# Define different management policies

def reactive_policy(obs, env):
    """Reactive policy - responds to immediate crises"""
    # Check for critical patients
    critical_count = 0
    for queue in env.patient_queues.values():
        critical_count += sum(1 for p in queue if p.severity >= 4)
    
    if critical_count > 5:
        return 33  # Activate emergency protocol
    elif critical_count > 3:
        return np.random.randint(0, 6)  # Assign nurse to random department
    
    # Check staff fatigue
    high_fatigue = sum(1 for d in env.doctors if d.fatigue > 80)
    if high_fatigue > 5:
        return 30  # Call additional staff
    
    # Check bed occupancy
    occupied_beds = sum(1 for b in env.beds if b.occupied)
    if occupied_beds / len(env.beds) > 0.9:
        return 31  # Discharge stable patients
    
    # Default action
    return 34  # Normal operation


def predictive_policy(obs, env):
    """Predictive policy - anticipates future needs"""
    # Analyze trends
    if env.current_time % 60 == 0:  # Every hour
        # Check utilization trend
        avg_util = sum(env.dept_utilization.values()) / len(env.dept_utilization)
        if avg_util > 0.7:
            # Preemptively call staff
            return 30
    
    # Maintain equipment
    if env.current_time % 120 == 0:  # Every 2 hours
        # Schedule maintenance
        return np.random.randint(18, 24)
    
    # Supply management
    low_meds = [m for m, c in env.medicine_inventory.items() if c < 30]
    if low_meds and env.current_time % 30 == 0:
        return np.random.randint(24, 30)  # Order supplies
    
    # Department optimization
    max_wait_dept = max(env.dept_wait_times, key=env.dept_wait_times.get)
    if env.dept_wait_times[max_wait_dept] > 30:
        return max_wait_dept.value  # Assign nurse to busiest department
    
    # Default to reactive actions for immediate needs
    return reactive_policy(obs, env)


def balanced_policy(obs, env):
    """Balanced policy - combines reactive and predictive strategies"""
    # 30% chance of predictive action
    if np.random.random() < 0.3:
        return predictive_policy(obs, env)
    else:
        return reactive_policy(obs, env)


def random_policy(obs, env):
    """Random policy - baseline for comparison"""
    return env.action_space.sample()


def optimal_heuristic_policy(obs, env):
    """Optimal heuristic policy - uses domain knowledge"""
    # Priority system
    priorities = []
    
    # Priority 1: Critical patients
    for dept, queue in env.patient_queues.items():
        critical = sum(1 for p in queue if p.severity == 5)
        if critical > 0:
            priorities.append((1, dept.value, critical))
    
    # Priority 2: High wait times
    for dept, wait_time in env.dept_wait_times.items():
        if wait_time > 45:
            priorities.append((2, dept.value, wait_time))
    
    # Priority 3: Equipment maintenance
    for i, eq in enumerate(env.equipment_types):
        if env.equipment_status[eq] < 0.3:
            priorities.append((3, 18 + i, env.equipment_status[eq]))
    
    # Priority 4: Low supplies
    for i, (med, count) in enumerate(env.medicine_inventory.items()):
        if count < 20:
            priorities.append((4, 24 + (i % 6), count))
    
    # Sort by priority
    priorities.sort(key=lambda x: (x[0], -x[2]))
    
    if priorities:
        # Take highest priority action
        priority_type, action_base, _ = priorities[0]
        
        if priority_type == 1:  # Critical patient
            # Assign resources to that department
            if env.current_time % 3 == 0:
                return action_base  # Assign nurse
            elif env.current_time % 3 == 1:
                return 6 + (action_base % 6)  # Reassign doctor
            else:
                return 33  # Emergency protocol
        
        elif priority_type == 2:  # High wait time
            return action_base  # Assign nurse to department
        
        elif priority_type == 3:  # Equipment issue
            return action_base  # Maintenance action
        
        elif priority_type == 4:  # Low supplies
            return action_base  # Order supplies
    
    # Check overall capacity
    avg_util = sum(env.dept_utilization.values()) / len(env.dept_utilization)
    if avg_util > 0.85:
        if env.current_time % 2 == 0:
            return 31  # Discharge stable
        else:
            return 30  # Call staff
    
    # Default maintenance
    if env.current_time % 180 == 0:
        return np.random.randint(18, 24)  # Regular maintenance
    
    return 34  # Normal operation


def main():
    """Main test execution"""
    print("Hospital Management Environment Test Suite")
    print("==========================================\n")
    
    # Create test runner
    runner = HospitalTestRunner(render=True)
    
    # Scenario 1: Normal Day
    print("\n1. NORMAL DAY SCENARIO")
    runner.run_scenario(
        "Normal Day",
        reactive_policy,
        episode_length=500
    )
    time.sleep(1)
    
    # Scenario 2: Mass Casualty Event
    print("\n2. MASS CASUALTY SCENARIO")
    runner.run_scenario(
        "Mass Casualty",
        optimal_heuristic_policy,
        episode_length=500,
        special_conditions={
            'mass_casualty': True,
            'casualty_count': 15
        }
    )
    time.sleep(1)
    
    # Scenario 3: Flu Outbreak
    print("\n3. FLU OUTBREAK SCENARIO")
    runner.run_scenario(
        "Flu Outbreak",
        predictive_policy,
        episode_length=500,
        special_conditions={
            'outbreak': 'flu'
        }
    )
    time.sleep(1)
    
    # Scenario 4: Staff Shortage
    print("\n4. STAFF SHORTAGE SCENARIO")
    runner.run_scenario(
        "Staff Shortage",
        balanced_policy,
        episode_length=500,
        special_conditions={
            'staff_shortage': True,
            'staff_reduction': 0.3
        }
    )
    time.sleep(1)
    
    # Scenario 5: Equipment Failure
    print("\n5. EQUIPMENT FAILURE SCENARIO")
    runner.run_scenario(
        "Equipment Failure",
        optimal_heuristic_policy,
        episode_length=500,
        special_conditions={
            'equipment_failure': True,
            'failed_equipment': ['ventilator', 'xray', 'mri']
        }
    )
    
    # Generate comparison plots
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE COMPARISON PLOTS")
    print("="*60)
    runner.plot_results()
    
    # Generate report
    print("\n" + "="*60)
    print("FINAL PERFORMANCE REPORT")
    print("="*60)
    
    for scenario, results in runner.results.items():
        print(f"\n{scenario}:")
        print(f"  Total Reward: {results['total_reward']:.2f}")
        print(f"  Mortality Rate: {results['final_deaths'] / max(results['final_treated'] + results['final_deaths'], 1):.2%}")
        print(f"  Efficiency Score: {results['final_treated'] / max(results['avg_wait_time'], 1):.2f}")
        print(f"  Resource Utilization: {results['avg_utilization']:.2%}")
    
    # Save results to file
    with open('hospital_test_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for scenario, results in runner.results.items():
            json_results[scenario] = {
                'total_reward': float(results['total_reward']),
                'final_deaths': int(results['final_deaths']),
                'final_treated': int(results['final_treated']),
                'avg_wait_time': float(results['avg_wait_time']),
                'avg_utilization': float(results['avg_utilization'])
            }
        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to 'hospital_test_results.json'")
    print("\nTest suite completed successfully!")


if __name__ == "__main__":
    main()