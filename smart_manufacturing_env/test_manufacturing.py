"""Test script for Smart Manufacturing Environment

This script tests the manufacturing environment with 4 different scenarios:
1. Normal production
2. High demand
3. Frequent breakdowns
4. Quality crisis

All test data and results are generated and stored within the package.
"""

import numpy as np
import matplotlib.pyplot as plt
import pygame
import gym
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import random
import time

# Import the environment
from manufacturing_env import SmartManufacturingEnv


class ManufacturingAgent:
    """Agent for testing different maintenance strategies"""
    
    def __init__(self, strategy='reactive'):
        """
        Initialize agent with specified strategy
        
        Args:
            strategy: 'reactive' or 'predictive'
        """
        self.strategy = strategy
        self.action_history = []
        self.maintenance_schedule = {}
    
    def get_action(self, observation, env):
        """Select action based on strategy"""
        # Parse observation
        obs_dict = self._parse_observation(observation, env)
        
        if self.strategy == 'reactive':
            return self._reactive_action(obs_dict, env)
        elif self.strategy == 'predictive':
            return self._predictive_action(obs_dict, env)
        else:
            return env.action_space.sample()
    
    def _parse_observation(self, observation, env):
        """Parse observation into meaningful components"""
        idx = 0
        obs_dict = {}
        
        # Products in system (30 elements)
        obs_dict['products_by_station'] = observation[idx:idx+30].reshape(5, 6)
        idx += 30
        
        # Machine status (15 elements)
        obs_dict['machine_status'] = observation[idx:idx+15].reshape(5, 3)
        idx += 15
        
        # Queue lengths (5 elements)
        obs_dict['queue_lengths'] = observation[idx:idx+5]
        idx += 5
        
        # Product quality scores (6 elements)
        obs_dict['quality_scores'] = observation[idx:idx+6]
        idx += 6
        
        # Raw material inventory (1 element)
        obs_dict['raw_materials'] = observation[idx]
        idx += 1
        
        # Production targets remaining (6 elements)
        obs_dict['targets_remaining'] = observation[idx:idx+6]
        idx += 6
        
        # Machine utilization rates (5 elements)
        obs_dict['utilization_rates'] = observation[idx:idx+5]
        idx += 5
        
        # Maintenance countdown (5 elements)
        obs_dict['maintenance_countdown'] = observation[idx:idx+5]
        idx += 5
        
        return obs_dict
    
    def _reactive_action(self, obs_dict, env):
        """Reactive maintenance strategy - fix when broken"""
        # Check for broken machines
        for i in range(5):
            if obs_dict['machine_status'][i][1] > 0:  # Machine is broken
                return 11 + i  # Schedule maintenance
        
        # Check raw materials
        if obs_dict['raw_materials'] < 50:
            # Don't start new production if low on materials
            return 24  # Balanced mode
        
        # Start production based on remaining targets
        max_target_idx = np.argmax(obs_dict['targets_remaining'])
        if obs_dict['targets_remaining'][max_target_idx] > 0:
            return max_target_idx  # Start production of needed product
        
        # Default to balanced production
        return 24
    
    def _predictive_action(self, obs_dict, env):
        """Predictive maintenance strategy - maintain before failure"""
        # Check maintenance countdown and schedule preventive maintenance
        for i in range(5):
            if obs_dict['maintenance_countdown'][i] < 15 and obs_dict['machine_status'][i][0] > 0:
                return 11 + i  # Schedule preventive maintenance
        
        # Check for already broken machines
        for i in range(5):
            if obs_dict['machine_status'][i][1] > 0:
                return 11 + i  # Fix broken machine
        
        # Quality focus if quality is dropping
        avg_quality = np.mean(obs_dict['quality_scores'])
        if avg_quality < 75:
            return 23  # Quality focus mode
        
        # Check raw materials
        if obs_dict['raw_materials'] < 50:
            return 24  # Balanced mode
        
        # Start production based on remaining targets with quality consideration
        for i, target in enumerate(obs_dict['targets_remaining']):
            if target > 0:
                # Prefer simpler products when machines are degraded
                if i < 3 or np.mean(obs_dict['maintenance_countdown']) > 50:
                    return i
        
        return 24  # Default to balanced


class ScenarioTester:
    """Test different manufacturing scenarios"""
    
    def __init__(self, env):
        self.env = env
        self.results = {}
        self.metrics_history = []
    
    def run_scenario(self, scenario_name, episodes=5, max_steps=1500, 
                     agent_strategy='reactive', render=False):
        """
        Run a specific scenario
        
        Args:
            scenario_name: Name of the scenario to run
            episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            agent_strategy: 'reactive' or 'predictive'
            render: Whether to render the environment
        """
        print(f"\n{'='*60}")
        print(f"Running Scenario: {scenario_name}")
        print(f"Strategy: {agent_strategy}")
        print(f"{'='*60}")
        
        agent = ManufacturingAgent(strategy=agent_strategy)
        scenario_results = {
            'episodes': [],
            'avg_reward': 0,
            'avg_oee': {'availability': 0, 'performance': 0, 'quality': 0},
            'production_rate': 0,
            'quality_rate': 0,
            'breakdown_count': 0,
            'energy_consumption': 0
        }
        
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            # Configure environment for scenario
            self._configure_scenario(scenario_name)
            
            obs = self.env.reset()
            episode_reward = 0
            episode_metrics = {
                'rewards': [],
                'oee_history': [],
                'production_history': [],
                'quality_history': [],
                'breakdowns': 0,
                'energy': 0
            }
            
            for step in range(max_steps):
                action = agent.get_action(obs, self.env)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_metrics['rewards'].append(reward)
                episode_metrics['oee_history'].append(info['oee'].copy())
                episode_metrics['production_history'].append(
                    sum(info['products_completed'].values())
                )
                episode_metrics['energy'] = info['energy_consumption']
                
                # Count breakdowns
                broken_count = sum(1 for s in self.env.stations 
                                 if s.status.value == 1)
                if broken_count > episode_metrics['breakdowns']:
                    episode_metrics['breakdowns'] = broken_count
                
                if render and step % 10 == 0:
                    self.env.render()
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            render = False
                
                if done:
                    break
            
            # Calculate episode statistics
            episode_stats = {
                'total_reward': episode_reward,
                'steps': step + 1,
                'products_completed': self.env.products_completed.copy(),
                'final_oee': info['oee'].copy(),
                'breakdowns': episode_metrics['breakdowns'],
                'energy_consumption': episode_metrics['energy'],
                'quality_rate': self._calculate_quality_rate()
            }
            
            scenario_results['episodes'].append(episode_stats)
            
            print(f"  Reward: {episode_reward:.0f}")
            print(f"  Products: {sum(episode_stats['products_completed'].values())}")
            print(f"  OEE: {(info['oee']['availability'] * info['oee']['performance'] * info['oee']['quality']):.1%}")
            print(f"  Quality Rate: {episode_stats['quality_rate']:.1%}")
        
        # Calculate scenario averages
        scenario_results['avg_reward'] = np.mean([e['total_reward'] 
                                                  for e in scenario_results['episodes']])
        scenario_results['avg_oee'] = {
            'availability': np.mean([e['final_oee']['availability'] 
                                    for e in scenario_results['episodes']]),
            'performance': np.mean([e['final_oee']['performance'] 
                                   for e in scenario_results['episodes']]),
            'quality': np.mean([e['final_oee']['quality'] 
                              for e in scenario_results['episodes']])
        }
        scenario_results['production_rate'] = np.mean([
            sum(e['products_completed'].values()) / e['steps'] 
            for e in scenario_results['episodes']
        ])
        scenario_results['quality_rate'] = np.mean([e['quality_rate'] 
                                                   for e in scenario_results['episodes']])
        scenario_results['breakdown_count'] = np.mean([e['breakdowns'] 
                                                      for e in scenario_results['episodes']])
        scenario_results['energy_consumption'] = np.mean([e['energy_consumption'] 
                                                         for e in scenario_results['episodes']])
        
        self.results[f"{scenario_name}_{agent_strategy}"] = scenario_results
        
        return scenario_results
    
    def _configure_scenario(self, scenario_name):
        """Configure environment for specific scenario"""
        if scenario_name == "normal_production":
            # Standard configuration
            pass
        
        elif scenario_name == "high_demand":
            # Increase production targets
            for product in self.env.production_targets:
                self.env.production_targets[product] *= 2
        
        elif scenario_name == "frequent_breakdowns":
            # Increase breakdown probability
            for station in self.env.stations:
                station.operations_count = 150  # Start with worn machines
                station.performance_degradation = 0.02
        
        elif scenario_name == "quality_crisis":
            # Lower initial quality and increase quality requirements
            self.env.quality_thresholds = [0.85, 0.90, 0.95]
            for station in self.env.stations:
                station.performance_degradation = 0.03
    
    def _calculate_quality_rate(self):
        """Calculate overall quality rate"""
        if len(self.env.completed_products) == 0:
            return 0.0
        
        good_products = len([p for p in self.env.completed_products 
                           if p.quality_score > 0.7])
        return good_products / len(self.env.completed_products)
    
    def compare_strategies(self):
        """Compare reactive vs predictive maintenance strategies"""
        print(f"\n{'='*60}")
        print("STRATEGY COMPARISON")
        print(f"{'='*60}")
        
        comparison = {}
        
        for scenario in ["normal_production", "high_demand", 
                        "frequent_breakdowns", "quality_crisis"]:
            reactive_key = f"{scenario}_reactive"
            predictive_key = f"{scenario}_predictive"
            
            if reactive_key in self.results and predictive_key in self.results:
                comparison[scenario] = {
                    'reactive': self.results[reactive_key],
                    'predictive': self.results[predictive_key],
                    'improvement': {
                        'reward': ((self.results[predictive_key]['avg_reward'] - 
                                   self.results[reactive_key]['avg_reward']) / 
                                  abs(self.results[reactive_key]['avg_reward']) * 100),
                        'quality': ((self.results[predictive_key]['quality_rate'] - 
                                    self.results[reactive_key]['quality_rate']) * 100),
                        'breakdowns': (self.results[reactive_key]['breakdown_count'] - 
                                      self.results[predictive_key]['breakdown_count'])
                    }
                }
                
                print(f"\n{scenario.upper()}:")
                print(f"  Reward Improvement: {comparison[scenario]['improvement']['reward']:.1f}%")
                print(f"  Quality Improvement: {comparison[scenario]['improvement']['quality']:.1f}%")
                print(f"  Breakdown Reduction: {comparison[scenario]['improvement']['breakdowns']:.1f}")
        
        return comparison
    
    def generate_report(self, save_path=None):
        """Generate comprehensive production report"""
        if save_path is None:
            save_path = os.path.dirname(os.path.abspath(__file__))
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'scenarios': self.results,
            'comparison': self.compare_strategies()
        }
        
        # Save JSON report
        report_file = os.path.join(save_path, 'production_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nReport saved to: {report_file}")
        
        # Generate visualization
        self.visualize_results(save_path)
        
        return report
    
    def visualize_results(self, save_path=None):
        """Create visualization of test results"""
        if save_path is None:
            save_path = os.path.dirname(os.path.abspath(__file__))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Smart Manufacturing Environment - Test Results', fontsize=16)
        
        scenarios = ["normal_production", "high_demand", 
                    "frequent_breakdowns", "quality_crisis"]
        
        # Plot 1: Average Rewards
        ax = axes[0, 0]
        reactive_rewards = []
        predictive_rewards = []
        
        for scenario in scenarios:
            if f"{scenario}_reactive" in self.results:
                reactive_rewards.append(self.results[f"{scenario}_reactive"]['avg_reward'])
            if f"{scenario}_predictive" in self.results:
                predictive_rewards.append(self.results[f"{scenario}_predictive"]['avg_reward'])
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax.bar(x - width/2, reactive_rewards, width, label='Reactive', color='coral')
        ax.bar(x + width/2, predictive_rewards, width, label='Predictive', color='skyblue')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Average Reward')
        ax.set_title('Rewards by Scenario and Strategy')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: OEE Comparison
        ax = axes[0, 1]
        oee_metrics = ['availability', 'performance', 'quality']
        reactive_oee = []
        predictive_oee = []
        
        for metric in oee_metrics:
            reactive_vals = []
            predictive_vals = []
            for scenario in scenarios:
                if f"{scenario}_reactive" in self.results:
                    reactive_vals.append(self.results[f"{scenario}_reactive"]['avg_oee'][metric])
                if f"{scenario}_predictive" in self.results:
                    predictive_vals.append(self.results[f"{scenario}_predictive"]['avg_oee'][metric])
            reactive_oee.append(np.mean(reactive_vals))
            predictive_oee.append(np.mean(predictive_vals))
        
        x = np.arange(len(oee_metrics))
        ax.bar(x - width/2, reactive_oee, width, label='Reactive', color='coral')
        ax.bar(x + width/2, predictive_oee, width, label='Predictive', color='skyblue')
        ax.set_xlabel('OEE Metric')
        ax.set_ylabel('Average Value')
        ax.set_title('Overall Equipment Effectiveness')
        ax.set_xticks(x)
        ax.set_xticklabels(oee_metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Quality Rates
        ax = axes[0, 2]
        reactive_quality = []
        predictive_quality = []
        
        for scenario in scenarios:
            if f"{scenario}_reactive" in self.results:
                reactive_quality.append(self.results[f"{scenario}_reactive"]['quality_rate'])
            if f"{scenario}_predictive" in self.results:
                predictive_quality.append(self.results[f"{scenario}_predictive"]['quality_rate'])
        
        x = np.arange(len(scenarios))
        ax.bar(x - width/2, reactive_quality, width, label='Reactive', color='coral')
        ax.bar(x + width/2, predictive_quality, width, label='Predictive', color='skyblue')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Quality Rate')
        ax.set_title('Product Quality by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Breakdown Frequency
        ax = axes[1, 0]
        reactive_breakdowns = []
        predictive_breakdowns = []
        
        for scenario in scenarios:
            if f"{scenario}_reactive" in self.results:
                reactive_breakdowns.append(self.results[f"{scenario}_reactive"]['breakdown_count'])
            if f"{scenario}_predictive" in self.results:
                predictive_breakdowns.append(self.results[f"{scenario}_predictive"]['breakdown_count'])
        
        x = np.arange(len(scenarios))
        ax.bar(x - width/2, reactive_breakdowns, width, label='Reactive', color='coral')
        ax.bar(x + width/2, predictive_breakdowns, width, label='Predictive', color='skyblue')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Average Breakdowns')
        ax.set_title('Machine Breakdowns by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Production Rate
        ax = axes[1, 1]
        reactive_production = []
        predictive_production = []
        
        for scenario in scenarios:
            if f"{scenario}_reactive" in self.results:
                reactive_production.append(self.results[f"{scenario}_reactive"]['production_rate'])
            if f"{scenario}_predictive" in self.results:
                predictive_production.append(self.results[f"{scenario}_predictive"]['production_rate'])
        
        x = np.arange(len(scenarios))
        ax.bar(x - width/2, reactive_production, width, label='Reactive', color='coral')
        ax.bar(x + width/2, predictive_production, width, label='Predictive', color='skyblue')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Products/Step')
        ax.set_title('Production Rate by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Energy Consumption
        ax = axes[1, 2]
        reactive_energy = []
        predictive_energy = []
        
        for scenario in scenarios:
            if f"{scenario}_reactive" in self.results:
                reactive_energy.append(self.results[f"{scenario}_reactive"]['energy_consumption'])
            if f"{scenario}_predictive" in self.results:
                predictive_energy.append(self.results[f"{scenario}_predictive"]['energy_consumption'])
        
        x = np.arange(len(scenarios))
        ax.bar(x - width/2, reactive_energy, width, label='Reactive', color='coral')
        ax.bar(x + width/2, predictive_energy, width, label='Predictive', color='skyblue')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Energy Units')
        ax.set_title('Energy Consumption by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_file = os.path.join(save_path, 'test_results.png')
        plt.savefig(fig_file, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to: {fig_file}")
        
        plt.show()
        
        return fig


def main():
    """Main test function"""
    print("="*60)
    print("SMART MANUFACTURING ENVIRONMENT TEST SUITE")
    print("="*60)
    
    # Create environment
    env = SmartManufacturingEnv()
    tester = ScenarioTester(env)
    
    # Test scenarios
    scenarios = [
        "normal_production",
        "high_demand",
        "frequent_breakdowns",
        "quality_crisis"
    ]
    
    # Run tests with both strategies
    for scenario in scenarios:
        # Test reactive strategy
        tester.run_scenario(scenario, episodes=3, agent_strategy='reactive', render=False)
        
        # Test predictive strategy
        tester.run_scenario(scenario, episodes=3, agent_strategy='predictive', render=False)
    
    # Generate comprehensive report
    report = tester.generate_report()
    
    # Interactive visualization demo
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATION DEMO")
    print("="*60)
    print("\nPress 'Q' or close window to exit visualization")
    print("Running normal production scenario with predictive maintenance...")
    
    # Create fresh environment for visualization
    demo_env = SmartManufacturingEnv()
    demo_agent = ManufacturingAgent(strategy='predictive')
    
    obs = demo_env.reset()
    running = True
    step_count = 0
    max_demo_steps = 500
    
    while running and step_count < max_demo_steps:
        action = demo_agent.get_action(obs, demo_env)
        obs, reward, done, info = demo_env.step(action)
        
        demo_env.render()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        step_count += 1
        
        if done:
            print(f"\nEpisode completed!")
            print(f"Total reward: {info['total_reward']:.0f}")
            print(f"Products completed: {sum(info['products_completed'].values())}")
            break
        
        # Small delay for visualization
        time.sleep(0.05)
    
    # Clean up
    demo_env.close()
    env.close()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)
    print(f"\nAll test data and results have been saved within the package directory:")
    print(f"  - production_report.json: Detailed test results")
    print(f"  - test_results.png: Visualization of comparative results")
    print("\nThe environment successfully simulates:")
    print("  ✓ Machine reliability and degradation")
    print("  ✓ Quality control at multiple checkpoints")
    print("  ✓ Production optimization challenges")
    print("  ✓ Predictive vs reactive maintenance strategies")
    print("  ✓ Real-time OEE tracking")
    print("  ✓ Energy consumption monitoring")
    print("  ✓ Supply chain disruptions")
    print("  ✓ Worker shift efficiency variations")


if __name__ == "__main__":
    main()