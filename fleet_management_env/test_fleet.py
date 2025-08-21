"""
Fleet Management Environment Testing Suite

Comprehensive testing with multiple scenarios, performance tracking,
and visualization of different agent strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import time
from datetime import datetime

from fleet_env import FleetManagementEnv, VehicleType, UrgencyLevel


@dataclass
class PerformanceMetrics:
    """Container for episode performance metrics"""
    episode_reward: float
    deliveries_completed: int
    total_deliveries: int
    delivery_rate: float
    fuel_consumed: float
    total_distance: int
    customer_satisfaction: float
    urgent_deliveries_missed: int
    average_delivery_time: float
    fuel_efficiency: float  # deliveries per fuel unit
    episode_length: int
    vehicles_exhausted: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RandomAgent:
    """Random action agent for baseline comparison"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        return self.action_space.sample()


class GreedyNearestAgent:
    """Greedy agent that always goes to nearest available delivery"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        # Parse observation to extract vehicle positions and delivery locations
        actions = []
        
        # Vehicle positions (first 6 values: 3 vehicles * 2 coordinates)
        vehicle_positions = []
        for i in range(3):
            x = int(observation[i * 2])
            y = int(observation[i * 2 + 1])
            vehicle_positions.append((x, y))
        
        # Vehicle fuel levels (next 3 values)
        fuel_levels = observation[6:9]
        
        # Vehicle cargo (next 3 values)
        cargo_used = observation[9:12]
        
        # Current deliveries assigned (next 3 values)
        assigned_deliveries = observation[12:15]
        
        # Delivery locations (next 24 values: 12 deliveries * 2 coordinates)
        delivery_locations = []
        for i in range(12):
            x = observation[15 + i * 2]
            y = observation[15 + i * 2 + 1]
            if x >= 0 and y >= 0:  # Valid delivery
                delivery_locations.append((int(x), int(y)))
            else:
                delivery_locations.append(None)
        
        # Delivery urgency (next 12 values)
        urgency_levels = observation[39:51]
        
        for vehicle_id in range(3):
            pos = vehicle_positions[vehicle_id]
            fuel = fuel_levels[vehicle_id]
            cargo = cargo_used[vehicle_id]
            assigned = int(assigned_deliveries[vehicle_id])
            
            action = 0  # Default: stay
            
            if fuel <= 0.1:  # Low fuel
                # Find nearest fuel station
                fuel_stations = [(5, 5), (20, 5), (5, 20)]
                nearest_station = min(fuel_stations, 
                                    key=lambda fs: abs(fs[0] - pos[0]) + abs(fs[1] - pos[1]))
                
                if pos == nearest_station:
                    action = 7  # Refuel
                else:
                    # Move towards fuel station
                    if nearest_station[0] > pos[0]:
                        action = 4  # Right
                    elif nearest_station[0] < pos[0]:
                        action = 3  # Left
                    elif nearest_station[1] > pos[1]:
                        action = 2  # Down
                    elif nearest_station[1] < pos[1]:
                        action = 1  # Up
            
            elif assigned >= 0:  # Has delivery assigned
                # Move towards delivery location (simplified - would need actual delivery info)
                action = np.random.choice([1, 2, 3, 4, 6])  # Move or attempt dropoff
            
            else:  # No delivery assigned
                # Find nearest available delivery
                available_deliveries = []
                for i, (loc, urgency) in enumerate(zip(delivery_locations, urgency_levels)):
                    if loc is not None and urgency >= 0:
                        distance = abs(loc[0] - pos[0]) + abs(loc[1] - pos[1])
                        priority = urgency * 10 + (1 / (distance + 1))  # Urgency + proximity
                        available_deliveries.append((i, loc, priority))
                
                if available_deliveries:
                    # Go to highest priority delivery
                    _, target_loc, _ = max(available_deliveries, key=lambda x: x[2])
                    
                    if pos == target_loc:
                        action = 5  # Pick up
                    else:
                        # Move towards target
                        if target_loc[0] > pos[0]:
                            action = 4  # Right
                        elif target_loc[0] < pos[0]:
                            action = 3  # Left
                        elif target_loc[1] > pos[1]:
                            action = 2  # Down
                        elif target_loc[1] < pos[1]:
                            action = 1  # Up
            
            actions.append(action)
        
        return actions


class FleetTestSuite:
    """Comprehensive testing suite for fleet management environment"""
    
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Test scenarios
        self.scenarios = {
            "light_traffic": {
                "description": "Light traffic conditions with standard delivery mix",
                "traffic_intensity": 0.1,
                "urgent_delivery_ratio": 0.2
            },
            "heavy_traffic": {
                "description": "Heavy traffic conditions testing fuel management",
                "traffic_intensity": 0.4,
                "urgent_delivery_ratio": 0.2
            },
            "mixed_urgency": {
                "description": "Mixed urgency deliveries testing prioritization",
                "traffic_intensity": 0.2,
                "urgent_delivery_ratio": 0.5
            }
        }
        
        self.agents = {
            "random": RandomAgent,
            "greedy_nearest": GreedyNearestAgent
        }
    
    def run_episode(self, env: FleetManagementEnv, agent, max_steps: int = 800, 
                   render: bool = False) -> PerformanceMetrics:
        """Run single episode and collect performance metrics"""
        
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        delivery_times = []
        initial_deliveries = len([d for d in env.delivery_requests if not d.completed])
        
        while steps < max_steps:
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render and steps % 10 == 0:  # Render every 10 steps
                env.render()
                time.sleep(0.1)
            
            # Track delivery times
            for delivery in env.delivery_requests:
                if (delivery.completed and delivery.delivery_time not in 
                    [dt[1] for dt in delivery_times]):
                    pickup_time = delivery.pickup_time if delivery.pickup_time > 0 else 0
                    delivery_time = delivery.delivery_time - pickup_time
                    delivery_times.append((delivery, delivery_time))
            
            if terminated or truncated:
                break
        
        # Calculate metrics
        completed_deliveries = sum(1 for d in env.delivery_requests if d.completed)
        delivery_rate = completed_deliveries / initial_deliveries if initial_deliveries > 0 else 0
        
        avg_delivery_time = (np.mean([dt[1] for dt in delivery_times]) 
                           if delivery_times else 0)
        
        fuel_efficiency = (completed_deliveries / max(env.episode_stats["fuel_consumed"], 1))
        
        customer_satisfaction = delivery_rate * 100  # Simplified metric
        
        vehicles_exhausted = sum(1 for v in env.vehicles if v.fuel <= 0)
        
        return PerformanceMetrics(
            episode_reward=total_reward,
            deliveries_completed=completed_deliveries,
            total_deliveries=initial_deliveries,
            delivery_rate=delivery_rate,
            fuel_consumed=env.episode_stats["fuel_consumed"],
            total_distance=env.episode_stats["total_distance"],
            customer_satisfaction=customer_satisfaction,
            urgent_deliveries_missed=env.episode_stats["urgent_deliveries_missed"],
            average_delivery_time=avg_delivery_time,
            fuel_efficiency=fuel_efficiency,
            episode_length=steps,
            vehicles_exhausted=vehicles_exhausted
        )
    
    def run_scenario_tests(self, scenario_name: str, num_episodes: int = 10, 
                          render_best: bool = True) -> Dict[str, List[PerformanceMetrics]]:
        """Run tests for a specific scenario across all agents"""
        
        print(f"\n=== Running {scenario_name.upper()} Scenario ===")
        print(f"Description: {self.scenarios[scenario_name]['description']}")
        
        results = {}
        
        for agent_name, agent_class in self.agents.items():
            print(f"\nTesting {agent_name} agent...")
            agent_results = []
            
            for episode in range(num_episodes):
                # Create environment
                env = FleetManagementEnv(render_mode="human" if render_best and episode == 0 else None)
                agent = agent_class(env.action_space)
                
                # Modify environment based on scenario
                self._configure_scenario(env, scenario_name)
                
                # Run episode
                metrics = self.run_episode(env, agent, render=(render_best and episode == 0))
                agent_results.append(metrics)
                
                print(f"  Episode {episode + 1}: Reward={metrics.episode_reward:.1f}, "
                      f"Deliveries={metrics.deliveries_completed}/{metrics.total_deliveries}, "
                      f"Rate={metrics.delivery_rate:.2f}")
                
                env.close()
            
            results[agent_name] = agent_results
        
        return results
    
    def _configure_scenario(self, env: FleetManagementEnv, scenario_name: str):
        """Configure environment for specific scenario"""
        scenario = self.scenarios[scenario_name]
        
        # Modify traffic intensity (simplified - would need more sophisticated implementation)
        traffic_prob = scenario["traffic_intensity"]
        if traffic_prob > 0.3:  # Heavy traffic
            env.traffic_grid = np.random.choice([1, 2], size=(5, 5), p=[0.3, 0.7])
        elif traffic_prob > 0.15:  # Medium traffic
            env.traffic_grid = np.random.choice([0, 1, 2], size=(5, 5), p=[0.4, 0.5, 0.1])
        else:  # Light traffic
            env.traffic_grid = np.random.choice([0, 1], size=(5, 5), p=[0.8, 0.2])
    
    def analyze_results(self, results: Dict[str, Dict[str, List[PerformanceMetrics]]]):
        """Analyze and visualize test results"""
        
        print("\n=== PERFORMANCE ANALYSIS ===")
        
        # Create comprehensive results dataframe
        all_data = []
        
        for scenario, scenario_results in results.items():
            for agent, episodes in scenario_results.items():
                for episode_idx, metrics in enumerate(episodes):
                    data = metrics.to_dict()
                    data['scenario'] = scenario
                    data['agent'] = agent
                    data['episode'] = episode_idx
                    all_data.append(data)
        
        df = pd.DataFrame(all_data)
        
        # Print summary statistics
        self._print_summary_stats(df)
        
        # Generate visualizations
        self._create_visualizations(df)
        
        # Save detailed results
        self._save_results(df, results)
        
        return df
    
    def _print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics for all scenarios and agents"""
        
        print("\n--- Summary Statistics ---")
        
        summary_metrics = ['episode_reward', 'delivery_rate', 'fuel_efficiency', 
                         'customer_satisfaction', 'average_delivery_time']
        
        for scenario in df['scenario'].unique():
            print(f"\nScenario: {scenario.upper()}")
            scenario_df = df[df['scenario'] == scenario]
            
            for agent in scenario_df['agent'].unique():
                agent_df = scenario_df[scenario_df['agent'] == agent]
                
                print(f"  {agent.title()} Agent:")
                for metric in summary_metrics:
                    if metric in agent_df.columns:
                        mean_val = agent_df[metric].mean()
                        std_val = agent_df[metric].std()
                        print(f"    {metric}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Best performing combinations
        print("\n--- Best Performing Combinations ---")
        best_reward = df.loc[df['episode_reward'].idxmax()]
        best_delivery = df.loc[df['delivery_rate'].idxmax()]
        best_efficiency = df.loc[df['fuel_efficiency'].idxmax()]
        
        print(f"Best Reward: {best_reward['agent']} on {best_reward['scenario']} "
              f"({best_reward['episode_reward']:.1f})")
        print(f"Best Delivery Rate: {best_delivery['agent']} on {best_delivery['scenario']} "
              f"({best_delivery['delivery_rate']:.3f})")
        print(f"Best Fuel Efficiency: {best_efficiency['agent']} on {best_efficiency['scenario']} "
              f"({best_efficiency['fuel_efficiency']:.3f})")
    
    def _create_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualizations of results"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Episode Rewards by Scenario and Agent
        plt.subplot(3, 3, 1)
        sns.boxplot(data=df, x='scenario', y='episode_reward', hue='agent')
        plt.title('Episode Rewards by Scenario')
        plt.xticks(rotation=45)
        
        # 2. Delivery Rate Comparison
        plt.subplot(3, 3, 2)
        sns.barplot(data=df, x='scenario', y='delivery_rate', hue='agent', ci=95)
        plt.title('Delivery Rate Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # 3. Fuel Efficiency
        plt.subplot(3, 3, 3)
        sns.boxplot(data=df, x='scenario', y='fuel_efficiency', hue='agent')
        plt.title('Fuel Efficiency (Deliveries/Fuel)')
        plt.xticks(rotation=45)
        
        # 4. Customer Satisfaction
        plt.subplot(3, 3, 4)
        sns.violinplot(data=df, x='scenario', y='customer_satisfaction', hue='agent')
        plt.title('Customer Satisfaction Score')
        plt.xticks(rotation=45)
        
        # 5. Average Delivery Time
        plt.subplot(3, 3, 5)
        sns.boxplot(data=df, x='scenario', y='average_delivery_time', hue='agent')
        plt.title('Average Delivery Time')
        plt.xticks(rotation=45)
        
        # 6. Episode Length Distribution
        plt.subplot(3, 3, 6)
        sns.histplot(data=df, x='episode_length', hue='agent', kde=True, alpha=0.6)
        plt.title('Episode Length Distribution')
        
        # 7. Reward vs Delivery Rate Scatter
        plt.subplot(3, 3, 7)
        sns.scatterplot(data=df, x='delivery_rate', y='episode_reward', 
                       hue='agent', style='scenario', s=100)
        plt.title('Reward vs Delivery Rate')
        
        # 8. Fuel Consumption vs Distance
        plt.subplot(3, 3, 8)
        sns.scatterplot(data=df, x='total_distance', y='fuel_consumed', 
                       hue='agent', style='scenario', s=100)
        plt.title('Fuel Consumption vs Distance')
        
        # 9. Performance Radar Chart (aggregate)
        plt.subplot(3, 3, 9)
        self._create_radar_chart(df, plt.gca())
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional detailed plots
        self._create_detailed_plots(df)
    
    def _create_radar_chart(self, df: pd.DataFrame, ax):
        """Create radar chart comparing agent performance"""
        
        # Aggregate metrics by agent
        metrics = ['episode_reward', 'delivery_rate', 'fuel_efficiency', 
                  'customer_satisfaction']
        
        agent_means = df.groupby('agent')[metrics].mean()
        
        # Normalize metrics to 0-1 scale for radar chart
        normalized = agent_means.copy()
        for col in metrics:
            normalized[col] = (normalized[col] - normalized[col].min()) / \
                             (normalized[col].max() - normalized[col].min())
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for agent in normalized.index:
            values = normalized.loc[agent].values
            values = np.concatenate((values, [values[0]]))
            
            ax.plot(angles, values, 'o-', linewidth=2, label=agent)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Comparison')
        ax.legend()
        ax.grid(True)
    
    def _create_detailed_plots(self, df: pd.DataFrame):
        """Create additional detailed analysis plots"""
        
        # Learning curves (if applicable)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards over time
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario]
            for agent in scenario_df['agent'].unique():
                agent_df = scenario_df[scenario_df['agent'] == agent]
                axes[0, 0].plot(agent_df['episode'], agent_df['episode_reward'], 
                              label=f"{agent}_{scenario}", marker='o', alpha=0.7)
        
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Delivery rate progression
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario]
            for agent in scenario_df['agent'].unique():
                agent_df = scenario_df[scenario_df['agent'] == agent]
                axes[0, 1].plot(agent_df['episode'], agent_df['delivery_rate'], 
                              label=f"{agent}_{scenario}", marker='s', alpha=0.7)
        
        axes[0, 1].set_title('Delivery Rate Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Delivery Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Fuel efficiency trends
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario]
            for agent in scenario_df['agent'].unique():
                agent_df = scenario_df[scenario_df['agent'] == agent]
                axes[1, 0].plot(agent_df['episode'], agent_df['fuel_efficiency'], 
                              label=f"{agent}_{scenario}", marker='^', alpha=0.7)
        
        axes[1, 0].set_title('Fuel Efficiency Over Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Fuel Efficiency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Customer satisfaction trends
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario]
            for agent in scenario_df['agent'].unique():
                agent_df = scenario_df[scenario_df['agent'] == agent]
                axes[1, 1].plot(agent_df['episode'], agent_df['customer_satisfaction'], 
                              label=f"{agent}_{scenario}", marker='d', alpha=0.7)
        
        axes[1, 1].set_title('Customer Satisfaction Over Time')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Satisfaction Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_results(self, df: pd.DataFrame, raw_results: Dict):
        """Save detailed results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DataFrame as CSV
        df.to_csv(f'{self.results_dir}/results_{timestamp}.csv', index=False)
        
        # Save raw results as JSON (convert metrics to dict)
        json_results = {}
        for scenario, scenario_results in raw_results.items():
            json_results[scenario] = {}
            for agent, episodes in scenario_results.items():
                json_results[scenario][agent] = [ep.to_dict() for ep in episodes]
        
        with open(f'{self.results_dir}/raw_results_{timestamp}.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Generate and save performance report
        self._generate_report(df, timestamp)
        
        print(f"\nResults saved to {self.results_dir}/")
    
    def _generate_report(self, df: pd.DataFrame, timestamp: str):
        """Generate comprehensive performance report"""
        
        report = []
        report.append("FLEET MANAGEMENT ENVIRONMENT - PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Episodes: {len(df)}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        
        best_agent = df.groupby('agent')['episode_reward'].mean().idxmax()
        best_scenario = df.groupby('scenario')['delivery_rate'].mean().idxmax()
        
        report.append(f"Best Overall Agent: {best_agent.title()}")
        report.append(f"Best Scenario for Delivery Rate: {best_scenario.title()}")
        report.append(f"Average Episode Reward: {df['episode_reward'].mean():.2f}")
        report.append(f"Average Delivery Rate: {df['delivery_rate'].mean():.3f}")
        report.append(f"Average Fuel Efficiency: {df['fuel_efficiency'].mean():.3f}")
        report.append("")
        
        # Detailed Analysis
        report.append("DETAILED ANALYSIS BY SCENARIO")
        report.append("-" * 35)
        
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario]
            report.append(f"\n{scenario.upper()} SCENARIO:")
            
            for agent in scenario_df['agent'].unique():
                agent_df = scenario_df[scenario_df['agent'] == agent]
                
                report.append(f"  {agent.title()} Agent:")
                report.append(f"    Episodes: {len(agent_df)}")
                report.append(f"    Avg Reward: {agent_df['episode_reward'].mean():.2f} ± {agent_df['episode_reward'].std():.2f}")
                report.append(f"    Delivery Rate: {agent_df['delivery_rate'].mean():.3f} ± {agent_df['delivery_rate'].std():.3f}")
                report.append(f"    Fuel Efficiency: {agent_df['fuel_efficiency'].mean():.3f} ± {agent_df['fuel_efficiency'].std():.3f}")
                report.append(f"    Customer Satisfaction: {agent_df['customer_satisfaction'].mean():.1f} ± {agent_df['customer_satisfaction'].std():.1f}")
        
        # Recommendations
        report.append("\nRECOMMENDations")
        report.append("-" * 15)
        
        # Find best performing combinations
        best_combinations = df.groupby(['scenario', 'agent']).agg({
            'episode_reward': 'mean',
            'delivery_rate': 'mean',
            'fuel_efficiency': 'mean'
        }).reset_index()
        
        best_reward_combo = best_combinations.loc[best_combinations['episode_reward'].idxmax()]
        best_delivery_combo = best_combinations.loc[best_combinations['delivery_rate'].idxmax()]
        
        report.append(f"1. For maximum reward: Use {best_reward_combo['agent']} in {best_reward_combo['scenario']} scenario")
        report.append(f"2. For best delivery rate: Use {best_delivery_combo['agent']} in {best_delivery_combo['scenario']} scenario")
        
        if df['fuel_efficiency'].std() > 0.1:
            report.append("3. Fuel efficiency varies significantly - consider fuel management strategies")
        
        if df['urgent_deliveries_missed'].mean() > 2:
            report.append("4. Urgent delivery performance needs improvement - prioritize time-sensitive orders")
        
        # Save report
        with open(f'{self.results_dir}/performance_report_{timestamp}.txt', 'w') as f:
            f.write('\n'.join(report))


def main():
    """Main testing function"""
    
    print("Fleet Management Environment - Comprehensive Test Suite")
    print("=" * 60)
    
    # Initialize test suite
    test_suite = FleetTestSuite()
    
    # Run all scenarios
    all_results = {}
    
    for scenario_name in test_suite.scenarios.keys():
        scenario_results = test_suite.run_scenario_tests(
            scenario_name, 
            num_episodes=5,  # Reduced for demonstration
            render_best=True
        )
        all_results[scenario_name] = scenario_results
    
    # Analyze and visualize results
    print("\nAnalyzing results...")
    results_df = test_suite.analyze_results(all_results)
    
    print("\n=== TEST SUITE COMPLETED ===")
    print(f"Total episodes run: {len(results_df)}")
    print(f"Results saved to: {test_suite.results_dir}/")
    
    # Quick performance summary
    print("\nQuick Performance Summary:")
    print("-" * 30)
    
    summary = results_df.groupby('agent').agg({
        'episode_reward': ['mean', 'std'],
        'delivery_rate': ['mean', 'std'],
        'fuel_efficiency': ['mean', 'std']
    }).round(3)
    
    print(summary)


if __name__ == "__main__":
    main()