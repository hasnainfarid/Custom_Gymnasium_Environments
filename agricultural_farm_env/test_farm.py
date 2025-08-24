"""
Test script for the Agricultural Farm Management Environment.
Includes enhanced visualization and comparison of different farming strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Import the environment
try:
    from farm_env import AgriculturalFarmEnv
except ImportError:
    from .farm_env import AgriculturalFarmEnv


class FarmingStrategy:
    """Base class for farming strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.action_history = []
        self.reward_history = []
        self.financial_history = []
        self.sustainability_metrics = []
    
    def select_action(self, observation: np.ndarray, info: Dict) -> int:
        """Select an action based on the strategy."""
        raise NotImplementedError
    
    def record_step(self, action: int, reward: float, info: Dict):
        """Record step information for analysis."""
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.financial_history.append(info.get("cash_flow", 0))
        self.sustainability_metrics.append({
            "carbon": info.get("carbon_sequestered", 0),
            "water_score": info.get("water_conservation_score", 0),
            "biodiversity": info.get("biodiversity_index", 0),
            "soil_health": info.get("soil_health_trend", 0)
        })


class ConventionalFarmingStrategy(FarmingStrategy):
    """Conventional intensive farming strategy."""
    
    def __init__(self):
        super().__init__("Conventional Farming")
        self.last_actions = {}
    
    def select_action(self, observation: np.ndarray, info: Dict) -> int:
        """High-input, high-output farming approach."""
        day = info.get("day", 0)
        season = info.get("season", "spring")
        
        # Intensive production mode every 30 days
        if day % 30 == 0:
            return 42
        
        # Plant crops aggressively in spring
        if season == "spring" and day % 5 < 4:
            return np.random.randint(0, 5)  # Plant in random field
        
        # Heavy fertilizer use
        if day % 15 == 7:
            return np.random.randint(10, 15)  # Fertilize fields
        
        # Frequent irrigation
        if day % 3 == 0:
            return np.random.randint(15, 20)  # Irrigate fields
        
        # Pesticide application when needed
        if day % 20 == 10:
            return np.random.randint(20, 25)  # Apply pesticides
        
        # Harvest when ready (check all fields)
        if day % 7 == 0:
            return np.random.randint(5, 10)  # Try harvesting
        
        # Equipment operation
        if day % 10 == 5:
            return np.random.randint(25, 30)  # Operate equipment
        
        # Default: soil testing
        return np.random.randint(30, 35)


class OrganicFarmingStrategy(FarmingStrategy):
    """Organic sustainable farming strategy."""
    
    def __init__(self):
        super().__init__("Organic Farming")
        self.rotation_schedule = {0: 0, 1: 4, 2: 2, 3: 1}  # Field to crop mapping
    
    def select_action(self, observation: np.ndarray, info: Dict) -> int:
        """Low-input, sustainable farming approach."""
        day = info.get("day", 0)
        season = info.get("season", "spring")
        
        # Sustainable farming mode every 20 days
        if day % 20 == 0:
            return 41
        
        # Crop rotation planning
        if day % 45 == 0:
            return 43
        
        # Plant according to rotation schedule
        if season in ["spring", "summer"] and day % 10 < 4:
            field_id = day % 4
            return field_id  # Plant in rotation
        
        # Minimal fertilizer use (organic)
        if day % 30 == 15:
            return np.random.randint(10, 12)  # Light fertilization
        
        # Efficient irrigation only when needed
        if day % 5 == 0 and np.random.random() < 0.5:
            return np.random.randint(15, 18)  # Selective irrigation
        
        # Avoid pesticides, use emergency protection instead
        if day % 40 == 20:
            return 40  # Weather protection instead of chemicals
        
        # Harvest at optimal times
        if day % 10 == 5:
            return np.random.randint(5, 10)
        
        # Regular equipment maintenance
        if day % 25 == 12:
            return 44
        
        # Market timing for better prices
        if day % 15 == 8:
            return np.random.randint(35, 40)
        
        # Default: soil improvement
        return np.random.randint(30, 35)


class SmartFarmingStrategy(FarmingStrategy):
    """Data-driven precision farming strategy."""
    
    def __init__(self):
        super().__init__("Smart Farming")
        self.field_status = {}
    
    def select_action(self, observation: np.ndarray, info: Dict) -> int:
        """Precision agriculture using data analysis."""
        day = info.get("day", 0)
        season = info.get("season", "spring")
        field_status = info.get("field_status", [])
        
        # Analyze field conditions from observation
        # Observation structure: first 60 elements are field data (15 per field)
        
        # Emergency responses
        if observation[70] < -10:  # Very cold temperature
            return 40  # Emergency weather protection
        
        # Check each field's status
        for i, field in enumerate(field_status):
            if i >= 4:
                break
                
            # Harvest if ready
            if field["stage"] == "HARVEST_READY":
                return 5 + i
            
            # Plant if empty and good season
            if field["crop"] == "None" and season in ["spring", "summer"]:
                return i
            
            # Check soil moisture (observation indices 5, 20, 35, 50)
            moisture_idx = i * 15 + 5
            if moisture_idx < len(observation) and observation[moisture_idx] < 0.3:
                return 15 + i  # Irrigate dry field
            
            # Check health
            if field["health"] < 0.7:
                return 20 + i  # Apply treatment
        
        # Periodic maintenance and optimization
        if day % 20 == 10:
            return 44  # Equipment maintenance
        
        if day % 30 == 15:
            return 41  # Sustainable mode
        
        if day % 25 == 5:
            return 43  # Rotation planning
        
        # Fertilize based on nutrient levels
        if day % 20 == 0:
            return np.random.randint(10, 15)
        
        # Market decisions
        if observation[504] > 1.2:  # Good market prices (normalized)
            return np.random.randint(35, 40)
        
        # Default: monitor soil
        return np.random.randint(30, 35)


def run_scenario(env: AgriculturalFarmEnv, strategy: FarmingStrategy, 
                 scenario_name: str, max_steps: int = 365) -> Dict:
    """Run a complete farming scenario with a given strategy."""
    print(f"\n{'='*60}")
    print(f"Running Scenario: {scenario_name}")
    print(f"Strategy: {strategy.name}")
    print(f"{'='*60}")
    
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    
    # Modify environment for specific scenarios
    if scenario_name == "Drought Year":
        env.weather.drought_days = 20
        env.reservoir_level = 0.3
    elif scenario_name == "Pest Outbreak":
        for field in env.field_sections:
            field.pest_level = 0.6
            field.disease_risk = 0.4
    elif scenario_name == "Market Crash":
        for crop_type in env.market.current_prices:
            env.market.current_prices[crop_type] *= 0.5
    
    # Run simulation
    for step in range(max_steps):
        action = strategy.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        strategy.record_step(action, reward, info)
        step_count += 1
        
        # Render periodically
        if env.render_mode == "human" and step % 10 == 0:
            env.render()
            # Handle pygame events to prevent window freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        # Print progress
        if step % 90 == 0:  # Every season
            print(f"  Season: {info['season']}, Day: {info['day']}, "
                  f"Cash: ${info['cash_flow']:,.0f}, "
                  f"Total Reward: {total_reward:.0f}")
        
        if terminated or truncated:
            break
    
    # Generate report
    report = {
        "scenario": scenario_name,
        "strategy": strategy.name,
        "total_reward": total_reward,
        "final_cash": info.get("cash_flow", 0),
        "total_revenue": info.get("total_revenue", 0),
        "total_expenses": info.get("total_expenses", 0),
        "profit_margin": info.get("profit_margin", 0),
        "carbon_sequestered": info.get("carbon_sequestered", 0),
        "water_conservation": info.get("water_conservation_score", 0),
        "biodiversity": info.get("biodiversity_index", 0),
        "soil_health": info.get("soil_health_trend", 0),
        "steps": step_count,
        "reward_history": strategy.reward_history,
        "financial_history": strategy.financial_history,
        "sustainability_metrics": strategy.sustainability_metrics
    }
    
    return report


def generate_comparison_plots(reports: List[Dict]):
    """Generate comparison plots for different strategies."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Agricultural Farming Strategy Comparison", fontsize=16)
    
    # Extract data for plotting
    strategies = [r["strategy"] for r in reports]
    scenarios = list(set(r["scenario"] for r in reports))
    
    # 1. Total Rewards Comparison
    ax = axes[0, 0]
    for scenario in scenarios:
        scenario_reports = [r for r in reports if r["scenario"] == scenario]
        rewards = [r["total_reward"] for r in scenario_reports]
        strats = [r["strategy"] for r in scenario_reports]
        ax.bar(strats, rewards, label=scenario, alpha=0.7)
    ax.set_title("Total Rewards")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Financial Performance
    ax = axes[0, 1]
    for scenario in scenarios[:1]:  # Show one scenario for clarity
        scenario_reports = [r for r in reports if r["scenario"] == scenario]
        for report in scenario_reports:
            ax.plot(report["financial_history"], label=f"{report['strategy']}", alpha=0.7)
    ax.set_title("Cash Flow Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Cash ($)")
    ax.legend()
    
    # 3. Profit Margins
    ax = axes[0, 2]
    for scenario in scenarios:
        scenario_reports = [r for r in reports if r["scenario"] == scenario]
        profits = [r["profit_margin"] * 100 for r in scenario_reports]
        strats = [r["strategy"] for r in scenario_reports]
        ax.bar(strats, profits, label=scenario, alpha=0.7)
    ax.set_title("Profit Margins")
    ax.set_ylabel("Profit Margin (%)")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Sustainability Metrics
    ax = axes[1, 0]
    metrics = ["carbon_sequestered", "water_conservation", "biodiversity", "soil_health"]
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, strategy in enumerate(set(r["strategy"] for r in reports)):
        strategy_reports = [r for r in reports if r["strategy"] == strategy and r["scenario"] == "Normal Year"]
        if strategy_reports:
            values = [strategy_reports[0].get(m, 0) for m in metrics]
            ax.bar(x + i * width, values, width, label=strategy)
    
    ax.set_title("Sustainability Metrics (Normal Year)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Carbon", "Water", "Biodiv", "Soil"])
    ax.legend()
    
    # 5. Reward Distribution
    ax = axes[1, 1]
    for report in reports[:3]:  # Show first 3 for clarity
        if len(report["reward_history"]) > 0:
            ax.hist(report["reward_history"], bins=30, alpha=0.5, 
                   label=f"{report['strategy']} ({report['scenario'][:7]}...)")
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Reward per Step")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    # 6. Strategy Performance Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create summary table
    summary_text = "Performance Summary:\n\n"
    for scenario in scenarios:
        summary_text += f"{scenario}:\n"
        scenario_reports = sorted(
            [r for r in reports if r["scenario"] == scenario],
            key=lambda x: x["total_reward"],
            reverse=True
        )
        for i, report in enumerate(scenario_reports, 1):
            summary_text += f"  {i}. {report['strategy']}: ${report['final_cash']:,.0f}\n"
        summary_text += "\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"farm_strategy_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved as: {filename}")
    
    plt.show()


def save_report(reports: List[Dict], filename: str = None):
    """Save detailed report to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"farm_report_{timestamp}.json"
    
    # Prepare report data (remove large arrays for readability)
    save_data = []
    for report in reports:
        save_report = report.copy()
        # Summarize long lists
        if "reward_history" in save_report:
            save_report["reward_summary"] = {
                "mean": np.mean(save_report["reward_history"]) if save_report["reward_history"] else 0,
                "std": np.std(save_report["reward_history"]) if save_report["reward_history"] else 0,
                "min": min(save_report["reward_history"]) if save_report["reward_history"] else 0,
                "max": max(save_report["reward_history"]) if save_report["reward_history"] else 0
            }
            del save_report["reward_history"]
        
        if "financial_history" in save_report:
            save_report["financial_summary"] = {
                "start": save_report["financial_history"][0] if save_report["financial_history"] else 0,
                "end": save_report["financial_history"][-1] if save_report["financial_history"] else 0,
                "min": min(save_report["financial_history"]) if save_report["financial_history"] else 0,
                "max": max(save_report["financial_history"]) if save_report["financial_history"] else 0
            }
            del save_report["financial_history"]
        
        if "sustainability_metrics" in save_report:
            if save_report["sustainability_metrics"]:
                final_metrics = save_report["sustainability_metrics"][-1]
                save_report["final_sustainability"] = final_metrics
            del save_report["sustainability_metrics"]
        
        save_data.append(save_report)
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Detailed report saved as: {filename}")


def demo_visualization():
    """Quick demo of the enhanced visualization."""
    print("\n" + "="*60)
    print("Enhanced Visualization Demo")
    print("="*60)
    
    env = AgriculturalFarmEnv(render_mode="human")
    obs, info = env.reset()
    
    print("\nEnhanced visualization features:")
    print("✨ Gradient backgrounds that change with seasons")
    print("🌾 Detailed crop visualization with growth stages")
    print("🌧️ Animated weather effects (rain, frost)")
    print("📊 Real-time performance graphs")
    print("💧 Liquid tank animations for water resources")
    print("🎯 Circular meters and gauges")
    print("📅 Seasonal calendar with planting schedules")
    print("💰 Financial dashboard with trend graphs")
    print("🌱 Sustainability indicators")
    print("⚠️  Smart notification system")
    print("\nRunning demo for 50 steps...")
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        
        if step % 10 == 0:
            print(f"  Step {step}: {info['season'].capitalize()}, Day {info['day']}")
        
        if terminated or truncated:
            break
    
    print("\nDemo complete! Close the window when done viewing.")
    
    # Keep window open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        env.render()
        time.sleep(0.1)
    
    env.close()

def main():
    """Main test function."""
    print("="*60)
    print("Agricultural Farm Management Environment Test")
    print("="*60)
    
    # Ask user for mode
    print("\nSelect mode:")
    print("1. Quick Visualization Demo")
    print("2. Full Strategy Comparison Test")
    
    try:
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
        if choice == "2":
            run_full_test()
        else:
            demo_visualization()
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        return

def run_full_test():
    """Run the full strategy comparison test."""
    
    # Initialize environment with rendering
    env = AgriculturalFarmEnv(render_mode="human")
    
    # Define scenarios to test
    scenarios = [
        "Normal Year",
        "Drought Year",
        "Pest Outbreak",
        "Market Crash"
    ]
    
    # Define strategies to test
    strategies = [
        ConventionalFarmingStrategy(),
        OrganicFarmingStrategy(),
        SmartFarmingStrategy()
    ]
    
    # Run tests
    all_reports = []
    
    for scenario in scenarios:
        for strategy in strategies:
            # Create fresh strategy instance for each run
            if strategy.name == "Conventional Farming":
                strategy = ConventionalFarmingStrategy()
            elif strategy.name == "Organic Farming":
                strategy = OrganicFarmingStrategy()
            else:
                strategy = SmartFarmingStrategy()
            
            report = run_scenario(env, strategy, scenario, max_steps=365)
            all_reports.append(report)
            
            # Print summary
            print(f"\nSummary for {strategy.name} in {scenario}:")
            print(f"  Total Reward: {report['total_reward']:.0f}")
            print(f"  Final Cash: ${report['final_cash']:,.0f}")
            print(f"  Profit Margin: {report['profit_margin']:.1%}")
            print(f"  Carbon Sequestered: {report['carbon_sequestered']:.1f} tons")
            print(f"  Water Conservation: {report['water_conservation']:.2f}")
            print(f"  Biodiversity: {report['biodiversity']:.2f}")
            print(f"  Soil Health Trend: {report['soil_health']:+.2f}")
    
    # Generate comparison plots
    generate_comparison_plots(all_reports)
    
    # Save detailed report
    save_report(all_reports)
    
    # Print final summary
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    print("\nBest Performers by Scenario:")
    for scenario in scenarios:
        scenario_reports = [r for r in all_reports if r["scenario"] == scenario]
        best = max(scenario_reports, key=lambda x: x["total_reward"])
        print(f"  {scenario}: {best['strategy']} (Reward: {best['total_reward']:.0f})")
    
    print("\nMost Sustainable Strategy:")
    sustainability_scores = {}
    for strategy_name in set(r["strategy"] for r in all_reports):
        strategy_reports = [r for r in all_reports if r["strategy"] == strategy_name]
        avg_sustainability = np.mean([
            r["carbon_sequestered"] + r["water_conservation"] + 
            r["biodiversity"] + r["soil_health"]
            for r in strategy_reports
        ])
        sustainability_scores[strategy_name] = avg_sustainability
    
    best_sustainable = max(sustainability_scores.items(), key=lambda x: x[1])
    print(f"  {best_sustainable[0]} (Score: {best_sustainable[1]:.2f})")
    
    print("\nMost Profitable Strategy:")
    profit_scores = {}
    for strategy_name in set(r["strategy"] for r in all_reports):
        strategy_reports = [r for r in all_reports if r["strategy"] == strategy_name]
        avg_profit = np.mean([r["profit_margin"] for r in strategy_reports])
        profit_scores[strategy_name] = avg_profit
    
    best_profitable = max(profit_scores.items(), key=lambda x: x[1])
    print(f"  {best_profitable[0]} (Avg Margin: {best_profitable[1]:.1%})")
    
    # Close environment
    env.close()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
