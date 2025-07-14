import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def calculate_episode_statistics(episode_data: List[Dict]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for an episode
    
    Args:
        episode_data: List of step data from an episode
        
    Returns:
        Dictionary containing episode statistics
    """
    if not episode_data:
        return {}
    
    # Extract key metrics
    total_rewards = [step.get('reward', 0) for step in episode_data]
    timesteps = [step.get('timestep', 0) for step in episode_data]
    waiting_customers = [step.get('info', {}).get('waiting_customers', 0) for step in episode_data]
    idle_waiters = [step.get('info', {}).get('idle_waiters', 0) for step in episode_data]
    kitchen_queue_lengths = [step.get('info', {}).get('kitchen_queue_length', 0) for step in episode_data]
    dirty_tables = [step.get('info', {}).get('dirty_tables', 0) for step in episode_data]
    
    # Calculate statistics
    stats = {
        'total_reward': sum(total_rewards),
        'average_reward_per_step': np.mean(total_rewards),
        'max_reward': max(total_rewards),
        'min_reward': min(total_rewards),
        'reward_std': np.std(total_rewards),
        'episode_length': len(episode_data),
        'average_waiting_customers': np.mean(waiting_customers),
        'max_waiting_customers': max(waiting_customers),
        'average_idle_waiters': np.mean(idle_waiters),
        'average_kitchen_queue_length': np.mean(kitchen_queue_lengths),
        'max_kitchen_queue_length': max(kitchen_queue_lengths),
        'average_dirty_tables': np.mean(dirty_tables),
        'max_dirty_tables': max(dirty_tables),
        'efficiency_score': _calculate_efficiency_score(episode_data)
    }
    
    # Add final episode stats if available
    if episode_data:
        final_info = episode_data[-1].get('info', {})
        episode_stats = final_info.get('episode_stats', {})
        stats.update({
            'customers_served': episode_stats.get('customers_served', 0),
            'customers_left': episode_stats.get('customers_left', 0),
            'tables_cleaned': episode_stats.get('tables_cleaned', 0),
            'orders_served': episode_stats.get('orders_served', 0),
            'average_wait_time': episode_stats.get('average_wait_time', 0.0),
            'customer_satisfaction_rate': _calculate_satisfaction_rate(episode_stats)
        })
    
    return stats

def _calculate_efficiency_score(episode_data: List[Dict]) -> float:
    """Calculate overall efficiency score for the episode"""
    if not episode_data:
        return 0.0
    
    # Extract efficiency metrics
    waiting_customers = [step.get('info', {}).get('waiting_customers', 0) for step in episode_data]
    idle_waiters = [step.get('info', {}).get('idle_waiters', 0) for step in episode_data]
    dirty_tables = [step.get('info', {}).get('dirty_tables', 0) for step in episode_data]
    kitchen_queue = [step.get('info', {}).get('kitchen_queue_length', 0) for step in episode_data]
    
    # Calculate efficiency components
    customer_efficiency = 1.0 - (np.mean(waiting_customers) / 10.0)  # Normalize by max expected
    waiter_efficiency = np.mean(idle_waiters) / 10.0  # Higher idle waiters = lower efficiency
    table_efficiency = 1.0 - (np.mean(dirty_tables) / 10.0)
    kitchen_efficiency = 1.0 - (np.mean(kitchen_queue) / 20.0)  # Normalize by reasonable max
    
    # Weighted average
    efficiency_score = (
        0.4 * customer_efficiency +
        0.3 * waiter_efficiency +
        0.2 * table_efficiency +
        0.1 * kitchen_efficiency
    )
    
    return max(0.0, min(1.0, efficiency_score))

def _calculate_satisfaction_rate(episode_stats: Dict) -> float:
    """Calculate customer satisfaction rate"""
    customers_served = episode_stats.get('customers_served', 0)
    customers_left = episode_stats.get('customers_left', 0)
    total_customers = customers_served + customers_left
    
    if total_customers == 0:
        return 0.0
    
    return customers_served / total_customers

def analyze_action_distribution(episode_data: List[Dict]) -> Dict[int, int]:
    """Analyze the distribution of actions taken during an episode"""
    action_counts = defaultdict(int)
    
    for step in episode_data:
        action = step.get('action')
        if action is not None:
            action_counts[action] += 1
    
    return dict(action_counts)

def calculate_reward_breakdown(episode_data: List[Dict]) -> Dict[str, float]:
    """Calculate breakdown of different reward components"""
    reward_components = {
        'seat_customer': 0.0,
        'serve_food': 0.0,
        'clean_table': 0.0,
        'efficiency_bonuses': 0.0,
        'penalties': 0.0
    }
    
    for step in episode_data:
        reward = step.get('reward', 0)
        info = step.get('info', {})
        
        # This is a simplified breakdown - in practice, you'd need to track
        # individual reward components in the environment
        if reward > 0:
            if info.get('waiting_customers', 0) < info.get('idle_waiters', 0):
                reward_components['efficiency_bonuses'] += reward
            else:
                reward_components['seat_customer'] += reward
        elif reward < 0:
            reward_components['penalties'] += abs(reward)
    
    return reward_components

def plot_episode_metrics(episode_data: List[Dict], save_path: str = None):
    """Create plots of episode metrics"""
    if not episode_data:
        print("No episode data to plot")
        return
    
    # Extract data
    timesteps = [step.get('timestep', i) for i, step in enumerate(episode_data)]
    rewards = [step.get('reward', 0) for step in episode_data]
    waiting_customers = [step.get('info', {}).get('waiting_customers', 0) for step in episode_data]
    idle_waiters = [step.get('info', {}).get('idle_waiters', 0) for step in episode_data]
    kitchen_queue = [step.get('info', {}).get('kitchen_queue_length', 0) for step in episode_data]
    dirty_tables = [step.get('info', {}).get('dirty_tables', 0) for step in episode_data]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Restaurant Environment Episode Metrics', fontsize=16)
    
    # Plot 1: Cumulative Reward
    axes[0, 0].plot(timesteps, np.cumsum(rewards))
    axes[0, 0].set_title('Cumulative Reward')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].grid(True)
    
    # Plot 2: Step Rewards
    axes[0, 1].plot(timesteps, rewards)
    axes[0, 1].set_title('Step Rewards')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True)
    
    # Plot 3: Waiting Customers
    axes[0, 2].plot(timesteps, waiting_customers)
    axes[0, 2].set_title('Waiting Customers')
    axes[0, 2].set_xlabel('Timestep')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].grid(True)
    
    # Plot 4: Idle Waiters
    axes[1, 0].plot(timesteps, idle_waiters)
    axes[1, 0].set_title('Idle Waiters')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True)
    
    # Plot 5: Kitchen Queue Length
    axes[1, 1].plot(timesteps, kitchen_queue)
    axes[1, 1].set_title('Kitchen Queue Length')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True)
    
    # Plot 6: Dirty Tables
    axes[1, 2].plot(timesteps, dirty_tables)
    axes[1, 2].set_title('Dirty Tables')
    axes[1, 2].set_xlabel('Timestep')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_performance_report(episode_data: List[Dict]) -> str:
    """Create a comprehensive performance report"""
    if not episode_data:
        return "No episode data available for analysis."
    
    stats = calculate_episode_statistics(episode_data)
    action_dist = analyze_action_distribution(episode_data)
    reward_breakdown = calculate_reward_breakdown(episode_data)
    
    # Action descriptions
    action_names = {
        0: "Seat Customer",
        1: "Serve Food", 
        2: "Clean Table",
        3: "Do Nothing"
    }
    
    report = []
    report.append("=" * 60)
    report.append("RESTAURANT ENVIRONMENT PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Episode Overview
    report.append("EPISODE OVERVIEW:")
    report.append(f"  Total Reward: {stats['total_reward']:.2f}")
    report.append(f"  Average Reward per Step: {stats['average_reward_per_step']:.3f}")
    report.append(f"  Episode Length: {stats['episode_length']} timesteps")
    report.append(f"  Efficiency Score: {stats['efficiency_score']:.3f}")
    report.append("")
    
    # Customer Service Metrics
    report.append("CUSTOMER SERVICE METRICS:")
    report.append(f"  Customers Served: {stats['customers_served']}")
    report.append(f"  Customers Left: {stats['customers_left']}")
    report.append(f"  Customer Satisfaction Rate: {stats['customer_satisfaction_rate']:.1%}")
    report.append(f"  Average Wait Time: {stats['average_wait_time']:.1f} timesteps")
    report.append("")
    
    # Operational Metrics
    report.append("OPERATIONAL METRICS:")
    report.append(f"  Tables Cleaned: {stats['tables_cleaned']}")
    report.append(f"  Orders Served: {stats['orders_served']}")
    report.append(f"  Average Waiting Customers: {stats['average_waiting_customers']:.1f}")
    report.append(f"  Average Idle Waiters: {stats['average_idle_waiters']:.1f}")
    report.append(f"  Average Kitchen Queue Length: {stats['average_kitchen_queue_length']:.1f}")
    report.append(f"  Average Dirty Tables: {stats['average_dirty_tables']:.1f}")
    report.append("")
    
    # Action Distribution
    report.append("ACTION DISTRIBUTION:")
    total_actions = sum(action_dist.values())
    for action_id, count in sorted(action_dist.items()):
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        action_name = action_names.get(action_id, f"Unknown Action {action_id}")
        report.append(f"  {action_name}: {count} ({percentage:.1f}%)")
    report.append("")
    
    # Reward Breakdown
    report.append("REWARD BREAKDOWN:")
    total_reward = sum(reward_breakdown.values())
    for component, value in reward_breakdown.items():
        percentage = (value / total_reward * 100) if total_reward > 0 else 0
        report.append(f"  {component.replace('_', ' ').title()}: {value:.2f} ({percentage:.1f}%)")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    if stats['average_waiting_customers'] > 3:
        report.append("  ⚠️  High number of waiting customers - consider more efficient seating")
    if stats['average_idle_waiters'] > 5:
        report.append("  ⚠️  Many idle waiters - consider better task assignment")
    if stats['average_dirty_tables'] > 3:
        report.append("  ⚠️  Many dirty tables - prioritize cleaning tasks")
    if stats['average_kitchen_queue_length'] > 5:
        report.append("  ⚠️  Long kitchen queue - consider serving food more quickly")
    if stats['customer_satisfaction_rate'] < 0.8:
        report.append("  ⚠️  Low customer satisfaction - focus on reducing wait times")
    
    if not any([stats['average_waiting_customers'] > 3, stats['average_idle_waiters'] > 5, 
                stats['average_dirty_tables'] > 3, stats['average_kitchen_queue_length'] > 5,
                stats['customer_satisfaction_rate'] < 0.8]):
        report.append("  ✅ Good overall performance!")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def validate_environment_state(env) -> List[str]:
    """Validate the current state of the environment for consistency"""
    errors = []
    
    # Check table consistency
    for table in env.tables:
        if table.occupied and not table.customer_id:
            errors.append(f"Table {table.table_id} is occupied but has no customer_id")
        if not table.occupied and table.customer_id:
            errors.append(f"Table {table.table_id} is not occupied but has customer_id")
    
    # Check customer consistency
    for customer_id, customer in env.customers.items():
        if customer.state.value == 'waiting' and customer_id not in env.waiting_customers:
            errors.append(f"Customer {customer_id} is waiting but not in waiting_customers list")
        if customer.state.value != 'waiting' and customer_id in env.waiting_customers:
            errors.append(f"Customer {customer_id} is not waiting but in waiting_customers list")
        if customer.table_id is not None:
            table = env.tables[customer.table_id]
            if not table.occupied or table.customer_id != customer_id:
                errors.append(f"Customer {customer_id} assigned to table {customer.table_id} but table state inconsistent")
    
    # Check waiter consistency
    for waiter in env.waiters:
        if waiter.state.value == 'busy' and waiter.task_remaining_time <= 0:
            errors.append(f"Waiter {waiter.waiter_id} is busy but task_remaining_time is {waiter.task_remaining_time}")
        if waiter.state.value == 'idle' and waiter.task_remaining_time > 0:
            errors.append(f"Waiter {waiter.waiter_id} is idle but task_remaining_time is {waiter.task_remaining_time}")
    
    return errors 