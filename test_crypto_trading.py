#!/usr/bin/env python3
"""
Test script for Crypto Trading Environment
Demonstrates various trading scenarios and environment capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from crypto_trading_env import CryptoTradingEnv, TradingConfig, MarketRegime
import random
import time
from typing import List, Dict, Any


class TradingAgent:
    """Simple trading agents for demonstration"""
    
    def __init__(self, strategy: str = "random"):
        self.strategy = strategy
        self.previous_price = None
        self.trend_buffer = []
        
    def get_action(self, observation: np.ndarray, action_space, info: Dict = None) -> Any:
        """Get action based on strategy"""
        if self.strategy == "random":
            return self._random_action(action_space)
        elif self.strategy == "momentum":
            return self._momentum_action(observation, action_space, info)
        elif self.strategy == "contrarian":
            return self._contrarian_action(observation, action_space, info)
        elif self.strategy == "rsi":
            return self._rsi_action(observation, action_space, info)
        else:
            return self._hold_action(action_space)
    
    def _random_action(self, action_space):
        """Random trading strategy"""
        return action_space.sample()
    
    def _momentum_action(self, observation, action_space, info):
        """Momentum-based strategy"""
        if info and 'current_price' in info:
            current_price = info['current_price']
            
            if self.previous_price is not None:
                price_change = (current_price - self.previous_price) / self.previous_price
                
                if hasattr(action_space, 'n'):  # Discrete
                    if price_change > 0.01:  # Strong upward momentum
                        return 2  # buy_large
                    elif price_change > 0.005:  # Moderate upward momentum
                        return 1  # buy_small
                    elif price_change < -0.01:  # Strong downward momentum
                        return 4  # sell_large
                    elif price_change < -0.005:  # Moderate downward momentum
                        return 3  # sell_small
                    else:
                        return 0  # hold
                else:  # Continuous
                    buy_signal = max(0, price_change * 10)
                    sell_signal = max(0, -price_change * 10)
                    return np.array([buy_signal, sell_signal], dtype=np.float32)
            
            self.previous_price = current_price
        
        return self._hold_action(action_space)
    
    def _contrarian_action(self, observation, action_space, info):
        """Contrarian strategy - buy low, sell high"""
        if info and 'market_psychology' in info:
            psychology = info['market_psychology']
            
            if hasattr(action_space, 'n'):  # Discrete
                if psychology < 0.3:  # Extreme fear - buy
                    return 2  # buy_large
                elif psychology < 0.4:  # Fear - buy small
                    return 1  # buy_small
                elif psychology > 0.7:  # Extreme greed - sell
                    return 4  # sell_large
                elif psychology > 0.6:  # Greed - sell small
                    return 3  # sell_small
                else:
                    return 0  # hold
            else:  # Continuous
                buy_signal = max(0, (0.5 - psychology) * 2)
                sell_signal = max(0, (psychology - 0.5) * 2)
                return np.array([buy_signal, sell_signal], dtype=np.float32)
        
        return self._hold_action(action_space)
    
    def _rsi_action(self, observation, action_space, info):
        """RSI-based strategy"""
        # RSI is the 4th technical indicator in observation
        # obs structure: [price_features, portfolio_features, technical_indicators]
        price_features_len = 50 * 5  # 250
        portfolio_features_len = 3
        rsi_idx = price_features_len + portfolio_features_len  # RSI is first technical indicator
        
        if len(observation) > rsi_idx:
            rsi = observation[rsi_idx] * 100  # Convert back from normalized [0,1] to [0,100]
            
            if hasattr(action_space, 'n'):  # Discrete
                if rsi < 30:  # Oversold - buy
                    return 2  # buy_large
                elif rsi < 40:  # Approaching oversold
                    return 1  # buy_small
                elif rsi > 70:  # Overbought - sell
                    return 4  # sell_large
                elif rsi > 60:  # Approaching overbought
                    return 3  # sell_small
                else:
                    return 0  # hold
            else:  # Continuous
                if rsi < 30:
                    return np.array([0.8, 0.0], dtype=np.float32)
                elif rsi > 70:
                    return np.array([0.0, 0.8], dtype=np.float32)
                else:
                    return np.array([0.0, 0.0], dtype=np.float32)
        
        return self._hold_action(action_space)
    
    def _hold_action(self, action_space):
        """Hold action"""
        if hasattr(action_space, 'n'):  # Discrete
            return 0
        else:  # Continuous
            return np.array([0.0, 0.0], dtype=np.float32)


def test_basic_functionality():
    """Test basic environment functionality"""
    print("=== Testing Basic Functionality ===")
    
    # Test both action types
    for action_type in ["discrete", "continuous"]:
        print(f"\nTesting {action_type} action space...")
        
        config = TradingConfig(initial_balance=10000.0, trading_fee_rate=0.001)
        env = CryptoTradingEnv(config=config, action_type=action_type)
        
        obs, info = env.reset(seed=42)
        print(f"Initial observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test a few random steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                print(f"Step {step}: Reward={reward:.2f}, Portfolio=${info['portfolio_value']:.2f}")
            
            if terminated or truncated:
                break
        
        print(f"Total reward over {step + 1} steps: {total_reward:.2f}")
        env.close()


def test_trading_strategies():
    """Test different trading strategies"""
    print("\n=== Testing Trading Strategies ===")
    
    strategies = ["random", "momentum", "contrarian", "rsi", "hold"]
    results = {}
    
    config = TradingConfig(initial_balance=10000.0, trading_fee_rate=0.001)
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        
        env = CryptoTradingEnv(config=config, action_type="discrete")
        agent = TradingAgent(strategy=strategy)
        
        obs, info = env.reset(seed=42)
        
        portfolio_values = [info.get('portfolio_value', config.initial_balance)]
        total_reward = 0
        trades_made = 0
        
        for step in range(200):
            action = agent.get_action(obs, env.action_space, info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            portfolio_values.append(info['portfolio_value'])
            
            if info.get('trade_info'):
                trades_made += 1
            
            if terminated or truncated:
                break
        
        final_portfolio = portfolio_values[-1]
        roi = (final_portfolio - config.initial_balance) / config.initial_balance * 100
        
        results[strategy] = {
            'final_portfolio': final_portfolio,
            'roi': roi,
            'total_reward': total_reward,
            'trades_made': trades_made,
            'portfolio_history': portfolio_values
        }
        
        print(f"Final portfolio: ${final_portfolio:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Trades made: {trades_made}")
        
        env.close()
    
    return results


def test_market_conditions():
    """Test environment under different market conditions"""
    print("\n=== Testing Market Conditions ===")
    
    config = TradingConfig(initial_balance=10000.0, volatility_base=0.03)
    
    # Test with different market regimes (simulated by different seeds)
    test_scenarios = [
        {"name": "Stable Market", "seed": 42, "steps": 300},
        {"name": "Volatile Market", "seed": 123, "steps": 300},
        {"name": "Trending Market", "seed": 456, "steps": 300},
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting {scenario['name']}...")
        
        env = CryptoTradingEnv(config=config, action_type="discrete")
        agent = TradingAgent(strategy="momentum")
        
        obs, info = env.reset(seed=scenario['seed'])
        
        price_history = []
        market_regimes = []
        psychology_history = []
        
        for step in range(scenario['steps']):
            action = agent.get_action(obs, env.action_space, info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            price_history.append(info['current_price'])
            market_regimes.append(info['market_regime'])
            psychology_history.append(info['market_psychology'])
            
            if terminated or truncated:
                break
        
        # Analyze market behavior
        price_volatility = np.std(np.diff(price_history)) / np.mean(price_history)
        unique_regimes = len(set(market_regimes))
        avg_psychology = np.mean(psychology_history)
        
        print(f"Price volatility: {price_volatility:.4f}")
        print(f"Market regimes encountered: {unique_regimes}")
        print(f"Average market psychology: {avg_psychology:.2f}")
        print(f"Final price: ${price_history[-1]:.2f}")
        
        env.close()


def test_visualization():
    """Test visualization capabilities"""
    print("\n=== Testing Visualization ===")
    
    try:
        config = TradingConfig(initial_balance=10000.0)
        env = CryptoTradingEnv(config=config, action_type="discrete", render_mode="rgb_array")
        
        obs, info = env.reset(seed=42)
        agent = TradingAgent(strategy="rsi")
        
        print("Running environment with visualization...")
        
        for step in range(50):
            action = agent.get_action(obs, env.action_space, info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render every 10 steps
            if step % 10 == 0:
                rgb_array = env.render()
                if rgb_array is not None:
                    print(f"Step {step}: Rendered frame shape {rgb_array.shape}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("Visualization test completed successfully!")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        print("This might be due to missing display or pygame dependencies")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")
    
    config = TradingConfig(initial_balance=1000.0, trading_fee_rate=0.01)  # High fees
    env = CryptoTradingEnv(config=config, action_type="continuous")
    
    obs, info = env.reset(seed=42)
    
    # Test extreme actions
    extreme_actions = [
        np.array([1.0, 0.0]),   # Maximum buy
        np.array([0.0, 1.0]),   # Maximum sell
        np.array([1.0, 1.0]),   # Conflicting signals
        np.array([-1.0, -1.0]), # Negative values
    ]
    
    print("Testing extreme actions...")
    for i, action in enumerate(extreme_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action {i+1}: Portfolio=${info['portfolio_value']:.2f}, "
              f"Cash=${info['cash']:.2f}, Holdings={info['holdings']:.6f}")
        
        if terminated:
            print("Environment terminated due to extreme action")
            break
    
    env.close()


def plot_strategy_comparison(results: Dict):
    """Plot comparison of different trading strategies"""
    print("\n=== Plotting Strategy Comparison ===")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        for strategy, data in results.items():
            ax1.plot(data['portfolio_history'], label=strategy, alpha=0.8)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Final ROI comparison
        strategies = list(results.keys())
        rois = [results[s]['roi'] for s in strategies]
        colors = ['green' if roi > 0 else 'red' for roi in rois]
        
        ax2.bar(strategies, rois, color=colors, alpha=0.7)
        ax2.set_title('Return on Investment (ROI)')
        ax2.set_ylabel('ROI (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Number of trades
        trades = [results[s]['trades_made'] for s in strategies]
        ax3.bar(strategies, trades, color='blue', alpha=0.7)
        ax3.set_title('Number of Trades')
        ax3.set_ylabel('Trades')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Risk-adjusted returns (Sharpe-like ratio)
        sharpe_ratios = []
        for strategy in strategies:
            portfolio_history = np.array(results[strategy]['portfolio_history'])
            returns = np.diff(portfolio_history) / portfolio_history[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)
        
        ax4.bar(strategies, sharpe_ratios, color='purple', alpha=0.7)
        ax4.set_title('Risk-Adjusted Returns (Sharpe-like)')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/strategy_comparison.png', dpi=150, bbox_inches='tight')
        print("Strategy comparison plot saved as 'strategy_comparison.png'")
        plt.close()
        
    except Exception as e:
        print(f"Plotting failed: {e}")


def run_performance_benchmark():
    """Benchmark environment performance"""
    print("\n=== Performance Benchmark ===")
    
    config = TradingConfig(initial_balance=10000.0)
    env = CryptoTradingEnv(config=config, action_type="discrete")
    
    # Warm up
    obs, info = env.reset(seed=42)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Benchmark
    num_steps = 1000
    start_time = time.time()
    
    obs, info = env.reset(seed=42)
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    end_time = time.time()
    
    steps_per_second = num_steps / (end_time - start_time)
    print(f"Performance: {steps_per_second:.2f} steps/second")
    print(f"Time per step: {(end_time - start_time) / num_steps * 1000:.2f} ms")
    
    env.close()


def main():
    """Run all tests"""
    print("Crypto Trading Environment Test Suite")
    print("=" * 50)
    
    try:
        # Basic functionality
        test_basic_functionality()
        
        # Trading strategies
        strategy_results = test_trading_strategies()
        
        # Market conditions
        test_market_conditions()
        
        # Visualization (optional)
        test_visualization()
        
        # Edge cases
        test_edge_cases()
        
        # Performance benchmark
        run_performance_benchmark()
        
        # Plot results
        if strategy_results:
            plot_strategy_comparison(strategy_results)
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
        # Summary
        print("\n=== Test Summary ===")
        if strategy_results:
            best_strategy = max(strategy_results.keys(), 
                              key=lambda k: strategy_results[k]['roi'])
            best_roi = strategy_results[best_strategy]['roi']
            print(f"Best performing strategy: {best_strategy} (ROI: {best_roi:.2f}%)")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()