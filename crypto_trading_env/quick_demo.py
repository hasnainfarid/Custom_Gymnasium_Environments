#!/usr/bin/env python3
"""
Quick Demo of Crypto Trading Environment
A simple demonstration showing how to use the environment.
"""

import numpy as np
from crypto_trading_env import CryptoTradingEnv, TradingConfig


def simple_rsi_strategy_demo():
    """Demonstrate a simple RSI-based trading strategy"""
    print("=== Crypto Trading Environment Demo ===")
    print("Running RSI-based trading strategy...")
    
    # Create environment with custom configuration
    config = TradingConfig(
        initial_balance=10000.0,
        trading_fee_rate=0.001,  # 0.1% trading fee
        slippage_rate=0.0005,    # 0.05% slippage
        history_length=50
    )
    
    env = CryptoTradingEnv(config=config, action_type="discrete")
    
    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"Initial portfolio value: ${info.get('portfolio_value', config.initial_balance):.2f}")
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    
    # Simple RSI strategy
    total_reward = 0
    portfolio_values = []
    trades_executed = 0
    
    for step in range(200):
        # Extract RSI from observation (simplified)
        # In real implementation, you'd parse the observation properly
        rsi_idx = 50 * 5 + 3  # After price history and portfolio features
        if len(observation) > rsi_idx:
            rsi = observation[rsi_idx] * 100  # Convert from [0,1] to [0,100]
            
            # RSI strategy
            if rsi < 30:  # Oversold - buy
                action = 2  # buy_large
            elif rsi > 70:  # Overbought - sell
                action = 4  # sell_large
            elif rsi < 40:  # Approaching oversold
                action = 1  # buy_small
            elif rsi > 60:  # Approaching overbought
                action = 3  # sell_small
            else:
                action = 0  # hold
        else:
            action = 0  # hold if we can't calculate RSI
        
        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        portfolio_values.append(info['portfolio_value'])
        
        if info.get('trade_info'):
            trades_executed += 1
            trade = info['trade_info']
            print(f"Step {step}: {trade['action'].upper()} - "
                  f"Amount: ${trade['amount']:.2f}, "
                  f"Price: ${trade['price']:.2f}, "
                  f"Portfolio: ${info['portfolio_value']:.2f}")
        
        # Print periodic updates
        if step % 50 == 0:
            print(f"Step {step}: Portfolio=${info['portfolio_value']:.2f}, "
                  f"Price=${info['current_price']:.2f}, "
                  f"Market={info['market_regime']}, "
                  f"RSI={rsi:.1f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    # Final results
    final_portfolio = portfolio_values[-1]
    roi = (final_portfolio - config.initial_balance) / config.initial_balance * 100
    
    print("\n=== Results ===")
    print(f"Initial balance: ${config.initial_balance:.2f}")
    print(f"Final portfolio: ${final_portfolio:.2f}")
    print(f"Return on Investment: {roi:.2f}%")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Trades executed: {trades_executed}")
    print(f"Steps completed: {step + 1}")
    
    # Performance metrics
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values)
        
        print(f"Volatility (annualized): {volatility:.2f}")
        print(f"Max drawdown: ${max_drawdown:.2f}")
        
        if volatility > 0:
            sharpe_ratio = (roi / 100) / volatility
            print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    
    env.close()
    return final_portfolio, roi


def test_both_action_types():
    """Test both continuous and discrete action spaces"""
    print("\n=== Testing Both Action Types ===")
    
    config = TradingConfig(initial_balance=10000.0)
    
    for action_type in ["discrete", "continuous"]:
        print(f"\nTesting {action_type} actions...")
        
        env = CryptoTradingEnv(config=config, action_type=action_type)
        observation, info = env.reset(seed=42)
        
        total_reward = 0
        for step in range(50):
            if action_type == "discrete":
                # Simple momentum strategy for discrete
                if step > 0 and info.get('current_price', 0) > prev_price:
                    action = 1  # buy_small
                elif step > 0 and info.get('current_price', 0) < prev_price:
                    action = 3  # sell_small
                else:
                    action = 0  # hold
            else:
                # Simple strategy for continuous
                if step > 0:
                    price_change = (info.get('current_price', 0) - prev_price) / prev_price
                    buy_signal = max(0, price_change * 5)
                    sell_signal = max(0, -price_change * 5)
                    action = np.array([buy_signal, sell_signal], dtype=np.float32)
                else:
                    action = np.array([0.0, 0.0], dtype=np.float32)
            
            prev_price = info.get('current_price', 0)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        final_value = info.get('portfolio_value', config.initial_balance)
        roi = (final_value - config.initial_balance) / config.initial_balance * 100
        
        print(f"Final portfolio: ${final_value:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Total reward: {total_reward:.2f}")
        
        env.close()


if __name__ == "__main__":
    try:
        # Run main demo
        final_portfolio, roi = simple_rsi_strategy_demo()
        
        # Test both action types
        test_both_action_types()
        
        print("\n=== Demo Complete ===")
        print("The crypto trading environment is working correctly!")
        print(f"RSI strategy achieved {roi:.2f}% return")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()