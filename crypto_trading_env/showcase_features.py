#!/usr/bin/env python3
"""
Showcase Features - Crypto Trading Environment
Demonstrates all key features of the trading environment in a compact format.
"""

import numpy as np
from crypto_trading_env import CryptoTradingEnv, TradingConfig, MarketRegime
import matplotlib.pyplot as plt


def showcase_environment_features():
    """Showcase all key features of the environment"""
    print("🚀 CRYPTO TRADING ENVIRONMENT SHOWCASE")
    print("=" * 60)
    
    # 1. Environment Creation with Custom Config
    print("\n📊 1. ENVIRONMENT CONFIGURATION")
    config = TradingConfig(
        initial_balance=10000.0,
        trading_fee_rate=0.002,    # 0.2% trading fee
        slippage_rate=0.001,       # 0.1% slippage
        history_length=50,
        volatility_base=0.025,     # Higher volatility
        market_psychology_factor=0.15
    )
    
    print(f"   💰 Initial Balance: ${config.initial_balance:,.2f}")
    print(f"   📈 Trading Fee: {config.trading_fee_rate*100:.2f}%")
    print(f"   🎯 Slippage: {config.slippage_rate*100:.2f}%")
    print(f"   📊 History Length: {config.history_length} candles")
    
    # 2. Both Action Spaces
    print("\n🎮 2. ACTION SPACES")
    
    for action_type in ["discrete", "continuous"]:
        env = CryptoTradingEnv(config=config, action_type=action_type)
        obs, info = env.reset(seed=42)
        
        print(f"\n   {action_type.upper()} Actions:")
        print(f"   - Action Space: {env.action_space}")
        print(f"   - Observation Shape: {obs.shape}")
        
        if action_type == "discrete":
            print("   - Actions: 0=hold, 1=buy_small, 2=buy_large, 3=sell_small, 4=sell_large")
        else:
            print("   - Actions: [buy_amount, sell_amount] in [-1, 1]")
        
        env.close()
    
    # 3. Market Simulation Features
    print("\n🌊 3. MARKET SIMULATION")
    env = CryptoTradingEnv(config=config, action_type="discrete")
    obs, info = env.reset(seed=123)
    
    regimes_seen = set()
    psychology_values = []
    price_changes = []
    
    for step in range(100):
        action = 0  # Hold to observe market
        obs, reward, terminated, truncated, info = env.step(action)
        
        regimes_seen.add(info['market_regime'])
        psychology_values.append(info['market_psychology'])
        price_changes.append(info['current_price'])
        
        if terminated or truncated:
            break
    
    print(f"   📊 Market Regimes Observed: {', '.join(regimes_seen)}")
    print(f"   🧠 Psychology Range: {min(psychology_values):.2f} - {max(psychology_values):.2f}")
    print(f"   💹 Price Volatility: {np.std(np.diff(price_changes))/np.mean(price_changes)*100:.2f}%")
    
    env.close()
    
    # 4. Technical Indicators
    print("\n📈 4. TECHNICAL INDICATORS")
    env = CryptoTradingEnv(config=config, action_type="discrete")
    obs, info = env.reset(seed=456)
    
    # Run for a bit to get meaningful indicators
    for _ in range(30):
        obs, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    
    # Extract indicators from observation
    price_features_len = 50 * 5  # OHLCV for 50 candles
    portfolio_features_len = 3
    indicators_start = price_features_len + portfolio_features_len
    
    rsi = obs[indicators_start] * 100  # Convert from [0,1] to [0,100]
    macd_line = obs[indicators_start + 1]
    bb_position = obs[indicators_start + 4]
    market_psychology = obs[indicators_start + 7]
    
    print(f"   📊 RSI: {rsi:.1f}")
    print(f"   📈 MACD Line: {macd_line:.4f}")
    print(f"   🎯 Bollinger Band Position: {bb_position:.2f}")
    print(f"   🧠 Market Psychology: {market_psychology:.2f}")
    
    env.close()
    
    # 5. Trading Mechanics
    print("\n💼 5. TRADING MECHANICS")
    env = CryptoTradingEnv(config=config, action_type="discrete")
    obs, info = env.reset(seed=789)
    
    # Get initial state
    initial_cash = env.cash
    initial_holdings = env.holdings
    
    print(f"   💰 Initial Cash: ${initial_cash:.2f}")
    print(f"   🪙 Initial Holdings: {initial_holdings:.6f} BTC")
    
    # Execute a buy order
    obs, reward, terminated, truncated, info = env.step(2)  # buy_large
    
    if info.get('trade_info'):
        trade = info['trade_info']
        print(f"\n   🛒 BUY ORDER EXECUTED:")
        print(f"   - Amount: ${trade['amount']:.2f}")
        print(f"   - Price: ${trade['price']:.2f}")
        print(f"   - Fee: ${trade['fee']:.2f}")
        print(f"   - Slippage: ${trade['slippage']:.2f}")
        print(f"   - Crypto Received: {trade['crypto_amount']:.6f} BTC")
    
    # Execute a sell order
    obs, reward, terminated, truncated, info = env.step(4)  # sell_large
    
    if info.get('trade_info'):
        trade = info['trade_info']
        print(f"\n   💰 SELL ORDER EXECUTED:")
        print(f"   - Crypto Sold: {trade['crypto_amount']:.6f} BTC")
        print(f"   - Price: ${trade['price']:.2f}")
        print(f"   - Fee: ${trade['fee']:.2f}")
        print(f"   - Slippage: ${trade['slippage']:.2f}")
        print(f"   - Cash Received: ${trade['amount']:.2f}")
    
    print(f"\n   📊 Final Portfolio: ${info['portfolio_value']:.2f}")
    
    env.close()
    
    # 6. Performance Metrics
    print("\n⚡ 6. PERFORMANCE BENCHMARK")
    env = CryptoTradingEnv(config=config, action_type="discrete")
    
    import time
    start_time = time.time()
    
    obs, info = env.reset()
    for _ in range(1000):
        action = np.random.choice(5)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    end_time = time.time()
    steps_per_second = 1000 / (end_time - start_time)
    
    print(f"   🚀 Performance: {steps_per_second:.0f} steps/second")
    print(f"   ⏱️  Time per step: {(end_time - start_time)/1000*1000:.2f} ms")
    
    env.close()
    
    print("\n✅ SHOWCASE COMPLETE!")
    print("🎯 Key Features Demonstrated:")
    print("   • Realistic market simulation with multiple regimes")
    print("   • Advanced trading mechanics (fees, slippage)")
    print("   • Technical indicators (RSI, MACD, Bollinger Bands)")
    print("   • Market psychology simulation")
    print("   • Both discrete and continuous action spaces")
    print("   • High-performance execution")
    print("   • Comprehensive state representation")


def create_feature_summary_plot():
    """Create a visual summary of key features"""
    try:
        print("\n📊 Creating Feature Summary Plot...")
        
        # Generate sample data from different strategies
        config = TradingConfig(initial_balance=10000.0)
        
        strategies = {
            'Random': [],
            'Momentum': [],
            'Contrarian': []
        }
        
        for strategy_name in strategies.keys():
            env = CryptoTradingEnv(config=config, action_type="discrete")
            obs, info = env.reset(seed=42)
            
            portfolio_values = [info['portfolio_value']]
            
            for step in range(100):
                if strategy_name == 'Random':
                    action = np.random.choice(5)
                elif strategy_name == 'Momentum':
                    action = 1 if step % 10 < 3 else 0  # Buy occasionally
                else:  # Contrarian
                    action = 3 if step % 15 < 2 else 0  # Sell occasionally
                
                obs, reward, terminated, truncated, info = env.step(action)
                portfolio_values.append(info['portfolio_value'])
                
                if terminated or truncated:
                    break
            
            strategies[strategy_name] = portfolio_values
            env.close()
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for strategy_name, values in strategies.items():
            ax.plot(values, label=strategy_name, linewidth=2, alpha=0.8)
        
        ax.set_title('Crypto Trading Environment - Strategy Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Trading Steps', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.text(0.02, 0.98, 'Features:\n• Market Psychology\n• Technical Indicators\n• Realistic Slippage\n• Multiple Market Regimes', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('/workspace/crypto_trading_showcase.png', dpi=150, bbox_inches='tight')
        print("   💾 Plot saved as 'crypto_trading_showcase.png'")
        plt.close()
        
    except Exception as e:
        print(f"   ❌ Plot creation failed: {e}")


if __name__ == "__main__":
    showcase_environment_features()
    create_feature_summary_plot()
    
    print("\n" + "="*60)
    print("🎉 CRYPTO TRADING ENVIRONMENT READY FOR USE!")
    print("📚 See README.md for detailed documentation")
    print("🧪 Run test_crypto_trading.py for comprehensive tests")
    print("🚀 Run quick_demo.py for a simple demonstration")