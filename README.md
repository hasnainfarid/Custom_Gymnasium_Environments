# Crypto Trading Gymnasium Environment

A comprehensive cryptocurrency trading environment built with OpenAI Gymnasium, featuring realistic market simulation, technical indicators, and advanced trading mechanics.

## Features

### Core Environment
- **Gymnasium Interface**: Standard RL environment with configurable action/observation spaces
- **Realistic Market Simulation**: Dynamic market regimes (bull runs, bear markets, crashes, sideways movement)
- **Market Psychology**: Fear & greed index affecting price movements
- **Advanced Trading Mechanics**: Transaction costs, slippage, and realistic volatility

### State Space
- **Price History**: OHLCV data for last 50 candles
- **Portfolio Information**: Cash, holdings, total portfolio value
- **Technical Indicators**: 
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- **Market Psychology**: Current fear/greed sentiment

### Action Space
- **Continuous**: `[buy_amount, sell_amount]` normalized to [-1, 1]
- **Discrete**: 5 actions (hold, buy_small, buy_large, sell_small, sell_large)

### Rewards
- Portfolio value change minus trading fees and slippage
- Small penalty for inaction to encourage active trading

### Visualization
- **Pygame Rendering**: Real-time candlestick charts with portfolio overlay
- **Matplotlib Charts**: Publication-ready plots for analysis
- **Market Information**: Current regime, psychology, and technical indicators

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from crypto_trading_env import CryptoTradingEnv, TradingConfig

# Create environment
config = TradingConfig(initial_balance=10000.0, trading_fee_rate=0.001)
env = CryptoTradingEnv(config=config, action_type="discrete")

# Reset and run
observation, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Run Demo

```bash
python quick_demo.py
```

### Run Comprehensive Tests

```bash
python test_crypto_trading.py
```

## Environment Configuration

```python
@dataclass
class TradingConfig:
    initial_balance: float = 10000.0      # Starting cash
    trading_fee_rate: float = 0.001       # 0.1% per trade
    slippage_rate: float = 0.0005         # 0.05% slippage
    history_length: int = 50              # Candles in state
    min_price: float = 100.0              # Min asset price
    max_price: float = 100000.0           # Max asset price
    volatility_base: float = 0.02         # Base volatility
    market_psychology_factor: float = 0.1  # Psychology influence
```

## Market Simulation

The environment simulates realistic crypto market conditions:

### Market Regimes
- **Bull Run**: Strong upward trends with moderate volatility
- **Bear Market**: Downward trends with increased volatility
- **Sideways**: Range-bound trading with low volatility
- **Crash**: Sudden price drops with extreme volatility
- **Recovery**: Bounce-back periods with high volatility

### Market Psychology
- Dynamic fear & greed index (0-1 scale)
- Influences price movements and volatility
- Updates based on recent price action
- Mean-reverting behavior

### Advanced Features
- **Slippage**: Price impact based on order size
- **Transaction Costs**: Configurable trading fees
- **Volume Impact**: Higher volume increases price stability
- **Regime Transitions**: Probabilistic market state changes

## Trading Strategies (Examples)

The test suite includes several example strategies:

1. **Random**: Baseline random trading
2. **Momentum**: Buy on upward price movement, sell on downward
3. **Contrarian**: Buy during fear, sell during greed
4. **RSI**: Oversold/overbought signals
5. **Hold**: Buy and hold benchmark

## Visualization

### Pygame Rendering
```python
env = CryptoTradingEnv(render_mode="human")
env.reset()
for step in range(100):
    action = env.action_space.sample()
    env.step(action)
    env.render()  # Shows candlestick chart + portfolio info
```

### Matplotlib Analysis
```python
env = CryptoTradingEnv(render_mode="rgb_array")
rgb_array = env.render()  # Returns numpy array for analysis
```

## File Structure

```
/workspace/
├── crypto_trading_env.py    # Main environment implementation
├── test_crypto_trading.py   # Comprehensive test suite
├── quick_demo.py           # Simple demonstration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Performance

- **Speed**: ~1000+ steps/second on modern hardware
- **Memory**: Efficient circular buffers for price history
- **Scalability**: Configurable history length and features

## Testing

The test suite covers:
- Basic functionality verification
- Multiple trading strategies comparison
- Different market conditions
- Edge cases and error handling
- Performance benchmarking
- Visualization capabilities

Run all tests:
```bash
python test_crypto_trading.py
```

## Advanced Usage

### Custom Market Conditions
```python
# Force specific market regime
env.market_sim.current_regime = MarketRegime.CRASH
env.market_sim.trend_strength = -0.8
```

### Custom Technical Indicators
```python
# Access price history for custom indicators
prices = np.array([candle[3] for candle in env.price_history])  # Close prices
custom_sma = np.mean(prices[-20:])  # 20-period SMA
```

### Multiple Environments
```python
# Run parallel environments for strategy comparison
envs = [CryptoTradingEnv() for _ in range(4)]
# ... parallel execution logic
```

## Contributing

Feel free to extend the environment with:
- Additional technical indicators
- More sophisticated market regimes
- Order book simulation
- Multi-asset trading
- News sentiment integration

## License

Open source - feel free to use and modify for your trading research and education.

## Disclaimer

This environment is for educational and research purposes only. It does not constitute financial advice. Real trading involves significant risk of loss.