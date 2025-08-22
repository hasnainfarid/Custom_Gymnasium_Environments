
<img width="1117" height="662" alt="Crypto" src="https://github.com/user-attachments/assets/c00a20df-4360-467e-b9a7-31fdaf864836" />

# Crypto Trading Environment

A reinforcement learning environment for cryptocurrency trading simulation built with Gymnasium, featuring realistic market dynamics and technical analysis capabilities.

## Features

- **Market Simulation**: Dynamic market regimes with realistic volatility
- **Technical Indicators**: RSI, MACD, Bollinger Bands integration
- **Trading Mechanics**: Transaction costs, slippage, and portfolio management
- **Visualization**: Real-time Pygame charts and Matplotlib analysis plots
- **Flexible Actions**: Both continuous and discrete action spaces

## Environment Specifications

- **Observation Space**: Price history, portfolio data, technical indicators
- **Action Space**: Continuous [buy_amount, sell_amount] or discrete 5 actions
- **Rewards**: Portfolio value changes with trading cost penalties
- **Market Features**: OHLCV data, fear/greed psychology, regime detection

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
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Run Demo
```bash
python quick_demo.py
```

### Run Tests
```bash
python test_crypto_trading.py
```

## Configuration

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

The environment simulates realistic crypto market conditions including bull runs, bear markets, crashes, and sideways movement with dynamic volatility and market psychology factors.

## Requirements

- Python 3.7+
- gymnasium>=0.28.0
- numpy>=1.19.0
- pygame>=2.0.0
- matplotlib>=3.3.0

## License

MIT License - see LICENSE file for details.

## Author

**Hasnain Fareed** - 2025
