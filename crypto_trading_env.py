"""
Crypto Trading Gymnasium Environment
A comprehensive trading environment with realistic market simulation, technical indicators, and visualization.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pygame
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    BULL_RUN = "bull_run"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    CRASH = "crash"
    RECOVERY = "recovery"


@dataclass
class TradingConfig:
    """Configuration for the trading environment"""
    initial_balance: float = 10000.0
    trading_fee_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005    # 0.05% slippage
    history_length: int = 50
    min_price: float = 100.0
    max_price: float = 100000.0
    volatility_base: float = 0.02
    market_psychology_factor: float = 0.1
    

class TechnicalIndicators:
    """Calculate technical indicators for trading state"""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        if len(prices) < period:
            current_price = prices[-1] if len(prices) > 0 else 1000.0
            return current_price * 1.02, current_price, current_price * 0.98
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD (macd_line, signal_line, histogram)"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        ema_fast = TechnicalIndicators._ema(prices, fast)
        ema_slow = TechnicalIndicators._ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # For signal line, we need MACD history
        if len(prices) < slow + signal:
            signal_line = 0.0
        else:
            macd_values = []
            for i in range(slow, len(prices) + 1):
                ema_f = TechnicalIndicators._ema(prices[:i], fast)
                ema_s = TechnicalIndicators._ema(prices[:i], slow)
                macd_values.append(ema_f - ema_s)
            
            signal_line = TechnicalIndicators._ema(np.array(macd_values), signal)
        
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) == 0:
            return 0.0
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema


class MarketSimulator:
    """Simulate realistic crypto market conditions"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_duration = 0
        self.trend_strength = 0.0
        self.market_psychology = 0.5  # 0 = extreme fear, 1 = extreme greed
        
    def generate_next_price(self, current_price: float, volume: float = 1.0) -> float:
        """Generate next price based on market regime and psychology"""
        # Update market regime periodically
        if random.random() < 0.01:  # 1% chance to change regime
            self._update_market_regime()
        
        # Base volatility adjusted by regime
        volatility = self._get_regime_volatility()
        
        # Market psychology influence
        psychology_drift = (self.market_psychology - 0.5) * self.config.market_psychology_factor
        
        # Trend component based on regime
        trend_component = self._get_trend_component()
        
        # Random walk component
        random_component = np.random.normal(0, volatility)
        
        # Volume impact (higher volume = more stability)
        volume_factor = 1.0 / (1.0 + volume * 0.1)
        
        # Calculate price change
        price_change_pct = (trend_component + psychology_drift + random_component) * volume_factor
        
        new_price = current_price * (1 + price_change_pct)
        
        # Ensure price stays within bounds
        new_price = np.clip(new_price, self.config.min_price, self.config.max_price)
        
        # Update market psychology based on price movement
        self._update_market_psychology(price_change_pct)
        
        return new_price
    
    def _update_market_regime(self):
        """Update market regime based on probabilities"""
        regime_transitions = {
            MarketRegime.BULL_RUN: [MarketRegime.SIDEWAYS, MarketRegime.CRASH],
            MarketRegime.BEAR_MARKET: [MarketRegime.SIDEWAYS, MarketRegime.RECOVERY],
            MarketRegime.SIDEWAYS: [MarketRegime.BULL_RUN, MarketRegime.BEAR_MARKET],
            MarketRegime.CRASH: [MarketRegime.RECOVERY, MarketRegime.BEAR_MARKET],
            MarketRegime.RECOVERY: [MarketRegime.BULL_RUN, MarketRegime.SIDEWAYS]
        }
        
        possible_regimes = regime_transitions[self.current_regime]
        self.current_regime = random.choice(possible_regimes)
        self.regime_duration = 0
        
        # Update trend strength
        if self.current_regime in [MarketRegime.BULL_RUN, MarketRegime.RECOVERY]:
            self.trend_strength = random.uniform(0.5, 1.0)
        elif self.current_regime in [MarketRegime.BEAR_MARKET, MarketRegime.CRASH]:
            self.trend_strength = random.uniform(-1.0, -0.5)
        else:
            self.trend_strength = random.uniform(-0.2, 0.2)
    
    def _get_regime_volatility(self) -> float:
        """Get volatility based on current market regime"""
        volatility_multipliers = {
            MarketRegime.BULL_RUN: 1.2,
            MarketRegime.BEAR_MARKET: 1.5,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.CRASH: 3.0,
            MarketRegime.RECOVERY: 2.0
        }
        
        return self.config.volatility_base * volatility_multipliers[self.current_regime]
    
    def _get_trend_component(self) -> float:
        """Get trend component based on regime"""
        regime_trends = {
            MarketRegime.BULL_RUN: 0.001,
            MarketRegime.BEAR_MARKET: -0.001,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.CRASH: -0.005,
            MarketRegime.RECOVERY: 0.002
        }
        
        base_trend = regime_trends[self.current_regime]
        return base_trend * self.trend_strength
    
    def _update_market_psychology(self, price_change_pct: float):
        """Update market psychology based on price movements"""
        # Fear and greed index update
        psychology_change = price_change_pct * 10  # Amplify for psychology
        self.market_psychology += psychology_change
        self.market_psychology = np.clip(self.market_psychology, 0.0, 1.0)
        
        # Add some mean reversion
        self.market_psychology += (0.5 - self.market_psychology) * 0.01


class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency Trading Environment
    
    State Space:
    - Price history (OHLCV for last 50 candles)
    - Portfolio value, cash, holdings
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Market psychology indicators
    
    Action Space:
    - Continuous: [buy_amount, sell_amount] normalized to [-1, 1]
    - Discrete: 0=hold, 1=buy_small, 2=buy_large, 3=sell_small, 4=sell_large
    
    Reward:
    - Portfolio value change minus trading fees and slippage
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[TradingConfig] = None,
        action_type: str = "continuous",  # "continuous" or "discrete"
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.config = config or TradingConfig()
        self.action_type = action_type
        self.render_mode = render_mode
        
        # Initialize market simulator
        self.market_sim = MarketSimulator(self.config)
        
        # Trading state
        self.cash = self.config.initial_balance
        self.holdings = 0.0
        self.price_history = []
        self.volume_history = []
        self.portfolio_history = []
        self.trade_history = []
        
        # Action and observation spaces
        self._setup_spaces()
        
        # Visualization
        self.screen = None
        self.clock = None
        self.screen_width = 1200
        self.screen_height = 800
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 1000
        
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Observation space: [price_features, portfolio_features, technical_indicators]
        # Price features: OHLCV for last 50 candles (50 * 5 = 250)
        # Portfolio features: cash, holdings, portfolio_value (3)
        # Technical indicators: RSI, MACD (3), Bollinger Bands (3), market psychology (1) = 7
        obs_dim = self.config.history_length * 5 + 3 + 7
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        if self.action_type == "continuous":
            # [buy_amount, sell_amount] normalized to [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )
        else:
            # Discrete actions: 0=hold, 1=buy_small, 2=buy_large, 3=sell_small, 4=sell_large
            self.action_space = spaces.Discrete(5)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset trading state
        self.cash = self.config.initial_balance
        self.holdings = 0.0
        self.current_step = 0
        
        # Initialize price history with some realistic starting data
        initial_price = 50000.0  # Starting BTC price
        self.price_history = []
        self.volume_history = []
        self.portfolio_history = []
        self.trade_history = []
        
        # Generate initial price history
        current_price = initial_price
        for _ in range(self.config.history_length):
            volume = random.uniform(0.5, 2.0)
            current_price = self.market_sim.generate_next_price(current_price, volume)
            
            # Generate OHLCV data
            high = current_price * random.uniform(1.0, 1.02)
            low = current_price * random.uniform(0.98, 1.0)
            open_price = current_price * random.uniform(0.99, 1.01)
            close_price = current_price
            
            self.price_history.append([open_price, high, low, close_price, volume])
            self.volume_history.append(volume)
        
        # Initialize portfolio history
        initial_portfolio_value = self.cash + self.holdings * current_price
        self.portfolio_history.append(initial_portfolio_value)
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one trading step"""
        # Execute action
        reward, trade_info = self._execute_action(action)
        
        # Generate next market data
        current_price = self.price_history[-1][3]  # Last close price
        volume = random.uniform(0.5, 2.0)
        new_price = self.market_sim.generate_next_price(current_price, volume)
        
        # Generate new OHLCV
        high = new_price * random.uniform(1.0, 1.02)
        low = new_price * random.uniform(0.98, 1.0)
        open_price = current_price
        close_price = new_price
        
        # Update history
        self.price_history.append([open_price, high, low, close_price, volume])
        self.volume_history.append(volume)
        
        # Keep history length constant
        if len(self.price_history) > self.config.history_length:
            self.price_history.pop(0)
            self.volume_history.pop(0)
        
        # Update portfolio history
        portfolio_value = self.cash + self.holdings * close_price
        self.portfolio_history.append(portfolio_value)
        
        # Store trade info
        if trade_info:
            self.trade_history.append({
                'step': self.current_step,
                'price': close_price,
                **trade_info
            })
        
        self.current_step += 1
        
        # Check termination conditions
        terminated = (
            self.current_step >= self.max_steps or
            portfolio_value <= 0 or
            portfolio_value >= self.config.initial_balance * 10
        )
        
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'current_price': close_price,
            'market_regime': self.market_sim.current_regime.value,
            'market_psychology': self.market_sim.market_psychology,
            'trade_info': trade_info
        }
    
    def _execute_action(self, action) -> Tuple[float, Optional[Dict]]:
        """Execute trading action and return reward and trade info"""
        current_price = self.price_history[-1][3]  # Current close price
        initial_portfolio_value = self.cash + self.holdings * current_price
        
        trade_info = None
        
        if self.action_type == "continuous":
            buy_amount, sell_amount = action
            
            # Convert to actual trade amounts
            max_buy = self.cash * 0.1  # Max 10% of cash per trade
            max_sell = self.holdings * 0.1  # Max 10% of holdings per trade
            
            buy_amount = np.clip(buy_amount, 0, 1) * max_buy
            sell_amount = np.clip(sell_amount, 0, 1) * max_sell
            
            if buy_amount > sell_amount and buy_amount > 0:
                # Execute buy
                trade_info = self._execute_buy(buy_amount, current_price)
            elif sell_amount > 0:
                # Execute sell
                trade_info = self._execute_sell(sell_amount, current_price)
        
        else:  # Discrete actions
            if action == 1:  # buy_small
                buy_amount = self.cash * 0.05
                trade_info = self._execute_buy(buy_amount, current_price)
            elif action == 2:  # buy_large
                buy_amount = self.cash * 0.2
                trade_info = self._execute_buy(buy_amount, current_price)
            elif action == 3:  # sell_small
                sell_amount = self.holdings * 0.05
                trade_info = self._execute_sell(sell_amount, current_price)
            elif action == 4:  # sell_large
                sell_amount = self.holdings * 0.2
                trade_info = self._execute_sell(sell_amount, current_price)
            # action == 0 is hold, do nothing
        
        # Calculate reward
        final_portfolio_value = self.cash + self.holdings * current_price
        reward = final_portfolio_value - initial_portfolio_value
        
        # Add small penalty for inaction to encourage trading
        if trade_info is None:
            reward -= 1.0
        
        return reward, trade_info
    
    def _execute_buy(self, amount: float, price: float) -> Optional[Dict]:
        """Execute buy order with fees and slippage"""
        if amount <= 0 or self.cash < amount:
            return None
        
        # Apply slippage (price increases for buy orders)
        slippage = price * self.config.slippage_rate * random.uniform(0.5, 1.5)
        effective_price = price + slippage
        
        # Calculate fees
        fee = amount * self.config.trading_fee_rate
        
        # Calculate how much crypto we can buy
        net_amount = amount - fee
        crypto_bought = net_amount / effective_price
        
        # Update portfolio
        self.cash -= amount
        self.holdings += crypto_bought
        
        return {
            'action': 'buy',
            'amount': amount,
            'crypto_amount': crypto_bought,
            'price': effective_price,
            'fee': fee,
            'slippage': slippage
        }
    
    def _execute_sell(self, crypto_amount: float, price: float) -> Optional[Dict]:
        """Execute sell order with fees and slippage"""
        if crypto_amount <= 0 or self.holdings < crypto_amount:
            return None
        
        # Apply slippage (price decreases for sell orders)
        slippage = price * self.config.slippage_rate * random.uniform(0.5, 1.5)
        effective_price = price - slippage
        
        # Calculate cash received
        cash_received = crypto_amount * effective_price
        fee = cash_received * self.config.trading_fee_rate
        net_cash = cash_received - fee
        
        # Update portfolio
        self.holdings -= crypto_amount
        self.cash += net_cash
        
        return {
            'action': 'sell',
            'amount': net_cash,
            'crypto_amount': crypto_amount,
            'price': effective_price,
            'fee': fee,
            'slippage': slippage
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        obs = []
        
        # Price features (OHLCV normalized)
        price_data = np.array(self.price_history)
        if len(price_data) > 0:
            # Normalize prices relative to current price
            current_price = price_data[-1, 3]
            normalized_prices = price_data / current_price
            obs.extend(normalized_prices.flatten())
        else:
            obs.extend([1.0] * (self.config.history_length * 5))
        
        # Portfolio features (normalized)
        current_price = self.price_history[-1][3] if self.price_history else 1.0
        portfolio_value = self.cash + self.holdings * current_price
        
        obs.extend([
            self.cash / self.config.initial_balance,
            self.holdings * current_price / self.config.initial_balance,
            portfolio_value / self.config.initial_balance
        ])
        
        # Technical indicators
        if len(self.price_history) > 0:
            close_prices = np.array([candle[3] for candle in self.price_history])
            
            # RSI (normalized to [0, 1])
            rsi = TechnicalIndicators.rsi(close_prices) / 100.0
            obs.append(rsi)
            
            # MACD (normalized)
            macd_line, signal_line, histogram = TechnicalIndicators.macd(close_prices)
            price_range = max(close_prices) - min(close_prices)
            if price_range > 0:
                obs.extend([
                    macd_line / price_range,
                    signal_line / price_range,
                    histogram / price_range
                ])
            else:
                obs.extend([0.0, 0.0, 0.0])
            
            # Bollinger Bands (normalized)
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close_prices)
            current_price = close_prices[-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.0
            obs.extend([bb_position, bb_width, (current_price - bb_middle) / bb_middle if bb_middle > 0 else 0.0])
        else:
            obs.extend([0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        
        # Market psychology
        obs.append(self.market_sim.market_psychology)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        """Render the environment using pygame"""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render for human viewing"""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Crypto Trading Environment")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill((20, 20, 30))
        
        # Draw candlestick chart
        self._draw_candlestick_chart()
        
        # Draw portfolio info
        self._draw_portfolio_info()
        
        # Draw market info
        self._draw_market_info()
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_rgb_array(self):
        """Render as RGB array"""
        # For simplicity, use matplotlib to generate the chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        if len(self.price_history) > 0:
            # Candlestick chart
            self._plot_candlesticks(ax1)
            
            # Portfolio chart
            if len(self.portfolio_history) > 1:
                ax2.plot(self.portfolio_history, color='green', linewidth=2)
                ax2.set_title('Portfolio Value')
                ax2.set_ylabel('Value ($)')
        
        plt.tight_layout()
        
        # Convert to RGB array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return buf
    
    def _draw_candlestick_chart(self):
        """Draw candlestick chart on pygame surface"""
        if len(self.price_history) < 2:
            return
        
        chart_rect = pygame.Rect(50, 50, 800, 400)
        pygame.draw.rect(self.screen, (40, 40, 50), chart_rect)
        
        # Calculate price range
        all_prices = []
        for candle in self.price_history[-50:]:  # Show last 50 candles
            all_prices.extend([candle[1], candle[2]])  # high, low
        
        if not all_prices:
            return
        
        min_price = min(all_prices)
        max_price = max(all_prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            return
        
        # Draw candles
        candle_width = chart_rect.width // len(self.price_history[-50:])
        
        for i, candle in enumerate(self.price_history[-50:]):
            open_price, high, low, close, volume = candle
            
            x = chart_rect.x + i * candle_width
            
            # Normalize prices to chart coordinates
            high_y = chart_rect.y + (1 - (high - min_price) / price_range) * chart_rect.height
            low_y = chart_rect.y + (1 - (low - min_price) / price_range) * chart_rect.height
            open_y = chart_rect.y + (1 - (open_price - min_price) / price_range) * chart_rect.height
            close_y = chart_rect.y + (1 - (close - min_price) / price_range) * chart_rect.height
            
            # Draw high-low line
            pygame.draw.line(self.screen, (200, 200, 200), (x + candle_width//2, high_y), (x + candle_width//2, low_y), 1)
            
            # Draw candle body
            body_top = min(open_y, close_y)
            body_height = abs(close_y - open_y)
            color = (0, 255, 0) if close > open_price else (255, 0, 0)
            
            pygame.draw.rect(self.screen, color, (x + 1, body_top, candle_width - 2, max(body_height, 1)))
    
    def _draw_portfolio_info(self):
        """Draw portfolio information"""
        font = pygame.font.Font(None, 24)
        
        current_price = self.price_history[-1][3] if self.price_history else 0
        portfolio_value = self.cash + self.holdings * current_price
        
        info_texts = [
            f"Cash: ${self.cash:.2f}",
            f"Holdings: {self.holdings:.6f} BTC",
            f"Current Price: ${current_price:.2f}",
            f"Portfolio Value: ${portfolio_value:.2f}",
            f"P&L: ${portfolio_value - self.config.initial_balance:.2f}",
            f"Step: {self.current_step}"
        ]
        
        y_offset = 500
        for text in info_texts:
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (900, y_offset))
            y_offset += 30
    
    def _draw_market_info(self):
        """Draw market information"""
        font = pygame.font.Font(None, 20)
        
        market_texts = [
            f"Market Regime: {self.market_sim.current_regime.value}",
            f"Market Psychology: {self.market_sim.market_psychology:.2f}",
            f"Trend Strength: {self.market_sim.trend_strength:.2f}"
        ]
        
        y_offset = 700
        for text in market_texts:
            surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (50, y_offset))
            y_offset += 25
    
    def _plot_candlesticks(self, ax):
        """Plot candlestick chart using matplotlib"""
        if len(self.price_history) < 2:
            return
        
        for i, candle in enumerate(self.price_history[-50:]):
            open_price, high, low, close, volume = candle
            
            color = 'green' if close > open_price else 'red'
            
            # Draw high-low line
            ax.plot([i, i], [low, high], color='black', linewidth=1)
            
            # Draw candle body
            height = abs(close - open_price)
            bottom = min(open_price, close)
            
            rect = Rectangle((i - 0.3, bottom), 0.6, height, 
                           facecolor=color, alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
        
        ax.set_title('Price Chart')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
    
    def close(self):
        """Close the environment"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


# Register the environment
gym.register(
    id='CryptoTrading-v0',
    entry_point='crypto_trading_env:CryptoTradingEnv',
    max_episode_steps=1000,
)