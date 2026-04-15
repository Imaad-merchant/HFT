"""Trading simulation environment compatible with gymnasium for RL training."""

import numpy as np
import pandas as pd
import duckdb
import json
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "aria.db"

NUM_ASSETS = 9
ASSET_LIST = ["ES", "NQ", "GC", "CL", "TLT", "SPY", "DXY", "VIX", "ZN"]
EPISODE_LENGTH = 390  # 1 trading day of 1-minute bars (6.5 hours)
INITIAL_CAPITAL = 1_000_000
TRANSACTION_COST_BPS = 0.5
SLIPPAGE_MIN_BPS = 0.5
SLIPPAGE_MAX_BPS = 2.0
ADVERSE_SELECTION_PROB = 0.3
ADVERSE_SELECTION_BPS = 1.0
MAX_DRAWDOWN_PENALTY_THRESHOLD = 0.05


class TradingEnvironment(gym.Env):
    """Multi-asset trading environment with realistic fills and risk-adjusted rewards."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, db_path: str = None, episode_length: int = EPISODE_LENGTH):
        super().__init__()
        self.db_path = str(db_path or DB_PATH)
        self.episode_length = episode_length

        # Load all data into memory for fast stepping
        self._load_data()

        # Observation: features + positions + pnl + sharpe
        # Estimate feature dim from data
        self.feature_dim = self._estimate_feature_dim()
        obs_dim = self.feature_dim + NUM_ASSETS + 2  # positions + unrealized_pnl + rolling_sharpe

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: 9 discrete directions + 9 continuous sizes
        # Represented as Box for PPO compatibility
        # First 9 values: position direction (-1, 0, 1) -> mapped from continuous
        # Last 9 values: position size (0 to 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_ASSETS * 2,), dtype=np.float32
        )

        self._reset_state()

    def _load_data(self):
        """Load OHLCV and feature data into memory."""
        try:
            con = duckdb.connect(self.db_path)

            self.price_data = {}
            for asset in ASSET_LIST:
                df = con.execute(
                    "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE asset = ? ORDER BY timestamp",
                    [asset]
                ).fetchdf()
                if not df.empty:
                    self.price_data[asset] = df

            # Load features
            self.feature_data = {}
            for asset in ASSET_LIST:
                fdf = con.execute(
                    "SELECT timestamp, feature_json FROM features_wide WHERE asset = ? ORDER BY timestamp",
                    [asset]
                ).fetchdf()
                if not fdf.empty:
                    self.feature_data[asset] = fdf

            con.close()
        except Exception:
            self.price_data = {}
            self.feature_data = {}

        # Find common timestamps across all assets with data
        available_assets = [a for a in ASSET_LIST if a in self.price_data and len(self.price_data[a]) > self.episode_length]
        if available_assets:
            ts_sets = [set(self.price_data[a]["timestamp"]) for a in available_assets]
            self.common_timestamps = sorted(set.intersection(*ts_sets)) if ts_sets else []
        else:
            self.common_timestamps = []

    def _estimate_feature_dim(self) -> int:
        """Estimate feature vector dimension from data."""
        for asset in ASSET_LIST:
            if asset in self.feature_data and not self.feature_data[asset].empty:
                try:
                    sample = json.loads(self.feature_data[asset]["feature_json"].iloc[-1])
                    return len(sample) * NUM_ASSETS + 20  # cross-asset + tda + macro
                except Exception:
                    pass
        return NUM_ASSETS * 30 + 20  # fallback estimate

    def _reset_state(self):
        """Reset internal state variables."""
        self.capital = INITIAL_CAPITAL
        self.positions = np.zeros(NUM_ASSETS)
        self.entry_prices = np.zeros(NUM_ASSETS)
        self.current_step = 0
        self.start_idx = 0
        self.peak_capital = INITIAL_CAPITAL
        self.trade_pnls = []
        self.total_trades = 0
        self.episode_pnl = 0.0

    def reset(self, seed=None, options=None):
        """Reset environment to a random starting point."""
        super().reset(seed=seed)
        self._reset_state()

        if len(self.common_timestamps) > self.episode_length + 10:
            max_start = len(self.common_timestamps) - self.episode_length - 1
            self.start_idx = self.np_random.integers(0, max_start)
        else:
            self.start_idx = 0

        obs = self._get_observation()
        return obs, {}

    def _get_current_prices(self) -> np.ndarray:
        """Get current close prices for all assets."""
        prices = np.zeros(NUM_ASSETS)
        if not self.common_timestamps:
            return prices

        idx = min(self.start_idx + self.current_step, len(self.common_timestamps) - 1)
        ts = self.common_timestamps[idx]

        for i, asset in enumerate(ASSET_LIST):
            if asset in self.price_data:
                mask = self.price_data[asset]["timestamp"] == ts
                rows = self.price_data[asset][mask]
                if not rows.empty:
                    prices[i] = rows["close"].iloc[0]

        return prices

    def _get_next_open_prices(self) -> np.ndarray:
        """Get next bar open prices for fill simulation."""
        prices = np.zeros(NUM_ASSETS)
        if not self.common_timestamps:
            return prices

        idx = min(self.start_idx + self.current_step + 1, len(self.common_timestamps) - 1)
        ts = self.common_timestamps[idx]

        for i, asset in enumerate(ASSET_LIST):
            if asset in self.price_data:
                mask = self.price_data[asset]["timestamp"] == ts
                rows = self.price_data[asset][mask]
                if not rows.empty:
                    prices[i] = rows["open"].iloc[0]

        return prices

    def _get_feature_vector(self) -> np.ndarray:
        """Build the full observation feature vector."""
        features = []

        if not self.common_timestamps:
            return np.zeros(self.feature_dim, dtype=np.float32)

        idx = min(self.start_idx + self.current_step, len(self.common_timestamps) - 1)
        ts = self.common_timestamps[idx]

        for asset in ASSET_LIST:
            if asset in self.feature_data:
                mask = self.feature_data[asset]["timestamp"] == ts
                rows = self.feature_data[asset][mask]
                if not rows.empty:
                    try:
                        feat_dict = json.loads(rows["feature_json"].iloc[0])
                        features.extend(feat_dict.values())
                        continue
                    except Exception:
                        pass
            # Pad with zeros if no features
            features.extend([0.0] * 30)

        # Pad or truncate to expected dim
        result = np.array(features[:self.feature_dim], dtype=np.float32)
        if len(result) < self.feature_dim:
            result = np.pad(result, (0, self.feature_dim - len(result)))

        return result

    def _get_observation(self) -> np.ndarray:
        """Build full observation: features + positions + pnl + sharpe."""
        feat = self._get_feature_vector()

        # Current unrealized PnL
        current_prices = self._get_current_prices()
        unrealized = np.sum(self.positions * (current_prices - self.entry_prices))
        total_equity = self.capital + unrealized

        # Rolling Sharpe from recent trades
        if len(self.trade_pnls) >= 5:
            recent = np.array(self.trade_pnls[-50:])
            sharpe = np.mean(recent) / (np.std(recent) + 1e-10)
        else:
            sharpe = 0.0

        obs = np.concatenate([
            feat,
            self.positions / 100.0,  # Normalize positions
            [unrealized / INITIAL_CAPITAL, sharpe],
        ]).astype(np.float32)

        return obs

    def _apply_slippage(self, price: float, direction: int, volatility: float = 0.01) -> float:
        """Apply realistic slippage to a fill price."""
        slippage_bps = np.random.uniform(SLIPPAGE_MIN_BPS, SLIPPAGE_MAX_BPS) * (1 + volatility)
        slippage = price * slippage_bps / 10000.0

        # Adverse selection: 30% chance price moves against you
        if np.random.random() < ADVERSE_SELECTION_PROB:
            slippage += price * ADVERSE_SELECTION_BPS / 10000.0

        return price + direction * slippage

    def step(self, action: np.ndarray):
        """Execute one step: apply action, compute reward."""
        directions = action[:NUM_ASSETS]
        sizes = np.clip(action[NUM_ASSETS:], 0, 1)

        # Discretize directions
        discrete_dirs = np.zeros(NUM_ASSETS)
        for i in range(NUM_ASSETS):
            if directions[i] > 0.33:
                discrete_dirs[i] = 1
            elif directions[i] < -0.33:
                discrete_dirs[i] = -1

        # Get fill prices (next bar open with slippage)
        fill_prices = self._get_next_open_prices()
        current_prices = self._get_current_prices()

        # Close existing positions and open new ones
        step_pnl = 0.0
        turnover = 0.0

        for i in range(NUM_ASSETS):
            target_pos = discrete_dirs[i] * sizes[i] * 100  # Scale position
            current_pos = self.positions[i]

            if abs(target_pos - current_pos) < 0.01:
                continue

            if fill_prices[i] <= 0:
                continue

            # Close existing position
            if current_pos != 0:
                close_price = self._apply_slippage(fill_prices[i], -int(np.sign(current_pos)))
                pnl = current_pos * (close_price - self.entry_prices[i])
                step_pnl += pnl
                self.trade_pnls.append(pnl)
                self.total_trades += 1

                # Transaction cost
                cost = abs(current_pos) * fill_prices[i] * TRANSACTION_COST_BPS / 10000.0
                step_pnl -= cost
                turnover += abs(current_pos) * fill_prices[i]

            # Open new position
            if target_pos != 0:
                entry = self._apply_slippage(fill_prices[i], int(np.sign(target_pos)))
                self.positions[i] = target_pos
                self.entry_prices[i] = entry

                cost = abs(target_pos) * fill_prices[i] * TRANSACTION_COST_BPS / 10000.0
                step_pnl -= cost
                turnover += abs(target_pos) * fill_prices[i]
            else:
                self.positions[i] = 0
                self.entry_prices[i] = 0

        self.capital += step_pnl
        self.episode_pnl += step_pnl

        # Track peak for drawdown
        unrealized = np.sum(self.positions * (current_prices - self.entry_prices))
        total_equity = self.capital + unrealized
        self.peak_capital = max(self.peak_capital, total_equity)

        # --- Reward computation ---
        # Risk-adjusted PnL
        if len(self.trade_pnls) >= 2:
            rolling_std = np.std(self.trade_pnls[-20:]) + 1e-10
            reward = step_pnl / rolling_std
        else:
            reward = step_pnl / (INITIAL_CAPITAL * 0.01)

        # Turnover penalty
        turnover_penalty = turnover / INITIAL_CAPITAL * 0.1
        reward -= turnover_penalty

        # Drawdown penalty
        drawdown = (self.peak_capital - total_equity) / self.peak_capital
        if drawdown > MAX_DRAWDOWN_PENALTY_THRESHOLD:
            reward -= drawdown * 10

        # Sharpe bonus
        if len(self.trade_pnls) >= 20:
            recent = np.array(self.trade_pnls[-20:])
            rolling_sharpe = np.mean(recent) / (np.std(recent) + 1e-10)
            if rolling_sharpe > 0.5:
                reward += 0.1

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = drawdown > 0.08  # Circuit breaker

        obs = self._get_observation()
        info = {
            "step_pnl": step_pnl,
            "total_equity": total_equity,
            "drawdown": drawdown,
            "total_trades": self.total_trades,
            "episode_pnl": self.episode_pnl,
        }

        return obs, float(reward), terminated, truncated, info

    def get_episode_stats(self) -> dict:
        """Get summary statistics for the completed episode."""
        total_equity = self.capital + np.sum(
            self.positions * (self._get_current_prices() - self.entry_prices)
        )
        total_return = (total_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
        max_dd = (self.peak_capital - total_equity) / self.peak_capital if self.peak_capital > 0 else 0

        sharpe = 0.0
        if len(self.trade_pnls) >= 5:
            pnls = np.array(self.trade_pnls)
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trade_count": self.total_trades,
            "final_equity": total_equity,
        }
