"""Topology-informed grid trading engine.

Uses algebraic topology (persistent homology) to:
- Detect when market regime favors grid trading (high H1 Betti = cyclical)
- Place grid levels at topologically significant support/resistance (H0 clusters)
- Adapt grid density based on persistence entropy (predictability)
- Widen/tighten grids using ATR-scaled topological amplitude
"""

import numpy as np
import pandas as pd
from loguru import logger
from features.tda import TDAFeatureExtractor, takens_embedding, compute_persistence_scipy

GRID_MIN_LEVELS = 5
GRID_MAX_LEVELS = 20
ATR_PERIOD = 20
GRID_PROFIT_TARGET_ATR = 0.3  # Take profit at 0.3 ATR per grid
GRID_ACTIVATION_THRESHOLD = 0.5  # Minimum topology score to activate


class GridLevel:
    """A single grid level with price, direction, and state."""

    def __init__(self, price, direction, strength=1.0):
        self.price = price
        self.direction = direction  # "buy" or "sell"
        self.strength = strength  # Topological persistence strength
        self.filled = False
        self.fill_price = 0.0
        self.pnl = 0.0


class TopologicalGridEngine:
    """Grid trading engine that uses persistent homology for level placement."""

    def __init__(self, tda=None):
        self.tda = tda or TDAFeatureExtractor()
        self.active_grids = {}  # asset -> list of GridLevel
        self.grid_pnl = {}  # asset -> cumulative pnl

    def compute_atr(self, highs, lows, closes, period=ATR_PERIOD):
        """Compute Average True Range."""
        if len(closes) < period + 1:
            return 0.0
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - np.roll(closes, 1)[-period:]),
                np.abs(lows[-period:] - np.roll(closes, 1)[-period:])
            )
        )
        return float(np.mean(tr))

    def build_grid(self, asset, prices, highs, lows, closes):
        """Build a topology-informed grid for an asset.

        The grid placement algorithm:
        1. Check if topology favors grid trading (H1 cycles)
        2. Extract topological support/resistance levels (H0 clusters)
        3. Fill gaps between topological levels with ATR-spaced grids
        4. Adjust density based on persistence entropy
        """
        if len(prices) < 60:
            return None, 0.0

        # Step 1: Check topological favorability
        topo_score, tda_features = self.tda.is_grid_favorable(prices[-60:])

        if topo_score < GRID_ACTIVATION_THRESHOLD:
            return None, topo_score

        # Step 2: Extract topological levels
        topo_levels, cyclicality = self.tda.extract_topological_levels(prices[-60:])

        # Step 3: Compute ATR for grid spacing
        atr = self.compute_atr(highs, lows, closes)
        if atr < 1e-10:
            return None, topo_score

        current_price = prices[-1]

        # Step 4: Determine grid range
        # Use topological amplitude to set range width
        amplitude = tda_features.get("amplitude", 1.0)
        complexity = tda_features.get("complexity_score", 5.0)

        # Range width: wider in high amplitude, tighter in low
        range_multiplier = max(2.0, min(6.0, amplitude * 2))
        grid_half_range = atr * range_multiplier

        grid_upper = current_price + grid_half_range
        grid_lower = current_price - grid_half_range

        # Step 5: Determine grid density from entropy
        entropy_h1 = tda_features.get("entropy_h1", 1.0)
        # Low entropy = predictable = more grids; high entropy = fewer grids
        n_levels = int(np.clip(
            GRID_MAX_LEVELS - entropy_h1 * 5,
            GRID_MIN_LEVELS, GRID_MAX_LEVELS
        ))

        # Step 6: Place grid levels
        levels = []

        # First, use topological levels if available
        for topo in topo_levels[:5]:
            if grid_lower <= topo["price"] <= grid_upper:
                direction = "buy" if topo["price"] < current_price else "sell"
                levels.append(GridLevel(
                    price=topo["price"],
                    direction=direction,
                    strength=topo["strength"]
                ))

        # Fill remaining levels with uniform ATR-spaced grid
        spacing = (grid_upper - grid_lower) / (n_levels + 1)
        for i in range(1, n_levels + 1):
            level_price = grid_lower + spacing * i
            # Skip if too close to an existing topological level
            if any(abs(level_price - l.price) < spacing * 0.3 for l in levels):
                continue
            direction = "buy" if level_price < current_price else "sell"
            levels.append(GridLevel(price=level_price, direction=direction, strength=0.5))

        # Sort by price
        levels.sort(key=lambda l: l.price)

        return levels, topo_score

    def generate_signal(self, asset, prices, highs, lows, closes):
        """Generate a grid trading signal for an asset.

        Returns a signal between -1 and 1:
        - Positive when price hits buy grid levels (below current)
        - Negative when price hits sell grid levels (above current)
        - Magnitude scaled by topological strength of the level
        """
        if len(prices) < 60:
            return 0.0, {}

        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else current_price

        # Build or update grid
        levels, topo_score = self.build_grid(asset, prices, highs, lows, closes)

        if levels is None:
            return 0.0, {"topo_score": topo_score, "grid_active": False}

        # Check which levels were crossed
        signal = 0.0
        filled_count = 0
        atr = self.compute_atr(highs, lows, closes)

        for level in levels:
            # Price crossed this level
            crossed_down = prev_price > level.price >= current_price
            crossed_up = prev_price < level.price <= current_price

            if level.direction == "buy" and crossed_down:
                # Price dropped to buy level — go long
                signal += 0.3 * level.strength
                filled_count += 1

            elif level.direction == "sell" and crossed_up:
                # Price rose to sell level — go short / take profit
                signal -= 0.3 * level.strength
                filled_count += 1

        # Mean-reversion component: when price is far from grid center
        grid_center = np.mean([l.price for l in levels])
        distance_from_center = (current_price - grid_center) / (atr + 1e-10)

        # Fade the extremes: buy when below center, sell when above
        reversion_signal = -distance_from_center * 0.15
        signal += reversion_signal

        signal = np.clip(signal, -1.0, 1.0)

        info = {
            "topo_score": topo_score,
            "grid_active": True,
            "n_levels": len(levels),
            "filled_count": filled_count,
            "grid_center": grid_center,
            "distance_from_center": distance_from_center,
            "cyclicality": topo_score,
        }

        return signal, info

    def get_grid_status(self, asset):
        """Get current grid status for an asset."""
        if asset not in self.active_grids:
            return {"active": False}
        levels = self.active_grids[asset]
        return {
            "active": True,
            "n_levels": len(levels),
            "filled": sum(1 for l in levels if l.filled),
            "pnl": self.grid_pnl.get(asset, 0.0),
        }
