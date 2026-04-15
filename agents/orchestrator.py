"""Autonomous agent orchestrator — runs the full ARIA system headlessly."""

import numpy as np
import pandas as pd
import duckdb
import json
import threading
import time
from datetime import datetime
from loguru import logger
from pathlib import Path

from data.fetcher import DataFetcher, ASSET_LABELS
from features.tda import TDAFeatureExtractor
from features.factory import FeatureFactory
from arbitrage.stat_arb import StatArbEngine
from macro.macro import MacroEngine
from environment.sim import TradingEnvironment, ASSET_LIST, NUM_ASSETS
from evolution.evolver import EvolutionarySearcher
from rl.agent import RLAgent
from monitor.decay import AlphaDecayMonitor

DB_PATH = Path(__file__).parent.parent / "aria.db"

MAX_POSITION_PCT = 0.20
MAX_PORTFOLIO_EXPOSURE = 1.0
MAX_DRAWDOWN_CIRCUIT_BREAKER = 0.08
RISK_OFF_REDUCTION = 0.50
EVENT_PROXIMITY_FLAT_MINUTES = 30


class Orchestrator:
    """Runs all ARIA agents and fuses their signals into paper trading decisions."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)

        # Initialize all components
        self.data_fetcher = DataFetcher(self.db_path)
        self.tda = TDAFeatureExtractor(self.db_path)
        self.factory = FeatureFactory(self.db_path)
        self.stat_arb = StatArbEngine(self.db_path)
        self.macro = MacroEngine(self.db_path)
        self.rl_agent = RLAgent(self.db_path)
        self.evolver = EvolutionarySearcher(self.db_path)
        self.decay_monitor = AlphaDecayMonitor(self.db_path)

        # State
        self.capital = 1_000_000
        self.positions = {}  # asset -> position size
        self.peak_capital = self.capital
        self.halted = False
        self.running = True

        self._init_trades_table()

    def _init_trades_table(self):
        """Create the trades and signals tables."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TIMESTAMP,
                asset VARCHAR,
                direction VARCHAR,
                size DOUBLE,
                price DOUBLE,
                pnl DOUBLE,
                reason VARCHAR,
                signal_source VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TIMESTAMP,
                asset VARCHAR,
                signal_source VARCHAR,
                direction DOUBLE,
                confidence DOUBLE,
                features_json VARCHAR
            )
        """)
        con.close()

    # --- Individual agent tasks ---

    def run_data_refresh(self):
        """DataAgent: refresh market data."""
        try:
            self.data_fetcher.refresh()
            self.data_fetcher.fill_gaps()
            logger.info("Data refresh complete")
        except Exception as e:
            logger.error("Data refresh failed: {}", e)

    def run_feature_recompute(self):
        """FeatureAgent: recompute all features."""
        try:
            self.factory.run()
            logger.info("Feature recompute complete")
        except Exception as e:
            logger.error("Feature recompute failed: {}", e)

    def run_tda(self):
        """TDAAgent: run TDA pipeline."""
        try:
            self.tda.run()
            logger.info("TDA pipeline complete")
        except Exception as e:
            logger.error("TDA pipeline failed: {}", e)

    def run_macro_update(self):
        """MacroAgent: update macro data and regime."""
        try:
            self.macro.run()
            logger.info("Macro update complete")
        except Exception as e:
            logger.error("Macro update failed: {}", e)

    def run_stat_arb_signals(self):
        """StatArbAgent: generate stat arb signals."""
        try:
            signals = self.stat_arb.generate_signals()
            return signals
        except Exception as e:
            logger.error("StatArb signal generation failed: {}", e)
            return []

    def run_decay_check(self):
        """DecayMonitor: check strategy health."""
        try:
            actions = self.decay_monitor.run()
            if self.decay_monitor.should_trigger_evolution():
                logger.info("Decay detected — triggering evolution search")
                threading.Thread(target=self.run_evolution, daemon=True).start()
            return actions
        except Exception as e:
            logger.error("Decay check failed: {}", e)
            return []

    def run_evolution(self):
        """EvolutionAgent: run evolutionary search."""
        try:
            logger.info("Starting evolutionary search (200 generations)")
            self.evolver.run(generations=200)
            logger.info("Evolution complete")
        except Exception as e:
            logger.error("Evolution failed: {}", e)

    def run_retrain(self, full: bool = True):
        """RetrainAgent: retrain RL model."""
        try:
            if full:
                logger.info("Starting full RL retraining (2M steps)")
                self.rl_agent.train(total_timesteps=2_000_000, from_scratch=True)
            else:
                logger.info("Starting incremental RL fine-tuning (10k steps)")
                self.rl_agent.fine_tune(timesteps=10_000)
        except Exception as e:
            logger.error("Retraining failed: {}", e)

    # --- Signal fusion and decision making ---

    def _get_rl_signals(self):
        """Get signals from the RL agent."""
        signals = {}
        try:
            env = TradingEnvironment(db_path=self.db_path)
            obs, _ = env.reset()
            action = self.rl_agent.predict(obs)

            for i, asset in enumerate(ASSET_LIST):
                direction = action[i]
                size = abs(action[NUM_ASSETS + i]) if len(action) > NUM_ASSETS + i else 0.5
                signals[asset] = float(direction * size)
        except Exception as e:
            logger.debug("RL signal generation failed: {}", e)

        return signals

    def _get_evo_signals(self):
        """Get signals from the best evolved genome."""
        signals = {}
        try:
            genome = self.evolver.get_best_genome()
            if genome is None:
                return signals

            for asset in ASSET_LIST:
                features = self.factory.get_feature_vector(asset)
                if features:
                    signal = self.evolver.get_genome_signal(genome, features)
                    signals[asset] = signal
        except Exception as e:
            logger.debug("Evo signal generation failed: {}", e)

        return signals

    def _get_stat_arb_signals(self):
        """Convert stat arb pair signals to per-asset signals."""
        signals = {}
        try:
            pair_signals = self.stat_arb.generate_signals()
            for sig in pair_signals:
                if sig["signal"] in ("ENTRY_LONG", "ENTRY_SHORT"):
                    direction = 1.0 if sig["signal"] == "ENTRY_LONG" else -1.0
                    # Long the first asset, short the second (or vice versa)
                    signals[sig["asset1"]] = signals.get(sig["asset1"], 0) + direction * 0.5
                    signals[sig["asset2"]] = signals.get(sig["asset2"], 0) - direction * 0.5
                elif sig["signal"] == "EXIT":
                    signals[sig["asset1"]] = 0.0
                    signals[sig["asset2"]] = 0.0
        except Exception as e:
            logger.debug("StatArb signal conversion failed: {}", e)

        return signals

    def fuse_signals(self):
        """Combine all signal sources weighted by their rolling Sharpe."""
        allocations = self.decay_monitor.get_strategy_allocations()

        rl_weight = allocations.get("rl_agent", 1.0)
        evo_weight = max(allocations.get(k, 0.5) for k in allocations if k.startswith("evo_")) if any(k.startswith("evo_") for k in allocations) else 0.5
        arb_weight = max(allocations.get(k, 0.5) for k in allocations if k.startswith("statarb_")) if any(k.startswith("statarb_") for k in allocations) else 0.5

        rl_signals = self._get_rl_signals()
        evo_signals = self._get_evo_signals()
        arb_signals = self._get_stat_arb_signals()

        # Weighted average
        fused = {}
        for asset in ASSET_LIST:
            components = []
            weights = []

            if asset in rl_signals:
                components.append(rl_signals[asset])
                weights.append(rl_weight)
            if asset in evo_signals:
                components.append(evo_signals[asset])
                weights.append(evo_weight)
            if asset in arb_signals:
                components.append(arb_signals[asset])
                weights.append(arb_weight)

            if components:
                total_weight = sum(weights) + 1e-10
                fused[asset] = sum(c * w for c, w in zip(components, weights)) / total_weight
            else:
                fused[asset] = 0.0

        # Apply macro regime filter
        macro_features = self.macro.get_macro_features()
        regime = macro_features.get("macro_regime", "NEUTRAL")
        if regime == "RISK_OFF":
            for asset in fused:
                fused[asset] *= RISK_OFF_REDUCTION
            logger.info("RISK_OFF regime — reducing all positions by {}%", int(RISK_OFF_REDUCTION * 100))

        # Apply event proximity filter
        event = self.macro.get_event_proximity()
        if event["hours_to_event"] < (EVENT_PROXIMITY_FLAT_MINUTES / 60.0):
            logger.info("High-impact event {} in {:.1f}h — going flat",
                        event["next_event"], event["hours_to_event"])
            for asset in fused:
                fused[asset] = 0.0

        # Log all signals
        now = pd.Timestamp.now()
        con = duckdb.connect(self.db_path)
        for asset in ASSET_LIST:
            for source, sigs in [("rl_agent", rl_signals), ("evolution", evo_signals), ("stat_arb", arb_signals)]:
                if asset in sigs:
                    sig_row = pd.DataFrame([{
                        "timestamp": now,
                        "asset": asset,
                        "signal_source": source,
                        "direction": sigs[asset],
                        "confidence": abs(sigs[asset]),
                        "features_json": "{}",
                    }])
                    con.execute("INSERT INTO signals SELECT * FROM sig_row")
        con.close()

        return fused

    def apply_risk_gate(self, target_positions: dict[str, float]):
        """Apply hard risk limits to target positions."""
        # Max per-asset position
        for asset in target_positions:
            max_size = self.capital * MAX_POSITION_PCT
            if abs(target_positions[asset]) > max_size:
                target_positions[asset] = np.sign(target_positions[asset]) * max_size

        # Max total exposure
        total_exposure = sum(abs(v) for v in target_positions.values())
        max_exposure = self.capital * MAX_PORTFOLIO_EXPOSURE
        if total_exposure > max_exposure:
            scale = max_exposure / total_exposure
            for asset in target_positions:
                target_positions[asset] *= scale

        # Circuit breaker
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.capital) / self.peak_capital
            if drawdown > MAX_DRAWDOWN_CIRCUIT_BREAKER:
                logger.critical("CIRCUIT BREAKER: Portfolio down {:.1f}% — halting all trading", drawdown * 100)
                self.halted = True
                return {a: 0.0 for a in target_positions}

        return target_positions

    def execute_paper_trades(self, target_positions: dict[str, float]):
        """Log paper trades and simulate fills."""
        if self.halted:
            logger.warning("Trading halted — no trades executed")
            return

        con = duckdb.connect(self.db_path)
        now = pd.Timestamp.now()

        for asset, target in target_positions.items():
            current = self.positions.get(asset, 0.0)
            if abs(target - current) < 0.01:
                continue

            # Get current price for fill simulation
            price_row = con.execute("""
                SELECT close FROM ohlcv WHERE asset = ?
                ORDER BY timestamp DESC LIMIT 1
            """, [asset]).fetchone()

            price = price_row[0] if price_row else 0.0
            if price <= 0:
                continue

            # Simulate fill at current price (next-bar-open approximation)
            direction = "LONG" if target > current else "SHORT" if target < current else "FLAT"
            size = abs(target - current)
            pnl = 0.0

            # If closing a position, compute PnL
            if current != 0 and np.sign(target) != np.sign(current):
                pnl = current * (price - self.positions.get(f"{asset}_entry", price))

            trade = pd.DataFrame([{
                "timestamp": now,
                "asset": asset,
                "direction": direction,
                "size": size,
                "price": price,
                "pnl": pnl,
                "reason": "signal_fusion",
                "signal_source": "orchestrator",
            }])
            con.execute("INSERT INTO trades SELECT * FROM trade")

            # Update positions
            self.positions[asset] = target
            if target != 0:
                self.positions[f"{asset}_entry"] = price

            self.capital += pnl
            self.peak_capital = max(self.peak_capital, self.capital)

            logger.info("Paper trade: {} {} {:.2f} @ {:.2f} (PnL: {:.2f})",
                        direction, asset, size, price, pnl)

        con.close()

    def run_trading_cycle(self):
        """Execute one full trading cycle: signals -> fusion -> risk -> execute."""
        if self.halted:
            return

        # Fuse signals from all sources
        fused = self.fuse_signals()

        # Convert to dollar positions
        target_positions = {}
        for asset, signal in fused.items():
            target_positions[asset] = signal * self.capital * 0.1  # 10% base allocation

        # Apply risk gate
        safe_positions = self.apply_risk_gate(target_positions)

        # Execute
        self.execute_paper_trades(safe_positions)

    def get_status(self) -> dict:
        """Get current system status."""
        coverage = self.data_fetcher.get_coverage()
        model_version = self.rl_agent.get_deployed_version()

        con = duckdb.connect(self.db_path)
        regime_row = con.execute(
            "SELECT regime FROM macro_regime ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        pair_count = con.execute("SELECT COUNT(*) FROM pairs WHERE status = 'ACTIVE'").fetchone()
        trade_count = con.execute("SELECT COUNT(*) FROM trades").fetchone()
        con.close()

        return {
            "capital": self.capital,
            "peak_capital": self.peak_capital,
            "halted": self.halted,
            "data_coverage": coverage,
            "model_version": model_version,
            "macro_regime": regime_row[0] if regime_row else "UNKNOWN",
            "active_pairs": pair_count[0] if pair_count else 0,
            "total_trades": trade_count[0] if trade_count else 0,
            "positions": {k: v for k, v in self.positions.items() if not k.endswith("_entry")},
        }

    def startup_summary(self):
        """Print startup summary."""
        status = self.get_status()
        logger.info("=" * 60)
        logger.info("ARIA - Adaptive Reinforcement Intelligence for Alpha")
        logger.info("=" * 60)
        logger.info("Capital: ${:,.0f}", status["capital"])
        logger.info("Model version: v{}", status["model_version"])
        logger.info("Macro regime: {}", status["macro_regime"])
        logger.info("Active stat arb pairs: {}", status["active_pairs"])
        logger.info("Total paper trades: {}", status["total_trades"])
        logger.info("Trading halted: {}", status["halted"])
        if status["data_coverage"]:
            logger.info("Data coverage:")
            for c in status["data_coverage"]:
                logger.info("  {}: {} bars ({} to {})", c["asset"], c["bars"], c["start"], c["end"])
        logger.info("=" * 60)
