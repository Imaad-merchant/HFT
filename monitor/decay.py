"""Alpha decay monitoring — tracks strategy health and triggers automated actions."""

import numpy as np
import pandas as pd
import duckdb
import json
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "aria.db"
HEALTH_FILE = Path(__file__).parent.parent / "aria_health.json"

# Thresholds
SHARPE_REDUCE_THRESHOLD = 0.3
SHARPE_SUSPEND_THRESHOLD = 0.0
SUSPEND_CONSECUTIVE_DAYS = 3
ALLOCATION_REDUCE_PCT = 0.25


class AlphaDecayMonitor:
    """Monitors strategy health and automates allocation/retirement decisions."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create monitoring tables."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS strategy_health (
                timestamp TIMESTAMP,
                strategy_name VARCHAR,
                strategy_type VARCHAR,
                sharpe_7d DOUBLE,
                sharpe_14d DOUBLE,
                sharpe_30d DOUBLE,
                win_rate DOUBLE,
                profit_factor DOUBLE,
                decay_rate DOUBLE,
                time_to_death_days DOUBLE,
                allocation_pct DOUBLE,
                status VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS alpha_events (
                timestamp TIMESTAMP,
                strategy_name VARCHAR,
                event_type VARCHAR,
                old_value VARCHAR,
                new_value VARCHAR,
                reason VARCHAR
            )
        """)
        con.close()

    def _compute_rolling_sharpe(self, pnls: np.ndarray, window: int) -> float:
        """Compute rolling Sharpe from PnL series."""
        if len(pnls) < window:
            return 0.0
        recent = pnls[-window:]
        return float(np.mean(recent) / (np.std(recent) + 1e-10) * np.sqrt(252))

    def _compute_win_rate(self, pnls: np.ndarray) -> float:
        """Compute win rate from PnL series."""
        if len(pnls) == 0:
            return 0.0
        return float(np.sum(pnls > 0) / len(pnls))

    def _compute_profit_factor(self, pnls: np.ndarray) -> float:
        """Compute profit factor (gross profit / gross loss)."""
        gains = np.sum(pnls[pnls > 0])
        losses = abs(np.sum(pnls[pnls < 0]))
        if losses == 0:
            return 10.0 if gains > 0 else 0.0
        return float(gains / losses)

    def _compute_decay_rate(self, sharpe_history: list[float]) -> float:
        """Compute decay rate: change in 7-day Sharpe per week."""
        if len(sharpe_history) < 2:
            return 0.0
        return float(sharpe_history[-1] - sharpe_history[0]) / max(len(sharpe_history), 1)

    def monitor_rl_agent(self) -> dict:
        """Monitor the RL agent's health."""
        con = duckdb.connect(self.db_path)

        # Get recent paper trades for RL agent
        trades = con.execute("""
            SELECT timestamp, pnl FROM trades
            WHERE signal_source = 'rl_agent'
            ORDER BY timestamp DESC LIMIT 500
        """).fetchdf()
        con.close()

        if trades.empty:
            return self._default_health("rl_agent", "RL")

        pnls = trades["pnl"].values[::-1]

        health = {
            "strategy_name": "rl_agent",
            "strategy_type": "RL",
            "sharpe_7d": self._compute_rolling_sharpe(pnls, 7 * 78),
            "sharpe_14d": self._compute_rolling_sharpe(pnls, 14 * 78),
            "sharpe_30d": self._compute_rolling_sharpe(pnls, 30 * 78),
            "win_rate": self._compute_win_rate(pnls),
            "profit_factor": self._compute_profit_factor(pnls),
        }

        return health

    def monitor_stat_arb(self) -> list[dict]:
        """Monitor all stat arb pairs."""
        con = duckdb.connect(self.db_path)
        pairs = con.execute("SELECT * FROM pairs WHERE status = 'ACTIVE'").fetchdf()

        results = []
        for _, pair in pairs.iterrows():
            key = f"statarb_{pair['asset1']}_{pair['asset2']}"
            trades = con.execute("""
                SELECT timestamp, pnl FROM trades
                WHERE signal_source = ? ORDER BY timestamp DESC LIMIT 200
            """, [key]).fetchdf()

            if trades.empty:
                results.append(self._default_health(key, "STAT_ARB"))
                continue

            pnls = trades["pnl"].values[::-1]
            results.append({
                "strategy_name": key,
                "strategy_type": "STAT_ARB",
                "sharpe_7d": self._compute_rolling_sharpe(pnls, 7 * 78),
                "sharpe_14d": self._compute_rolling_sharpe(pnls, 14 * 78),
                "sharpe_30d": self._compute_rolling_sharpe(pnls, 30 * 78),
                "win_rate": self._compute_win_rate(pnls),
                "profit_factor": self._compute_profit_factor(pnls),
            })

        con.close()
        return results

    def monitor_evolved_strategies(self) -> list[dict]:
        """Monitor evolved genome strategies."""
        con = duckdb.connect(self.db_path)
        genomes = con.execute("""
            SELECT strategy_id, fitness FROM evolved_strategies
            WHERE status IN ('ACTIVE', 'CANDIDATE') ORDER BY fitness DESC LIMIT 10
        """).fetchdf()

        results = []
        for _, genome in genomes.iterrows():
            key = f"evo_{genome['strategy_id']}"
            trades = con.execute("""
                SELECT timestamp, pnl FROM trades
                WHERE signal_source = ? ORDER BY timestamp DESC LIMIT 200
            """, [key]).fetchdf()

            if trades.empty:
                results.append(self._default_health(key, "EVOLUTION"))
                continue

            pnls = trades["pnl"].values[::-1]
            results.append({
                "strategy_name": key,
                "strategy_type": "EVOLUTION",
                "sharpe_7d": self._compute_rolling_sharpe(pnls, 7 * 78),
                "sharpe_14d": self._compute_rolling_sharpe(pnls, 14 * 78),
                "sharpe_30d": self._compute_rolling_sharpe(pnls, 30 * 78),
                "win_rate": self._compute_win_rate(pnls),
                "profit_factor": self._compute_profit_factor(pnls),
            })

        con.close()
        return results

    def _default_health(self, name: str, stype: str) -> dict:
        """Return default health for strategies with no trade data."""
        return {
            "strategy_name": name,
            "strategy_type": stype,
            "sharpe_7d": 0.0, "sharpe_14d": 0.0, "sharpe_30d": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0,
        }

    def check_and_act(self) -> list[dict]:
        """Run health checks and take automated actions."""
        actions = []

        # Gather all strategy health data
        all_health = []
        all_health.append(self.monitor_rl_agent())
        all_health.extend(self.monitor_stat_arb())
        all_health.extend(self.monitor_evolved_strategies())

        con = duckdb.connect(self.db_path)
        now = pd.Timestamp.now()

        for health in all_health:
            name = health["strategy_name"]
            s7 = health["sharpe_7d"]

            # Determine allocation and status
            allocation = 1.0
            status = "ACTIVE"

            # Check for reduced allocation
            if s7 < SHARPE_REDUCE_THRESHOLD and s7 > SHARPE_SUSPEND_THRESHOLD:
                allocation = ALLOCATION_REDUCE_PCT
                status = "REDUCED"
                actions.append({
                    "strategy": name,
                    "action": "REDUCE",
                    "reason": f"7-day Sharpe={s7:.3f} < {SHARPE_REDUCE_THRESHOLD}",
                })

            # Check for suspension
            if s7 < SHARPE_SUSPEND_THRESHOLD:
                # Check if this has been negative for consecutive days
                recent = con.execute("""
                    SELECT sharpe_7d FROM strategy_health
                    WHERE strategy_name = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, [name, SUSPEND_CONSECUTIVE_DAYS]).fetchdf()

                if len(recent) >= SUSPEND_CONSECUTIVE_DAYS:
                    if all(recent["sharpe_7d"] < SHARPE_SUSPEND_THRESHOLD):
                        allocation = 0.0
                        status = "SUSPENDED"
                        actions.append({
                            "strategy": name,
                            "action": "SUSPEND",
                            "reason": f"7-day Sharpe < 0 for {SUSPEND_CONSECUTIVE_DAYS} consecutive checks",
                        })

            # Compute decay rate from historical sharpe
            sharpe_hist = con.execute("""
                SELECT sharpe_7d FROM strategy_health
                WHERE strategy_name = ?
                ORDER BY timestamp DESC LIMIT 10
            """, [name]).fetchdf()

            decay_rate = 0.0
            time_to_death = 999.0
            if not sharpe_hist.empty:
                decay_rate = self._compute_decay_rate(sharpe_hist["sharpe_7d"].tolist()[::-1])
                if decay_rate < 0 and s7 > 0:
                    time_to_death = s7 / abs(decay_rate)

            # Store health snapshot
            health_row = pd.DataFrame([{
                "timestamp": now,
                "strategy_name": name,
                "strategy_type": health["strategy_type"],
                "sharpe_7d": s7,
                "sharpe_14d": health["sharpe_14d"],
                "sharpe_30d": health["sharpe_30d"],
                "win_rate": health["win_rate"],
                "profit_factor": health["profit_factor"],
                "decay_rate": decay_rate,
                "time_to_death_days": time_to_death,
                "allocation_pct": allocation,
                "status": status,
            }])
            con.execute("INSERT INTO strategy_health SELECT * FROM health_row")

            # Log events
            for action_info in actions:
                if action_info["strategy"] == name:
                    event = pd.DataFrame([{
                        "timestamp": now,
                        "strategy_name": name,
                        "event_type": action_info["action"],
                        "old_value": "ACTIVE",
                        "new_value": status,
                        "reason": action_info["reason"],
                    }])
                    con.execute("INSERT INTO alpha_events SELECT * FROM event")

        con.close()

        # Log actions
        for a in actions:
            logger.warning("Alpha decay action: {} -> {} ({})", a["strategy"], a["action"], a["reason"])

        return actions

    def get_strategy_allocations(self) -> dict[str, float]:
        """Get current allocation percentages for all strategies."""
        con = duckdb.connect(self.db_path)
        df = con.execute("""
            SELECT strategy_name, allocation_pct, status FROM strategy_health
            WHERE timestamp = (SELECT MAX(timestamp) FROM strategy_health)
        """).fetchdf()
        con.close()

        allocations = {}
        if not df.empty:
            for _, row in df.iterrows():
                allocations[row["strategy_name"]] = row["allocation_pct"]

        return allocations

    def write_health_json(self):
        """Write health snapshot to aria_health.json."""
        con = duckdb.connect(self.db_path)

        # Latest health for all strategies
        health = con.execute("""
            SELECT * FROM strategy_health
            WHERE timestamp = (SELECT MAX(timestamp) FROM strategy_health)
        """).fetchdf()

        # Recent events
        events = con.execute("""
            SELECT * FROM alpha_events
            ORDER BY timestamp DESC LIMIT 20
        """).fetchdf()

        # Model info
        model_info = con.execute("""
            SELECT version, eval_sharpe, promoted FROM retraining_log
            ORDER BY version DESC LIMIT 1
        """).fetchone()

        con.close()

        output = {
            "timestamp": datetime.now().isoformat(),
            "strategies": health.to_dict("records") if not health.empty else [],
            "recent_events": events.to_dict("records") if not events.empty else [],
            "model_version": model_info[0] if model_info else 0,
            "model_sharpe": model_info[1] if model_info else 0,
            "model_promoted": model_info[2] if model_info else False,
        }

        # Convert timestamps to strings for JSON
        for strategy in output["strategies"]:
            for k, v in strategy.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    strategy[k] = v.isoformat()

        for event in output["recent_events"]:
            for k, v in event.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    event[k] = v.isoformat()

        with open(HEALTH_FILE, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info("Written health snapshot to {}", HEALTH_FILE)

    def should_trigger_evolution(self) -> bool:
        """Check if any strategy suspension should trigger a new evolutionary search."""
        con = duckdb.connect(self.db_path)
        suspended = con.execute("""
            SELECT COUNT(*) FROM strategy_health
            WHERE status = 'SUSPENDED'
            AND timestamp > CURRENT_TIMESTAMP - INTERVAL 1 HOUR
        """).fetchone()
        con.close()
        return suspended[0] > 0 if suspended else False

    def run(self):
        """Run full monitoring cycle."""
        actions = self.check_and_act()
        self.write_health_json()
        return actions
