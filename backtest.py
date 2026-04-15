"""ARIA Backtester — walk-forward simulation over historical data with full performance report."""

import numpy as np
import pandas as pd
import duckdb
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arbitrage.grid import TopologicalGridEngine

DB_PATH = Path(__file__).parent / "aria.db"
OUTPUT_DIR = Path(__file__).parent / "backtest_results"

ASSETS = ["ES", "NQ", "GC", "CL", "TLT", "SPY", "DXY", "VIX", "ZN"]
INITIAL_CAPITAL = 1_000_000
TRANSACTION_COST_BPS = 1.0  # round-trip
SLIPPAGE_BPS = 1.0
MAX_POSITION_PCT = 0.20
MAX_PORTFOLIO_EXPOSURE = 1.0


class Backtester:
    """Runs ARIA strategies over historical data and produces performance metrics."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grid_engine = TopologicalGridEngine()

    def load_prices(self):
        """Load all OHLCV data from DuckDB."""
        con = duckdb.connect(self.db_path)
        prices = {}
        for asset in ASSETS:
            df = con.execute(
                "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE asset = ? ORDER BY timestamp",
                [asset]
            ).fetchdf()
            if not df.empty:
                df = df.set_index("timestamp")
                prices[asset] = df
        con.close()
        logger.info("Loaded prices for {} assets", len(prices))
        return prices

    def load_features(self):
        """Load precomputed features from DuckDB."""
        con = duckdb.connect(self.db_path)
        features = {}
        for asset in ASSETS:
            df = con.execute(
                "SELECT timestamp, feature_json FROM features_wide WHERE asset = ? ORDER BY timestamp",
                [asset]
            ).fetchdf()
            if not df.empty:
                features[asset] = df.set_index("timestamp")
        con.close()
        logger.info("Loaded features for {} assets", len(features))
        return features

    def trend_following_signal(self, prices, asset, idx):
        """Trend following: long above 50h MA, short below. Core alpha source."""
        if idx < 60:
            return 0.0
        closes = prices[asset]["close"].iloc[:idx + 1]
        ma_fast = closes.iloc[-10:].mean()
        ma_slow = closes.iloc[-50:].mean()
        atr = (prices[asset]["high"].iloc[idx - 20:idx] - prices[asset]["low"].iloc[idx - 20:idx]).mean()
        if atr < 1e-10:
            return 0.0
        # Normalized distance from slow MA
        signal = (ma_fast - ma_slow) / atr
        return np.clip(signal * 0.3, -1, 1)

    def vol_adjusted_reversion(self, prices, asset, idx):
        """Mean reversion only in low-vol, range-bound conditions."""
        if idx < 60:
            return 0.0
        closes = prices[asset]["close"].iloc[idx - 60:idx]
        ret_vol = closes.pct_change().std()
        # Only mean-revert in low vol (bottom 40th percentile)
        if idx < 200:
            return 0.0
        hist_vol = prices[asset]["close"].iloc[:idx].pct_change().rolling(60).std()
        vol_pct = (hist_vol < ret_vol).mean()
        if vol_pct > 0.6:
            return 0.0  # Not low vol — skip
        z = (closes.iloc[-1] - closes.mean()) / (closes.std() + 1e-10)
        if abs(z) < 1.5:
            return 0.0  # Not extreme enough
        return np.clip(-z * 0.2, -0.7, 0.7)

    def cross_asset_signal(self, prices, idx, lookback=40):
        """Cross-asset signal: risk-on/off from ES, VIX, TLT."""
        signals = {}
        if idx < lookback:
            return {a: 0.0 for a in ASSETS}

        es_ret = 0.0
        if "ES" in prices and idx < len(prices["ES"]):
            es_window = prices["ES"]["close"].iloc[max(0, idx - lookback):idx]
            if len(es_window) >= 2:
                es_ret = (es_window.iloc[-1] / es_window.iloc[0]) - 1

        vix_level = 20.0
        if "VIX" in prices and idx < len(prices["VIX"]):
            v = prices["VIX"]["close"].iloc[min(idx, len(prices["VIX"]) - 1)]
            vix_level = v

        risk_on = es_ret > 0.02 and vix_level < 18
        risk_off = es_ret < -0.02 and vix_level > 28

        for asset in ASSETS:
            if asset == "VIX":
                signals[asset] = 0.0
            elif asset in ("TLT", "ZN", "GC"):
                signals[asset] = -0.2 if risk_on else (0.4 if risk_off else 0.0)
            elif asset in ("ES", "NQ", "SPY"):
                signals[asset] = 0.3 if risk_on else (-0.3 if risk_off else 0.0)
            else:
                signals[asset] = 0.0
        return signals

    def feature_signal(self, features, asset, idx):
        """Signal from precomputed features — regime-adaptive."""
        if asset not in features:
            return 0.0

        feat_df = features[asset]
        if idx >= len(feat_df):
            return 0.0

        try:
            feat_json = feat_df["feature_json"].iloc[idx]
            feats = json.loads(feat_json)
        except (IndexError, json.JSONDecodeError):
            return 0.0

        hurst = feats.get("hurst", 0.5)
        trend = feats.get("trend_strength", 0)

        if hurst > 0.55 and trend > 30:
            signal = feats.get("ret_20", 0) * 4.0
        elif hurst < 0.42:
            z60 = feats.get("zscore_60", 0)
            signal = -z60 * 0.4 if abs(z60) > 1.5 else 0.0
        else:
            signal = feats.get("ret_10", 0) * 1.5

        # High vol dampening
        rv = feats.get("rv_regime", 0.5)
        if rv > 0.8:
            signal *= 0.4

        return np.clip(signal, -1, 1)

    def grid_signal(self, prices, asset, idx):
        """Topology-informed grid trading signal (cached every 20 bars)."""
        if idx < 60:
            return 0.0

        cache_key = f"{asset}_{idx // 20}"
        if not hasattr(self, '_grid_cache'):
            self._grid_cache = {}

        if cache_key not in self._grid_cache:
            close_arr = prices[asset]["close"].iloc[max(0, idx - 59):idx + 1].values
            high_arr = prices[asset]["high"].iloc[max(0, idx - 59):idx + 1].values
            low_arr = prices[asset]["low"].iloc[max(0, idx - 59):idx + 1].values
            signal, info = self.grid_engine.generate_signal(
                asset, close_arr, high_arr, low_arr, close_arr
            )
            self._grid_cache[cache_key] = signal
            # Evict old cache entries
            if len(self._grid_cache) > 200:
                old_keys = list(self._grid_cache.keys())[:100]
                for k in old_keys:
                    del self._grid_cache[k]

        return self._grid_cache[cache_key]

    def run(self, start_pct=0.3, end_pct=1.0):
        """Run the full backtest over historical data."""
        prices = self.load_prices()
        features = self.load_features()

        if not prices:
            logger.error("No price data — run ARIA first to populate the database")
            return

        # Align timestamps by flooring to nearest hour
        available = [a for a in ASSETS if a in prices]
        for asset in available:
            prices[asset].index = prices[asset].index.floor("h")
            prices[asset] = prices[asset][~prices[asset].index.duplicated(keep="last")]

        if available and features:
            for asset in list(features.keys()):
                features[asset].index = features[asset].index.floor("h")
                features[asset] = features[asset][~features[asset].index.duplicated(keep="last")]

        # Use union of timestamps — trade each asset whenever it has data
        all_ts = set()
        for a in available:
            all_ts.update(prices[a].index)
        common_ts = sorted(all_ts)

        if len(common_ts) < 100:
            logger.error("Not enough timestamps: {} (assets: {})", len(common_ts), available)
            return

        logger.info("Trading {} assets: {}", len(available), ", ".join(available))

        # Use portion of data for backtest (skip early data used for warmup)
        start_idx = int(len(common_ts) * start_pct)
        end_idx = int(len(common_ts) * end_pct)
        test_timestamps = common_ts[start_idx:end_idx]

        logger.info("Backtesting {} to {} ({} bars)",
                     test_timestamps[0], test_timestamps[-1], len(test_timestamps))

        # State
        capital = INITIAL_CAPITAL
        positions = {a: 0.0 for a in available}
        entry_prices = {a: 0.0 for a in available}

        # Tracking
        equity_curve = []
        trade_log = []
        daily_returns = []
        peak_equity = capital
        max_drawdown = 0.0

        for step, ts in enumerate(test_timestamps):
            # Get current prices
            current_prices = {}
            for asset in available:
                if ts in prices[asset].index:
                    current_prices[asset] = prices[asset].loc[ts, "close"]

            if not current_prices:
                continue

            # Find index in the full price array for signal computation
            price_idx = {}
            for asset in available:
                if ts in prices[asset].index:
                    loc = prices[asset].index.get_loc(ts)
                    if isinstance(loc, slice):
                        loc = loc.start
                    price_idx[asset] = loc

            # Generate signals from multiple sources
            combined_signals = {a: 0.0 for a in available}

            for asset in [a for a in available if a in price_idx]:
                if asset == "VIX":
                    continue  # VIX is signal only, don't trade it
                idx = price_idx[asset]
                trend = self.trend_following_signal(prices, asset, idx)
                feat = self.feature_signal(features, asset, idx)
                grid = self.grid_signal(prices, asset, idx)

                # Regime-adaptive signal fusion:
                # Strong trend -> follow trend, ignore grid
                # No trend + grid active -> use grid (mean-reversion)
                # Weak everything -> use features only
                if abs(trend) > 0.15:
                    # Clear trend: trend-follow, grid dampens
                    combined_signals[asset] = trend * 0.60 + feat * 0.30 + grid * 0.10
                elif abs(grid) > 0.05:
                    # Range-bound, grid is active: grid dominates
                    combined_signals[asset] = grid * 0.50 + feat * 0.30 + trend * 0.20
                else:
                    # Quiet market: light feature-based positioning
                    combined_signals[asset] = feat * 0.50 + trend * 0.30 + grid * 0.20

            # Cross-asset overlay (lighter touch)
            ref_idx = price_idx.get("ES", 0)
            cross = self.cross_asset_signal(prices, ref_idx)
            for asset in available:
                combined_signals[asset] = combined_signals.get(asset, 0) + cross.get(asset, 0) * 0.15
                combined_signals[asset] = np.clip(combined_signals.get(asset, 0), -1, 1)

            # Signal threshold — only trade when conviction is high enough
            SIGNAL_THRESHOLD = 0.12
            target_positions = {}
            for asset in available:
                sig = combined_signals.get(asset, 0.0)
                if abs(sig) < SIGNAL_THRESHOLD:
                    sig = 0.0  # No position if signal is weak
                target = sig * capital * MAX_POSITION_PCT
                target_positions[asset] = target

            # Enforce max exposure
            total_exp = sum(abs(v) for v in target_positions.values())
            max_exp = capital * MAX_PORTFOLIO_EXPOSURE
            if total_exp > max_exp and total_exp > 0:
                scale = max_exp / total_exp
                target_positions = {a: v * scale for a, v in target_positions.items()}

            # Execute trades — mark-to-market PnL on existing positions first
            step_pnl = 0.0
            for asset in available:
                if asset not in current_prices:
                    continue

                price = current_prices[asset]
                old_pos = positions[asset]
                new_pos = target_positions.get(asset, 0.0)

                # Mark-to-market: PnL from price change on existing position
                if old_pos != 0 and entry_prices[asset] > 0:
                    price_change = (price - entry_prices[asset]) / entry_prices[asset]
                    step_pnl += old_pos * price_change
                    entry_prices[asset] = price  # Reset entry to current for next bar

                # Only rebalance if position change is significant (5% of capital)
                if abs(new_pos - old_pos) < capital * 0.05:
                    continue

                # Transaction costs on the trade
                trade_value = abs(new_pos - old_pos)
                cost = trade_value * (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0
                step_pnl -= cost

                # Update position
                positions[asset] = new_pos
                entry_prices[asset] = price

                trade_log.append({
                    "timestamp": ts,
                    "asset": asset,
                    "old_pos": old_pos,
                    "new_pos": new_pos,
                    "price": price,
                    "signal": combined_signals.get(asset, 0.0),
                })

            capital += step_pnl
            equity_curve.append({"timestamp": ts, "equity": capital, "pnl": step_pnl})

            # Track drawdown
            peak_equity = max(peak_equity, capital)
            dd = (peak_equity - capital) / peak_equity
            max_drawdown = max(max_drawdown, dd)

        # Build results
        eq_df = pd.DataFrame(equity_curve)
        trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

        results = self.compute_metrics(eq_df, trade_df, max_drawdown)
        self.print_report(results)
        self.plot_results(eq_df, results)

        return results

    def compute_metrics(self, eq_df, trade_df, max_drawdown):
        """Compute performance metrics."""
        if eq_df.empty:
            return {}

        total_return = (eq_df["equity"].iloc[-1] / INITIAL_CAPITAL) - 1
        pnls = eq_df["pnl"].values

        # Sharpe (annualized, assuming hourly bars ~6.5hrs/day * 252 days)
        bars_per_year = 6.5 * 252
        sharpe = (np.mean(pnls) / (np.std(pnls) + 1e-10)) * np.sqrt(bars_per_year)

        # Sortino
        downside = pnls[pnls < 0]
        sortino = (np.mean(pnls) / (np.std(downside) + 1e-10)) * np.sqrt(bars_per_year) if len(downside) > 0 else 0

        # Calmar
        calmar = (total_return * (bars_per_year / len(pnls))) / (max_drawdown + 1e-10)

        # Win rate
        winning = pnls[pnls > 0]
        losing = pnls[pnls < 0]
        win_rate = len(winning) / (len(winning) + len(losing) + 1e-10)

        # Profit factor
        gross_profit = np.sum(winning)
        gross_loss = abs(np.sum(losing))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        # Trade stats
        n_trades = len(trade_df) if not trade_df.empty else 0

        # Monthly returns
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
        eq_df = eq_df.set_index("timestamp")
        monthly = eq_df["equity"].resample("ME").last().pct_change().dropna()

        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "final_equity": eq_df["equity"].iloc[-1],
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": n_trades,
            "n_bars": len(eq_df),
            "start_date": eq_df.index[0],
            "end_date": eq_df.index[-1],
            "avg_monthly_return": monthly.mean() * 100 if len(monthly) > 0 else 0,
            "best_month": monthly.max() * 100 if len(monthly) > 0 else 0,
            "worst_month": monthly.min() * 100 if len(monthly) > 0 else 0,
            "monthly_returns": monthly,
            "equity_df": eq_df,
        }

    def print_report(self, results):
        """Print performance report to console."""
        if not results:
            logger.error("No results to report")
            return

        print("\n" + "=" * 60)
        print("  ARIA BACKTEST REPORT")
        print("=" * 60)
        print(f"  Period:          {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}")
        print(f"  Bars tested:     {results['n_bars']:,}")
        print(f"  Trades:          {results['n_trades']:,}")
        print("-" * 60)
        print(f"  Total Return:    {results['total_return_pct']:+.2f}%")
        print(f"  Final Equity:    ${results['final_equity']:,.0f}")
        print(f"  Sharpe Ratio:    {results['sharpe']:.3f}")
        print(f"  Sortino Ratio:   {results['sortino']:.3f}")
        print(f"  Calmar Ratio:    {results['calmar']:.3f}")
        print(f"  Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate:        {results['win_rate']*100:.1f}%")
        print(f"  Profit Factor:   {results['profit_factor']:.2f}")
        print("-" * 60)
        print(f"  Avg Month:       {results['avg_monthly_return']:+.2f}%")
        print(f"  Best Month:      {results['best_month']:+.2f}%")
        print(f"  Worst Month:     {results['worst_month']:+.2f}%")
        print("=" * 60)
        print(f"\n  Charts saved to: {self.output_dir}/")
        print()

    def plot_results(self, eq_df, results):
        """Generate performance charts."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
        fig.suptitle("ARIA Backtest Results", fontsize=16, fontweight="bold")

        equity_df = results["equity_df"]

        # 1. Equity curve
        ax1 = axes[0]
        ax1.plot(equity_df.index, equity_df["equity"], color="#2196F3", linewidth=1.2, label="Portfolio Equity")
        ax1.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
        ax1.fill_between(equity_df.index, INITIAL_CAPITAL, equity_df["equity"],
                         where=equity_df["equity"] >= INITIAL_CAPITAL, alpha=0.15, color="green")
        ax1.fill_between(equity_df.index, INITIAL_CAPITAL, equity_df["equity"],
                         where=equity_df["equity"] < INITIAL_CAPITAL, alpha=0.15, color="red")
        ax1.set_ylabel("Equity ($)")
        ax1.set_title(f"Equity Curve — Sharpe: {results['sharpe']:.2f} | Return: {results['total_return_pct']:+.1f}% | MaxDD: {results['max_drawdown_pct']:.1f}%")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # 2. Drawdown
        ax2 = axes[1]
        peak = equity_df["equity"].cummax()
        drawdown = (peak - equity_df["equity"]) / peak * 100
        ax2.fill_between(equity_df.index, 0, drawdown, color="red", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown")
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)

        # 3. Monthly returns bar chart
        ax3 = axes[2]
        monthly = results["monthly_returns"]
        if len(monthly) > 0:
            colors = ["green" if r >= 0 else "red" for r in monthly.values]
            ax3.bar(monthly.index, monthly.values * 100, width=20, color=colors, alpha=0.7)
            ax3.axhline(y=0, color="gray", linewidth=0.5)
            ax3.set_ylabel("Return (%)")
            ax3.set_title("Monthly Returns")
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = self.output_dir / "backtest_report.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Chart saved to {}", chart_path)

        # Save equity curve CSV
        csv_path = self.output_dir / "equity_curve.csv"
        equity_df.to_csv(csv_path)
        logger.info("Equity curve saved to {}", csv_path)


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

    bt = Backtester()
    results = bt.run(start_pct=0.3, end_pct=1.0)


if __name__ == "__main__":
    main()
