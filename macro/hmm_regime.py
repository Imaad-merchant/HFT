"""HMM-based market regime detector and short-horizon dump probability forecaster.

Fits a Gaussian Hidden Markov Model on multi-asset return / volatility / VIX
features, labels the latent states by mean ES return (BULL / NORMAL / BEAR /
CRASH), then rolls the learned transition matrix forward 1-2 steps to produce
the probability that the market sits in a "dump" regime tomorrow and the day
after.

This is a probabilistic regime forecast, not a crystal ball — it tells you how
similar today's tape looks to historical setups that preceded sharp downside,
and it propagates the empirical transition probabilities forward. It does not
"know" any specific future event.

CLI usage:
    python -m macro.hmm_regime fit       # fit and persist the model
    python -m macro.hmm_regime predict   # print current regime + 1-2d dump probs
    python -m macro.hmm_regime run       # fit (if needed) then predict
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

DB_PATH = Path(__file__).parent.parent / "aria.db"
MODEL_DIR = Path(__file__).parent.parent / "models"

# Equity index futures we run separate HMMs for. The detector is fit once per
# primary asset; ES and NQ are the standard pair for US equity regime work.
DEFAULT_PRIMARIES = ["ES", "NQ"]

# Cross-asset confirmation features. The current primary is excluded from this
# list at runtime so it doesn't get duplicated as both the primary and a support.
SUPPORT_ASSETS = ["ES", "NQ", "VIX", "TLT", "GC", "DXY"]

# Slow-moving macro series that condition the regime. These are pulled from
# the macro_data table populated by MacroEngine.fetch_fred_data() if present.
FRED_FEATURES = [
    "fed_funds_rate",
    "yield_10y",
    "yield_2y",
    "yield_curve_spread",
    "cpi",
    "unemployment",
]

# Hours of news headlines to roll into the per-bar sentiment feature.
SENTIMENT_LOOKBACK_HOURS = 24

# Number of latent regimes. Four is the sweet spot for equity index data:
# strong-bull / drift / pullback / crash. More states overfits short samples.
N_STATES = 4

# Labels are assigned post-fit by sorting states on mean ES log-return.
REGIME_LABELS = ["CRASH", "BEAR", "NORMAL", "BULL"]
DUMP_REGIMES = {"CRASH", "BEAR"}

# Trading bars per "day" — depends on the bar interval ARIA fetched. The
# fetcher uses 1h bars over 2y history, which is ~6.5 bars per RTH session.
# We auto-detect the bar cadence from the data and convert horizons properly.


@dataclass
class HMMForecast:
    """One forecast snapshot from the HMM regime model."""

    timestamp: str
    asset: str                              # primary asset this forecast is for
    current_regime: str
    current_state_id: int
    state_posterior: list[float]            # P(state=i | obs up to now)
    regime_posterior: dict[str, float]      # P(regime=label | obs up to now)
    p_dump_1d: float                        # P(dump regime ~1 trading day ahead)
    p_dump_2d: float                        # P(dump regime ~2 trading days ahead)
    p_dump_now: float                       # P(currently in dump regime)
    expected_1d_return: float               # state-mixture expected primary log-return
    expected_2d_return: float
    bars_per_day: int
    n_train_bars: int
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


class HMMRegimeDetector:
    """Gaussian HMM over multi-asset returns + vol + VIX features."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        primary_asset: str = "ES",
        n_states: int = N_STATES,
        use_macro: bool = True,
        use_sentiment: bool = True,
    ):
        self.db_path = str(db_path or DB_PATH)
        self.primary_asset = primary_asset
        self.n_states = n_states
        self.use_macro = use_macro
        self.use_sentiment = use_sentiment
        self.model = None  # hmmlearn.hmm.GaussianHMM
        self.feature_cols: list[str] = []
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.state_label_map: dict[int, str] = {}
        self.state_mean_return: dict[int, float] = {}
        self.bars_per_day: int = 7
        self.model_path = MODEL_DIR / f"hmm_regime_{primary_asset.lower()}.pkl"
        self._init_db()

    # ------------------------------------------------------------------ DB

    def _init_db(self) -> None:
        con = duckdb.connect(self.db_path)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS hmm_forecast (
                timestamp TIMESTAMP,
                asset VARCHAR,
                current_regime VARCHAR,
                current_state_id INTEGER,
                p_dump_now DOUBLE,
                p_dump_1d DOUBLE,
                p_dump_2d DOUBLE,
                expected_1d_return DOUBLE,
                expected_2d_return DOUBLE,
                payload_json VARCHAR,
                PRIMARY KEY (timestamp, asset)
            )
            """
        )
        con.close()
        MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # -------------------------------------------------------------- features

    def _load_aligned_closes(self) -> pd.DataFrame:
        """Pull closes for primary + support assets aligned on a common index."""
        con = duckdb.connect(self.db_path)
        frames: dict[str, pd.Series] = {}
        # Always include the primary first, then everything else from SUPPORT_ASSETS
        # excluding the primary itself to avoid duplicating it.
        asset_list = [self.primary_asset] + [a for a in SUPPORT_ASSETS if a != self.primary_asset]
        for asset in asset_list:
            df = con.execute(
                "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp",
                [asset],
            ).fetchdf()
            if df.empty:
                logger.warning("HMM[{}]: no rows for {} in ohlcv — skipping", self.primary_asset, asset)
                continue
            df = df.set_index("timestamp")
            frames[asset] = df["close"].astype(float)
        con.close()

        if self.primary_asset not in frames:
            raise RuntimeError(
                f"HMM[{self.primary_asset}] fit needs OHLCV for {self.primary_asset}; "
                f"none found in {self.db_path}"
            )

        wide = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="any")
        return wide

    def _detect_bars_per_day(self, index: pd.DatetimeIndex) -> int:
        """Estimate trading bars per day from the median bar interval."""
        if len(index) < 3:
            return 7
        diffs = np.diff(index.values).astype("timedelta64[s]").astype(float)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 7
        median_seconds = float(np.median(diffs))
        if median_seconds <= 0:
            return 7
        # 6.5 RTH hours per US session -> bars/day
        bars = int(round((6.5 * 3600) / median_seconds))
        return max(1, min(bars, 78))  # clamp to [1, 78] (78 = 5-min bars)

    def _load_macro_panel(self) -> pd.DataFrame:
        """Pull FRED macro series from `macro_data` and pivot into a wide daily panel.

        Returns an empty frame if the table doesn't exist or has no rows.
        Series come in at heterogeneous cadences (daily yields, monthly CPI,
        quarterly GDP) — we keep them as they are and forward-fill onto the
        price bar timeline downstream.
        """
        try:
            con = duckdb.connect(self.db_path)
            df = con.execute(
                """
                SELECT date, series_name, value
                FROM macro_data
                WHERE series_name IN ({})
                ORDER BY date
                """.format(",".join(f"'{s}'" for s in FRED_FEATURES))
            ).fetchdf()
            con.close()
        except Exception as e:
            logger.debug("HMM: macro_data not available ({}) — skipping FRED features", e)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        wide = df.pivot_table(index="date", columns="series_name", values="value", aggfunc="last")
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        # Synthesize the slope column even if FRED's T10Y2Y is missing.
        if "yield_10y" in wide.columns and "yield_2y" in wide.columns and "yield_curve_spread" not in wide.columns:
            wide["yield_curve_spread"] = wide["yield_10y"] - wide["yield_2y"]
        return wide

    def _load_sentiment_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Build a per-bar sentiment series: rolling mean of last 24h headlines.

        Returns a Series of zeros if `news_sentiment` is empty/missing.
        """
        try:
            con = duckdb.connect(self.db_path)
            df = con.execute(
                """
                SELECT timestamp, sentiment_score
                FROM news_sentiment
                ORDER BY timestamp
                """
            ).fetchdf()
            con.close()
        except Exception as e:
            logger.debug("HMM: news_sentiment not available ({}) — skipping sentiment", e)
            return pd.Series(0.0, index=index)

        if df.empty:
            return pd.Series(0.0, index=index)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        # 24-hour rolling mean of compound sentiment
        rolling = df["sentiment_score"].rolling(f"{SENTIMENT_LOOKBACK_HOURS}h").mean()
        # Project onto the price bar timeline with last-known carry forward
        projected = rolling.reindex(rolling.index.union(index)).sort_index().ffill().reindex(index)
        return projected.fillna(0.0)

    def _build_features(self, closes: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
        """Construct the HMM observation matrix from prices + (optional) macro + sentiment."""
        df = pd.DataFrame(index=closes.index)

        pri = closes[self.primary_asset]
        df["pri_logret"] = np.log(pri).diff()

        # Realised vol (rolling std of 1-bar log returns)
        df["pri_vol_20"] = df["pri_logret"].rolling(20).std()
        df["pri_vol_60"] = df["pri_logret"].rolling(60).std()

        # Vol-of-vol — accelerates before crashes
        df["pri_volvol"] = df["pri_vol_20"].rolling(20).std()

        # Drawdown from rolling high (negative when stressed)
        roll_max = pri.rolling(60, min_periods=10).max()
        df["pri_dd_60"] = (pri / roll_max) - 1.0

        if "VIX" in closes.columns:
            vix = closes["VIX"]
            df["vix_level"] = vix
            df["vix_chg"] = vix.diff()
            df["vix_z"] = (vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-9)

        if "TLT" in closes.columns:
            tlt = closes["TLT"]
            df["tlt_logret"] = np.log(tlt).diff()
            # Stocks down + bonds up = classic risk-off
            df["pri_tlt_diff"] = df["pri_logret"] - df["tlt_logret"]

        # Cross-asset confirmations from the *other* equity index. When the
        # primary is ES we add NQ, and vice versa — gives the model both an
        # idiosyncratic and a systemic signal.
        if self.primary_asset != "ES" and "ES" in closes.columns:
            df["es_logret"] = np.log(closes["ES"]).diff()
        if self.primary_asset != "NQ" and "NQ" in closes.columns:
            df["nq_logret"] = np.log(closes["NQ"]).diff()

        if "DXY" in closes.columns:
            df["dxy_logret"] = np.log(closes["DXY"]).diff()

        if "GC" in closes.columns:
            df["gc_logret"] = np.log(closes["GC"]).diff()

        # ----- Optional FRED macro features -----
        if self.use_macro:
            macro_panel = self._load_macro_panel()
            if not macro_panel.empty:
                # Forward-fill the daily/monthly panel onto the bar timeline
                macro_aligned = (
                    macro_panel.reindex(macro_panel.index.union(df.index))
                    .sort_index()
                    .ffill()
                    .reindex(df.index)
                )
                added = []
                for col in FRED_FEATURES:
                    if col in macro_aligned.columns:
                        series = macro_aligned[col].astype(float)
                        df[f"macro_{col}"] = series
                        # 30-day rate of change captures regime transitions
                        df[f"macro_{col}_chg30"] = series.diff(30 * max(1, self.bars_per_day))
                        added.append(col)
                if added:
                    logger.debug("HMM: fused FRED features {}", added)

        # ----- Optional news sentiment feature -----
        if self.use_sentiment:
            sent = self._load_sentiment_series(df.index)
            if (sent != 0).any():
                df["news_sentiment_24h"] = sent
                # 5-day delta — accelerating negative sentiment is a stress tell
                window = max(1, 5 * max(1, self.bars_per_day))
                df["news_sentiment_chg5d"] = sent - sent.shift(window)
                logger.debug("HMM: fused news_sentiment_24h feature")

        df = df.dropna()
        feature_cols = list(df.columns)
        return df.values, feature_cols, df.index

    # ------------------------------------------------------------------ fit

    def fit(self) -> dict:
        """Fit the HMM on all available history and persist the model."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise RuntimeError(
                "hmmlearn is required for HMMRegimeDetector. Install with: pip install hmmlearn"
            ) from e

        closes = self._load_aligned_closes()
        # Detect bar cadence first so the feature builder sizes its rolling
        # windows consistently between fit and predict.
        self.bars_per_day = self._detect_bars_per_day(closes.index)
        X, cols, idx = self._build_features(closes)
        if len(X) < 200:
            raise RuntimeError(
                f"HMM fit needs at least 200 aligned bars; got {len(X)}. "
                f"Run the data fetcher first (orch.data_fetcher.fetch_all('2y'))."
            )

        self.feature_cols = cols

        # Z-score features so the diagonal Gaussian is well-conditioned
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-9
        Xz = (X - self.feature_mean) / self.feature_std

        logger.info(
            "HMM: fitting GaussianHMM n_states={} on {} bars x {} features (bars/day={})",
            self.n_states, len(Xz), Xz.shape[1], self.bars_per_day,
        )

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=200,
            tol=1e-4,
            random_state=42,
            init_params="stmc",
        )
        model.fit(Xz)
        self.model = model

        # --- Label states by a left-tail-aware score ---
        # Naively ranking by mean primary log-return mislabels high-vol regimes:
        # a stress state with a few positive shocks can have a positive mean
        # yet still be a "dump" regime. Rank by `mean - 2*vol` instead so high
        # volatility pushes a state toward CRASH/BEAR regardless of sign.
        state_seq = model.predict(Xz)
        pri_logret_idx = self.feature_cols.index("pri_logret")
        per_state_mean: dict[int, float] = {}
        per_state_vol: dict[int, float] = {}
        per_state_score: dict[int, float] = {}
        for s in range(self.n_states):
            mask = state_seq == s
            if mask.any():
                per_state_mean[s] = float(X[mask, pri_logret_idx].mean())
                per_state_vol[s] = float(X[mask, pri_logret_idx].std())
            else:
                per_state_mean[s] = 0.0
                per_state_vol[s] = 0.0
            per_state_score[s] = per_state_mean[s] - 2.0 * per_state_vol[s]

        ordered = sorted(per_state_score.items(), key=lambda kv: kv[1])  # worst -> best
        labels = REGIME_LABELS[: self.n_states] if self.n_states <= len(REGIME_LABELS) else [
            f"S{i}" for i in range(self.n_states)
        ]
        self.state_label_map = {state_id: labels[rank] for rank, (state_id, _) in enumerate(ordered)}
        self.state_mean_return = per_state_mean

        logger.info("HMM[{}]: state labels (rank by mean - 2*vol): {}", self.primary_asset, self.state_label_map)
        logger.info("HMM[{}]: per-state mean log-return: {}", self.primary_asset, {k: round(v, 6) for k, v in per_state_mean.items()})
        logger.info("HMM[{}]: per-state vol  log-return: {}", self.primary_asset, {k: round(v, 6) for k, v in per_state_vol.items()})
        logger.info("HMM[{}]: transition matrix:\n{}", self.primary_asset, np.round(model.transmat_, 3))

        self._save()
        return {
            "n_train_bars": int(len(Xz)),
            "bars_per_day": int(self.bars_per_day),
            "state_labels": self.state_label_map,
            "state_mean_return": self.state_mean_return,
        }

    # --------------------------------------------------------------- persist

    def _save(self) -> None:
        payload = {
            "primary_asset": self.primary_asset,
            "model": self.model,
            "feature_cols": self.feature_cols,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "state_label_map": self.state_label_map,
            "state_mean_return": self.state_mean_return,
            "bars_per_day": self.bars_per_day,
            "n_states": self.n_states,
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("HMM[{}]: saved model to {}", self.primary_asset, self.model_path)

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]
        self.feature_mean = payload["feature_mean"]
        self.feature_std = payload["feature_std"]
        self.state_label_map = payload["state_label_map"]
        self.state_mean_return = payload["state_mean_return"]
        self.bars_per_day = payload.get("bars_per_day", 7)
        self.n_states = payload.get("n_states", N_STATES)
        # Sanity-check the saved primary_asset matches what was requested
        saved_primary = payload.get("primary_asset", self.primary_asset)
        if saved_primary != self.primary_asset:
            logger.warning(
                "HMM: saved model is for {} but instance asks for {}. Refit recommended.",
                saved_primary, self.primary_asset,
            )
        return True

    # -------------------------------------------------------------- predict

    def predict(self, persist: bool = True) -> HMMForecast:
        """Run a forecast: current regime posterior + 1d/2d dump probability."""
        if self.model is None and not self.load():
            raise RuntimeError("No HMM model loaded — call fit() first.")

        closes = self._load_aligned_closes()
        X, cols, idx = self._build_features(closes)
        if cols != self.feature_cols:
            raise RuntimeError(
                f"Feature columns changed since fit: model expects {self.feature_cols}, got {cols}. "
                "Refit the model."
            )
        if len(X) == 0:
            raise RuntimeError("No features available for prediction.")

        Xz = (X - self.feature_mean) / self.feature_std

        # Posterior over states given the full observation history
        gamma = self.model.predict_proba(Xz)  # shape (T, n_states)
        current = gamma[-1]                   # P(state_T = i | obs_{1..T})
        current_state_id = int(np.argmax(current))
        current_regime = self.state_label_map.get(current_state_id, f"S{current_state_id}")

        # Roll forward k bars under the learned transition matrix
        T = self.model.transmat_  # (n_states, n_states)
        k1 = max(1, self.bars_per_day)
        k2 = max(2, 2 * self.bars_per_day)

        def step_forward(p: np.ndarray, k: int) -> np.ndarray:
            out = p.copy()
            for _ in range(k):
                out = out @ T
            return out

        p1 = step_forward(current, k1)
        p2 = step_forward(current, k2)

        # P(dump) = sum of probabilities over CRASH/BEAR labelled states
        dump_state_ids = {sid for sid, lab in self.state_label_map.items() if lab in DUMP_REGIMES}

        def dump_prob(p: np.ndarray) -> float:
            return float(sum(p[s] for s in dump_state_ids))

        p_dump_now = dump_prob(current)
        p_dump_1d = dump_prob(p1)
        p_dump_2d = dump_prob(p2)

        # Expected ES log-return = state-mixture * per-state mean * horizon
        mean_vec = np.array([self.state_mean_return.get(s, 0.0) for s in range(self.n_states)])
        exp_1d = float((p1 @ mean_vec) * k1)
        exp_2d = float((p2 @ mean_vec) * k2)

        regime_posterior: dict[str, float] = {}
        for sid, p in enumerate(current):
            label = self.state_label_map.get(sid, f"S{sid}")
            regime_posterior[label] = regime_posterior.get(label, 0.0) + float(p)

        note = self._build_note(p_dump_now, p_dump_1d, p_dump_2d, current_regime)

        forecast = HMMForecast(
            timestamp=pd.Timestamp(idx[-1]).isoformat(),
            asset=self.primary_asset,
            current_regime=current_regime,
            current_state_id=current_state_id,
            state_posterior=[float(x) for x in current],
            regime_posterior=regime_posterior,
            p_dump_1d=p_dump_1d,
            p_dump_2d=p_dump_2d,
            p_dump_now=p_dump_now,
            expected_1d_return=exp_1d,
            expected_2d_return=exp_2d,
            bars_per_day=int(self.bars_per_day),
            n_train_bars=int(len(Xz)),
            note=note,
        )

        if persist:
            self._persist_forecast(forecast)

        return forecast

    @staticmethod
    def _build_note(p_now: float, p1: float, p2: float, regime: str) -> str:
        peak = max(p_now, p1, p2)
        if peak >= 0.65:
            tone = "ELEVATED — current tape resembles historical pre-dump regimes"
        elif peak >= 0.40:
            tone = "MODERATE — some stress features present, but not dominant"
        else:
            tone = "LOW — current dynamics look closer to drift/bull regimes"
        return f"{tone} (current regime: {regime})"

    def _persist_forecast(self, fc: HMMForecast) -> None:
        con = duckdb.connect(self.db_path)
        try:
            con.execute(
                "DELETE FROM hmm_forecast WHERE timestamp = ? AND asset = ?",
                [pd.Timestamp(fc.timestamp), fc.asset],
            )
            con.execute(
                """
                INSERT INTO hmm_forecast VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    pd.Timestamp(fc.timestamp),
                    fc.asset,
                    fc.current_regime,
                    fc.current_state_id,
                    fc.p_dump_now,
                    fc.p_dump_1d,
                    fc.p_dump_2d,
                    fc.expected_1d_return,
                    fc.expected_2d_return,
                    json.dumps(fc.to_dict()),
                ],
            )
        finally:
            con.close()

    # ----------------------------------------------------------------- run

    def run(self) -> HMMForecast:
        """Full cycle: fit if no model exists, then predict."""
        if not self.load():
            self.fit()
        return self.predict()

    # ----------------------------------------------------------- horizon

    def forecast_horizon(self, n_days: int = 30) -> pd.DataFrame:
        """Roll the current state posterior forward N trading days bar-by-bar.

        Returns one row per future bar with columns:
          - bar, day:                bar index and fractional trading-day index
          - p_<LABEL>:               P(regime = label) at that future bar
          - p_dump:                  P(CRASH or BEAR) at that future bar
          - e_ret_per_bar:           state-mixture expected log-return for that bar
          - cum_e_logret:            cumulative expected log-return from now to that bar
        """
        if self.model is None and not self.load():
            raise RuntimeError("No HMM model loaded — call fit() first.")

        closes = self._load_aligned_closes()
        X, cols, idx = self._build_features(closes)
        if cols != self.feature_cols:
            raise RuntimeError(
                f"Feature columns changed since fit: model expects {self.feature_cols}, got {cols}. "
                "Refit the model."
            )
        if len(X) == 0:
            raise RuntimeError("No features available for forecast horizon.")

        Xz = (X - self.feature_mean) / self.feature_std
        gamma = self.model.predict_proba(Xz)
        current = gamma[-1].copy()

        T = self.model.transmat_
        bars_per_day = max(1, self.bars_per_day)
        total_bars = int(n_days * bars_per_day)

        mean_vec = np.array([self.state_mean_return.get(s, 0.0) for s in range(self.n_states)])
        dump_state_ids = [sid for sid, lab in self.state_label_map.items() if lab in DUMP_REGIMES]
        regime_columns = REGIME_LABELS[: self.n_states]

        rows = []
        p = current
        cum_logret = 0.0

        for bar_idx in range(total_bars + 1):
            day = bar_idx / bars_per_day
            regime_probs = {label: 0.0 for label in regime_columns}
            for sid, prob in enumerate(p):
                label = self.state_label_map.get(sid, f"S{sid}")
                if label in regime_probs:
                    regime_probs[label] += float(prob)

            p_dump = float(sum(p[s] for s in dump_state_ids))
            e_ret = float(p @ mean_vec)
            cum_logret += e_ret if bar_idx > 0 else 0.0

            row = {
                "bar": bar_idx,
                "day": day,
                "p_dump": p_dump,
                "e_ret_per_bar": e_ret,
                "cum_e_logret": cum_logret,
            }
            for label in regime_columns:
                row[f"p_{label}"] = regime_probs[label]
            rows.append(row)

            # Step the chain forward one bar (skip on the last iteration is fine — we don't use it)
            p = p @ T

        return pd.DataFrame(rows)

    def expected_first_passage_to_dump(self) -> Optional[float]:
        """Expected number of trading days until the chain first enters a dump state.

        Computes the mean first-passage time to the dump set analytically by
        treating the dump states as absorbing: solve `(I - Q) h = 1` on the
        transient submatrix Q, then look up the entry for the current state.

        Returns:
          - 0.0 if the current state is already a dump state
          - float days if reachable from the current state via positive-probability
            transitions, the linear system is well-conditioned, and the solution
            is finite and non-negative
          - None otherwise (model not loaded, dump set unreachable, singular
            system, numerical blow-up)
        """
        if self.model is None and not self.load():
            return None

        closes = self._load_aligned_closes()
        X, cols, idx = self._build_features(closes)
        if cols != self.feature_cols or len(X) == 0:
            return None
        Xz = (X - self.feature_mean) / self.feature_std
        gamma = self.model.predict_proba(Xz)
        current_state = int(np.argmax(gamma[-1]))

        dump_state_ids = {sid for sid, lab in self.state_label_map.items() if lab in DUMP_REGIMES}
        if current_state in dump_state_ids:
            return 0.0

        T = self.model.transmat_
        n = T.shape[0]

        # ---- Reachability check ----
        # If no positive-probability path from current_state into the dump set
        # exists, the analytic first-passage time is +infinity. Detect this by
        # BFS on the directed graph of nonzero transitions.
        EPSILON = 1e-9
        visited: set[int] = {current_state}
        stack: list[int] = [current_state]
        reached_dump = False
        while stack:
            s = stack.pop()
            for j in range(n):
                if T[s, j] > EPSILON and j not in visited:
                    if j in dump_state_ids:
                        reached_dump = True
                        stack.clear()
                        break
                    visited.add(j)
                    stack.append(j)
        if not reached_dump:
            return None

        # ---- Solve (I - Q) h = 1 on the transient submatrix ----
        transient = [s for s in range(n) if s not in dump_state_ids]
        if not transient:
            return None
        Q = T[np.ix_(transient, transient)]
        A = np.eye(len(transient)) - Q

        # Reject ill-conditioned systems before they produce nonsense
        if not np.all(np.isfinite(A)) or np.linalg.cond(A) > 1e10:
            return None

        try:
            h = np.linalg.solve(A, np.ones(len(transient)))
        except np.linalg.LinAlgError:
            return None

        state_to_idx = {s: i for i, s in enumerate(transient)}
        if current_state not in state_to_idx:
            return None

        bars = float(h[state_to_idx[current_state]])
        if not np.isfinite(bars) or bars < 0:
            return None

        return bars / max(1, self.bars_per_day)

    def plot_forecast(
        self,
        n_days: int = 30,
        output_dir: str | Path = "dashboard",
    ) -> Path:
        """Render a 3-panel forecast chart and save to dashboard/.

        Panels:
          1. Stacked area of regime probabilities over the forecast horizon
          2. P(dump) decay curve with the expected-first-passage marker
          3. Expected cumulative log-return path

        Returns the PNG path. Also writes a `.csv` with the raw trajectory.
        """
        # Lazy import — matplotlib is optional and only needed for visuals
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        df = self.forecast_horizon(n_days=n_days)
        fpt_days = self.expected_first_passage_to_dump()

        # Pull the current row to label the chart
        current_row = df.iloc[0]
        regime_columns = [f"p_{lab}" for lab in REGIME_LABELS[: self.n_states] if f"p_{lab}" in df.columns]
        current_probs = {col[2:]: float(current_row[col]) for col in regime_columns}
        current_regime = max(current_probs.items(), key=lambda kv: kv[1])[0]

        if fpt_days is None:
            fpt_label = "dump unreachable from current state"
        elif fpt_days == 0.0:
            fpt_label = "currently in dump state"
        else:
            fpt_label = f"~{fpt_days:.1f} trading days"

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        fig.suptitle(
            f"HMM Regime Forecast — {self.primary_asset}\n"
            f"Current: {current_regime}   |   "
            f"Expected first dump transition: {fpt_label}   |   "
            f"Horizon: {n_days} trading days",
            fontsize=12, fontweight="bold",
        )

        # Stable colour scheme so ES and NQ panels look the same
        REGIME_COLORS = {
            "BULL":   "#2ecc71",
            "NORMAL": "#f1c40f",
            "BEAR":   "#e67e22",
            "CRASH":  "#c0392b",
        }

        days = df["day"].values

        # ---- Panel 1: stacked regime probabilities ----
        ax1 = axes[0]
        # Order BULL -> CRASH so the dump regimes are at the top of the stack
        ordered = [lab for lab in ["BULL", "NORMAL", "BEAR", "CRASH"] if f"p_{lab}" in df.columns]
        stacks = [df[f"p_{lab}"].values for lab in ordered]
        colors = [REGIME_COLORS.get(lab, "#95a5a6") for lab in ordered]
        ax1.stackplot(days, *stacks, labels=ordered, colors=colors, alpha=0.9)
        ax1.set_ylabel("Regime probability")
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, max(1.0, float(days.max())))
        ax1.legend(loc="upper right", framealpha=0.92, ncol=len(ordered), fontsize=9)
        ax1.set_title("Forward regime distribution (rolled under fitted transition matrix)", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ---- Panel 2: P(dump) decay curve + first-passage marker ----
        ax2 = axes[1]
        ax2.plot(days, df["p_dump"].values, color="#c0392b", linewidth=2.5,
                 label="P(dump = CRASH ∪ BEAR)")
        ax2.fill_between(days, 0, df["p_dump"].values, color="#c0392b", alpha=0.18)
        ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="50% threshold")
        if fpt_days is not None and fpt_days > 0 and fpt_days <= float(days.max()):
            ax2.axvline(fpt_days, color="black", linestyle=":", linewidth=1.6,
                        label=f"E[first dump] = {fpt_days:.1f}d")
        # Mark the *current* P(dump) value
        ax2.scatter([0], [df["p_dump"].iloc[0]], color="black", zorder=5, s=40)
        ax2.set_ylabel("P(dump)")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right", framealpha=0.92, fontsize=9)
        ax2.set_title("Probability of being in a CRASH or BEAR regime", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # ---- Panel 3: expected cumulative log-return path ----
        ax3 = axes[2]
        cum_pct = df["cum_e_logret"].values * 100.0
        ax3.plot(days, cum_pct, color="#2980b9", linewidth=2.5)
        ax3.fill_between(days, 0, cum_pct, where=(cum_pct >= 0),
                         interpolate=True, color="#2ecc71", alpha=0.3, label="positive drift")
        ax3.fill_between(days, 0, cum_pct, where=(cum_pct < 0),
                         interpolate=True, color="#c0392b", alpha=0.3, label="negative drift")
        ax3.axhline(0.0, color="gray", linestyle="-", linewidth=0.5)
        ax3.set_xlabel("Trading days from now")
        ax3.set_ylabel("Cumulative E[log-return] (%)")
        ax3.set_title("State-mixture expected return path", fontsize=10)
        ax3.legend(loc="upper right", framealpha=0.92, fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        png_path = output_dir / f"hmm_forecast_{self.primary_asset.lower()}.png"
        csv_path = output_dir / f"hmm_forecast_{self.primary_asset.lower()}.csv"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        df.to_csv(csv_path, index=False)

        logger.info("HMM[{}]: saved visualization {} (data {})", self.primary_asset, png_path, csv_path)
        return png_path


# --------------------------------------------------------------------- CLI

def _print_forecast(fc: HMMForecast) -> None:
    print("=" * 64)
    print(f"ARIA HMM Regime Forecast — {fc.asset}")
    print("=" * 64)
    print(f"As of bar:        {fc.timestamp}")
    print(f"Bars/day:         {fc.bars_per_day}   Train bars: {fc.n_train_bars}")
    print(f"Current regime:   {fc.current_regime}  (state {fc.current_state_id})")
    print()
    print("Regime posterior (now):")
    for label, p in sorted(fc.regime_posterior.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(p * 40))
        print(f"  {label:<7} {p:6.1%}  {bar}")
    print()
    print(f"P(dump now):       {fc.p_dump_now:6.1%}")
    print(f"P(dump in ~1 day): {fc.p_dump_1d:6.1%}   E[ret 1d] = {fc.expected_1d_return:+.3%}")
    print(f"P(dump in ~2 day): {fc.p_dump_2d:6.1%}   E[ret 2d] = {fc.expected_2d_return:+.3%}")
    print()
    print(fc.note)
    print("=" * 64)


def run_all(assets: list[str] | None = None, db_path: str | Path | None = None) -> dict[str, HMMForecast]:
    """Fit (if needed) and predict for every primary asset, return per-asset forecasts."""
    out: dict[str, HMMForecast] = {}
    for asset in assets or DEFAULT_PRIMARIES:
        try:
            det = HMMRegimeDetector(db_path=db_path, primary_asset=asset)
            fc = det.run()
            out[asset] = fc
        except Exception as e:
            logger.error("HMM[{}]: run failed: {}", asset, e)
    return out


def _parse_asset_arg(argv: list[str]) -> list[str]:
    """Look for `--asset ES,NQ` or `--asset ES`; default to DEFAULT_PRIMARIES."""
    for i, tok in enumerate(argv):
        if tok == "--asset" and i + 1 < len(argv):
            return [a.strip().upper() for a in argv[i + 1].split(",") if a.strip()]
    return list(DEFAULT_PRIMARIES)


def _parse_int_arg(argv: list[str], flag: str, default: int) -> int:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                pass
    return default


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "run"
    assets = _parse_asset_arg(argv)

    if cmd == "fit":
        results = {}
        for a in assets:
            det = HMMRegimeDetector(primary_asset=a)
            results[a] = det.fit()
        print(json.dumps(results, indent=2, default=str))
        return 0
    if cmd == "predict":
        forecasts = {}
        for a in assets:
            det = HMMRegimeDetector(primary_asset=a)
            if not det.load():
                print(f"No saved model for {a} — run `python -m macro.hmm_regime fit --asset {a}` first.",
                      file=sys.stderr)
                return 2
            forecasts[a] = det.predict()
        for fc in forecasts.values():
            _print_forecast(fc)
        _print_summary(forecasts)
        return 0
    if cmd == "run":
        forecasts = run_all(assets)
        for fc in forecasts.values():
            _print_forecast(fc)
        _print_summary(forecasts)
        return 0
    if cmd == "visualize":
        n_days = _parse_int_arg(argv, "--days", 30)
        out_dir = Path("dashboard")
        for i, tok in enumerate(argv):
            if tok == "--out" and i + 1 < len(argv):
                out_dir = Path(argv[i + 1])
        paths: list[Path] = []
        for a in assets:
            det = HMMRegimeDetector(primary_asset=a)
            if not det.load():
                print(f"No saved model for {a} — fitting first...", file=sys.stderr)
                det.fit()
            paths.append(det.plot_forecast(n_days=n_days, output_dir=out_dir))
            fpt = det.expected_first_passage_to_dump()
            if fpt is None:
                fpt_str = "n/a"
            elif fpt == 0.0:
                fpt_str = "currently in dump"
            else:
                fpt_str = f"~{fpt:.1f} trading days"
            print(f"  {a}: E[first dump transition] = {fpt_str}")
        print("Saved visualizations:")
        for p in paths:
            print(f"  {p}")
        return 0

    print(f"Unknown command: {cmd}. Use fit | predict | run | visualize [--asset ES,NQ] [--days N]",
          file=sys.stderr)
    return 2


def _print_summary(forecasts: dict[str, HMMForecast]) -> None:
    """Cross-asset dump summary so you can compare ES vs NQ at a glance."""
    if len(forecasts) <= 1:
        return
    print()
    print("Cross-asset dump probability summary")
    print("-" * 64)
    print(f"{'asset':<6} {'regime':<7} {'P(now)':>8} {'P(1d)':>8} {'P(2d)':>8} {'E[1d]':>10} {'E[2d]':>10}")
    for asset, fc in forecasts.items():
        print(
            f"{asset:<6} {fc.current_regime:<7} "
            f"{fc.p_dump_now:>7.1%} {fc.p_dump_1d:>7.1%} {fc.p_dump_2d:>7.1%} "
            f"{fc.expected_1d_return:>+9.3%} {fc.expected_2d_return:>+9.3%}"
        )
    print("-" * 64)
    print(
        "NOTE: HMM dump probabilities are conditional similarity scores, not\n"
        "directional crash calls. They tell you how much today's tape resembles\n"
        "historical stress regimes and how the transition matrix rolls forward."
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
