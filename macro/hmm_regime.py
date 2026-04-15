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
MODEL_PATH = MODEL_DIR / "hmm_regime.pkl"

# Assets used to build the joint regime feature vector. ES is the primary; the
# others provide cross-asset confirmation that a stressed state is broad-based.
PRIMARY_ASSET = "ES"
SUPPORT_ASSETS = ["NQ", "VIX", "TLT", "GC", "DXY"]

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
    current_regime: str
    current_state_id: int
    state_posterior: list[float]            # P(state=i | obs up to now)
    regime_posterior: dict[str, float]      # P(regime=label | obs up to now)
    p_dump_1d: float                        # P(dump regime ~1 trading day ahead)
    p_dump_2d: float                        # P(dump regime ~2 trading days ahead)
    p_dump_now: float                       # P(currently in dump regime)
    expected_1d_return: float               # state-mixture expected ES log-return
    expected_2d_return: float
    bars_per_day: int
    n_train_bars: int
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


class HMMRegimeDetector:
    """Gaussian HMM over multi-asset returns + vol + VIX features."""

    def __init__(self, db_path: str | Path | None = None, n_states: int = N_STATES):
        self.db_path = str(db_path or DB_PATH)
        self.n_states = n_states
        self.model = None  # hmmlearn.hmm.GaussianHMM
        self.feature_cols: list[str] = []
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.state_label_map: dict[int, str] = {}
        self.state_mean_return: dict[int, float] = {}
        self.bars_per_day: int = 7
        self._init_db()

    # ------------------------------------------------------------------ DB

    def _init_db(self) -> None:
        con = duckdb.connect(self.db_path)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS hmm_forecast (
                timestamp TIMESTAMP PRIMARY KEY,
                current_regime VARCHAR,
                current_state_id INTEGER,
                p_dump_now DOUBLE,
                p_dump_1d DOUBLE,
                p_dump_2d DOUBLE,
                expected_1d_return DOUBLE,
                expected_2d_return DOUBLE,
                payload_json VARCHAR
            )
            """
        )
        con.close()
        MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # -------------------------------------------------------------- features

    def _load_aligned_closes(self) -> pd.DataFrame:
        """Pull closes for primary + support assets aligned on a common index."""
        con = duckdb.connect(self.db_path)
        frames = {}
        for asset in [PRIMARY_ASSET] + SUPPORT_ASSETS:
            df = con.execute(
                "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp",
                [asset],
            ).fetchdf()
            if df.empty:
                logger.warning("HMM: no rows for {} in ohlcv — skipping", asset)
                continue
            df = df.set_index("timestamp")
            frames[asset] = df["close"].astype(float)
        con.close()

        if PRIMARY_ASSET not in frames:
            raise RuntimeError(
                f"HMM regime fit needs OHLCV for {PRIMARY_ASSET}; none found in {self.db_path}"
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

    def _build_features(self, closes: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
        """Construct the HMM observation matrix from aligned close prices."""
        df = pd.DataFrame(index=closes.index)

        es = closes[PRIMARY_ASSET]
        df["es_logret"] = np.log(es).diff()

        # Realised vol (rolling std of 1-bar log returns)
        df["es_vol_20"] = df["es_logret"].rolling(20).std()
        df["es_vol_60"] = df["es_logret"].rolling(60).std()

        # Vol-of-vol — accelerates before crashes
        df["es_volvol"] = df["es_vol_20"].rolling(20).std()

        # Drawdown from rolling high (negative when stressed)
        roll_max = es.rolling(60, min_periods=10).max()
        df["es_dd_60"] = (es / roll_max) - 1.0

        if "VIX" in closes.columns:
            vix = closes["VIX"]
            df["vix_level"] = vix
            df["vix_chg"] = vix.diff()
            df["vix_z"] = (vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-9)

        if "TLT" in closes.columns:
            tlt = closes["TLT"]
            df["tlt_logret"] = np.log(tlt).diff()
            # Stocks down + bonds up = classic risk-off
            df["es_tlt_diff"] = df["es_logret"] - df["tlt_logret"]

        if "NQ" in closes.columns:
            df["nq_logret"] = np.log(closes["NQ"]).diff()

        if "DXY" in closes.columns:
            df["dxy_logret"] = np.log(closes["DXY"]).diff()

        if "GC" in closes.columns:
            df["gc_logret"] = np.log(closes["GC"]).diff()

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
        X, cols, idx = self._build_features(closes)
        if len(X) < 200:
            raise RuntimeError(
                f"HMM fit needs at least 200 aligned bars; got {len(X)}. "
                f"Run the data fetcher first (orch.data_fetcher.fetch_all('2y'))."
            )

        self.bars_per_day = self._detect_bars_per_day(idx)
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

        # --- Label states by mean ES log-return ---
        # Decode the most likely state sequence and compute the empirical mean
        # ES log-return per state. Sort ascending -> CRASH .. BULL.
        state_seq = model.predict(Xz)
        es_logret_idx = self.feature_cols.index("es_logret")
        # The features in X are *raw* (not z-scored), so use the raw matrix.
        per_state_mean = {}
        for s in range(self.n_states):
            mask = state_seq == s
            per_state_mean[s] = float(X[mask, es_logret_idx].mean()) if mask.any() else 0.0

        ordered = sorted(per_state_mean.items(), key=lambda kv: kv[1])  # worst -> best
        labels = REGIME_LABELS[: self.n_states] if self.n_states <= len(REGIME_LABELS) else [
            f"S{i}" for i in range(self.n_states)
        ]
        self.state_label_map = {state_id: labels[rank] for rank, (state_id, _) in enumerate(ordered)}
        self.state_mean_return = per_state_mean

        logger.info("HMM: state labels by mean ES log-return: {}", self.state_label_map)
        logger.info("HMM: per-state mean ES log-return: {}", {k: round(v, 6) for k, v in per_state_mean.items()})
        logger.info("HMM: transition matrix:\n{}", np.round(model.transmat_, 3))

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
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(payload, f)
        logger.info("HMM: saved model to {}", MODEL_PATH)

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            return False
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]
        self.feature_mean = payload["feature_mean"]
        self.feature_std = payload["feature_std"]
        self.state_label_map = payload["state_label_map"]
        self.state_mean_return = payload["state_mean_return"]
        self.bars_per_day = payload.get("bars_per_day", 7)
        self.n_states = payload.get("n_states", N_STATES)
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
            con.execute("DELETE FROM hmm_forecast WHERE timestamp = ?", [pd.Timestamp(fc.timestamp)])
            con.execute(
                """
                INSERT INTO hmm_forecast VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    pd.Timestamp(fc.timestamp),
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


# --------------------------------------------------------------------- CLI

def _print_forecast(fc: HMMForecast) -> None:
    print("=" * 64)
    print("ARIA HMM Regime Forecast")
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
    print(
        "NOTE: this is a probabilistic regime forecast from a Gaussian HMM, not a\n"
        "directional crash call. It says how much today's tape resembles historical\n"
        "stress regimes and propagates the empirical transition matrix forward."
    )


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "run"
    det = HMMRegimeDetector()

    if cmd == "fit":
        info = det.fit()
        print(json.dumps(info, indent=2, default=str))
        return 0
    if cmd == "predict":
        if not det.load():
            print("No saved model — run `python -m macro.hmm_regime fit` first.", file=sys.stderr)
            return 2
        fc = det.predict()
        _print_forecast(fc)
        return 0
    if cmd == "run":
        fc = det.run()
        _print_forecast(fc)
        return 0

    print(f"Unknown command: {cmd}. Use fit | predict | run", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
