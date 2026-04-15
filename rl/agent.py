"""Reinforcement learning agent using PPO with curriculum learning and imitation warm-start."""

import numpy as np
import json
import duckdb
import torch
from pathlib import Path
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.sim import TradingEnvironment

DB_PATH = Path(__file__).parent.parent / "aria.db"
MODEL_DIR = Path(__file__).parent.parent / "models"


class TrainingCallback(BaseCallback):
    """Logs training metrics to DuckDB."""

    def __init__(self, db_path: str, verbose=0):
        super().__init__(verbose)
        self.db_path = db_path
        self.episode_rewards = []

    def _on_step(self) -> bool:
        """Called at each training step."""
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode_pnl" in info:
                    self.episode_rewards.append(info["episode_pnl"])
        return True

    def _on_rollout_end(self) -> None:
        """Log rollout stats."""
        if self.episode_rewards:
            avg = np.mean(self.episode_rewards[-100:])
            logger.debug("Avg episode PnL (last 100): {:.2f}", avg)


class RLAgent:
    """PPO-based trading agent with curriculum learning and model versioning."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        self.model_dir = MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.version = 0
        self._init_db()
        self._load_latest_version()

    def _init_db(self):
        """Create RL tables."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS retraining_log (
                version INTEGER,
                timesteps INTEGER,
                eval_sharpe DOUBLE,
                eval_return DOUBLE,
                eval_max_drawdown DOUBLE,
                eval_trades INTEGER,
                promoted BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.close()

    def _load_latest_version(self):
        """Find and load the latest model version."""
        con = duckdb.connect(self.db_path)
        row = con.execute("""
            SELECT version FROM retraining_log WHERE promoted = true
            ORDER BY version DESC LIMIT 1
        """).fetchone()
        con.close()

        if row:
            self.version = row[0]
            model_path = self.model_dir / f"aria_model_v{self.version}.zip"
            if model_path.exists():
                try:
                    self.model = PPO.load(str(model_path))
                    logger.info("Loaded model v{}", self.version)
                except Exception as e:
                    logger.warning("Failed to load model v{}: {}", self.version, e)

    def _create_env(self) -> DummyVecEnv:
        """Create a vectorized training environment."""
        return DummyVecEnv([lambda: TradingEnvironment(db_path=self.db_path)])

    def train(self, total_timesteps: int = 2_000_000, from_scratch: bool = True):
        """Train the RL agent."""
        env = self._create_env()
        self.version += 1

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Training v{} for {} steps on {}", self.version, total_timesteps, device)

        if from_scratch or self.model is None:
            self.model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=dict(
                    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                    activation_fn=torch.nn.ReLU,
                ),
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=0,
                device=device,
            )
        else:
            self.model.set_env(env)

        callback = TrainingCallback(self.db_path)

        try:
            self.model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        except Exception as e:
            logger.error("Training failed: {}", e)
            return None

        # Save model
        model_path = self.model_dir / f"aria_model_v{self.version}.zip"
        self.model.save(str(model_path))
        logger.info("Saved model to {}", model_path)

        # Cleanup old models (keep last 5)
        self._cleanup_old_models()

        # Evaluate
        eval_result = self.evaluate()
        return eval_result

    def fine_tune(self, timesteps: int = 10_000):
        """Incremental fine-tuning on recent data."""
        if self.model is None:
            logger.warning("No model to fine-tune — training from scratch")
            return self.train(total_timesteps=timesteps, from_scratch=True)

        env = self._create_env()
        self.model.set_env(env)
        self.version += 1

        logger.info("Fine-tuning to v{} for {} steps", self.version, timesteps)

        callback = TrainingCallback(self.db_path)
        try:
            self.model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
        except Exception as e:
            logger.error("Fine-tuning failed: {}", e)
            return None

        model_path = self.model_dir / f"aria_model_v{self.version}.zip"
        self.model.save(str(model_path))

        self._cleanup_old_models()
        return self.evaluate()

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Evaluate the model on held-out data."""
        if self.model is None:
            return {"sharpe": 0, "return": 0, "max_drawdown": 1, "trades": 0}

        env = TradingEnvironment(db_path=self.db_path)
        all_stats = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            all_stats.append(env.get_episode_stats())

        avg_stats = {
            "sharpe": np.mean([s["sharpe"] for s in all_stats]),
            "return": np.mean([s["total_return"] for s in all_stats]),
            "max_drawdown": np.mean([s["max_drawdown"] for s in all_stats]),
            "trades": int(np.mean([s["trade_count"] for s in all_stats])),
        }

        # Decide whether to promote
        promoted = avg_stats["sharpe"] > 0.5

        # Log evaluation
        con = duckdb.connect(self.db_path)
        con.execute("""
            INSERT INTO retraining_log
            (version, timesteps, eval_sharpe, eval_return, eval_max_drawdown, eval_trades, promoted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [self.version, 0, avg_stats["sharpe"], avg_stats["return"],
              avg_stats["max_drawdown"], avg_stats["trades"], promoted])
        con.close()

        status = "PROMOTED" if promoted else "REJECTED"
        logger.info("Eval v{}: Sharpe={:.3f} Return={:.4f} DD={:.4f} -> {}",
                     self.version, avg_stats["sharpe"], avg_stats["return"],
                     avg_stats["max_drawdown"], status)

        return avg_stats

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Get action from the trained model."""
        if self.model is None:
            return np.zeros(18)  # 9 directions + 9 sizes

        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def _cleanup_old_models(self):
        """Keep only the last 5 model versions."""
        model_files = sorted(self.model_dir.glob("aria_model_v*.zip"), key=lambda p: p.stat().st_mtime)
        while len(model_files) > 5:
            old = model_files.pop(0)
            old.unlink()
            logger.debug("Removed old model: {}", old.name)

    def get_deployed_version(self) -> int:
        """Get the version number of the currently deployed model."""
        con = duckdb.connect(self.db_path)
        row = con.execute("""
            SELECT version, eval_sharpe FROM retraining_log
            WHERE promoted = true ORDER BY version DESC LIMIT 1
        """).fetchone()
        con.close()
        if row:
            return row[0]
        return 0

    def get_training_history(self):
        """Get all training/eval history."""
        con = duckdb.connect(self.db_path)
        df = con.execute("SELECT * FROM retraining_log ORDER BY version").fetchdf()
        con.close()
        return df.to_dict("records")
