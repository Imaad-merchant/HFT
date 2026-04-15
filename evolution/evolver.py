"""Evolutionary strategy search — genome-based rule discovery with walk-forward validation."""

import numpy as np
import pandas as pd
import duckdb
import json
import copy
from loguru import logger
from pathlib import Path
from tqdm import tqdm

DB_PATH = Path(__file__).parent.parent / "aria.db"

POPULATION_SIZE = 100
GENOME_MIN_SIZE = 20
GENOME_MAX_SIZE = 50
NUM_GENERATIONS = 200
SURVIVORS = 20
MUTATION_RATE = 0.1
OOS_SHARPE_RATIO = 0.5  # OOS must be at least 50% of IS Sharpe
MAX_DRAWDOWN_LIMIT = 0.15
EVAL_DAYS = 30

REGIME_FILTERS = ["RISK_ON", "RISK_OFF", "STAGFLATION", "NEUTRAL", "ALL"]


class Genome:
    """A strategy genome: feature-weight-threshold-regime triplets."""

    def __init__(self, genes: list[dict] = None):
        self.genes = genes or []
        self.fitness = -np.inf
        self.is_sharpe = 0.0
        self.oos_sharpe = 0.0
        self.max_drawdown = 0.0

    def to_dict(self) -> dict:
        """Serialize genome."""
        return {
            "genes": self.genes,
            "fitness": self.fitness,
            "is_sharpe": self.is_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "max_drawdown": self.max_drawdown,
        }

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        """Deserialize genome."""
        g = Genome(d["genes"])
        g.fitness = d.get("fitness", -np.inf)
        g.is_sharpe = d.get("is_sharpe", 0.0)
        g.oos_sharpe = d.get("oos_sharpe", 0.0)
        g.max_drawdown = d.get("max_drawdown", 0.0)
        return g


class EvolutionarySearcher:
    """Discovers rule-based strategies via evolutionary search with walk-forward validation."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        self.feature_names = []
        self.population = []
        self._init_db()

    def _init_db(self):
        """Create evolution tables."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS evolved_strategies (
                strategy_id INTEGER,
                genome_json VARCHAR,
                fitness DOUBLE,
                is_sharpe DOUBLE,
                oos_sharpe DOUBLE,
                max_drawdown DOUBLE,
                generation INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR DEFAULT 'CANDIDATE'
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS evolution_log (
                generation INTEGER,
                best_fitness DOUBLE,
                avg_fitness DOUBLE,
                worst_fitness DOUBLE,
                population_size INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.close()

    def _load_feature_names(self):
        """Load available feature names from the database."""
        con = duckdb.connect(self.db_path)
        row = con.execute(
            "SELECT feature_json FROM features_wide ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        con.close()

        if row:
            features = json.loads(row[0])
            self.feature_names = list(features.keys())
        else:
            # Fallback feature names
            self.feature_names = [f"feat_{i}" for i in range(50)]

    def _random_gene(self) -> dict:
        """Create a random gene (feature-weight-threshold-regime triplet)."""
        return {
            "feature": np.random.choice(self.feature_names) if self.feature_names else "ret_1",
            "weight": np.random.uniform(-2.0, 2.0),
            "threshold": np.random.uniform(-2.0, 2.0),
            "regime_filter": np.random.choice(REGIME_FILTERS),
        }

    def _random_genome(self) -> Genome:
        """Create a random genome."""
        size = np.random.randint(GENOME_MIN_SIZE, GENOME_MAX_SIZE + 1)
        genes = [self._random_gene() for _ in range(size)]
        return Genome(genes)

    def _evaluate_genome(self, genome: Genome, price_data: dict, feature_data: dict,
                         timestamps: list, split_ratio: float = 0.7) -> float:
        """Evaluate a genome with walk-forward validation."""
        if not timestamps or not price_data:
            return -np.inf

        split_idx = int(len(timestamps) * split_ratio)
        is_timestamps = timestamps[:split_idx]
        oos_timestamps = timestamps[split_idx:]

        is_sharpe = self._simulate_genome(genome, price_data, feature_data, is_timestamps)
        oos_sharpe = self._simulate_genome(genome, price_data, feature_data, oos_timestamps)

        # Curve-fit filter
        if oos_sharpe < OOS_SHARPE_RATIO * is_sharpe:
            return -np.inf

        # Drawdown filter
        if genome.max_drawdown > MAX_DRAWDOWN_LIMIT:
            return -np.inf

        genome.is_sharpe = is_sharpe
        genome.oos_sharpe = oos_sharpe
        genome.fitness = oos_sharpe

        return oos_sharpe

    def _simulate_genome(self, genome: Genome, price_data: dict,
                         feature_data: dict, timestamps: list) -> float:
        """Run a genome through historical data and compute Sharpe."""
        if not timestamps:
            return 0.0

        pnls = []
        capital = 1_000_000
        peak = capital
        max_dd = 0.0

        assets = list(price_data.keys())

        for t_idx in range(1, len(timestamps)):
            ts = timestamps[t_idx]
            prev_ts = timestamps[t_idx - 1]

            # Get features for this timestamp
            features = {}
            for asset in assets:
                if asset in feature_data:
                    mask = feature_data[asset]["timestamp"] == ts
                    rows = feature_data[asset][mask]
                    if not rows.empty:
                        try:
                            features[asset] = json.loads(rows["feature_json"].iloc[0])
                        except Exception:
                            features[asset] = {}

            # Compute aggregate signal from genome
            total_signal = 0.0
            active_genes = 0

            for gene in genome.genes:
                feat_name = gene["feature"]
                weight = gene["weight"]
                threshold = gene["threshold"]

                # Check across all assets
                for asset in assets:
                    if asset not in features:
                        continue
                    feat_val = features[asset].get(feat_name, 0.0)
                    if feat_val > threshold:
                        total_signal += weight
                        active_genes += 1
                    elif feat_val < -threshold:
                        total_signal -= weight
                        active_genes += 1

            if active_genes == 0:
                continue

            signal = np.clip(total_signal / active_genes, -1, 1)

            # Simplified PnL: signal * average return across assets
            returns = []
            for asset in assets:
                if asset in price_data:
                    curr_mask = price_data[asset]["timestamp"] == ts
                    prev_mask = price_data[asset]["timestamp"] == prev_ts
                    curr_rows = price_data[asset][curr_mask]
                    prev_rows = price_data[asset][prev_mask]
                    if not curr_rows.empty and not prev_rows.empty:
                        ret = (curr_rows["close"].iloc[0] / prev_rows["close"].iloc[0]) - 1
                        returns.append(ret)

            if returns:
                avg_return = np.mean(returns)
                step_pnl = signal * avg_return * capital * 0.1  # 10% capital per trade
                step_pnl -= abs(signal) * capital * 0.00005  # transaction cost
                pnls.append(step_pnl)
                capital += step_pnl

                peak = max(peak, capital)
                dd = (peak - capital) / peak
                max_dd = max(max_dd, dd)

        genome.max_drawdown = max_dd

        if len(pnls) < 10:
            return 0.0

        pnl_arr = np.array(pnls)
        sharpe = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-10) * np.sqrt(252)
        return sharpe

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Combine genes from two parents."""
        genes1 = copy.deepcopy(parent1.genes)
        genes2 = copy.deepcopy(parent2.genes)

        # Randomly select genes from each parent
        all_genes = genes1 + genes2
        np.random.shuffle(all_genes)
        child_size = np.random.randint(GENOME_MIN_SIZE, GENOME_MAX_SIZE + 1)
        child_genes = all_genes[:child_size]

        return Genome(child_genes)

    def _mutate(self, genome: Genome) -> Genome:
        """Mutate a genome's weights, thresholds, or features."""
        genes = copy.deepcopy(genome.genes)

        for i in range(len(genes)):
            if np.random.random() < MUTATION_RATE:
                mutation_type = np.random.choice(["weight", "threshold", "feature", "regime"])
                if mutation_type == "weight":
                    genes[i]["weight"] += np.random.normal(0, 0.5)
                    genes[i]["weight"] = np.clip(genes[i]["weight"], -3.0, 3.0)
                elif mutation_type == "threshold":
                    genes[i]["threshold"] += np.random.normal(0, 0.3)
                    genes[i]["threshold"] = np.clip(genes[i]["threshold"], -3.0, 3.0)
                elif mutation_type == "feature" and self.feature_names:
                    genes[i]["feature"] = np.random.choice(self.feature_names)
                elif mutation_type == "regime":
                    genes[i]["regime_filter"] = np.random.choice(REGIME_FILTERS)

        return Genome(genes)

    def run(self, generations: int = NUM_GENERATIONS):
        """Run the full evolutionary search."""
        self._load_feature_names()

        # Load evaluation data
        con = duckdb.connect(self.db_path)
        assets = [r[0] for r in con.execute("SELECT DISTINCT asset FROM ohlcv").fetchall()]

        price_data = {}
        feature_data = {}
        for asset in assets:
            pdf = con.execute(
                "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp", [asset]
            ).fetchdf()
            if not pdf.empty:
                price_data[asset] = pdf

            fdf = con.execute(
                "SELECT timestamp, feature_json FROM features_wide WHERE asset = ? ORDER BY timestamp",
                [asset]
            ).fetchdf()
            if not fdf.empty:
                feature_data[asset] = fdf
        con.close()

        # Find common timestamps
        if price_data:
            ts_sets = [set(df["timestamp"]) for df in price_data.values()]
            timestamps = sorted(set.intersection(*ts_sets))
            # Use last EVAL_DAYS worth of data
            if len(timestamps) > EVAL_DAYS * 78:
                timestamps = timestamps[-(EVAL_DAYS * 78):]
        else:
            logger.warning("No price data for evolution")
            return

        # Initialize population
        logger.info("Initializing population of {} genomes", POPULATION_SIZE)
        self.population = [self._random_genome() for _ in range(POPULATION_SIZE)]

        fitness_history = []

        for gen in tqdm(range(generations), desc="Evolution"):
            # Evaluate all genomes
            for genome in self.population:
                self._evaluate_genome(genome, price_data, feature_data, timestamps)

            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness, reverse=True)

            # Log generation stats
            fitnesses = [g.fitness for g in self.population if g.fitness > -np.inf]
            best = max(fitnesses) if fitnesses else -np.inf
            avg = np.mean(fitnesses) if fitnesses else -np.inf
            worst = min(fitnesses) if fitnesses else -np.inf

            fitness_history.append({
                "generation": gen,
                "best_fitness": best,
                "avg_fitness": avg,
                "worst_fitness": worst,
                "population_size": len(self.population),
            })

            if gen % 10 == 0:
                logger.info("Gen {}: best={:.4f} avg={:.4f} pop={}", gen, best, avg, len(self.population))

            # Selection: keep top survivors
            survivors = self.population[:SURVIVORS]

            # Create next generation
            next_gen = list(survivors)  # Elitism

            while len(next_gen) < POPULATION_SIZE:
                p1, p2 = np.random.choice(survivors, size=2, replace=False)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_gen.append(child)

            self.population = next_gen

        # Save top genomes to DB
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        top_genomes = self.population[:10]

        con = duckdb.connect(self.db_path)
        con.execute("BEGIN TRANSACTION")
        # Clear old candidates
        con.execute("DELETE FROM evolved_strategies WHERE status = 'CANDIDATE'")

        for i, genome in enumerate(top_genomes):
            if genome.fitness > -np.inf:
                con.execute("""
                    INSERT INTO evolved_strategies
                    (strategy_id, genome_json, fitness, is_sharpe, oos_sharpe, max_drawdown, generation)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [i, json.dumps(genome.to_dict()), genome.fitness,
                      genome.is_sharpe, genome.oos_sharpe, genome.max_drawdown, generations])
                logger.info("Saved genome {}: fitness={:.4f} IS={:.4f} OOS={:.4f}",
                            i, genome.fitness, genome.is_sharpe, genome.oos_sharpe)

        # Save fitness history
        history_df = pd.DataFrame(fitness_history)
        history_df.to_csv(Path(self.db_path).parent / "evolution_fitness.csv", index=False)

        # Log to DB
        log_df = pd.DataFrame(fitness_history)
        con.execute("INSERT INTO evolution_log SELECT * FROM log_df")
        con.execute("COMMIT")
        con.close()

        logger.info("Evolution complete. {} viable genomes saved.", sum(1 for g in top_genomes if g.fitness > -np.inf))

    def get_best_genome(self):
        """Retrieve the best evolved genome from DB."""
        con = duckdb.connect(self.db_path)
        row = con.execute("""
            SELECT genome_json FROM evolved_strategies
            WHERE status IN ('CANDIDATE', 'ACTIVE')
            ORDER BY fitness DESC LIMIT 1
        """).fetchone()
        con.close()

        if row:
            return Genome.from_dict(json.loads(row[0]))
        return None

    def get_genome_signal(self, genome: Genome, features: dict) -> float:
        """Get trading signal from a genome given current features."""
        total_signal = 0.0
        active = 0

        for gene in genome.genes:
            feat_val = features.get(gene["feature"], 0.0)
            if feat_val > gene["threshold"]:
                total_signal += gene["weight"]
                active += 1
            elif feat_val < -gene["threshold"]:
                total_signal -= gene["weight"]
                active += 1

        if active == 0:
            return 0.0

        return np.clip(total_signal / active, -1.0, 1.0)
