from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence, Tuple
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from atlas_sb import (
    RadiusConfig,
    atlas_sb_from_statistics,
    covariance_of_rows,
    opnorm,
    prepare_rolling_statistics,
    realized_gaussian_nll,
    spectral_clip,
    sym,
    theoretical_tau,
)


UCI_URL = (
    "https://archive.ics.uci.edu/static/public/270/"
    "gas+sensor+array+drift+dataset+at+different+concentrations.zip"
)


def download_archive(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(UCI_URL, path)
    return path


def parse_batch(text: str, batch: int) -> pd.DataFrame:
    rows: List[dict] = []
    for row_number, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if ";" not in tokens[0]:
            raise ValueError(f"malformed row in batch {batch}: {line[:80]}")
        gas_text, concentration_text = tokens[0].split(";", 1)
        features = np.zeros(128, dtype=float)
        for token in tokens[1:]:
            index_text, value_text = token.split(":", 1)
            index = int(index_text) - 1
            if not 0 <= index < 128:
                raise ValueError(f"feature index out of range: {index + 1}")
            features[index] = float(value_text)
        row = {
            "batch": batch,
            "row_in_batch": row_number,
            "gas": int(float(gas_text)),
            "concentration": float(concentration_text),
        }
        row.update({f"x{j + 1}": float(value) for j, value in enumerate(features)})
        rows.append(row)
    return pd.DataFrame(rows)


def load_uci(path: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        candidates: List[Tuple[int, str]] = []
        for name in archive.namelist():
            match = re.search(r"batch(\d+)\.dat$", name, flags=re.IGNORECASE)
            if match:
                candidates.append((int(match.group(1)), name))
        if len(candidates) != 10:
            raise RuntimeError(f"expected 10 batch files, found {len(candidates)}")
        for batch, name in sorted(candidates):
            raw = archive.read(name)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            frames.append(parse_batch(text, batch))
    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["batch", "row_in_batch"]).reset_index(drop=True)


def condition_design(gas: np.ndarray, concentration: np.ndarray) -> np.ndarray:
    gas = gas.astype(int)
    if np.min(gas) < 1 or np.max(gas) > 6:
        raise ValueError("gas labels must lie in 1,...,6")
    log_c = np.log1p(concentration.astype(float))
    one_hot = np.zeros((len(gas), 6), dtype=float)
    one_hot[np.arange(len(gas)), gas - 1] = 1.0
    return np.column_stack(
        [
            np.ones(len(gas)),
            one_hot,
            one_hot * log_c[:, None],
            one_hot * (log_c[:, None] ** 2),
        ]
    )


def fit_ridge(design: np.ndarray, response: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ response)


def raw_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"x{j}" for j in range(1, 129)]].to_numpy(float)


class FrozenPilot:
    def __init__(
        self,
        coefficients: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        pilot_rank: int,
        monitor_vectors: np.ndarray,
        monitor_values: np.ndarray,
    ) -> None:
        self.coefficients = coefficients
        self.mean = mean
        self.scale = scale
        self.pilot_rank = pilot_rank
        self.monitor_vectors = monitor_vectors
        self.monitor_values = monitor_values


def fit_frozen_pilot(
    calibration: pd.DataFrame,
    explained_variance: float = 0.80,
    pilot_rank_cap: int = 16,
    monitor_dimension: int = 24,
) -> FrozenPilot:
    raw = raw_feature_matrix(calibration)
    design = condition_design(
        calibration["gas"].to_numpy(), calibration["concentration"].to_numpy()
    )
    coefficients = fit_ridge(design, raw)
    residual = raw - design @ coefficients
    mean = residual.mean(axis=0)
    scale = np.maximum(residual.std(axis=0, ddof=1), 1e-8)
    standardized = (residual - mean) / scale
    covariance = covariance_of_rows(standardized)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    cumulative = np.cumsum(np.maximum(values, 0.0))
    cumulative /= cumulative[-1]
    requested = int(np.searchsorted(cumulative, explained_variance)) + 1
    pilot_rank = min(pilot_rank_cap, requested, len(values) - monitor_dimension)
    monitor_vectors = vectors[:, pilot_rank : pilot_rank + monitor_dimension]
    monitor_values = np.maximum(
        values[pilot_rank : pilot_rank + monitor_dimension], 1e-6
    )
    return FrozenPilot(
        coefficients, mean, scale, pilot_rank, monitor_vectors, monitor_values
    )


def residual_coordinates(frame: pd.DataFrame, pilot: FrozenPilot) -> np.ndarray:
    raw = raw_feature_matrix(frame)
    design = condition_design(
        frame["gas"].to_numpy(), frame["concentration"].to_numpy()
    )
    residual = raw - design @ pilot.coefficients
    standardized = (residual - pilot.mean) / pilot.scale
    return standardized @ pilot.monitor_vectors / np.sqrt(pilot.monitor_values)


def calibration_envelopes(
    residual_by_batch: Mapping[int, np.ndarray], safety: float = 2.0
) -> Tuple[float, float]:
    maxima = []
    standardized_norms = []
    for batch in (1, 2):
        rows = residual_by_batch[batch]
        covariance = covariance_of_rows(rows)
        maxima.append(float(np.max(np.linalg.eigvalsh(covariance))))
        standardized_norms.extend(np.linalg.norm(rows, axis=1).tolist())
    M = max(1.25, safety * max(maxima))
    # K is a declared calibration envelope, not a distribution-free estimate.
    median_norm = float(np.median(standardized_norms))
    K = max(1.0, min(3.0, safety * median_norm / math.sqrt(residual_by_batch[1].shape[1])))
    return K, M


def batch_memory_windows(
    batches: Mapping[int, np.ndarray], last_batch: int, memories: Sequence[int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    history = np.vstack([batches[b] for b in range(1, last_batch + 1)])
    counts: Dict[int, int] = {}
    for memory in sorted(memories):
        if memory <= last_batch:
            counts[memory] = int(
                sum(len(batches[b]) for b in range(last_batch - memory + 1, last_batch + 1))
            )
    return history, counts


def mean_nll(forecast: np.ndarray, rows: np.ndarray) -> float:
    return float(np.mean([realized_gaussian_nll(forecast, row) for row in rows]))


def covariance_discrepancy(forecast: np.ndarray, rows: np.ndarray) -> float:
    target = covariance_of_rows(rows)
    return opnorm(forecast - target) / max(opnorm(target), 1e-12)


def run_analysis(
    data: pd.DataFrame,
    out: Path,
    bootstrap_replications: int,
    c_det: float,
    kappa: float,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    calibration = data[data["batch"].isin([1, 2])].copy()
    pilot = fit_frozen_pilot(calibration)
    residual_by_batch: Dict[int, np.ndarray] = {
        batch: residual_coordinates(data[data["batch"] == batch], pilot)
        for batch in range(1, 11)
    }
    m = residual_by_batch[1].shape[1]
    K, M = calibration_envelopes(residual_by_batch)
    memories = [1, 2, 4, 8]
    gammas = (0.5, 0.25, 0.125, 0.0625, 0.03125)
    config = RadiusConfig(
        K=K,
        M=M,
        alpha=0.10,
        horizon=len(data),
        n_windows=len(memories),
        gammas=gammas,
        c_tau=1.0,
        c_det=c_det,
        kappa=kappa,
        c_bias=4.0,
    )
    tau = theoretical_tau(m, config)
    rows: List[dict] = []
    forecast_cache: Dict[Tuple[int, str], np.ndarray] = {}
    target_cache: Dict[int, np.ndarray] = {}
    hedge_log_weights = {memory: 0.0 for memory in memories}
    hedge_eta = 0.05

    for last_batch in range(2, 10):
        target_batch = last_batch + 1
        history, memory_counts = batch_memory_windows(
            residual_by_batch, last_batch, memories
        )
        stats = prepare_rolling_statistics(history, tau)
        count_to_memory = {count: memory for memory, count in memory_counts.items()}
        atlas = atlas_sb_from_statistics(
            stats,
            len(history),
            sorted(count_to_memory),
            config,
            lower=1.0,
        )
        selected_memory = count_to_memory[atlas.selected_window]
        fixed_forecasts = {
            count_to_memory[count]: spectral_clip(scatter, 1.0, M)
            for count, scatter in atlas.scatters.items()
        }
        available = sorted(fixed_forecasts)
        weights = np.array([math.exp(hedge_log_weights[memory]) for memory in available])
        weights /= np.sum(weights)
        hedge = sym(
            sum(weight * fixed_forecasts[memory] for weight, memory in zip(weights, available))
        )
        expanding = spectral_clip(covariance_of_rows(history), 1.0, M)
        target = residual_by_batch[target_batch]
        target_covariance = covariance_of_rows(target)
        local_oracle_memory = min(
            available,
            key=lambda memory: opnorm(fixed_forecasts[memory] - target_covariance),
        )
        forecasts: Dict[str, np.ndarray] = {
            "Pilot": np.eye(m),
            "Expanding": expanding,
            "Hedge": hedge,
            "ATLAS-SB": atlas.forecast,
            "Local oracle": fixed_forecasts[local_oracle_memory],
        }
        forecasts.update(
            {f"Fixed-{memory}-batch": fixed_forecasts[memory] for memory in available}
        )
        target_cache[target_batch] = target
        negative_residual_share = float(
            np.mean(np.linalg.eigvalsh(target_covariance) < 1.0)
        )
        for method, forecast in forecasts.items():
            forecast_cache[(target_batch, method)] = forecast
            inv = np.linalg.inv(forecast)
            energy = np.einsum("ni,ij,nj->n", target, inv, target, optimize=True)
            rows.append(
                {
                    "origin_batch": last_batch,
                    "target_batch": target_batch,
                    "method": method,
                    "selected_memory": selected_memory if method == "ATLAS-SB" else np.nan,
                    "target_observations": len(target),
                    "mean_nll": mean_nll(forecast, target),
                    "relative_covariance_discrepancy": covariance_discrepancy(
                        forecast, target
                    ),
                    "mean_standardized_energy": float(energy.mean() / m),
                    "negative_residual_share": negative_residual_share,
                }
            )
        fixed_losses = {
            memory: mean_nll(fixed_forecasts[memory], target) for memory in available
        }
        minimum = min(fixed_losses.values())
        for memory, loss in fixed_losses.items():
            hedge_log_weights[memory] -= hedge_eta * min(30.0, loss - minimum)
        print(
            f"origin batch {last_batch}: selected {selected_memory} batch(es), "
            f"target {target_batch}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("method", as_index=False)
        .agg(
            mean_nll=("mean_nll", "mean"),
            mean_covariance_discrepancy=("relative_covariance_discrepancy", "mean"),
            mean_standardized_energy=("mean_standardized_energy", "mean"),
            forecast_origins=("target_batch", "nunique"),
        )
        .sort_values("mean_nll")
    )
    results.to_csv(out / "batch_ahead_results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    rng = np.random.default_rng(20260730)
    methods = sorted(results["method"].unique())
    target_batches = sorted(target_cache)
    bootstrap_rows: List[dict] = []
    for replication in range(bootstrap_replications):
        sampled_batches = rng.choice(target_batches, size=len(target_batches), replace=True)
        scores: Dict[str, List[float]] = {method: [] for method in methods}
        for target_batch in sampled_batches:
            target = target_cache[int(target_batch)]
            indices = rng.integers(0, len(target), size=len(target))
            sampled = target[indices]
            for method in methods:
                key = (int(target_batch), method)
                if key in forecast_cache:
                    scores[method].append(mean_nll(forecast_cache[key], sampled))
        for method, values in scores.items():
            if values:
                bootstrap_rows.append(
                    {
                        "replication": replication,
                        "method": method,
                        "mean_nll": float(np.mean(values)),
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(out / "hierarchical_bootstrap.csv", index=False)
    atlas = bootstrap[bootstrap["method"] == "ATLAS-SB"][
        ["replication", "mean_nll"]
    ].rename(columns={"mean_nll": "atlas_nll"})
    comparison_rows: List[dict] = []
    for method in methods:
        if method == "ATLAS-SB":
            continue
        other = bootstrap[bootstrap["method"] == method][
            ["replication", "mean_nll"]
        ].rename(columns={"mean_nll": "other_nll"})
        merged = atlas.merge(other, on="replication", how="inner")
        difference = merged["atlas_nll"] - merged["other_nll"]
        comparison_rows.append(
            {
                "competitor": method,
                "mean_difference": float(difference.mean()),
                "ci_2_5": float(difference.quantile(0.025)),
                "ci_97_5": float(difference.quantile(0.975)),
                "atlas_win_probability": float((difference < 0).mean()),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        out / "paired_bootstrap_comparisons.csv", index=False
    )
    composition = data.groupby(["batch", "gas"], as_index=False).size()
    composition.to_csv(out / "composition_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "observations": len(data),
                "features": 128,
                "calibration_batches": "1,2",
                "evaluation_batches": "3-10",
                "pilot_rank": pilot.pilot_rank,
                "monitor_dimension": m,
                "K_calibration_envelope": K,
                "M_calibration_envelope": M,
                "tau": tau,
                "candidate_memories": "1,2,4,8 batches",
                "bootstrap_replications": bootstrap_replications,
                "c_det": c_det,
                "kappa": kappa,
            }
        ]
    ).to_csv(out / "metadata.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", type=Path, default=Path("data/gas_sensor_array_drift.zip")
    )
    parser.add_argument("--out", type=Path, default=Path("generated/uci"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--c-det", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=0.25)
    args = parser.parse_args()
    archive = download_archive(args.archive)
    data = load_uci(archive)
    run_analysis(data, args.out, args.bootstrap, args.c_det, args.kappa)


if __name__ == "__main__":
    main()
