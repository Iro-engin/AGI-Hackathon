from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import run_uci as base
from atlas_sb import (
    RadiusConfig,
    atlas_sb_from_statistics,
    covariance_of_rows,
    opnorm,
    prepare_rolling_statistics,
    spectral_clip,
    sym,
    theoretical_tau,
)


def vectorized_mean_nll(forecast: np.ndarray, rows: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(forecast)
    if sign <= 0:
        raise ValueError("forecast must be positive definite")
    inverse = np.linalg.inv(forecast)
    quadratic = np.einsum("ni,ij,nj->n", rows, inverse, rows, optimize=True)
    return float(logdet + quadratic.mean())


def precision_aggregate(
    forecasts: Dict[int, np.ndarray], weights: np.ndarray
) -> np.ndarray:
    precision = sum(
        float(weight) * np.linalg.inv(forecasts[memory])
        for weight, memory in zip(weights, sorted(forecasts))
    )
    return sym(np.linalg.inv(precision))


def covariance_aggregate(
    forecasts: Dict[int, np.ndarray], weights: np.ndarray
) -> np.ndarray:
    return sym(
        sum(
            float(weight) * forecasts[memory]
            for weight, memory in zip(weights, sorted(forecasts))
        )
    )


def run_analysis(
    data: pd.DataFrame,
    out: Path,
    bootstrap_replications: int,
    c_det: float,
    kappa: float,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    calibration_batches = (1, 2, 3, 4)
    evaluation_batches = tuple(range(5, 11))
    calibration = data[data["batch"].isin(calibration_batches)].copy()
    pilot = base.fit_frozen_pilot(calibration)
    residual_by_batch: Dict[int, np.ndarray] = {
        batch: base.residual_coordinates(data[data["batch"] == batch], pilot)
        for batch in range(1, 11)
    }
    m = residual_by_batch[1].shape[1]
    K, M = base.calibration_envelopes(
        residual_by_batch, calibration_batches
    )
    memories = [1, 2, 4]
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
    log_weights = {memory: 0.0 for memory in memories}
    learning_rate = 0.05

    for target_batch in evaluation_batches:
        last_batch = target_batch - 1
        history, memory_counts = base.batch_memory_windows(
            residual_by_batch, last_batch, memories
        )
        stats = prepare_rolling_statistics(history, tau)
        count_to_memory = {count: memory for memory, count in memory_counts.items()}
        certified = atlas_sb_from_statistics(
            stats,
            len(history),
            sorted(count_to_memory),
            config,
            lower=1.0,
        )
        certified_memory = count_to_memory[certified.selected_window]
        fixed_forecasts = {
            count_to_memory[count]: spectral_clip(scatter, 1.0, M)
            for count, scatter in certified.scatters.items()
        }
        available = sorted(fixed_forecasts)
        weights = np.array([math.exp(log_weights[memory]) for memory in available])
        weights /= np.sum(weights)
        atlas_forecast = precision_aggregate(fixed_forecasts, weights)
        covariance_mix = covariance_aggregate(fixed_forecasts, weights)
        dominant_memory = available[int(np.argmax(weights))]
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
            "Covariance mix": covariance_mix,
            "ATLAS-SB": atlas_forecast,
            "Certified selector": certified.forecast,
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
            inverse = np.linalg.inv(forecast)
            energy = np.einsum("ni,ij,nj->n", target, inverse, target, optimize=True)
            rows.append(
                {
                    "origin_batch": last_batch,
                    "target_batch": target_batch,
                    "method": method,
                    "selected_memory": (
                        dominant_memory
                        if method == "ATLAS-SB"
                        else certified_memory
                        if method == "Certified selector"
                        else np.nan
                    ),
                    "target_observations": len(target),
                    "mean_nll": vectorized_mean_nll(forecast, target),
                    "relative_covariance_discrepancy": base.covariance_discrepancy(
                        forecast, target
                    ),
                    "mean_standardized_energy": float(energy.mean() / m),
                    "negative_residual_share": negative_residual_share,
                }
            )
        fixed_losses = {
            memory: vectorized_mean_nll(fixed_forecasts[memory], target)
            for memory in available
        }
        minimum = min(fixed_losses.values())
        for memory, loss in fixed_losses.items():
            log_weights[memory] -= learning_rate * min(30.0, loss - minimum)
        print(
            f"origin batch {last_batch}: predictive {dominant_memory}, "
            f"certified {certified_memory}, target {target_batch}",
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
    bootstrap_rows: List[dict] = []
    for replication in range(bootstrap_replications):
        sampled_batches = rng.choice(
            evaluation_batches, size=len(evaluation_batches), replace=True
        )
        scores: Dict[str, List[float]] = {method: [] for method in methods}
        for target_batch in sampled_batches:
            target = target_cache[int(target_batch)]
            indices = rng.integers(0, len(target), size=len(target))
            sampled = target[indices]
            for method in methods:
                scores[method].append(
                    vectorized_mean_nll(
                        forecast_cache[(int(target_batch), method)], sampled
                    )
                )
        for method, values in scores.items():
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
    comparisons: List[dict] = []
    for method in methods:
        if method == "ATLAS-SB":
            continue
        other = bootstrap[bootstrap["method"] == method][
            ["replication", "mean_nll"]
        ].rename(columns={"mean_nll": "other_nll"})
        merged = atlas.merge(other, on="replication", how="inner")
        difference = merged["atlas_nll"] - merged["other_nll"]
        comparisons.append(
            {
                "competitor": method,
                "mean_difference": float(difference.mean()),
                "ci_2_5": float(difference.quantile(0.025)),
                "ci_97_5": float(difference.quantile(0.975)),
                "atlas_win_probability": float((difference < 0).mean()),
            }
        )
    pd.DataFrame(comparisons).to_csv(
        out / "paired_bootstrap_comparisons.csv", index=False
    )
    data.groupby(["batch", "gas"], as_index=False).size().to_csv(
        out / "composition_audit.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "observations": len(data),
                "features": 128,
                "calibration_batches": "1-4",
                "evaluation_batches": "5-10",
                "pilot_rank": pilot.pilot_rank,
                "monitor_dimension": m,
                "K_calibration_envelope": K,
                "M_calibration_envelope": M,
                "tau": tau,
                "candidate_memories": "1,2,4 batches",
                "bootstrap_replications": bootstrap_replications,
                "learning_rate": learning_rate,
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
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--c-det", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=0.25)
    args = parser.parse_args()
    data = base.load_uci(base.download_archive(args.archive))
    run_analysis(data, args.out, args.bootstrap, args.c_det, args.kappa)


if __name__ == "__main__":
    main()
