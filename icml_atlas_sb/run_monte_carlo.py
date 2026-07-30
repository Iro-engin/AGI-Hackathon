from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from atlas_sb import (
    AtlasForecast,
    RadiusConfig,
    atlas_sb_from_statistics,
    covariance_of_rows,
    expected_gaussian_nll_excess,
    opnorm,
    prepare_rolling_statistics,
    realized_gaussian_nll,
    spectral_clip,
    sym,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    max_eigenvalue: float


def unit_rotation(m: int, angle: float, i: int, j: int) -> np.ndarray:
    vector = np.zeros(m)
    vector[i] = math.cos(angle)
    vector[j] = math.sin(angle)
    return vector


def covariance_path(name: str, T: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    covariances = np.repeat(np.eye(m)[None, :, :], T, axis=0)
    ranks = np.zeros(T, dtype=int)
    for t in range(T):
        z = t / max(1, T - 1)
        covariance = np.eye(m)
        rank = 0
        if name == "stationary":
            v = unit_rotation(m, 0.0, 0, 1)
            covariance += 2.0 * np.outer(v, v)
            rank = 1
        elif name == "smooth_rotation":
            v = unit_rotation(m, 2.0 * math.pi * z, 0, 1)
            strength = 2.0 + 0.5 * math.sin(4.0 * math.pi * z)
            covariance += strength * np.outer(v, v)
            rank = 1
        elif name == "birth_death":
            if 0.30 <= z < 0.70:
                v = unit_rotation(m, math.pi / 5.0, 0, 1)
                covariance += 5.0 * np.outer(v, v)
                rank = 1
        elif name == "mixed":
            if z < 0.20:
                v = unit_rotation(m, 0.0, 0, 1)
                covariance += 2.0 * np.outer(v, v)
                rank = 1
            elif z < 0.45:
                local = (z - 0.20) / 0.25
                v = unit_rotation(m, 0.5 * math.pi * local, 0, 1)
                covariance += 3.0 * np.outer(v, v)
                rank = 1
            elif z < 0.62:
                local = (z - 0.45) / 0.17
                ramp = max(0.0, min(1.0, local / 0.12, (1.0 - local) / 0.12))
                v1 = unit_rotation(m, math.pi / 2.0, 0, 1)
                v2 = unit_rotation(m, math.pi / 3.0, 2, 3)
                covariance += 6.0 * ramp * np.outer(v1, v1)
                covariance += 4.0 * ramp * np.outer(v2, v2)
                rank = 2 if ramp > 0 else 0
            elif z < 0.76:
                pass
            else:
                local = (z - 0.76) / 0.24
                v = unit_rotation(m, 4.0 * math.pi * local, 0, 2)
                strength = 3.0 + 0.5 * math.sin(8.0 * math.pi * local)
                covariance += strength * np.outer(v, v)
                rank = 1
        else:
            raise ValueError(f"unknown scenario: {name}")
        covariances[t] = sym(covariance)
        ranks[t] = rank
    return covariances, ranks


def sample_bounded_stream(covariances: np.ndarray, seed: int) -> np.ndarray:
    """Conditionally sub-Gaussian bounded innovations with exact covariance.

    Rademacher innovations make the no-clipping envelope auditable:
    ||C_t^{1/2} eps_t||^2 <= m lambda_max(C_t).
    """

    rng = np.random.default_rng(seed)
    T, m, _ = covariances.shape
    observations = np.empty((T, m))
    for t, covariance in enumerate(covariances):
        values, vectors = np.linalg.eigh(covariance)
        root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
        innovations = rng.choice(np.array([-1.0, 1.0]), size=m)
        observations[t] = root @ innovations
    return observations


def scenario_M(covariances: np.ndarray) -> float:
    return float(max(np.max(np.linalg.eigvalsh(c)) for c in covariances))


def global_fixed_choice(seed_level: pd.DataFrame, metric: str) -> Dict[Tuple[str, int], str]:
    fixed = seed_level[seed_level["method"].str.startswith("Fixed-")]
    means = fixed.groupby(["scenario", "dimension", "method"])[metric].mean()
    result: Dict[Tuple[str, int], str] = {}
    for (scenario, dimension), group in means.groupby(level=[0, 1]):
        result[(scenario, int(dimension))] = str(group.idxmin()[2])
    return result


def run_one(
    scenario: str,
    dimension: int,
    seed: int,
    T: int,
    windows: Sequence[int],
    alpha: float,
    c_det: float,
    kappa: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    covariances, ranks = covariance_path(scenario, T, dimension)
    M = max(1.01, scenario_M(covariances))
    observations = sample_bounded_stream(covariances, seed)
    # Deterministic boundedness: ||y_t||^2 <= m M, so clipping never fires.
    tau = math.sqrt(dimension * M) * (1.0 + 1e-10)
    config = RadiusConfig(
        K=1.0,
        M=M,
        alpha=alpha,
        horizon=T,
        n_windows=len(windows),
        gammas=(0.5, 0.25, 0.125, 0.0625, 0.03125),
        c_tau=1.0,
        c_det=c_det,
        kappa=kappa,
        c_bias=0.0,
    )
    stats = prepare_rolling_statistics(observations, tau)
    cumulative_covariance = np.concatenate(
        [np.zeros((1, dimension, dimension)), np.cumsum(covariances, axis=0)],
        axis=0,
    )
    burnin = max(windows)
    method_rows: List[dict] = []
    audit_rows: List[dict] = []
    selector_rows: List[dict] = []
    hedge_log_weights = np.zeros(len(windows))
    hedge_eta = 0.05
    path_covered = True

    for t in range(burnin - 1, T - 1):
        start_clock = time.perf_counter()
        atlas: AtlasForecast = atlas_sb_from_statistics(
            stats, t + 1, windows, config, lower=1.0
        )
        atlas_ms = 1000.0 * (time.perf_counter() - start_clock)
        available = sorted(atlas.scatters)
        fixed_forecasts = {
            h: spectral_clip(atlas.scatters[h], 1.0, M) for h in available
        }
        weights = np.exp(hedge_log_weights[: len(available)] - np.max(hedge_log_weights[: len(available)]))
        weights /= np.sum(weights)
        hedge_forecast = sym(
            sum(weight * fixed_forecasts[h] for weight, h in zip(weights, available))
        )
        target_covariance = covariances[t + 1]
        target_observation = observations[t + 1]
        local_oracle_h = min(
            available,
            key=lambda h: opnorm(fixed_forecasts[h] - target_covariance),
        )
        forecasts: Dict[str, np.ndarray] = {
            "Pilot": np.eye(dimension),
            "Hedge": hedge_forecast,
            "ATLAS-SB": atlas.forecast,
            "Local oracle": fixed_forecasts[local_oracle_h],
        }
        forecasts.update({f"Fixed-{h}": fixed_forecasts[h] for h in available})
        for method, forecast in forecasts.items():
            method_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "dimension": dimension,
                    "time": t,
                    "method": method,
                    "selected_window": atlas.selected_window if method == "ATLAS-SB" else np.nan,
                    "relative_op_error": opnorm(forecast - target_covariance) / opnorm(target_covariance),
                    "expected_nll_excess": expected_gaussian_nll_excess(forecast, target_covariance),
                    "realized_nll": realized_gaussian_nll(forecast, target_observation),
                    "update_ms": atlas_ms if method == "ATLAS-SB" else np.nan,
                }
            )

        all_scale_hits = True
        for h in available:
            start = t + 1 - h
            population_window = sym(
                (cumulative_covariance[t + 1] - cumulative_covariance[start]) / h
            )
            error = opnorm(atlas.scatters[h] - population_window)
            hit = error <= atlas.stabilized_radii[h]
            all_scale_hits = all_scale_hits and hit
            audit_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "dimension": dimension,
                    "time": t,
                    "window": h,
                    "centered_error": error,
                    "stabilized_radius": atlas.stabilized_radii[h],
                    "empirical_radius": atlas.empirical_radii[h],
                    "deterministic_radius": atlas.deterministic_radii[h],
                    "coverage": int(hit),
                    "empirical_tightens": int(
                        atlas.empirical_radii[h] < atlas.deterministic_radii[h]
                    ),
                }
            )
        path_covered = path_covered and all_scale_hits

        selected_h = atlas.selected_window
        selected_start = t + 1 - selected_h
        selected_population = sym(
            (cumulative_covariance[t + 1] - cumulative_covariance[selected_start])
            / selected_h
        )
        drift = opnorm(selected_population - target_covariance)
        total_radius = atlas.stabilized_radii[selected_h] + drift
        selected_values = np.linalg.eigvalsh(atlas.scatters[selected_h])
        certified_rank = int(np.sum(selected_values > 1.0 + 2.0 * total_radius))
        true_rank = int(ranks[t + 1])
        selector_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "dimension": dimension,
                "time": t,
                "selected_window": selected_h,
                "true_rank": true_rank,
                "certified_rank": certified_rank,
                "rank_false_positive": int(certified_rank > true_rank),
                "rank_exact": int(certified_rank == true_rank),
                "known_drift": drift,
                "total_radius": total_radius,
                "path_covered_so_far": int(path_covered),
            }
        )

        fixed_losses = np.array(
            [realized_gaussian_nll(fixed_forecasts[h], target_observation) for h in available]
        )
        fixed_losses = np.clip(fixed_losses - np.min(fixed_losses), 0.0, 30.0)
        hedge_log_weights[: len(available)] -= hedge_eta * fixed_losses

    selector = pd.DataFrame(selector_rows)
    selector["whole_path_coverage"] = int(path_covered)
    return pd.DataFrame(method_rows), pd.DataFrame(audit_rows), selector


def aggregate_results(results: pd.DataFrame, audits: pd.DataFrame, selectors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_level = (
        results.groupby(["seed", "scenario", "dimension", "method"], as_index=False)
        .agg(
            relative_op_error=("relative_op_error", "mean"),
            expected_nll_excess=("expected_nll_excess", "mean"),
            realized_nll=("realized_nll", "mean"),
            update_ms=("update_ms", "mean"),
        )
    )
    summary = (
        seed_level.groupby(["scenario", "dimension", "method"], as_index=False)
        .agg(
            op_mean=("relative_op_error", "mean"),
            op_se=("relative_op_error", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
            nll_mean=("expected_nll_excess", "mean"),
            nll_se=("expected_nll_excess", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
            realized_nll_mean=("realized_nll", "mean"),
            update_ms=("update_ms", "mean"),
        )
    )

    audit_summary = pd.DataFrame(
        [
            {
                "pointwise_scale_coverage": float(audits["coverage"].mean()),
                "whole_path_coverage": float(
                    selectors.groupby(["seed", "scenario", "dimension"])[
                        "whole_path_coverage"
                    ].first().mean()
                ),
                "empirical_tightening_rate": float(audits["empirical_tightens"].mean()),
                "rank_false_positive_rate": float(selectors["rank_false_positive"].mean()),
                "exact_rank_rate": float(selectors["rank_exact"].mean()),
            }
        ]
    )

    choices = global_fixed_choice(seed_level, "relative_op_error")
    paired_rows: List[dict] = []
    for (scenario, dimension), best_method in choices.items():
        block = seed_level[
            (seed_level["scenario"] == scenario)
            & (seed_level["dimension"] == dimension)
        ]
        atlas = block[block["method"] == "ATLAS-SB"][
            ["seed", "relative_op_error", "expected_nll_excess"]
        ].rename(
            columns={
                "relative_op_error": "atlas_op",
                "expected_nll_excess": "atlas_nll",
            }
        )
        benchmark = block[block["method"] == best_method][
            ["seed", "relative_op_error", "expected_nll_excess"]
        ].rename(
            columns={
                "relative_op_error": "benchmark_op",
                "expected_nll_excess": "benchmark_nll",
            }
        )
        merged = atlas.merge(benchmark, on="seed", how="inner")
        for metric in ("op", "nll"):
            diff = merged[f"atlas_{metric}"] - merged[f"benchmark_{metric}"]
            paired_rows.append(
                {
                    "scenario": scenario,
                    "dimension": dimension,
                    "benchmark": best_method,
                    "metric": metric,
                    "mean_difference": float(diff.mean()),
                    "se_difference": float(diff.std(ddof=1) / math.sqrt(len(diff))),
                    "t_stat": float(diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))))
                    if diff.std(ddof=1) > 0
                    else np.nan,
                }
            )
    return summary, audit_summary, pd.DataFrame(paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("generated/monte_carlo"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--T", type=int, default=6000)
    parser.add_argument("--c-det", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=0.25)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = ["stationary", "smooth_rotation", "birth_death", "mixed"]
    dimensions = [4, 8, 16]
    windows = [256, 512, 1024, 2048]
    all_results: List[pd.DataFrame] = []
    all_audits: List[pd.DataFrame] = []
    all_selectors: List[pd.DataFrame] = []
    for scenario in scenarios:
        for dimension in dimensions:
            for seed in range(args.seeds):
                result, audit, selector = run_one(
                    scenario=scenario,
                    dimension=dimension,
                    seed=seed,
                    T=args.T,
                    windows=windows,
                    alpha=0.05,
                    c_det=args.c_det,
                    kappa=args.kappa,
                )
                all_results.append(result)
                all_audits.append(audit)
                all_selectors.append(selector)
                print(f"completed {scenario} m={dimension} seed={seed}", flush=True)
    results = pd.concat(all_results, ignore_index=True)
    audits = pd.concat(all_audits, ignore_index=True)
    selectors = pd.concat(all_selectors, ignore_index=True)
    summary, audit_summary, paired = aggregate_results(results, audits, selectors)
    results.to_csv(args.out / "strict_results.csv", index=False)
    audits.to_csv(args.out / "coverage_audit.csv", index=False)
    selectors.to_csv(args.out / "selector_audit.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    audit_summary.to_csv(args.out / "audit_summary.csv", index=False)
    paired.to_csv(args.out / "paired_differences.csv", index=False)
    print(summary.to_string(index=False))
    print(audit_summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
