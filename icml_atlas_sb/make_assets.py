from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MC_ORDER = [
    "Pilot",
    "Fixed-256",
    "Fixed-512",
    "Fixed-1024",
    "Fixed-2048",
    "Covariance mix",
    "ATLAS-SB",
    "Certified selector",
    "Local oracle",
]
UCI_ORDER = [
    "Pilot",
    "Fixed-1-batch",
    "Fixed-2-batch",
    "Fixed-4-batch",
    "Expanding",
    "Covariance mix",
    "ATLAS-SB",
    "Certified selector",
    "Local oracle",
]


def mean_se(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(float)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def format_mean_se(mean: float, se: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {se:.{digits}f}"


def monte_carlo_assets(mc_dir: Path, output: Path) -> Dict[str, float | str]:
    results = pd.read_csv(mc_dir / "strict_results.csv")
    audits = pd.read_csv(mc_dir / "coverage_audit.csv")
    selectors = pd.read_csv(mc_dir / "selector_audit.csv")
    seed_level = (
        results.groupby(["seed", "scenario", "dimension", "method"], as_index=False)
        .agg(
            op=("relative_op_error", "mean"),
            nll=("expected_nll_excess", "mean"),
            update_ms=("update_ms", "mean"),
        )
    )
    rows: List[dict] = []
    for method in MC_ORDER:
        block = seed_level[seed_level["method"] == method]
        if block.empty:
            continue
        op_mean, op_se = mean_se(block["op"])
        nll_mean, nll_se = mean_se(block["nll"])
        runtime = (
            float(block["update_ms"].dropna().mean())
            if block["update_ms"].notna().any()
            else np.nan
        )
        rows.append(
            {
                "method": method,
                "op_mean": op_mean,
                "op_se": op_se,
                "nll_mean": nll_mean,
                "nll_se": nll_se,
                "runtime": runtime,
            }
        )
    table = pd.DataFrame(rows)
    feasible = table[table["method"] != "Local oracle"]
    best_op = float(feasible["op_mean"].min())
    best_nll = float(feasible["nll_mean"].min())
    lines = [
        r"\begin{table}[t]",
        r"\caption{Strict $t\mapsto t+1$ Monte Carlo, averaged equally over seeds, paths, and dimensions (lower is better). Every fixed window is shown.}",
        r"\label{tab:mc}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Rel. operator error & Excess NLL & Update ms \\",
        r"\midrule",
    ]
    for row in rows:
        label = "Local oracle$^\\dagger$" if row["method"] == "Local oracle" else row["method"]
        op_text = format_mean_se(row["op_mean"], row["op_se"])
        nll_text = format_mean_se(row["nll_mean"], row["nll_se"])
        if row["method"] != "Local oracle" and abs(row["op_mean"] - best_op) < 1e-12:
            op_text = r"\textbf{" + op_text + "}"
        if row["method"] != "Local oracle" and abs(row["nll_mean"] - best_nll) < 1e-12:
            nll_text = r"\textbf{" + nll_text + "}"
        runtime = "--" if np.isnan(row["runtime"]) else f"{row['runtime']:.2f}"
        lines.append(f"{label} & {op_text} & {nll_text} & {runtime} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{-1mm}",
        r"\begin{flushleft}\tiny $^\dagger$Uses $C_{t+1}$ to choose a fixed window and is infeasible. ATLAS-SB is the precision aggregate; the certified selector is reported separately.\end{flushleft}",
        r"\end{table}",
    ]
    (output / "mc_table.tex").write_text("\n".join(lines), encoding="utf-8")

    pointwise = float(audits["coverage"].mean())
    path = float(
        selectors.groupby(["seed", "scenario", "dimension"])["whole_path_coverage"]
        .first()
        .mean()
    )
    tightening = float(audits["empirical_tightens"].mean())
    rank_fp = float(selectors["rank_false_positive"].mean())
    rank_exact = float(selectors["rank_exact"].mean())
    audit_lines = [
        r"\begin{table}[t]",
        r"\caption{Certificate audit in simulation. Rank thresholds add the known local drift.}",
        r"\label{tab:audit}",
        r"\centering\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Diagnostic & Estimate \\",
        r"\midrule",
        f"Pointwise scale--time coverage & {100 * pointwise:.1f}\\% \\",
        f"Whole-path simultaneous coverage & {100 * path:.1f}\\% \\",
        f"Empirical radius below deterministic radius & {100 * tightening:.1f}\\% \\",
        f"Certified-rank false-positive rate & {100 * rank_fp:.2f}\\% \\",
        f"Exact certified-rank rate & {100 * rank_exact:.1f}\\% \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (output / "audit_table.tex").write_text("\n".join(audit_lines), encoding="utf-8")

    atlas = seed_level[seed_level["method"] == "ATLAS-SB"]
    block_rows: List[dict] = []
    for (scenario, dimension), block in seed_level.groupby(["scenario", "dimension"]):
        fixed = block[block["method"].str.startswith("Fixed-")]
        fixed_mean = fixed.groupby("method")["op"].mean()
        atlas_value = float(
            atlas[(atlas["scenario"] == scenario) & (atlas["dimension"] == dimension)]["op"].mean()
        )
        block_rows.append(
            {
                "scenario": scenario,
                "dimension": int(dimension),
                "best_fixed": str(fixed_mean.idxmin()),
                "atlas_gap_pct": 100.0 * (atlas_value / float(fixed_mean.min()) - 1.0),
            }
        )
    blocks = pd.DataFrame(block_rows)
    within_five = float(np.mean(blocks["atlas_gap_pct"] <= 5.0))
    within_ten = float(np.mean(blocks["atlas_gap_pct"] <= 10.0))
    median_gap = float(blocks["atlas_gap_pct"].median())
    best_blocks = int(np.sum(blocks["atlas_gap_pct"] <= 0.0))

    mixed = selectors[
        (selectors["scenario"] == "mixed")
        & (selectors["dimension"] == 8)
        & (selectors["seed"] == selectors["seed"].min())
    ].sort_values("time")
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(
        mixed["time"],
        mixed["dominant_predictive_window"],
        linewidth=0.9,
        label="predictive dominant",
    )
    ax.plot(
        mixed["time"],
        mixed["selected_window"],
        linewidth=0.9,
        linestyle="--",
        label="certified selector",
    )
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Forecast origin")
    ax.set_ylabel("Window")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "selected_window.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = pd.read_csv(mc_dir / "summary.csv")
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    for method, marker in (("ATLAS-SB", "o"), ("Local oracle", "s"), ("Certified selector", "^")):
        block = (
            summary[summary["method"] == method]
            .groupby("dimension", as_index=False)["op_mean"]
            .mean()
            .sort_values("dimension")
        )
        if not block.empty:
            ax.plot(block["dimension"], block["op_mean"], marker=marker, label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Residual dimension $m$")
    ax.set_ylabel("Relative operator error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "dimension_scaling.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "mc_pointwise_coverage": pointwise,
        "mc_path_coverage": path,
        "mc_tightening": tightening,
        "mc_rank_fp": rank_fp,
        "mc_rank_exact": rank_exact,
        "mc_within_five": within_five,
        "mc_within_ten": within_ten,
        "mc_median_gap": median_gap,
        "mc_best_blocks": best_blocks,
    }


def uci_assets(uci_dir: Path, output: Path) -> Dict[str, float | str]:
    results = pd.read_csv(uci_dir / "batch_ahead_results.csv")
    summary = pd.read_csv(uci_dir / "summary.csv")
    comparisons = pd.read_csv(uci_dir / "paired_bootstrap_comparisons.csv")
    metadata = pd.read_csv(uci_dir / "metadata.csv").iloc[0]
    lookup = {row["method"]: row for _, row in summary.iterrows()}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Public UCI gas-sensor drift data: strict next-batch forecasts. Batches 1--4 fix preprocessing, pilot, and envelopes; batches 5--10 are evaluated once.}",
        r"\label{tab:uci}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Mean NLL & Cov. discrepancy & Energy \\",
        r"\midrule",
    ]
    feasible_nll = float(summary[summary["method"] != "Local oracle"]["mean_nll"].min())
    for method in UCI_ORDER:
        if method not in lookup:
            continue
        row = lookup[method]
        label = "Local oracle$^\\dagger$" if method == "Local oracle" else method
        nll = f"{row['mean_nll']:.3f}"
        if method != "Local oracle" and abs(row["mean_nll"] - feasible_nll) < 1e-12:
            nll = r"\textbf{" + nll + "}"
        lines.append(
            f"{label} & {nll} & {row['mean_covariance_discrepancy']:.3f} & {row['mean_standardized_energy']:.3f} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{-1mm}",
        r"\begin{flushleft}\tiny $^\dagger$Chooses a fixed memory using the target-batch sample covariance and is infeasible. Within-batch row order is not treated as time.\end{flushleft}",
        r"\end{table}",
    ]
    (output / "uci_table.tex").write_text("\n".join(lines), encoding="utf-8")

    atlas_row = summary[summary["method"] == "ATLAS-SB"].iloc[0]
    best_fixed = summary[summary["method"].str.startswith("Fixed-")].sort_values("mean_nll").iloc[0]
    comparison = comparisons[comparisons["competitor"] == best_fixed["method"]]
    if comparison.empty:
        ci_low = ci_high = win_probability = float("nan")
    else:
        item = comparison.iloc[0]
        ci_low = float(item["ci_2_5"])
        ci_high = float(item["ci_97_5"])
        win_probability = float(item["atlas_win_probability"])

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for method in ("Pilot", best_fixed["method"], "Covariance mix", "ATLAS-SB", "Certified selector"):
        block = results[results["method"] == method].sort_values("target_batch")
        if not block.empty:
            ax.plot(block["target_batch"], block["mean_nll"], marker="o", label=method)
    ax.set_xlabel("Target batch")
    ax.set_ylabel("Mean Gaussian NLL")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(output / "uci_batch_nll.pdf", bbox_inches="tight")
    plt.close(fig)

    selected = results[results["method"].isin(["ATLAS-SB", "Certified selector"])].sort_values("target_batch")
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    for method, style in (("ATLAS-SB", "-"), ("Certified selector", "--")):
        block = selected[selected["method"] == method]
        ax.step(
            block["target_batch"],
            block["selected_memory"],
            where="mid",
            linestyle=style,
            label=method,
        )
    ax.set_xlabel("Target batch")
    ax.set_ylabel("Selected batches")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "uci_selected_memory.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "uci_atlas_nll": float(atlas_row["mean_nll"]),
        "uci_best_fixed": str(best_fixed["method"]),
        "uci_best_fixed_nll": float(best_fixed["mean_nll"]),
        "uci_diff_ci_low": ci_low,
        "uci_diff_ci_high": ci_high,
        "uci_win_probability": win_probability,
        "uci_pilot_rank": int(metadata["pilot_rank"]),
        "uci_monitor_dimension": int(metadata["monitor_dimension"]),
    }


def write_macros(values: Dict[str, float | str], output: Path) -> None:
    mapping = {
        "MCPointCoverage": f"{100 * float(values['mc_pointwise_coverage']):.1f}\\%",
        "MCPathCoverage": f"{100 * float(values['mc_path_coverage']):.1f}\\%",
        "MCTightening": f"{100 * float(values['mc_tightening']):.1f}\\%",
        "MCRankFPR": f"{100 * float(values['mc_rank_fp']):.2f}\\%",
        "MCRankExact": f"{100 * float(values['mc_rank_exact']):.1f}\\%",
        "MCWithinFive": f"{100 * float(values['mc_within_five']):.1f}\\%",
        "MCWithinTen": f"{100 * float(values['mc_within_ten']):.1f}\\%",
        "MCMedianGap": f"{float(values['mc_median_gap']):.1f}\\%",
        "MCBestBlocks": str(values["mc_best_blocks"]),
        "UCIAtlasNLL": f"{float(values['uci_atlas_nll']):.3f}",
        "UCIBestFixed": latex_escape(str(values["uci_best_fixed"])),
        "UCIBestFixedNLL": f"{float(values['uci_best_fixed_nll']):.3f}",
        "UCIDiffCILow": f"{float(values['uci_diff_ci_low']):.3f}",
        "UCIDiffCIHigh": f"{float(values['uci_diff_ci_high']):.3f}",
        "UCIWinProbability": f"{100 * float(values['uci_win_probability']):.1f}\\%",
        "UCIPilotRank": str(values["uci_pilot_rank"]),
        "UCIMonitorDimension": str(values["uci_monitor_dimension"]),
    }
    (output / "result_macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in mapping.items())
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=Path, default=Path("generated/monte_carlo"))
    parser.add_argument("--uci", type=Path, default=Path("generated/uci"))
    parser.add_argument("--out", type=Path, default=Path("paper/generated"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    values: Dict[str, float | str] = {}
    values.update(monte_carlo_assets(args.mc, args.out))
    values.update(uci_assets(args.uci, args.out))
    write_macros(values, args.out)
    (args.out / "claim_audit.json").write_text(
        json.dumps(values, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out / "results_summary.md").write_text(
        "# ATLAS-SB result audit\n\n"
        + "\n".join(f"- **{key}**: {value}" for key, value in sorted(values.items()))
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(values, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
