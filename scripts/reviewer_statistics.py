from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


BOOTSTRAP_SEED = 20261003


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if not finite.size:
        return adjusted
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan, np.nan
    estimate = float(statistic(values))
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    if statistic is np.mean:
        boot = samples.mean(axis=1)
    elif statistic is np.median:
        boot = np.median(samples, axis=1)
    else:
        boot = np.apply_along_axis(statistic, 1, samples)
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)


def final_search_rows(df: pd.DataFrame) -> pd.DataFrame:
    search = df[df["audit"].isin(["greedy", "multistart_greedy", "beam"])].copy()
    if search.empty:
        return search
    search["target_flipped"] = _as_bool(search["target_flipped"])
    keys = ["run_label", "dataset", "seed", "classifier", "audit", "target_index"]
    keys = [column for column in keys if column in search.columns]
    search = search.sort_values(keys + ["repeat"])
    successes = search[search["target_flipped"]].groupby(keys, as_index=False).first()
    failed_keys = search.groupby(keys, as_index=False)["target_flipped"].max()
    failed_keys = failed_keys[~failed_keys["target_flipped"]][keys]
    failures = (
        search.merge(failed_keys, on=keys, how="inner")
        .groupby(keys, as_index=False)
        .last()
    )
    return pd.concat([successes, failures], ignore_index=True)


def summarize_target_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = {
        "target_flip_rate": ("target_flipped", np.mean),
        "mean_collateral_flip_rate": ("collateral_flip_rate", np.mean),
        "mean_removed_fraction": ("removed_fraction", np.mean),
    }
    if "independent_prediction_changed" in frame.columns:
        metrics["independent_prediction_change_rate"] = (
            "independent_prediction_changed",
            np.mean,
        )
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key))
        row["n_target_units"] = int(len(group))
        for output_name, (column, statistic) in metrics.items():
            estimate, low, high = bootstrap_ci(
                group[column].to_numpy(dtype=float),
                statistic,
                n_boot=n_boot,
                rng=rng,
            )
            row[output_name] = estimate
            row[f"{output_name}_ci95_low"] = low
            row[f"{output_name}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon_tests(main: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Matched target-level structured-vs-random tests.

    The random outcome is first averaged over repeated draws for each target. The
    target cell (seed + target index) is therefore the pairing and inference unit.
    """
    selected = main[np.isclose(main["budget"], budget)].copy()
    selected["target_flipped"] = _as_bool(selected["target_flipped"]).astype(float)
    keys = ["dataset", "seed", "classifier", "target_index"]
    target = (
        selected.groupby(keys + ["audit"], as_index=False)["target_flipped"]
        .mean()
        .pivot(index=keys, columns="audit", values="target_flipped")
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for (dataset, classifier), group in target.groupby(["dataset", "classifier"]):
        for strategy in ("nearest", "same_class"):
            paired = group[["random", strategy]].dropna()
            differences = paired[strategy].to_numpy() - paired["random"].to_numpy()
            nonzero = differences[differences != 0]
            if nonzero.size:
                result = wilcoxon(
                    paired[strategy],
                    paired["random"],
                    zero_method="pratt",
                    alternative="two-sided",
                    method="approx",
                )
                ranks = rankdata(np.abs(nonzero), method="average")
                positive = float(ranks[nonzero > 0].sum())
                negative = float(ranks[nonzero < 0].sum())
                rank_biserial = (positive - negative) / (positive + negative)
            else:
                result = None
                rank_biserial = 0.0
            sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "structured_strategy": strategy,
                    "budget": budget,
                    "pairing_unit": "cell-level matched target (seed + target_index)",
                    "n_pairs": int(len(paired)),
                    "mean_paired_difference": float(differences.mean()),
                    "cohens_dz": float(differences.mean() / sd) if sd > 0 else np.nan,
                    "matched_rank_biserial": float(rank_biserial),
                    "wilcoxon_statistic": float(result.statistic) if result else 0.0,
                    "p_value_two_sided": float(result.pvalue) if result else 1.0,
                    "test": "Wilcoxon signed-rank; Pratt zeros; normal approximation",
                    "null_hypothesis": "median paired structured-minus-random flip propensity is zero",
                }
            )
    output = pd.DataFrame(rows)
    output["p_value_bh"] = bh_adjust(output["p_value_two_sided"])
    output["reject_fdr_0_05"] = output["p_value_bh"] <= 0.05
    return output


def poststratified_celltypist(
    target_level: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    stratum_to_weight = {
        "changed_by_majority_vote": "changed_by_majority_vote_prevalence",
        "stable_under_majority_vote": "stable_under_majority_vote_prevalence",
    }
    metrics = [
        "target_flipped",
        "independent_prediction_changed",
        "collateral_flip_rate",
    ]
    rows: list[dict[str, object]] = []
    for (strategy, budget), condition in target_level.groupby(["strategy", "budget"]):
        estimates = {metric: [] for metric in metrics}
        boot_values = {metric: np.zeros(n_boot, dtype=float) for metric in metrics}
        valid_seeds: list[int] = []
        for seed, seed_group in condition.groupby("seed"):
            clean_row = clean[clean["seed"] == seed]
            if clean_row.empty:
                continue
            valid_seeds.append(int(seed))
            for metric in metrics:
                seed_estimate = 0.0
                seed_boot = np.zeros(n_boot, dtype=float)
                for stratum, weight_column in stratum_to_weight.items():
                    values = seed_group.loc[
                        seed_group["target_stratum"] == stratum, metric
                    ].to_numpy(dtype=float)
                    if not values.size:
                        raise ValueError(f"Missing {stratum} targets for seed {seed}")
                    weight = float(clean_row.iloc[0][weight_column])
                    seed_estimate += weight * float(values.mean())
                    sampled = rng.choice(values, size=(n_boot, len(values)), replace=True)
                    seed_boot += weight * sampled.mean(axis=1)
                estimates[metric].append(seed_estimate)
                boot_values[metric] += seed_boot
        if not valid_seeds:
            continue
        row: dict[str, object] = {
            "strategy": strategy,
            "budget": float(budget),
            "estimand": "post-stratified all-cell query-cohort rate",
            "seeds": len(valid_seeds),
        }
        for metric in metrics:
            estimate = float(np.mean(estimates[metric]))
            boot = boot_values[metric] / len(valid_seeds)
            low, high = np.quantile(boot, [0.025, 0.975])
            output_name = {
                "target_flipped": "target_flip_rate",
                "independent_prediction_changed": "independent_prediction_change_rate",
                "collateral_flip_rate": "mean_collateral_flip_rate",
            }[metric]
            row[output_name] = estimate
            row[f"{output_name}_ci95_low"] = float(low)
            row[f"{output_name}_ci95_high"] = float(high)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reviewer-requested confidence intervals, matched inference, and FDR control"
    )
    parser.add_argument("--main", default="outputs/iccke/per_target_results.csv")
    parser.add_argument("--search", default="outputs/search_v2/per_target_results.csv")
    parser.add_argument(
        "--celltypist", default="outputs/celltypist_validation_v2/per_target_results.csv"
    )
    parser.add_argument(
        "--celltypist-clean",
        default="outputs/celltypist_validation_v2/clean_cohort_summary.csv",
    )
    parser.add_argument("--output-dir", default="outputs/reviewer_statistics")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(args.main)
    main_df["target_flipped"] = _as_bool(main_df["target_flipped"])
    main_target = (
        main_df.groupby(
            ["dataset", "seed", "classifier", "audit", "budget", "target_index"],
            as_index=False,
        )
        .agg(
            target_flipped=("target_flipped", "mean"),
            collateral_flip_rate=("collateral_flip_rate", "mean"),
            removed_fraction=("removed_fraction", "mean"),
        )
    )
    main_summary = summarize_target_metrics(
        main_target,
        ["dataset", "classifier", "audit", "budget"],
        n_boot=args.bootstrap_replicates,
        rng=rng,
    )
    main_summary.to_csv(out / "main_metrics_target_bootstrap_ci.csv", index=False)

    tests = paired_wilcoxon_tests(main_df, budget=0.05)
    tests.to_csv(out / "main_paired_tests_bh.csv", index=False)

    search_df = final_search_rows(pd.read_csv(args.search))
    search_summary = summarize_target_metrics(
        search_df,
        ["dataset", "classifier", "audit"],
        n_boot=args.bootstrap_replicates,
        rng=rng,
    )
    successful = search_df[search_df["target_flipped"]].copy()
    success_rows: list[dict[str, object]] = []
    for key, group in successful.groupby(["dataset", "classifier", "audit"]):
        row = dict(zip(["dataset", "classifier", "audit"], key))
        row["n_successes"] = len(group)
        for name, column, statistic in (
            ("median_successful_removed_fraction", "removed_fraction", np.median),
            ("mean_successful_collateral_flip_rate", "collateral_flip_rate", np.mean),
        ):
            estimate, low, high = bootstrap_ci(
                group[column].to_numpy(dtype=float),
                statistic,
                n_boot=args.bootstrap_replicates,
                rng=rng,
            )
            row[name] = estimate
            row[f"{name}_ci95_low"] = low
            row[f"{name}_ci95_high"] = high
        success_rows.append(row)
    search_summary.merge(
        pd.DataFrame(success_rows),
        on=["dataset", "classifier", "audit"],
        how="left",
    ).to_csv(out / "search_v2_metrics_target_bootstrap_ci.csv", index=False)

    cell = pd.read_csv(args.celltypist)
    cell["target_flipped"] = _as_bool(cell["target_flipped"]).astype(float)
    cell["independent_prediction_changed"] = _as_bool(
        cell["independent_prediction_changed"]
    ).astype(float)
    cell_target = (
        cell.groupby(
            ["seed", "target_index", "target_stratum", "strategy", "budget"],
            as_index=False,
        )
        .agg(
            target_flipped=("target_flipped", "mean"),
            independent_prediction_changed=("independent_prediction_changed", "mean"),
            collateral_flip_rate=("collateral_flip_rate", "mean"),
            removed_fraction=("removed_fraction", "mean"),
        )
    )
    cell_summary = summarize_target_metrics(
        cell_target,
        ["target_stratum", "strategy", "budget"],
        n_boot=args.bootstrap_replicates,
        rng=rng,
    )
    cell_summary.to_csv(out / "celltypist_targeted_metrics_bootstrap_ci.csv", index=False)

    clean_path = Path(args.celltypist_clean)
    if clean_path.exists():
        clean = pd.read_csv(clean_path)
        poststratified_celltypist(
            cell_target,
            clean,
            n_boot=args.bootstrap_replicates,
            rng=rng,
        ).to_csv(out / "celltypist_poststratified_population_estimates.csv", index=False)

    protocol = {
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95% confidence interval",
            "resampling_unit": "target cell; repeated random removals are averaged within target before resampling",
        },
        "main_hypothesis_tests": {
            "test": "two-sided Wilcoxon signed-rank test with Pratt handling of zero differences and normal approximation",
            "pairing_unit": "cell-level matched target identified by dataset, seed, classifier, and target_index",
            "random_baseline": "mean target-flip indicator across the 10 random removal repeats for the same target and budget",
            "effect_sizes": ["Cohen's dz", "matched rank-biserial correlation"],
            "multiplicity": "Benjamini-Hochberg FDR correction across the eight prespecified 5% structured-vs-random comparisons",
        },
        "celltypist_population_estimate": {
            "method": "post-stratification",
            "strata": ["clean base/majority disagreement", "clean base/majority agreement"],
            "weights": "observed stratum prevalence in each full 1,200-cell clean query cohort",
            "scope": "descriptive all-cell estimate under the tested removal intervention, not a natural unperturbed failure rate",
        },
    }
    (out / "statistical_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"Saved reviewer statistics to {out.resolve()}")


if __name__ == "__main__":
    main()
