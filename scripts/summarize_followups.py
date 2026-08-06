from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_outputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for raw in paths:
        path = Path(raw)
        csv = path / "per_target_results.csv" if path.is_dir() else path
        if not csv.exists():
            print(f"Skipping missing file: {csv}")
            continue
        frame = pd.read_csv(csv)
        if "run_label" not in frame.columns:
            frame["run_label"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No per_target_results.csv files were found.")
    return pd.concat(frames, ignore_index=True)


def greedy_budget_summary(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    greedy = df[df["audit"] == "greedy"].copy()
    if greedy.empty:
        return pd.DataFrame()
    keys = ["run_label", "dataset", "seed", "classifier", "target_index"]
    rows = []
    for key, group in greedy.groupby(keys, dropna=False):
        group = group.sort_values("removed_fraction")
        first_success = group[group["target_flipped"].astype(bool)]
        first_fraction = float(first_success.iloc[0]["removed_fraction"]) if not first_success.empty else np.nan
        max_fraction = float(group["removed_fraction"].max())
        row = dict(zip(keys, key))
        row["first_success_fraction"] = first_fraction
        row["max_evaluated_fraction"] = max_fraction
        row["ever_success"] = bool(not first_success.empty)
        for threshold in thresholds:
            eligible = group[group["removed_fraction"] <= threshold + 1e-12]
            row[f"success_at_{threshold:g}"] = bool(eligible["target_flipped"].astype(bool).any())
        rows.append(row)
    target = pd.DataFrame(rows)
    agg_cols = ["run_label", "dataset", "classifier"]
    aggregations = {
        "n_targets": ("target_index", "size"),
        "ever_success_rate": ("ever_success", "mean"),
        "median_first_success_fraction": ("first_success_fraction", "median"),
        "mean_max_evaluated_fraction": ("max_evaluated_fraction", "mean"),
    }
    for threshold in thresholds:
        aggregations[f"success_at_{threshold:g}"] = (f"success_at_{threshold:g}", "mean")
    return target.groupby(agg_cols, dropna=False).agg(**aggregations).reset_index()


def structured_summary(df: pd.DataFrame) -> pd.DataFrame:
    nongreedy = df[df["audit"] != "greedy"].copy()
    if nongreedy.empty:
        return pd.DataFrame()
    return (
        nongreedy.groupby(["run_label", "dataset", "classifier", "audit", "budget"], dropna=False)
        .agg(
            n_rows=("target_index", "size"),
            n_targets=("target_index", "nunique"),
            target_flip_rate=("target_flipped", "mean"),
            mean_removed_fraction=("removed_fraction", "mean"),
            mean_collateral_flip_rate=("collateral_flip_rate", "mean"),
            mean_margin=("target_margin_original_class", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Output directories or per-target CSV files.")
    parser.add_argument("--output-dir", default="outputs/followup_summary")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.01, 0.02, 0.05, 0.10])
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_outputs(args.paths)
    greedy = greedy_budget_summary(data, args.thresholds)
    structured = structured_summary(data)
    greedy.to_csv(outdir / "greedy_matched_budget_summary.csv", index=False)
    structured.to_csv(outdir / "structured_ablation_summary.csv", index=False)
    metadata = {
        "input_rows": int(len(data)),
        "run_labels": sorted(data["run_label"].dropna().astype(str).unique().tolist()),
        "thresholds": args.thresholds,
    }
    (outdir / "summary_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Summary written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
