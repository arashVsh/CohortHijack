from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def final_search_rows(df: pd.DataFrame) -> pd.DataFrame:
    search = df[df["audit"].isin(["greedy", "multistart_greedy", "beam"])].copy()
    if search.empty:
        return search
    keys = ["run_label", "dataset", "seed", "classifier", "audit", "target_index"]
    keys = [k for k in keys if k in search.columns]
    search = search.sort_values(keys + ["repeat"])
    successful = search[search["target_flipped"].astype(bool)]
    first_success = successful.groupby(keys, as_index=False).first()
    failed_keys = search.groupby(keys, as_index=False)["target_flipped"].max()
    failed_keys = failed_keys[~failed_keys["target_flipped"].astype(bool)][keys]
    failed = search.merge(failed_keys, on=keys, how="inner").groupby(keys, as_index=False).last()
    return pd.concat([first_success, failed], ignore_index=True)


def bootstrap_rate(values: np.ndarray, seed: int = 2026, n_boot: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def broad_lineage(dataset: str, label: str) -> str:
    text = str(label).lower()
    if dataset == "pbmc3k":
        if "t cell" in text or "cd4" in text or "cd8" in text:
            return "T cell"
        if "nk" in text:
            return "NK cell"
        if "b cell" in text or text.startswith("b "):
            return "B cell"
        if "mono" in text:
            return "Monocyte"
        if "dendritic" in text or "dc" in text:
            return "Dendritic"
        if "megakary" in text:
            return "Megakaryocyte"
    if dataset == "paul15":
        if "ery" in text:
            return "Erythroid"
        if "gran" in text or "neut" in text:
            return "Granulocyte"
        if "mono" in text:
            return "Monocyte"
        if "meg" in text:
            return "Megakaryocyte"
        if "baso" in text or "eos" in text or "mast" in text:
            return "Basophil/Eosinophil/Mast"
        if "lymph" in text:
            return "Lymphoid"
    return "Other/Unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/publication_analysis")
    args = parser.parse_args()

    frames = []
    for item in args.inputs:
        path = Path(item)
        csv = path / "per_target_results.csv" if path.is_dir() else path
        if csv.exists():
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError("No per_target_results.csv files found.")
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Aggregate attack rates with target-level bootstrap confidence intervals.
    rows = []
    structured = df[df["audit"].isin(["random", "same_class", "nearest"])].copy()
    if not structured.empty:
        target_level = structured.groupby(
            ["run_label", "dataset", "classifier", "audit", "budget", "seed", "target_index"],
            as_index=False,
        )["target_flipped"].mean()
        for key, group in target_level.groupby(["run_label", "dataset", "classifier", "audit", "budget"]):
            mean, lo, hi = bootstrap_rate(group["target_flipped"].to_numpy())
            rows.append(dict(zip(["run_label", "dataset", "classifier", "audit", "budget"], key)) |
                        {"n_targets": len(group), "flip_rate": mean, "ci95_low": lo, "ci95_high": hi})
    search_final = final_search_rows(df)
    if not search_final.empty:
        for key, group in search_final.groupby(["run_label", "dataset", "classifier", "audit"]):
            mean, lo, hi = bootstrap_rate(group["target_flipped"].astype(float).to_numpy())
            rows.append(dict(zip(["run_label", "dataset", "classifier", "audit"], key)) |
                        {"budget": np.nan, "n_targets": len(group), "flip_rate": mean,
                         "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(rows).to_csv(out / "attack_rates_with_ci.csv", index=False)

    # Per-class vulnerability and class-frequency association.
    analysis_rows = search_final if not search_final.empty else structured
    if not analysis_rows.empty:
        class_df = analysis_rows.groupby(
            ["run_label", "dataset", "classifier", "audit", "target_true_label"], as_index=False
        ).agg(
            n_targets=("target_index", "nunique"),
            flip_rate=("target_flipped", "mean"),
            mean_class_frequency=("target_class_frequency", "mean"),
            mean_clean_confidence=("target_clean_confidence", "mean"),
            mean_removed_fraction=("removed_fraction", "mean"),
            mean_collateral_flip_rate=("collateral_flip_rate", "mean"),
        )
        class_df.to_csv(out / "per_class_vulnerability.csv", index=False)

        corr_rows = []
        unique_targets = analysis_rows.drop_duplicates(
            ["run_label", "dataset", "classifier", "audit", "seed", "target_index"]
        )
        for key, group in unique_targets.groupby(["run_label", "dataset", "classifier", "audit"]):
            rho_freq, p_freq = spearmanr(group["target_class_frequency"], group["target_flipped"].astype(int))
            rho_conf, p_conf = spearmanr(group["target_clean_confidence"], group["target_flipped"].astype(int))
            corr_rows.append(dict(zip(["run_label", "dataset", "classifier", "audit"], key)) |
                             {"rho_class_frequency": rho_freq, "p_class_frequency": p_freq,
                              "rho_clean_confidence": rho_conf, "p_clean_confidence": p_conf,
                              "n_targets": len(group)})
        pd.DataFrame(corr_rows).to_csv(out / "vulnerability_correlations.csv", index=False)

        transitions = analysis_rows[analysis_rows["target_flipped"].astype(bool)].copy()
        if not transitions.empty:
            transitions["source_lineage"] = [broad_lineage(d, l) for d, l in zip(transitions.dataset, transitions.target_clean_label)]
            transitions["destination_lineage"] = [broad_lineage(d, l) for d, l in zip(transitions.dataset, transitions.target_new_label)]
            transitions["transition_scope"] = np.where(
                transitions["source_lineage"] == transitions["destination_lineage"],
                "within_lineage", "cross_lineage"
            )
            trans_summary = transitions.groupby(
                ["run_label", "dataset", "classifier", "audit", "source_lineage", "destination_lineage", "transition_scope"],
                as_index=False,
            ).agg(n=("target_index", "nunique"))
            trans_summary.to_csv(out / "biological_transition_summary.csv", index=False)
            transitions[["run_label", "dataset", "seed", "classifier", "audit", "target_index",
                         "target_true_label", "target_clean_label", "target_new_label", "source_lineage",
                         "destination_lineage", "transition_scope", "removed_fraction",
                         "collateral_flip_rate"]].to_csv(out / "successful_transition_cases.csv", index=False)

    summary = {
        "input_files": [str(x) for x in args.inputs],
        "rows_loaded": int(len(df)),
        "search_final_targets": int(len(search_final)),
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved publication analyses to {out.resolve()}")


if __name__ == "__main__":
    main()
