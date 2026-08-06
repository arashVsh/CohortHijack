from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_flip_curves(df: pd.DataFrame, out: Path) -> None:
    subset = df[df["audit"].isin(["random", "same_class", "nearest", "greedy"])].copy()
    grouped = subset.groupby(["dataset", "classifier", "audit", "budget"], as_index=False)["target_flipped"].mean()
    for dataset in grouped["dataset"].unique():
        d = grouped[grouped["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(7, 5))
        for (classifier, audit), g in d.groupby(["classifier", "audit"]):
            g = g.sort_values("budget")
            ax.plot(g["budget"] * 100, g["target_flipped"], marker="o", label=f"{classifier}: {audit}")
        ax.set_xlabel("Removed query cells (%)")
        ax.set_ylabel("Target flip rate")
        ax.set_title(f"Target-preserving composition audit: {dataset}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"flip_rate_{dataset}.pdf", bbox_inches="tight")
        fig.savefig(out / f"flip_rate_{dataset}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_clean_performance(df: pd.DataFrame, out: Path) -> None:
    long = df.melt(
        id_vars=["dataset", "seed", "classifier"],
        value_vars=["base_accuracy", "refined_accuracy", "base_macro_f1", "refined_macro_f1"],
        var_name="metric",
        value_name="value",
    )
    summary = long.groupby(["dataset", "classifier", "metric"], as_index=False)["value"].mean()
    for dataset in summary["dataset"].unique():
        d = summary[summary["dataset"] == dataset].copy()
        labels = [f"{r.classifier}\n{r.metric}" for r in d.itertuples()]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(d)), d["value"])
        ax.set_xticks(range(len(d)), labels, rotation=35, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(f"Clean annotation performance: {dataset}")
        fig.tight_layout()
        fig.savefig(out / f"clean_performance_{dataset}.pdf", bbox_inches="tight")
        fig.savefig(out / f"clean_performance_{dataset}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    per_target = pd.read_csv(root / "per_target_results.csv")
    clean = pd.read_csv(root / "clean_performance.csv")
    save_flip_curves(per_target, out)
    save_clean_performance(clean, out)
    print(f"Figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
