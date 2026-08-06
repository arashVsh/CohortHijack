from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from contextcell.utils import atomic_write_csv, atomic_write_json, load_yaml, stable_seed


def require_celltypist():
    try:
        import celltypist
        from celltypist import models
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency with: pip install celltypist") from exc
    return celltypist, models


def load_pbmc(max_cells: int, seed: int):
    raw = sc.datasets.pbmc3k().copy()
    processed = sc.datasets.pbmc3k_processed().copy()
    common = raw.obs_names.intersection(processed.obs_names)
    raw = raw[common].copy()
    raw.obs["reference_label"] = processed[common].obs["louvain"].astype(str).to_numpy()
    if raw.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        labels = raw.obs["reference_label"].astype(str).to_numpy()
        chosen = []
        per_class = max(1, max_cells // len(np.unique(labels)))
        for label in sorted(np.unique(labels)):
            idx = np.flatnonzero(labels == label)
            chosen.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
        chosen = np.asarray(sorted(chosen[:max_cells]), dtype=int)
        raw = raw[chosen].copy()
    sc.pp.normalize_total(raw, target_sum=1e4)
    sc.pp.log1p(raw)
    raw.var_names_make_unique()
    return raw


def extract_labels(result) -> tuple[np.ndarray, np.ndarray]:
    frame = result.predicted_labels
    if "predicted_labels" not in frame.columns:
        raise KeyError(f"CellTypist result lacks predicted_labels column: {list(frame.columns)}")
    base = frame["predicted_labels"].astype(str).to_numpy()
    majority_col = "majority_voting" if "majority_voting" in frame.columns else "predicted_labels"
    majority = frame[majority_col].astype(str).to_numpy()
    return base, majority


def annotate(celltypist, adata, model: str, use_gpu: bool):
    result = celltypist.annotate(
        adata,
        model=model,
        majority_voting=True,
        use_GPU=use_gpu,
    )
    return extract_labels(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/celltypist_validation.yaml")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["output_dir"])
    checkpoints = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    celltypist, models = require_celltypist()
    models.download_models(model=[cfg["model"]])

    all_rows = []
    for seed in cfg["seeds"]:
        seed = int(seed)
        adata = load_pbmc(int(cfg["max_cells"]), seed)
        base, majority = annotate(celltypist, adata, cfg["model"], bool(cfg.get("use_gpu", False)))
        pca_adata = adata.copy()
        sc.pp.highly_variable_genes(pca_adata, n_top_genes=min(1500, pca_adata.n_vars), flavor="seurat", subset=True)
        sc.pp.scale(pca_adata, max_value=10)
        sc.tl.pca(pca_adata, n_comps=min(40, pca_adata.n_obs - 1, pca_adata.n_vars - 1), random_state=seed)
        Xp = np.asarray(pca_adata.obsm["X_pca"], dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=min(101, adata.n_obs)).fit(Xp)
        neighbor_order = nn.kneighbors(Xp, return_distance=False)[:, 1:]

        changed = np.flatnonzero(base != majority)
        candidates = changed if cfg.get("target_policy") == "changed_by_majority_vote" and changed.size else np.arange(adata.n_obs)
        rng = np.random.default_rng(stable_seed("celltypist_targets", seed))
        n_targets = min(int(cfg["n_targets_per_seed"]), len(candidates))
        targets = np.sort(rng.choice(candidates, size=n_targets, replace=False))
        atomic_write_json({"targets": targets.tolist()}, checkpoints / f"seed_{seed}" / "target_manifest.json")

        for target in tqdm(targets, desc=f"celltypist/{seed}"):
            target = int(target)
            target_dir = checkpoints / f"seed_{seed}"
            csv_path = target_dir / f"target_{target:06d}.csv"
            done_path = target_dir / f"target_{target:06d}.done.json"
            if not args.fresh and bool(cfg.get("resume", True)) and csv_path.exists() and done_path.exists():
                all_rows.append(pd.read_csv(csv_path))
                continue
            target_name = adata.obs_names[target]
            non_target = np.delete(np.arange(adata.n_obs), target)
            rows = []
            target_rng = np.random.default_rng(stable_seed("celltypist_audit", seed, target))
            for budget in cfg["budgets"]:
                remove_n = max(1, int(round(float(budget) * len(non_target))))
                for strategy in cfg["strategies"]:
                    repeats = int(cfg["random_repeats"]) if strategy == "random" else 1
                    for rep in range(repeats):
                        if strategy == "random":
                            removed = target_rng.choice(non_target, size=remove_n, replace=False)
                        elif strategy == "same_majority_label":
                            pool = non_target[majority[non_target] == majority[target]]
                            if not len(pool):
                                continue
                            removed = target_rng.choice(pool, size=min(remove_n, len(pool)), replace=False)
                        elif strategy == "nearest":
                            removed = neighbor_order[target][:remove_n]
                        else:
                            raise ValueError(strategy)
                        keep = np.ones(adata.n_obs, dtype=bool)
                        keep[removed] = False
                        subset = adata[keep].copy()
                        _, attacked_majority = annotate(celltypist, subset, cfg["model"], bool(cfg.get("use_gpu", False)))
                        pos = int(np.flatnonzero(subset.obs_names == target_name)[0])
                        new_label = str(attacked_majority[pos])
                        rows.append({
                            "seed": seed,
                            "target_index": target,
                            "target_obs_name": str(target_name),
                            "target_reference_label": str(adata.obs.iloc[target]["reference_label"]),
                            "target_base_label": str(base[target]),
                            "target_clean_majority_label": str(majority[target]),
                            "strategy": strategy,
                            "repeat": rep,
                            "budget": float(budget),
                            "removed_count": int(len(removed)),
                            "removed_fraction": float(len(removed) / len(non_target)),
                            "target_new_label": new_label,
                            "target_flipped": bool(new_label != majority[target]),
                        })
            target_df = pd.DataFrame(rows)
            atomic_write_csv(target_df, csv_path)
            atomic_write_json({"status": "complete", "target": target, "rows": len(rows)}, done_path)
            all_rows.append(target_df)
            combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
            atomic_write_csv(combined, out / "per_target_results.csv")

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    atomic_write_csv(combined, out / "per_target_results.csv")
    if not combined.empty:
        aggregate = combined.groupby(["strategy", "budget"], as_index=False).agg(
            n=("target_index", "size"), target_flip_rate=("target_flipped", "mean"),
            mean_removed_fraction=("removed_fraction", "mean"),
        )
        atomic_write_csv(aggregate, out / "aggregate_results.csv")
    atomic_write_json({"completed_targets": int(combined[["seed", "target_index"]].drop_duplicates().shape[0]) if not combined.empty else 0}, out / "summary.json")


if __name__ == "__main__":
    main()
