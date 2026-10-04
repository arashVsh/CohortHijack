from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextcell.utils import (
    atomic_write_csv,
    atomic_write_json,
    load_yaml,
    stable_seed,
)


def require_celltypist():
    try:
        import celltypist
        from celltypist import models
    except ImportError as exc:
        raise RuntimeError(
            "CellTypist is required for this optional validation. Install it with: pip install celltypist"
        ) from exc
    return celltypist, models


def sha256_array(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def load_pbmc(max_cells: int, seed: int):
    raw = sc.datasets.pbmc3k().copy()
    processed = sc.datasets.pbmc3k_processed().copy()
    common = raw.obs_names.intersection(processed.obs_names)
    raw = raw[common].copy()
    raw.obs["reference_label"] = processed[common].obs["louvain"].astype(str).to_numpy()

    if raw.n_obs > max_cells:
        rng = np.random.default_rng(stable_seed("celltypist_pbmc_subsample", seed))
        labels = raw.obs["reference_label"].astype(str).to_numpy()
        unique = sorted(np.unique(labels))
        base_quota = max_cells // len(unique)
        chosen: list[int] = []
        leftovers: list[int] = []
        for label in unique:
            idx = np.flatnonzero(labels == label)
            take = min(base_quota, len(idx))
            if take:
                selected = rng.choice(idx, size=take, replace=False)
                chosen.extend(selected.tolist())
                leftovers.extend(
                    np.setdiff1d(idx, selected, assume_unique=False).tolist()
                )
        remaining = max_cells - len(chosen)
        if remaining > 0 and leftovers:
            chosen.extend(
                rng.choice(
                    np.asarray(leftovers),
                    size=min(remaining, len(leftovers)),
                    replace=False,
                ).tolist()
            )
        raw = raw[np.asarray(sorted(chosen), dtype=int)].copy()

    # CellTypist expects log1p-normalized expression with 10,000 counts per cell.
    sc.pp.normalize_total(raw, target_sum=1e4)
    sc.pp.log1p(raw)
    raw.var_names_make_unique()
    return raw


def extract_annotation(result) -> dict[str, np.ndarray]:
    frame = result.predicted_labels
    required = {"predicted_labels", "over_clustering", "majority_voting"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"CellTypist result is missing columns: {sorted(missing)}")
    probs = result.probability_matrix
    return {
        "base": frame["predicted_labels"].astype(str).to_numpy(),
        "cluster": frame["over_clustering"].astype(str).to_numpy(),
        "majority": frame["majority_voting"].astype(str).to_numpy(),
        "confidence": probs.max(axis=1).to_numpy(dtype=float),
    }


def annotate(celltypist, adata, model: str, use_gpu: bool):
    kwargs: dict[str, Any] = {
        "filename": adata,
        "model": model,
        "majority_voting": True,
        "use_GPU": use_gpu,
    }
    result = celltypist.annotate(**kwargs)
    return extract_annotation(result)


def pca_neighbor_order(adata, seed: int, max_neighbors: int) -> np.ndarray:
    work = adata.copy()
    sc.pp.highly_variable_genes(
        work,
        n_top_genes=min(1500, work.n_vars),
        flavor="seurat",
        subset=True,
    )
    sc.pp.scale(work, max_value=10)
    n_comps = min(40, work.n_obs - 1, work.n_vars - 1)
    sc.tl.pca(work, n_comps=n_comps, random_state=seed)
    xp = np.asarray(work.obsm["X_pca"], dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(max_neighbors + 1, adata.n_obs)).fit(xp)
    return nn.kneighbors(xp, return_distance=False)[:, 1:]


def choose_targets(
    clean: dict[str, np.ndarray], cfg: dict[str, Any], seed: int
) -> pd.DataFrame:
    changed = np.flatnonzero(clean["base"] != clean["majority"])
    stable = np.flatnonzero(clean["base"] == clean["majority"])
    rng = np.random.default_rng(stable_seed("celltypist_target_strata", seed))

    requested_changed = int(cfg["targets_changed_per_seed"])
    requested_stable = int(cfg["targets_stable_per_seed"])
    n_changed = min(requested_changed, len(changed))
    n_stable = min(requested_stable, len(stable))

    selected_changed = (
        rng.choice(changed, size=n_changed, replace=False)
        if n_changed
        else np.array([], dtype=int)
    )
    selected_stable = (
        rng.choice(stable, size=n_stable, replace=False)
        if n_stable
        else np.array([], dtype=int)
    )

    rows = [
        {"target_index": int(i), "target_stratum": "changed_by_majority_vote"}
        for i in selected_changed
    ] + [
        {"target_index": int(i), "target_stratum": "stable_under_majority_vote"}
        for i in selected_stable
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["target_stratum", "target_index"])
        .reset_index(drop=True)
    )


def removal_set(
    strategy: str,
    target: int,
    non_target: np.ndarray,
    remove_n: int,
    clean: dict[str, np.ndarray],
    neighbor_order: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if strategy == "random":
        return np.asarray(
            rng.choice(non_target, size=min(remove_n, len(non_target)), replace=False),
            dtype=int,
        )
    if strategy == "same_majority_label":
        pool = non_target[clean["majority"][non_target] == clean["majority"][target]]
        if len(pool) == 0:
            return np.array([], dtype=int)
        return np.asarray(
            rng.choice(pool, size=min(remove_n, len(pool)), replace=False), dtype=int
        )
    if strategy == "nearest":
        order = neighbor_order[target]
        order = order[order != target]
        return np.asarray(order[: min(remove_n, len(order))], dtype=int)
    raise ValueError(f"Unknown strategy: {strategy}")


def write_combined_outputs(out: Path, frames: list[pd.DataFrame]) -> None:
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    atomic_write_csv(combined, out / "per_target_results.csv")
    if combined.empty:
        return

    group_cols = ["target_stratum", "strategy", "budget"]
    aggregate = combined.groupby(group_cols, as_index=False).agg(
        evaluations=("target_index", "size"),
        unique_targets=("target_index", "nunique"),
        target_flip_rate=("target_flipped", "mean"),
        independent_prediction_change_rate=("independent_prediction_changed", "mean"),
        mean_collateral_flip_rate=("collateral_flip_rate", "mean"),
        mean_removed_fraction=("removed_fraction", "mean"),
    )
    atomic_write_csv(aggregate, out / "aggregate_results.csv")

    seed_aggregate = combined.groupby(["seed", *group_cols], as_index=False).agg(
        evaluations=("target_index", "size"),
        unique_targets=("target_index", "nunique"),
        target_flip_rate=("target_flipped", "mean"),
        independent_prediction_change_rate=("independent_prediction_changed", "mean"),
        mean_collateral_flip_rate=("collateral_flip_rate", "mean"),
    )
    atomic_write_csv(seed_aggregate, out / "aggregate_by_seed.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crash-safe CellTypist majority-voting cohort-removal validation"
    )
    parser.add_argument("--config", default="configs/celltypist_validation_v2.yaml")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = Path(cfg["output_dir"])
    checkpoints = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)

    celltypist, models = require_celltypist()
    models.download_models(model=[cfg["model"]])

    all_frames: list[pd.DataFrame] = []
    clean_cohort_rows: list[dict[str, Any]] = []
    run_start = time.time()

    for seed_value in cfg["seeds"]:
        seed = int(seed_value)
        seed_dir = checkpoints / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        adata = load_pbmc(int(cfg["max_cells"]), seed)
        clean = annotate(
            celltypist,
            adata,
            str(cfg["model"]),
            bool(cfg.get("use_gpu", False)),
        )
        neighbor_order = pca_neighbor_order(
            adata, seed, int(cfg.get("max_neighbor_candidates", 250))
        )
        changed_mask = clean["base"] != clean["majority"]
        reference = adata.obs["reference_label"].astype(str).to_numpy()
        clean_cohort_rows.append(
            {
                "seed": seed,
                "n_cells": int(adata.n_obs),
                "changed_by_majority_vote_count": int(changed_mask.sum()),
                "stable_under_majority_vote_count": int((~changed_mask).sum()),
                "changed_by_majority_vote_prevalence": float(changed_mask.mean()),
                "stable_under_majority_vote_prevalence": float((~changed_mask).mean()),
                "base_reference_accuracy": float((clean["base"] == reference).mean()),
                "majority_reference_accuracy": float(
                    (clean["majority"] == reference).mean()
                ),
                "base_majority_disagreement_rate": float(changed_mask.mean()),
                "mean_independent_confidence": float(clean["confidence"].mean()),
            }
        )

        manifest_path = seed_dir / "target_manifest.csv"
        if manifest_path.exists() and not args.fresh:
            manifest = pd.read_csv(manifest_path)
        else:
            manifest = choose_targets(clean, cfg, seed)
            manifest["target_obs_name"] = [
                str(adata.obs_names[int(i)]) for i in manifest["target_index"]
            ]
            manifest["clean_base_label"] = [
                str(clean["base"][int(i)]) for i in manifest["target_index"]
            ]
            manifest["clean_majority_label"] = [
                str(clean["majority"][int(i)]) for i in manifest["target_index"]
            ]
            manifest["clean_confidence"] = [
                float(clean["confidence"][int(i)]) for i in manifest["target_index"]
            ]
            atomic_write_csv(manifest, manifest_path)

        atomic_write_json(
            {
                "seed": seed,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "model": str(cfg["model"]),
                "targets": int(len(manifest)),
                "changed_targets": int(
                    (manifest["target_stratum"] == "changed_by_majority_vote").sum()
                ),
                "stable_targets": int(
                    (manifest["target_stratum"] == "stable_under_majority_vote").sum()
                ),
                "changed_by_majority_vote_count": int(changed_mask.sum()),
                "stable_under_majority_vote_count": int((~changed_mask).sum()),
                "base_majority_disagreement_rate": float(changed_mask.mean()),
                "base_reference_accuracy": float((clean["base"] == reference).mean()),
                "majority_reference_accuracy": float(
                    (clean["majority"] == reference).mean()
                ),
            },
            seed_dir / "clean_summary.json",
        )

        for item in tqdm(
            manifest.itertuples(index=False),
            total=len(manifest),
            desc=f"celltypist-v2/{seed}",
        ):
            target = int(item.target_index)
            target_dir = seed_dir / "targets"
            csv_path = target_dir / f"target_{target:06d}.csv"
            done_path = target_dir / f"target_{target:06d}.done.json"

            if (
                not args.fresh
                and bool(cfg.get("resume", True))
                and csv_path.exists()
                and done_path.exists()
            ):
                all_frames.append(pd.read_csv(csv_path))
                continue

            target_obs_name = str(adata.obs_names[target])
            target_vector_before = np.asarray(
                adata[target].X.toarray()
                if hasattr(adata[target].X, "toarray")
                else adata[target].X
            )
            target_hash_before = sha256_array(target_vector_before)
            non_target = np.delete(np.arange(adata.n_obs), target)
            target_rng = np.random.default_rng(
                stable_seed("celltypist_v2_audit", seed, target)
            )
            rows: list[dict[str, Any]] = []

            for budget_value in cfg["budgets"]:
                budget = float(budget_value)
                remove_n = max(1, int(round(budget * len(non_target))))
                for strategy in cfg["strategies"]:
                    repeats = int(cfg["random_repeats"]) if strategy == "random" else 1
                    for repeat in range(repeats):
                        removed = removal_set(
                            strategy,
                            target,
                            non_target,
                            remove_n,
                            clean,
                            neighbor_order,
                            target_rng,
                        )
                        if len(removed) == 0:
                            continue

                        keep = np.ones(adata.n_obs, dtype=bool)
                        keep[removed] = False
                        subset = adata[keep].copy()
                        attacked = annotate(
                            celltypist,
                            subset,
                            str(cfg["model"]),
                            bool(cfg.get("use_gpu", False)),
                        )
                        positions = np.flatnonzero(subset.obs_names == target_obs_name)
                        if len(positions) != 1:
                            raise RuntimeError(
                                f"Target {target_obs_name} was not uniquely retained"
                            )
                        pos = int(positions[0])

                        target_vector_after = np.asarray(
                            subset[pos].X.toarray()
                            if hasattr(subset[pos].X, "toarray")
                            else subset[pos].X
                        )
                        target_hash_after = sha256_array(target_vector_after)
                        if target_hash_before != target_hash_after:
                            raise AssertionError(
                                "Target expression vector changed during companion-cell removal"
                            )

                        retained_original_indices = np.flatnonzero(keep)
                        clean_majority_retained = clean["majority"][
                            retained_original_indices
                        ]
                        collateral_mask = retained_original_indices != target
                        collateral_flips = (
                            attacked["majority"][collateral_mask]
                            != clean_majority_retained[collateral_mask]
                        )

                        rows.append(
                            {
                                "seed": seed,
                                "target_index": target,
                                "target_obs_name": target_obs_name,
                                "target_stratum": str(item.target_stratum),
                                "target_reference_label": str(
                                    adata.obs.iloc[target]["reference_label"]
                                ),
                                "target_clean_base_label": str(clean["base"][target]),
                                "target_clean_majority_label": str(
                                    clean["majority"][target]
                                ),
                                "target_clean_cluster": str(clean["cluster"][target]),
                                "target_clean_confidence": float(
                                    clean["confidence"][target]
                                ),
                                "strategy": strategy,
                                "repeat": int(repeat),
                                "budget": budget,
                                "removed_count": int(len(removed)),
                                "removed_fraction": float(
                                    len(removed) / len(non_target)
                                ),
                                "target_attacked_base_label": str(
                                    attacked["base"][pos]
                                ),
                                "target_attacked_majority_label": str(
                                    attacked["majority"][pos]
                                ),
                                "target_attacked_cluster": str(
                                    attacked["cluster"][pos]
                                ),
                                "target_attacked_confidence": float(
                                    attacked["confidence"][pos]
                                ),
                                "target_flipped": bool(
                                    attacked["majority"][pos]
                                    != clean["majority"][target]
                                ),
                                "independent_prediction_changed": bool(
                                    attacked["base"][pos] != clean["base"][target]
                                ),
                                "cluster_changed": bool(
                                    attacked["cluster"][pos] != clean["cluster"][target]
                                ),
                                "collateral_flip_count": int(collateral_flips.sum()),
                                "collateral_evaluated": int(collateral_mask.sum()),
                                "collateral_flip_rate": (
                                    float(collateral_flips.mean())
                                    if collateral_mask.any()
                                    else 0.0
                                ),
                                "target_expression_sha256": target_hash_before,
                            }
                        )

            target_df = pd.DataFrame(rows)
            atomic_write_csv(target_df, csv_path)
            atomic_write_json(
                {
                    "status": "complete",
                    "target": target,
                    "target_obs_name": target_obs_name,
                    "rows": int(len(target_df)),
                },
                done_path,
            )
            all_frames.append(target_df)
            write_combined_outputs(out, all_frames)

    write_combined_outputs(out, all_frames)
    atomic_write_csv(pd.DataFrame(clean_cohort_rows), out / "clean_cohort_summary.csv")
    combined = (
        pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    )
    atomic_write_json(
        {
            "completed_targets": (
                int(combined[["seed", "target_index"]].drop_duplicates().shape[0])
                if not combined.empty
                else 0
            ),
            "evaluations": int(len(combined)),
            "seeds": [int(s) for s in cfg["seeds"]],
            "model": str(cfg["model"]),
            "runtime_seconds": float(time.time() - run_start),
            "resume_enabled": bool(cfg.get("resume", True)),
        },
        out / "summary.json",
    )


if __name__ == "__main__":
    main()
