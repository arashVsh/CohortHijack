from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .audits import (
    AuditContext,
    beam_search_removal,
    choose_candidate_pool,
    evaluate_removal,
    greedy_worst_case_removal,
    multistart_greedy_removal,
)
from .data import filter_and_cap_classes, load_dataset, make_split, preprocess_features
from .models import fit_classifier
from .refinement import refine_predictions, target_margin
from .utils import (
    atomic_write_csv,
    atomic_write_json,
    save_environment,
    save_yaml,
    set_seed,
    stable_seed,
)


def _select_targets(
    y_true: np.ndarray,
    clean_labels: np.ndarray,
    clean_probs: np.ndarray,
    n_targets: int,
    policy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    correct = np.flatnonzero(clean_labels == y_true)
    if not correct.size:
        correct = np.arange(y_true.size)
    if policy == "vulnerable":
        confidence = clean_probs[correct].max(axis=1)
        order = np.argsort(confidence)
        pool = correct[order[: min(len(order), max(n_targets * 3, n_targets))]]
        if pool.size > n_targets:
            pool = rng.choice(pool, size=n_targets, replace=False)
        return np.sort(pool)
    if correct.size > n_targets:
        correct = rng.choice(correct, size=n_targets, replace=False)
    return np.sort(correct)


def _aggregate(per_target: pd.DataFrame) -> pd.DataFrame:
    if per_target.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "seed", "classifier", "audit", "budget"]
    if "run_label" in per_target.columns:
        group_cols.insert(0, "run_label")
    return (
        per_target.groupby(group_cols, dropna=False)
        .agg(
            n=("target_index", "size"),
            target_flip_rate=("target_flipped", "mean"),
            mean_removed_fraction=("removed_fraction", "mean"),
            mean_collateral_flip_rate=("collateral_flip_rate", "mean"),
            mean_original_class_probability=("target_probability_original_class", "mean"),
            mean_original_class_margin=("target_margin_original_class", "mean"),
        )
        .reset_index()
    )


def _checkpoint_dir(outdir: Path, dataset: str, seed: int, classifier: str) -> Path:
    return outdir / "checkpoints" / dataset / f"seed_{seed}" / classifier


def _target_checkpoint_paths(model_checkpoint_dir: Path, target_idx: int) -> tuple[Path, Path]:
    stem = f"target_{target_idx:06d}"
    return model_checkpoint_dir / f"{stem}.csv", model_checkpoint_dir / f"{stem}.done.json"


def _load_completed_target_rows(checkpoint_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not checkpoint_root.exists():
        return pd.DataFrame()
    for done_path in sorted(checkpoint_root.rglob("target_*.done.json")):
        csv_path = done_path.with_name(done_path.name.replace(".done.json", ".csv"))
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_clean_rows(checkpoint_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not checkpoint_root.exists():
        return pd.DataFrame()
    for path in sorted(checkpoint_root.rglob("clean_metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def _write_combined_outputs(outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_root = outdir / "checkpoints"
    per_target = _load_completed_target_rows(checkpoint_root)
    clean_df = _load_clean_rows(checkpoint_root)
    aggregate = _aggregate(per_target)
    atomic_write_csv(per_target, outdir / "per_target_results.csv")
    atomic_write_csv(clean_df, outdir / "clean_performance.csv")
    atomic_write_csv(aggregate, outdir / "aggregate_results.csv")
    return per_target, clean_df, aggregate


def _load_or_select_targets(
    cfg: dict[str, Any],
    dataset_name: str,
    seed: int,
    classifier_name: str,
    y_test: np.ndarray,
    clean_labels: np.ndarray,
    clean_probs: np.ndarray,
    model_checkpoint_dir: Path,
) -> np.ndarray:
    audit_cfg = cfg["audit"]
    manifest_root_value = audit_cfg.get("target_manifest_root")
    if manifest_root_value:
        manifest_path = (
            Path(manifest_root_value)
            / "checkpoints"
            / dataset_name
            / f"seed_{seed}"
            / classifier_name
            / "target_manifest.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Requested target manifest does not exist: {manifest_path}. "
                "Run the primary experiment first or remove audit.target_manifest_root."
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        targets = np.asarray(payload["targets"], dtype=int)
        requested = int(audit_cfg.get("n_targets_per_run", len(targets)))
        targets = targets[:requested]
        if np.any(targets < 0) or np.any(targets >= len(y_test)):
            raise ValueError(f"Target manifest {manifest_path} is incompatible with this split.")
    else:
        target_rng = np.random.default_rng(stable_seed("targets", dataset_name, seed, classifier_name))
        targets = _select_targets(
            y_test,
            clean_labels,
            clean_probs,
            int(audit_cfg["n_targets_per_run"]),
            str(audit_cfg.get("target_policy", "vulnerable")),
            target_rng,
        )
    atomic_write_json(
        {"targets": [int(x) for x in targets]},
        model_checkpoint_dir / "target_manifest.json",
    )
    return targets


def _enabled_audits(audit_cfg: dict[str, Any]) -> set[str]:
    enabled = audit_cfg.get("enabled", ["random", "same_class", "nearest", "greedy"])
    valid = {"random", "same_class", "nearest", "greedy", "multistart_greedy", "beam"}
    result = {str(x) for x in enabled}
    unknown = result - valid
    if unknown:
        raise ValueError(f"Unknown audits requested: {sorted(unknown)}")
    return result


def run_experiment(cfg: dict[str, Any]) -> Path:
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(exist_ok=True)
    resume = bool(cfg.get("runtime", {}).get("resume", True))

    if not resume and (outdir / "checkpoints").exists():
        shutil.rmtree(outdir / "checkpoints")

    save_yaml(cfg, outdir / "run_config.yaml")
    save_environment(outdir / "environment.json")

    for dataset_name in cfg["datasets"]:
        raw = load_dataset(dataset_name)
        for seed_value in cfg["seeds"]:
            seed = int(seed_value)
            set_seed(seed)
            data_rng = np.random.default_rng(stable_seed("data", dataset_name, seed))
            adata = filter_and_cap_classes(
                raw.adata,
                raw.label_key,
                int(cfg["min_class_cells"]),
                int(cfg["max_cells_per_class"]),
                seed,
            )
            adata = preprocess_features(adata, int(cfg["n_hvg"]), int(cfg["n_pcs"]), seed)
            X_train, X_test, y_train, y_test, test_original_idx = make_split(
                adata, raw.label_key, float(cfg["test_size"]), seed
            )

            for classifier_name in cfg["classifiers"]:
                model_checkpoint_dir = _checkpoint_dir(outdir, dataset_name, seed, classifier_name)
                model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

                model = fit_classifier(classifier_name, X_train, y_train, seed)
                base_probs = model.predict_proba(X_test)
                base_labels = model.classes_[np.argmax(base_probs, axis=1)]
                ref_cfg = cfg["refinement"]
                clean = refine_predictions(
                    X_test,
                    base_probs,
                    model.classes_,
                    k=int(ref_cfg["k"]),
                    self_weight=float(ref_cfg["self_weight"]),
                    neighbor_weight=float(ref_cfg["neighbor_weight"]),
                    confidence_weighted=bool(ref_cfg["confidence_weighted"]),
                )

                clean_row = {
                    "run_label": str(cfg.get("run_label", cfg.get("experiment_name", "experiment"))),
                    "dataset": dataset_name,
                    "seed": seed,
                    "classifier": classifier_name,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_classes": len(np.unique(y_train)),
                    "base_accuracy": accuracy_score(y_test, base_labels),
                    "base_macro_f1": f1_score(y_test, base_labels, average="macro", zero_division=0),
                    "refined_accuracy": accuracy_score(y_test, clean.labels),
                    "refined_macro_f1": f1_score(y_test, clean.labels, average="macro", zero_division=0),
                    "base_refined_disagreement": np.mean(base_labels != clean.labels),
                }
                atomic_write_json(clean_row, model_checkpoint_dir / "clean_metrics.json")

                if cfg["runtime"].get("save_models", False):
                    model_path = outdir / "models" / f"{dataset_name}_{seed}_{classifier_name}.joblib"
                    tmp_model_path = model_path.with_suffix(model_path.suffix + ".tmp")
                    joblib.dump(model.estimator, tmp_model_path)
                    tmp_model_path.replace(model_path)

                ctx = AuditContext(
                    X=X_test,
                    base_probabilities=base_probs,
                    classes=model.classes_,
                    clean=clean,
                    k=int(ref_cfg["k"]),
                    self_weight=float(ref_cfg["self_weight"]),
                    neighbor_weight=float(ref_cfg["neighbor_weight"]),
                    confidence_weighted=bool(ref_cfg["confidence_weighted"]),
                )
                audit_cfg = cfg["audit"]
                enabled_audits = _enabled_audits(audit_cfg)
                targets = _load_or_select_targets(
                    cfg, dataset_name, seed, classifier_name, y_test, clean.labels,
                    clean.probabilities, model_checkpoint_dir
                )

                for target_idx_value in tqdm(
                    targets,
                    desc=f"{dataset_name}/{seed}/{classifier_name}",
                    leave=False,
                ):
                    target_idx = int(target_idx_value)
                    target_csv, target_done = _target_checkpoint_paths(model_checkpoint_dir, target_idx)
                    if resume and target_done.exists() and target_csv.exists():
                        continue

                    # A target-specific RNG makes a resumed run bitwise reproducible:
                    # skipping completed targets cannot change later random samples.
                    rng = np.random.default_rng(
                        stable_seed("audit", dataset_name, seed, classifier_name, target_idx)
                    )
                    target_rows: list[dict[str, Any]] = []
                    original_label = str(clean.labels[target_idx])
                    original_class_pos = int(np.flatnonzero(model.classes_ == original_label)[0])
                    n_context = len(y_test) - 1
                    all_non_target = np.delete(np.arange(len(y_test)), target_idx)
                    collateral_n = min(int(audit_cfg["collateral_sample_size"]), all_non_target.size)
                    collateral_sample = rng.choice(all_non_target, size=collateral_n, replace=False)

                    common = {
                        "run_label": str(cfg.get("run_label", cfg.get("experiment_name", "experiment"))),
                        "dataset": dataset_name,
                        "seed": seed,
                        "classifier": classifier_name,
                        "target_index": target_idx,
                        "target_original_dataset_index": int(test_original_idx[target_idx]),
                        "target_true_label": str(y_test[target_idx]),
                        "target_clean_label": original_label,
                        "target_base_label": str(base_labels[target_idx]),
                        "target_clean_confidence": float(clean.probabilities[target_idx].max()),
                        "target_clean_margin": target_margin(clean.probabilities, target_idx, original_class_pos),
                        "target_class_frequency": int(np.sum(y_test == y_test[target_idx])),
                    }

                    for budget_value in audit_cfg["budgets"]:
                        budget = float(budget_value)
                        remove_n = max(1, int(round(budget * n_context)))
                        remove_n = min(remove_n, n_context)

                        if "random" in enabled_audits:
                            for rep in range(int(audit_cfg.get("random_repeats", 0))):
                                removed = rng.choice(all_non_target, size=remove_n, replace=False)
                                outcome = evaluate_removal(ctx, target_idx, removed, collateral_sample)
                                target_rows.append({**common, "audit": "random", "repeat": rep, "budget": budget,
                                    "removed_count": len(outcome.removed), "removed_fraction": len(outcome.removed)/n_context,
                                    "target_new_label": outcome.target_label,
                                    "target_probability_original_class": outcome.target_probability_original_class,
                                    "target_margin_original_class": outcome.target_margin_original_class,
                                    "target_flipped": outcome.target_flipped,
                                    "collateral_flip_rate": outcome.collateral_flip_rate})

                        if "same_class" in enabled_audits:
                            same = all_non_target[clean.labels[all_non_target] == original_label]
                            if same.size:
                                chosen = rng.choice(same, size=min(remove_n, same.size), replace=False)
                                outcome = evaluate_removal(ctx, target_idx, chosen, collateral_sample)
                                target_rows.append({**common, "audit": "same_class", "repeat": 0, "budget": budget,
                                    "removed_count": len(outcome.removed), "removed_fraction": len(outcome.removed)/n_context,
                                    "target_new_label": outcome.target_label,
                                    "target_probability_original_class": outcome.target_probability_original_class,
                                    "target_margin_original_class": outcome.target_margin_original_class,
                                    "target_flipped": outcome.target_flipped,
                                    "collateral_flip_rate": outcome.collateral_flip_rate})

                        if "nearest" in enabled_audits:
                            dist = np.linalg.norm(X_test - X_test[target_idx], axis=1)
                            nearest = np.argsort(dist)
                            nearest = nearest[nearest != target_idx][:remove_n]
                            outcome = evaluate_removal(ctx, target_idx, nearest, collateral_sample)
                            target_rows.append({**common, "audit": "nearest", "repeat": 0, "budget": budget,
                                "removed_count": len(outcome.removed), "removed_fraction": len(outcome.removed)/n_context,
                                "target_new_label": outcome.target_label,
                                "target_probability_original_class": outcome.target_probability_original_class,
                                "target_margin_original_class": outcome.target_margin_original_class,
                                "target_flipped": outcome.target_flipped,
                                "collateral_flip_rate": outcome.collateral_flip_rate})

                    def _search_limits() -> tuple[np.ndarray, int, int]:
                        pool = choose_candidate_pool(ctx, target_idx, int(audit_cfg["candidate_pool_size"]))
                        group_size = int(audit_cfg["greedy_group_size"])
                        max_steps = int(audit_cfg["max_greedy_steps"])
                        max_fraction = audit_cfg.get("max_removed_fraction")
                        if max_fraction is not None:
                            fraction_steps = max(1, int(np.ceil(float(max_fraction) * n_context / group_size)))
                            max_steps = min(max_steps, fraction_steps)
                        return pool, group_size, max_steps

                    def _append_trajectory(audit_name: str, trajectory, extra: dict[str, Any] | None = None) -> None:
                        extra = extra or {}
                        for step, outcome in enumerate(trajectory, start=1):
                            target_rows.append({**common, **extra, "audit": audit_name, "repeat": step,
                                "budget": len(outcome.removed)/n_context,
                                "removed_count": len(outcome.removed), "removed_fraction": len(outcome.removed)/n_context,
                                "target_new_label": outcome.target_label,
                                "target_probability_original_class": outcome.target_probability_original_class,
                                "target_margin_original_class": outcome.target_margin_original_class,
                                "target_flipped": outcome.target_flipped,
                                "collateral_flip_rate": outcome.collateral_flip_rate})

                    if "greedy" in enabled_audits:
                        pool, group_size, max_steps = _search_limits()
                        trajectory = greedy_worst_case_removal(
                            ctx, target_idx, pool, max_steps, group_size, collateral_sample
                        )
                        _append_trajectory("greedy", trajectory)

                    if "multistart_greedy" in enabled_audits:
                        pool, group_size, max_steps = _search_limits()
                        search_rng = np.random.default_rng(
                            stable_seed("multistart", dataset_name, seed, classifier_name, target_idx)
                        )
                        trajectory, chosen_start = multistart_greedy_removal(
                            ctx, target_idx, pool, max_steps, group_size, collateral_sample,
                            n_starts=int(audit_cfg.get("multistart_restarts", 8)), rng=search_rng,
                        )
                        _append_trajectory("multistart_greedy", trajectory, {"search_restart": chosen_start})

                    if "beam" in enabled_audits:
                        pool, group_size, max_steps = _search_limits()
                        search_rng = np.random.default_rng(
                            stable_seed("beam", dataset_name, seed, classifier_name, target_idx)
                        )
                        trajectory = beam_search_removal(
                            ctx, target_idx, pool, max_steps, group_size, collateral_sample,
                            beam_width=int(audit_cfg.get("beam_width", 4)),
                            branching_factor=int(audit_cfg.get("beam_branching_factor", 12)),
                            rng=search_rng,
                        )
                        _append_trajectory("beam", trajectory)

                    target_df = pd.DataFrame(target_rows)
                    atomic_write_csv(target_df, target_csv)
                    atomic_write_json(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "classifier": classifier_name,
                            "target_index": target_idx,
                            "rows": len(target_df),
                            "status": "complete",
                        },
                        target_done,
                    )
                    # Refresh human-friendly combined CSVs after every completed target.
                    _write_combined_outputs(outdir)

                atomic_write_json(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "classifier": classifier_name,
                        "n_targets": len(targets),
                        "status": "complete",
                    },
                    model_checkpoint_dir / "model_run.done.json",
                )
                _write_combined_outputs(outdir)

    per_target, clean_df, _ = _write_combined_outputs(outdir)
    summary = {
        "runs": len(clean_df),
        "audit_rows": len(per_target),
        "completed_targets": int(per_target[["dataset", "seed", "classifier", "target_index"]].drop_duplicates().shape[0]) if not per_target.empty else 0,
        "datasets": cfg["datasets"],
        "seeds": cfg["seeds"],
        "classifiers": cfg["classifiers"],
        "resume_enabled": resume,
    }
    atomic_write_json(summary, outdir / "summary.json")
    return outdir
