from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .refinement import RefinementResult, refine_predictions, target_margin


@dataclass
class AuditContext:
    X: np.ndarray
    base_probabilities: np.ndarray
    classes: np.ndarray
    clean: RefinementResult
    k: int
    self_weight: float
    neighbor_weight: float
    confidence_weighted: bool


@dataclass
class RemovalOutcome:
    removed: np.ndarray
    target_label: str
    target_probability_original_class: float
    target_margin_original_class: float
    target_flipped: bool
    collateral_flip_rate: float


def evaluate_removal(
    ctx: AuditContext,
    target_idx: int,
    remove_idx: Iterable[int],
    collateral_sample: np.ndarray | None = None,
) -> RemovalOutcome:
    remove = np.unique(np.asarray(list(remove_idx), dtype=int))
    if np.any(remove == target_idx):
        raise ValueError("Target cell cannot be removed.")
    n = ctx.X.shape[0]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[remove] = False
    kept = np.flatnonzero(keep_mask)
    target_new_pos_arr = np.flatnonzero(kept == target_idx)
    if target_new_pos_arr.size != 1:
        raise RuntimeError("Target must remain exactly once.")
    target_new_pos = int(target_new_pos_arr[0])

    result = refine_predictions(
        ctx.X[kept],
        ctx.base_probabilities[kept],
        ctx.classes,
        k=ctx.k,
        self_weight=ctx.self_weight,
        neighbor_weight=ctx.neighbor_weight,
        confidence_weighted=ctx.confidence_weighted,
    )
    original_label = ctx.clean.labels[target_idx]
    original_class_pos = int(np.flatnonzero(ctx.classes == original_label)[0])
    new_label = result.labels[target_new_pos]

    if collateral_sample is None:
        collateral_original_idx = kept[kept != target_idx]
    else:
        collateral_original_idx = np.intersect1d(kept, collateral_sample, assume_unique=False)
        collateral_original_idx = collateral_original_idx[collateral_original_idx != target_idx]
    old_to_new = {old: new for new, old in enumerate(kept.tolist())}
    if collateral_original_idx.size:
        new_positions = np.asarray([old_to_new[int(i)] for i in collateral_original_idx], dtype=int)
        collateral_rate = float(np.mean(result.labels[new_positions] != ctx.clean.labels[collateral_original_idx]))
    else:
        collateral_rate = 0.0

    return RemovalOutcome(
        removed=remove,
        target_label=str(new_label),
        target_probability_original_class=float(result.probabilities[target_new_pos, original_class_pos]),
        target_margin_original_class=target_margin(result.probabilities, target_new_pos, original_class_pos),
        target_flipped=bool(new_label != original_label),
        collateral_flip_rate=collateral_rate,
    )


def choose_candidate_pool(ctx: AuditContext, target_idx: int, pool_size: int) -> np.ndarray:
    direct = ctx.clean.neighbor_indices[target_idx]
    second = ctx.clean.neighbor_indices[direct].reshape(-1) if direct.size else np.array([], dtype=int)
    pool = np.unique(np.concatenate([direct, second]))
    pool = pool[pool != target_idx]
    if pool.size >= pool_size:
        return pool[:pool_size]
    # Supplement with nearest points in feature space.
    dist = np.linalg.norm(ctx.X - ctx.X[target_idx], axis=1)
    nearest = np.argsort(dist)
    nearest = nearest[nearest != target_idx]
    return np.unique(np.concatenate([pool, nearest]))[:pool_size]


def greedy_worst_case_removal(
    ctx: AuditContext,
    target_idx: int,
    candidate_pool: np.ndarray,
    max_steps: int,
    group_size: int,
    collateral_sample: np.ndarray | None,
) -> list[RemovalOutcome]:
    selected: list[int] = []
    remaining = [int(i) for i in candidate_pool if int(i) != target_idx]
    trajectory: list[RemovalOutcome] = []

    for _ in range(max_steps):
        if not remaining:
            break
        # Evaluate contiguous candidate chunks to control runtime.
        groups = [remaining[i : i + group_size] for i in range(0, len(remaining), group_size)]
        scored: list[tuple[tuple[float, float, float], list[int], RemovalOutcome]] = []
        for group in groups:
            outcome = evaluate_removal(ctx, target_idx, selected + group, collateral_sample)
            score = (
                1.0 if outcome.target_flipped else 0.0,
                -outcome.target_margin_original_class,
                -outcome.collateral_flip_rate,
            )
            scored.append((score, group, outcome))
        scored.sort(key=lambda z: z[0], reverse=True)
        _, best_group, best_outcome = scored[0]
        selected.extend(best_group)
        remaining = [i for i in remaining if i not in best_group]
        trajectory.append(best_outcome)
        if best_outcome.target_flipped:
            break
    return trajectory



def _outcome_score(outcome: RemovalOutcome) -> tuple[float, float, float]:
    """Higher is better: prefer flips, then lower original-class margin, then lower collateral damage."""
    return (
        1.0 if outcome.target_flipped else 0.0,
        -outcome.target_margin_original_class,
        -outcome.collateral_flip_rate,
    )


def _partition_groups(items: list[int], group_size: int) -> list[list[int]]:
    return [items[i : i + group_size] for i in range(0, len(items), group_size) if items[i : i + group_size]]


def multistart_greedy_removal(
    ctx: AuditContext,
    target_idx: int,
    candidate_pool: np.ndarray,
    max_steps: int,
    group_size: int,
    collateral_sample: np.ndarray | None,
    n_starts: int,
    rng: np.random.Generator,
) -> tuple[list[RemovalOutcome], int]:
    """Run greedy search from several deterministic random candidate orderings.

    Returns the best trajectory and the restart index that produced it. A successful
    trajectory is ranked by fewer removals, then final score. Otherwise, the final
    outcome score is used.
    """
    base = [int(i) for i in candidate_pool if int(i) != target_idx]
    candidates: list[tuple[tuple[float, ...], int, list[RemovalOutcome]]] = []
    for start in range(max(1, n_starts)):
        order = base.copy()
        if start > 0:
            rng.shuffle(order)
        selected: list[int] = []
        remaining = order.copy()
        trajectory: list[RemovalOutcome] = []
        for _ in range(max_steps):
            if not remaining:
                break
            groups = _partition_groups(remaining, group_size)
            scored: list[tuple[tuple[float, float, float], list[int], RemovalOutcome]] = []
            for group in groups:
                outcome = evaluate_removal(ctx, target_idx, selected + group, collateral_sample)
                scored.append((_outcome_score(outcome), group, outcome))
            scored.sort(key=lambda z: z[0], reverse=True)
            _, best_group, best_outcome = scored[0]
            selected.extend(best_group)
            remaining = [i for i in remaining if i not in best_group]
            trajectory.append(best_outcome)
            if best_outcome.target_flipped:
                break
        if trajectory:
            final = trajectory[-1]
            if final.target_flipped:
                rank = (1.0, -float(len(final.removed)), *_outcome_score(final)[1:])
            else:
                rank = (0.0, -float(len(final.removed)), *_outcome_score(final)[1:])
            candidates.append((rank, start, trajectory))
    if not candidates:
        return [], 0
    candidates.sort(key=lambda z: z[0], reverse=True)
    _, start, trajectory = candidates[0]
    return trajectory, start


@dataclass
class _BeamState:
    selected: tuple[int, ...]
    used_groups: tuple[int, ...]
    trajectory: list[RemovalOutcome]
    outcome: RemovalOutcome | None


def beam_search_removal(
    ctx: AuditContext,
    target_idx: int,
    candidate_pool: np.ndarray,
    max_steps: int,
    group_size: int,
    collateral_sample: np.ndarray | None,
    beam_width: int = 4,
    branching_factor: int = 12,
    rng: np.random.Generator | None = None,
) -> list[RemovalOutcome]:
    """Beam search over candidate-cell groups.

    The candidate pool is partitioned into groups once. At each depth, each beam
    state expands using its best immediate candidate groups; the globally best
    states survive. The returned trajectory is the best successful path (fewest
    removals, then score), or the best non-successful path.
    """
    order = [int(i) for i in candidate_pool if int(i) != target_idx]
    if rng is not None:
        # Keep nearest-neighbour ordering as one component while breaking ties and
        # avoiding the same fixed contiguous grouping in every run.
        blocks = [order[i:i+group_size] for i in range(0, len(order), group_size)]
        rng.shuffle(blocks)
        order = [x for block in blocks for x in block]
    groups = _partition_groups(order, group_size)
    if not groups:
        return []

    beam = [_BeamState(selected=tuple(), used_groups=tuple(), trajectory=[], outcome=None)]
    successful: list[_BeamState] = []
    for _depth in range(max_steps):
        expanded: list[_BeamState] = []
        for state in beam:
            available = [g for g in range(len(groups)) if g not in state.used_groups]
            local: list[tuple[tuple[float, float, float], int, RemovalOutcome]] = []
            for gi in available:
                selected = list(state.selected) + groups[gi]
                outcome = evaluate_removal(ctx, target_idx, selected, collateral_sample)
                local.append((_outcome_score(outcome), gi, outcome))
            local.sort(key=lambda z: z[0], reverse=True)
            for _, gi, outcome in local[: max(1, branching_factor)]:
                new_state = _BeamState(
                    selected=tuple(list(state.selected) + groups[gi]),
                    used_groups=tuple(list(state.used_groups) + [gi]),
                    trajectory=state.trajectory + [outcome],
                    outcome=outcome,
                )
                if outcome.target_flipped:
                    successful.append(new_state)
                else:
                    expanded.append(new_state)
        if successful:
            successful.sort(
                key=lambda st: (-len(st.selected), *_outcome_score(st.outcome)),
                reverse=True,
            )
            return successful[0].trajectory
        if not expanded:
            break
        expanded.sort(key=lambda st: _outcome_score(st.outcome), reverse=True)
        beam = expanded[: max(1, beam_width)]

    final_states = beam + successful
    if not final_states:
        return []
    final_states.sort(key=lambda st: _outcome_score(st.outcome), reverse=True)
    return final_states[0].trajectory
