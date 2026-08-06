# CellTypist Majority-Voting Validation V2

This experiment validates the CohortHijack threat model using CellTypist's actual
query-cohort over-clustering and majority-voting refinement.

## Design

- Dataset: PBMC3K, deterministically balanced to at most 1,200 cells.
- Model: `Immune_All_Low.pkl`.
- Seeds: 13, 37, 73.
- Targets per seed: 10 cells changed by clean majority voting and 10 cells left
  unchanged by clean majority voting.
- Removal budgets: 1% and 2% of non-target query cells.
- Strategies: random, same-majority-label, and nearest-cell removal.
- Random repeats: 5 per target and budget.

The target expression vector is hashed before and after every intervention. The
run aborts if the target changes. It also records whether the independent
CellTypist prediction changes; this should remain zero because only the query
cohort context is modified.

## Install

```powershell
pip install celltypist
```

## Run

```powershell
python scripts/run_celltypist_validation_v2.py --config configs/celltypist_validation_v2.yaml
```

Restart the same command after interruption. Completed targets are skipped.

## Outputs

- `outputs/celltypist_validation_v2/per_target_results.csv`
- `outputs/celltypist_validation_v2/aggregate_results.csv`
- `outputs/celltypist_validation_v2/aggregate_by_seed.csv`
- `outputs/celltypist_validation_v2/summary.json`
- target-level checkpoints under `outputs/celltypist_validation_v2/checkpoints/`

## Primary checks

1. Structured removal should exceed random removal.
2. Majority-voted labels may change while independent CellTypist labels remain
   unchanged.
3. Changed-by-majority-vote targets are expected to be more vulnerable than
   stable targets.
4. Collateral flip rates should remain small relative to target flip rates.
