# ContextCell Search V2

## Motivation
The pool-240 follow-up reduced success despite a larger budget, showing that one-step greedy choices are path-dependent. Search V2 tests whether exploring multiple trajectories recovers successful sparse removals.

## Methods
- **Multi-start greedy:** eight deterministic candidate-group orderings; retain the best trajectory, prioritizing a successful flip with the fewest removed cells.
- **Beam search:** beam width 4, branching factor 10, group size 3, up to 5% removal.
- **Paired evaluation:** Paul15, logistic regression and calibrated linear SVM, seeds 13/37/73, original 100 target manifests.
- **Safety/reproducibility:** target never removed or modified; frozen model and PCA representation; target-level atomic checkpoints.

## Commands
Smoke test:

```powershell
python scripts/run_experiments.py --config configs/search_v2_smoke.yaml
```

Full run:

```powershell
python scripts/run_experiments.py --config configs/search_v2.yaml
```

Resume after interruption by running the same command. Do not use `--fresh`.

Summary:

```powershell
python scripts/summarize_search_v2.py
```

## Primary comparisons
Compare final per-target outcomes for original greedy, pool-240 greedy, multi-start greedy, and beam search. Report success rate, success at <=1%, <=2%, and <=5% removal, minimum successful removal fraction, collateral flip rate, and paired target wins/losses.
