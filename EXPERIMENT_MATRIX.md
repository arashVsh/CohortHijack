# ICCKE experiment matrix

## Core factorial design

| Dimension | Levels |
|---|---|
| Dataset | PBMC3K, Paul15 hematopoiesis |
| Seed | 13, 37, 73 |
| Classifier | Logistic regression, calibrated linear SVM |
| Prediction | Independent base, cohort-refined kNN |
| Audit | Random, same-class, nearest-neighbour, greedy worst-case |
| Random budget | 0.5%, 1%, 2%, 5%, 10% |

This yields 12 primary model runs and at least 6 clean-performance comparisons per dataset.

## Required tables

1. Dataset statistics after filtering.
2. Clean base and refined Accuracy/Macro-F1, mean ± standard deviation over seeds.
3. Target flip rate by audit and budget.
4. Mean collateral flip rate by audit and budget.
5. Greedy minimum discovered removal budget among successful targets.
6. Results stratified by class frequency and clean confidence quartile.

## Recommended ablations

1. k = 10, 25, 50.
2. confidence-weighted versus unweighted neighbours.
3. self:neighbour weight = 2:1, 1:1, 1:2.
4. vulnerable-target selection versus uniform target selection.
5. candidate pool sizes 50, 120, 250.

## Statistical analysis

- Report mean and standard deviation across three seeds.
- For paired random-versus-greedy target outcomes, use a paired bootstrap confidence interval on the flip-rate difference.
- For successful attacks, report median and IQR of removal budgets.
- Correct multiple per-class comparisons with Benjamini-Hochberg.

## Added publication follow-ups

| Study | Datasets | Classifiers | Seeds | Targets | Purpose |
|---|---|---:|---:|---:|---|
| Matched greedy | Paul15 | 2 | 3 | 100/run | Compare greedy success at up to 10% removal with structured baselines |
| Neighborhood-size ablation | PBMC3K, Paul15 | 2 | 3 | 50/run | Test sensitivity to `k=10,25,50` |
| Context-weight ablation | PBMC3K, Paul15 | 2 | 3 | 50/run | Establish the no-context negative control and dose response |
| Weighting ablation | PBMC3K, Paul15 | 2 | 3 | 50/run | Compare confidence-weighted and uniform neighbor aggregation |
| Search-pool ablation | Paul15 | 2 | 3 | 50/run | Compare pools of 60, 120, and 240 candidates |
| Group-size ablation | Paul15 | 2 | 3 | 50/run | Compare group sizes 3 and 5 under matched removal constraints |

The `neighbor_weight=0` condition is the key negative control: with no contextual contribution, companion-cell removal should not alter the protected target's prediction.
