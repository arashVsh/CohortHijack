# Publication-completion experiments

This version adds the remaining experiments needed to complete the ICCKE evidence package without rerunning completed primary or Search V2 outputs.

## A. Mechanism ablations (publication critical)

These reuse all original target manifests and evaluate both datasets, both classifiers, three seeds, and 100 paired targets per run.

```powershell
python scripts/run_publication_ablations.py --config configs/publication_ablations.yaml
```

Variants:

- `context_weight_0`: negative control. Companion removal must produce zero target flips because refinement ignores neighbors.
- `context_weight_0p5`, `context_weight_2`: dose-response of contextual dependence.
- `refine_k10`, `refine_k50`: neighborhood-size sensitivity.
- `uniform_neighbor_voting`: confidence-weighting ablation.

Run highest-priority controls first:

```powershell
python scripts/run_publication_ablations.py --config configs/publication_ablations.yaml --only context_weight_0 context_weight_0p5 context_weight_2
```

## B. Search-design ablations

These use Paul15, both classifiers, all three seeds, and the first 50 original paired targets.

```powershell
python scripts/run_publication_search_ablations.py --config configs/publication_search_ablations.yaml
```

Variants isolate restart count and removal-group size:

- 1, 4, and 16 multi-start trajectories;
- group sizes 1 and 5, compared with the Search V2 default of 3 and 8 restarts.

## C. Biological and statistical analysis

No new model runs are required.

```powershell
python scripts/analyze_publication_results.py `
  --inputs outputs/iccke outputs/search_v2 outputs/publication_ablations/context_weight_0 `
  --output-dir outputs/publication_analysis
```

The script produces bootstrap 95% confidence intervals, per-cell-type vulnerability, class-frequency and confidence correlations, and within-lineage versus cross-lineage transition summaries.

## D. Optional real CellTypist validation

Install the optional dependency:

```powershell
pip install celltypist
```

Then run:

```powershell
python scripts/run_celltypist_validation.py --config configs/celltypist_validation.yaml
```

This audit uses CellTypist's actual majority-voting mode on normalized PBMC3K data. It keeps each target cell unchanged and reruns annotation after random, same-majority-label, or nearest-cell removal. It is deliberately small (30 paired targets total) because CellTypist recomputes over-clustering for every modified cohort.

CellTypist validation downloads the official `Immune_All_Low.pkl` model on first use and therefore requires internet access.

## Crash recovery

Every experiment checkpoints after each target. Restart the same command after interruption. Never pass `--fresh` unless intentionally discarding that experiment's checkpoints.
