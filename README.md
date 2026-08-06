# CohortHijack

**Robustness of single cell annotation to companion cell removal**

CohortHijack is a target preserving robustness audit for single cell RNA sequencing annotation. It tests whether the refined label of a target cell can be changed by removing a small set of other cells from the query cohort.

The target cell is never edited or removed. Its expression vector, base classifier, model parameters, and independent prediction remain fixed. Only the cells available to neighborhood or cluster based refinement are changed.

We use **audit** for the complete evaluation protocol and **attack** for a specific removal set selected to change a target label.

## Main findings

The released implementation supports the experiments reported in the paper:

- PBMC3K and Paul15 datasets loaded through Scanpy
- multinomial logistic regression and calibrated linear SVM classifiers
- random, nearest cell, and same class removal
- greedy, multi-start greedy, and beam search
- context weight, neighborhood size, and voting rule ablations
- CellTypist majority voting validation
- crash safe target level checkpointing and deterministic resume

In the completed experiments, structured removals were stronger than random removal on Paul15. Multi-start search changed 24.33% of linear SVM targets and 19.67% of logistic regression targets, with median removal fractions of 0.64% and 1.27%. Mean collateral changes remained below 0.4% in the controlled pipeline. When neighborhood refinement was disabled, no target flips occurred. CellTypist independent predictions remained unchanged across all validation runs, while majority voted labels sometimes changed after small cohort removals.

## Figures

The repository includes the three paper figures as PDF files:

### CohortHijack overview

[Open the overview figure](Figures/CohortHijack_overview.png)

This figure contrasts ordinary cohort based annotation with a CohortHijack audit. It shows that the target and its independent prediction remain unchanged while the refined label can change after companion cell removal.

### Structured removal versus random removal

[Open the main results figure](Figures/figure1_structured_vs_random.png)

This figure compares random, nearest cell, and same class removal at the 5% budget across both datasets and classifiers.

### Context weight ablation

[Open the context weight figure](Figures/figure_context_weight_compact.png)

This figure shows the Paul15 same class removal results as the contribution of neighboring cells increases.

## Repository structure

```text
CohortHijack/
├── configs/                 Experiment configurations
├── contextcell/             Core data, model, refinement, and audit code
├── Figures/                 Paper figures in PDF format
├── scripts/                 Experiment, analysis, and plotting commands
├── tests/                   Invariant and safety tests
├── data/                    Downloaded datasets, ignored by Git
├── outputs/                 Generated results, ignored by Git
├── requirements.txt
└── README.md
```

## Installation

Python 3.10, 3.11, or 3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the installation:

```bash
pytest -q
```

The Scanpy datasets are downloaded automatically on first use. CellTypist also downloads the configured pretrained model on its first validation run.

## Quick start

Run the smoke configuration before launching a full experiment:

```bash
python scripts/run_experiments.py --config configs/smoke.yaml
```

The run writes its results under `outputs/`. If it finishes successfully, the environment is ready for the main experiments.

## Main controlled experiment

Run the complete PBMC3K and Paul15 audit:

```bash
python scripts/run_experiments.py --config configs/iccke.yaml
```

The command evaluates the configured datasets, classifiers, seeds, targets, removal strategies, and budgets.

Create the standard result plots:

```bash
python scripts/make_plots.py --results outputs/iccke
```

### Resume after interruption

The runner saves an atomic checkpoint after every completed target. To resume, run the same command again:

```bash
python scripts/run_experiments.py --config configs/iccke.yaml
```

Completed targets are skipped automatically. Do not use `--fresh` when resuming.

To intentionally discard checkpoints and rerun the configuration:

```bash
python scripts/run_experiments.py --config configs/iccke.yaml --fresh
```

## Search V2

Run a short test first:

```bash
python scripts/run_experiments.py --config configs/search_v2_smoke.yaml
```

Run the full multi-start greedy and beam search evaluation:

```bash
python scripts/run_experiments.py --config configs/search_v2.yaml
```

Summarize the completed search runs:

```bash
python scripts/summarize_search_v2.py \
  --primary outputs/iccke \
  --followup outputs/followup_matched_greedy \
  --search-v2 outputs/search_v2 \
  --output outputs/search_v2_summary
```

PowerShell users can place the command on one line or use the backtick character for line continuation.

## Mechanism ablations

Run the configured ablation suite:

```bash
python scripts/run_ablation_suite.py --config configs/ablation_suite.yaml
```

Run selected variants only:

```bash
python scripts/run_ablation_suite.py \
  --config configs/ablation_suite.yaml \
  --only refine_k10 context_weight_0 context_weight_2 unweighted_neighbors
```

The ablations vary neighborhood size, context weight, voting rule, candidate pool size, and search group size while reusing the main target manifests.

## CellTypist validation

Run the real tool validation with CellTypist majority voting:

```bash
python scripts/run_celltypist_validation_v2.py \
  --config configs/celltypist_validation_v2.yaml
```

This experiment:

1. annotates PBMC3K using the configured CellTypist immune model
2. selects context sensitive and initially stable targets
3. removes random, nearest, or same label companion cells
4. reruns over-clustering and majority voting
5. verifies that the target expression hash and independent prediction remain unchanged
6. records target and collateral majority vote changes

The run is checkpointed per target and can be resumed with the same command.

## Publication analysis

Analyze one or more completed output directories:

```bash
python scripts/analyze_publication_results.py \
  --inputs outputs/iccke outputs/search_v2 \
  --output-dir outputs/publication_analysis
```

The exact available tables depend on which experiments have been completed. Generated files may include target level summaries, aggregate flip rates, confidence intervals, class level vulnerability, and correlation analyses.

## Key output files

A controlled experiment output directory typically contains:

```text
outputs/iccke/
├── run_config.yaml
├── environment.json
├── clean_performance.csv
├── per_target_results.csv
├── aggregate_results.csv
├── figures/
├── models/
└── checkpoints/
```

Important files:

- `clean_performance.csv`: base and refined clean metrics
- `per_target_results.csv`: intervention level target results
- `aggregate_results.csv`: grouped summaries across targets
- `environment.json`: software and runtime information
- `checkpoints/`: target manifests and crash safe completion markers

## Reproducibility and invariants

The implementation enforces the following conditions:

- the target is excluded from every removal set
- the target feature vector is checked before and after intervention
- trained base models are not updated during an audit
- base probabilities are reused after removal
- only the query neighborhood or clustering stage is recomputed
- random splitting, target selection, and removal sampling are seeded
- the same target manifests are reused for matched follow-up experiments

## Data and generated files

Datasets, trained models, checkpoints, and generated outputs should not be committed to Git. The source repository should contain the code, configurations, tests, documentation, and selected paper figures only.

## Citation

A BibTeX entry will be added after the paper receives its final publication information. Until then, please cite the repository and the accompanying preprint or conference submission.

## License

See [LICENSE](LICENSE).
