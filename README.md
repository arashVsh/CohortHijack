# ContextCell publication analysis

## Run the statistical analysis

```powershell
python scripts/run_essential_analysis.py `
  --main outputs/iccke `
  --search-v2 outputs/search_v2 `
  --ablations outputs/publication_ablations `
  --celltypist outputs/celltypist_validation_v2 `
  --output-dir outputs/publication_analysis
```

## Generate three paper figures

```powershell
python scripts/make_paper_figures.py `
  --analysis-dir outputs/publication_analysis `
  --output-dir outputs/paper_figures
```

The script writes both 300-dpi PNG and vector PDF versions.

## Recommended paper figures

1. `figure1_structured_vs_random`: central controlled-pipeline result.
2. `figure2_context_weight_ablation`: mechanism/negative-control result.
3. `figure3_celltypist_validation`: real-tool external validation.

For a two-figure paper, use Figures 1 and 3, and move Figure 2 to an ablation table.
