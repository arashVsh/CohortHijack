# Reviewer-driven revision material

This file provides manuscript-ready replacements grounded in the released result
files. The repository does not contain the LaTeX source for `Paper.pdf`, so these
edits could not be applied directly to the typeset manuscript.

## Scope correction for the abstract

Replace the quantitative part of the abstract with:

> In a targeted robustness audit on correctly annotated, lower-confidence Paul15
> cells, the Search V2 multi-start procedure changed 24.33% of 300 linear-SVM
> targets (95% target-bootstrap CI, 19.67%-29.33%) and 19.67% of 300
> logistic-regression targets (95% CI, 15.33%-24.33%). These are
> search-optimized vulnerability rates for the prespecified audit subset, not
> estimates of failure prevalence in an unconstrained cell population. Successful
> searches required median removals of 0.64% and 1.27%, respectively, and had mean
> collateral label-change rates below 0.4%. In a separate CellTypist majority-vote
> experiment, the target's independent prediction was unchanged in all 840
> intervention evaluations. Majority-voted labels were most sensitive in the
> deliberately selected context-sensitive stratum; post-stratification to the
> observed clean-cohort stratum prevalences yielded all-cell intervention flip-rate
> estimates of 10.45%-18.53% across the tested strategies and 1%-2% budgets. These
> findings establish a target-preserving failure mode and motivate stability
> reporting, but do not imply that routine unperturbed cohorts fail at the targeted
> Search V2 rates.

## Refinement equation clarification

Replace the paragraph around Equations (1)-(2) with:

> Let \(p_{ic}\) be the frozen base-classifier probability for cell \(i\) and
> class \(c\), and let \(q_j=\max_d p_{jd}\) be neighbor \(j\)'s base confidence.
> The confidence-weighted neighborhood distribution is
> \(h_{ic}=\sum_{j\in N_k(i)}q_jp_{jc}/\sum_{j\in N_k(i)}q_j\). For uniform
> refinement, every \(q_j\) is set to one. Because both \(p_i\) and \(h_i\) sum to
> one, the implemented mixture is
> \(r_{ic}=(\alpha p_{ic}+\lambda h_{ic})/(\alpha+\lambda)\), for non-negative
> \(\alpha,\lambda\) with \(\alpha+\lambda>0\). The final label is
> \(\arg\max_c r_{ic}\), with NumPy's deterministic first-index rule for an exact
> tie. The main experiment sets \(\alpha=\lambda=1\). Setting \(\lambda=0\)
> recovers the frozen base prediction and makes companion-cell removal unable to
> change the controlled pipeline's output.

This wording is algebraically equivalent to the implementation in
`contextcell/refinement.py` and removes ambiguity about normalization and the role
of neighbor confidence.

## Target selection and estimands

Add at the end of the target-selection subsection:

> The sampling design is intentionally outcome-focused. Within every dataset,
> classifier, and seed, targets were restricted to cells that were correctly
> classified after clean refinement and then sampled from the lower-confidence
> portion of that eligible set. Accordingly, the primary estimand is the
> intervention flip rate among selected lower-confidence targets under a specified
> audit or search configuration. It is not the natural prevalence of label changes
> among all cells. Search V2 further optimizes the removal set separately for each
> target, so its success rate is a worst-case audit quantity. We report it separately
> from non-search removal results and from post-stratified CellTypist estimates.

## Statistical analysis subsection

Add this subsection to Methods:

> **Statistical analysis.** The inferential unit was the target cell, uniquely
> identified within a dataset/classifier run by its seed and target index. Repeated
> random-removal outcomes were averaged within target before analysis; removal
> repeats were never treated as independent observations. Primary rates and mean
> collateral/removal fractions use two-sided percentile 95% confidence intervals
> from 10,000 non-parametric bootstrap samples of target cells (fixed bootstrap
> seed 20261003). Confidence intervals for median successful removal fractions
> bootstrap successful target cells. At the prespecified 5% budget, each structured
> method was compared with the random baseline on the same targets using a
> two-sided Wilcoxon signed-rank test with Pratt handling of zero differences and
> the normal approximation. The null hypothesis was that the median paired
> structured-minus-random target-flip propensity equaled zero. We report the exact
> numerical p-value returned by the test, Cohen's \(d_z\), and the matched
> rank-biserial correlation. Benjamini-Hochberg correction controlled the false
> discovery rate across the eight prespecified structured-versus-random
> comparisons. Per-class summaries and search trajectories were treated as
> descriptive and were not subjected to unreported marker-, category-, or
> iteration-wise significance testing.

## Revised main companion-cell removal results

Use the following text:

> At the 5% budget on the selected lower-confidence Paul15 targets, linear-SVM
> flip rates were 1.63% for random removal (95% CI, 0.83%-2.57%), 11.33% for
> nearest-cell removal (8.00%-15.00%), and 17.39% for same-class removal
> (13.04%-21.74%). Logistic-regression rates were 1.37% (0.63%-2.27%), 9.33%
> (6.00%-12.67%), and 15.00% (11.33%-19.33%), respectively. In matched target-level
> tests, the nearest-minus-random difference was 9.70 percentage points for linear
> SVM (\(d_z=0.310\), matched rank-biserial \(r=0.915\), raw \(p=0.00003237\),
> BH-adjusted \(p=0.00005989\)) and 7.97 points for logistic regression
> (\(d_z=0.295\), \(r=0.951\), raw \(p=0.00001657\), adjusted
> \(p=0.00004419\)). The same-class-minus-random difference was 15.82 points for
> linear SVM (\(d_z=0.447\), \(r=0.999\), raw \(p=2.16\times10^{-12}\), adjusted
> \(p=1.72\times10^{-11}\)) and 13.63 points for logistic regression
> (\(d_z=0.408\), \(r=0.998\), raw \(p=7.68\times10^{-11}\), adjusted
> \(p=3.07\times10^{-10}\)). These comparisons concern the targeted audit set.

The complete table also shows that, after correction, the PBMC3K linear-SVM
comparisons remained significant, whereas the PBMC3K logistic-regression
comparisons did not (adjusted \(p=0.0520\) and \(0.0833\)).

## Revised Search V2 paragraph

> Search V2 was evaluated only on Paul15 and only on 300 selected
> lower-confidence targets per classifier across three seeds. Multi-start greedy
> search changed 24.33% of linear-SVM targets (95% target-bootstrap CI,
> 19.67%-29.33%) and 19.67% of logistic-regression targets (15.33%-24.33%). Beam
> search changed 19.33% (15.00%-24.00%) and 15.67% (11.67%-20.00%), respectively.
> Among successful multi-start attacks, median removal fractions were 0.64% and
> 1.27%, and mean collateral change rates were 0.389% and 0.390%. These quantities
> characterize success under the stated search budget and selected-target policy;
> they are not baseline rates for an unperturbed or randomly sampled population.

## Revised CellTypist paragraph

> The CellTypist validation deliberately sampled 30 targets whose clean
> independent and majority-voted labels disagreed and 30 whose labels agreed.
> Consequently, the 26.67%-50.00% rates in the context-sensitive stratum are
> conditional, targeted estimates. The context-sensitive cells comprised 26.33%-
> 29.42% of the three complete 1,200-cell clean query cohorts. Weighting the two
> stratum-specific estimates by these observed prevalences gave post-stratified
> all-cell intervention flip rates of 10.45% (95% stratified target-bootstrap CI,
> 6.23%-15.20%) for random removal at 1%, 11.57% (7.11%-16.72%) for random removal
> at 2%, 15.40% (9.37%-22.27%) and 18.53% (10.43%-27.71%) for nearest-cell removal,
> and 17.76% (10.32%-26.38%) and 16.39% (10.19%-23.30%) for same-majority-label
> removal. These remain intervention-based estimates rather than natural
> unperturbed failure rates. Across all 840 evaluations, the independent CellTypist
> prediction never changed; the observed changes were confined to over-clustering
> and majority voting.

## Revised conclusion

> CohortHijack establishes the feasibility of a target-preserving failure mode in
> which companion-cell removal changes a cohort-refined annotation without altering
> the target expression vector or frozen base prediction. The largest rates in this
> study arise from Search V2 on selected lower-confidence Paul15 targets and should
> be interpreted as worst-case audit success rates, not population-wide pipeline
> failure probabilities. Non-search and post-stratified CellTypist analyses show a
> more qualified pattern: susceptibility varies substantially by dataset,
> classifier, target stratum, and refinement mechanism. These findings support
> reporting independent and refined labels separately and evaluating per-cell label
> stability under controlled cohort perturbations. Broader claims require
> representative sampling, additional datasets and annotation systems, and external
> validation.

## Figure, table, and reference presentation corrections

The rendered PDF shows two presentation issues that should be corrected in the
LaTeX source before resubmission:

1. Hyperlinks around figure and table references are rendered as red boxes. Use
   `\hypersetup{hidelinks}` (or the venue's approved link style) so cross-references
   do not appear as proofreading marks.
2. Figure 2's four diagonal x-axis labels are crowded at column width. Use two-line
   labels (dataset on the first line, classifier on the second), reduce the rotation,
   and add the target-bootstrap 95% confidence intervals from
   `main_metrics_target_bootstrap_ci.csv` as error bars.

Table I should be retitled "Search V2 results on selected lower-confidence Paul15
targets" and should add `n=300 per classifier` plus 95% CIs for success. Table II
should state `n=30 targets per stratum` and label its rates as stratum-conditional.
All figure/table callouts should be generated with `\label`/`\ref` rather than
hard-coded numbering. The reference list should also be checked against the target
venue's policy for preprints and unpublished 2026 items.

## Generated evidence files

- `outputs/reviewer_statistics/main_metrics_target_bootstrap_ci.csv`
- `outputs/reviewer_statistics/main_paired_tests_bh.csv`
- `outputs/reviewer_statistics/search_v2_metrics_target_bootstrap_ci.csv`
- `outputs/reviewer_statistics/celltypist_targeted_metrics_bootstrap_ci.csv`
- `outputs/reviewer_statistics/celltypist_poststratified_population_estimates.csv`
- `outputs/reviewer_statistics/statistical_protocol.json`
- `outputs/celltypist_validation_v2/clean_cohort_summary.csv`
