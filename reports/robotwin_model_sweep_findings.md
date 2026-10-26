# RoboTwin: three sequential modeling experiments

24 October 2026. Independent exploratory analysis of pinned public evaluation means. All three experiments were specified before inspecting their results. No robot trials, simulation rollouts, robot-policy training/inference, or paid API calls.

**Finding:** expanded linear capacity does not improve the primary result because inner validation still selects low ranks. The bounded model modestly improves success-rate MAE, with clearer evidence in randomized scenes. Direct ranking training improves primary agreement by only 0.10 points clean / 0.20 randomized over the anchor average, with both descriptive paired intervals including zero. Increasing anchor coverage produces a larger observed ranking gain than changing model class.

## Setup and scope

We retain the 25 jointly trained RoboTwin configurations, 50 known task identities and 17 conservative ancestry proxies. Clean and randomized conditions remain separate. The same three grouped outer folds, three seeded partitions and five random anchor orders are used. Both policies in each ranking comparison are held out together and belong to different groups.

Training coverage uses 20%, 60% and 100% of available groups, with the first three nested subset orders from the preceding learning-curve study. Anchor budgets are 5, 10 and 20. The primary result is full training coverage and ten anchors evaluated on the remaining 40 tasks, preserving the original endpoint. The controlled scaling comparison always evaluates the last 30 tasks of the seeded order, so changing anchor count does not change its target set.

An observation is a published policy-task mean. Trial denominators remain unverified; no episode labels or binomial likelihood are reconstructed. More training cells here means more evaluated policies supplied to the meta-predictor. The underlying robot policies are fixed.

## 1. Higher-dimensional linear models

OriginalLatent retains rank caps 1/2/4 and ridge 1/0.1/0.01. LinearRidge adds ridge 10 as a control. LinearWide also allows rank caps 8 and 12. Centered SVD learns task loadings; ridge regression infers a held-out policy’s coordinates from anchors. Inner grouped CV selects family-macro MAE. Requested rank is capped by the available numerical rank, which can be smaller in inner folds.

At full training coverage, neither rank 8 nor 12 is selected at any of the three anchor budgets. The wider linear model and wider-ridge control reproduce the original model’s primary 10-anchor results. These data do not establish that greater latent dimension helps; low-dimensional choices winning in this small sample do not prove higher-dimensional structure is absent.

## 2. Bounded nonlinear model

LogitLatent clips published proportions to [0.01,0.99] for numerical stability, transforms them to logits, fits centered SVD factors in that space, infers coordinates by ridge regression on logit anchors, and returns sigmoid outputs. It uses the same expanded rank/ridge grid and selects probability-space MAE in inner grouped CV.

This is a bounded nonlinear prediction rule, with squared-error factor fitting in logit space. It does not jointly optimize a probability-space matrix-factorization likelihood and does not add interactions between latent coordinates. The clipping threshold is a fixed numerical choice, not an inferred trial-count correction.

## 3. Direct pairwise ranking training

RankGlobal learns one correction to the anchor average; RankTask learns a separate correction for each known task. Their utility is mean(anchor scores) plus a linear function of centered anchors. Training minimizes pairwise logistic loss for distinct-group policy pairs whose observed success gap is at least 0.10, with fixed temperature 0.10. Pair weights are inverse products of the groups’ policy counts, normalized within task. L2 regularization shrinks the correction toward zero.

Inner grouped validation selects shrinkage from infinity, 100, 10, 1 and 0.1 using task-macro material-gap ranking error. Infinity exactly recovers the anchor-only ranking baseline. Inner ranking comparisons never cross folds. If no eligible cross-family validation pair exists, the selection is flagged unavailable and falls back to infinity. This happens at the smallest training budgets and must not be interpreted as evidence of absent task structure.

The outputs are uncalibrated ranking utilities. We do not report their numeric difference from success rates as MAE. Training includes the known tasks’ available training-policy labels; only held-out policy anchors enter prediction. Optimizer failures raise errors rather than being silently included.

## Primary absolute results

MAE averages repeats within policy, policies within group and groups equally. Ranking error pools disjoint folds within a design, averages designs within task and then tasks equally. Observed gaps must be at least ten percentage points; predicted ties cost 0.5.

| Method | Clean MAE (pp) | Randomized MAE (pp) | Clean ranking agreement | Randomized ranking agreement |
|---|---:|---:|---:|---:|
| Anchor average | 17.25 | 17.78 | 82.54% | 91.51% |
| Additive | 10.80 | 11.71 | 81.98% | 91.49% |
| Scalar | 11.46 | 13.53 | 82.18% | 91.61% |
| Original latent | 10.10 | 11.57 | 82.13% | 91.52% |
| Wider ridge grid | 10.10 | 11.57 | 82.13% | 91.52% |
| Higher-dimensional | 10.10 | 11.57 | 82.13% | 91.52% |
| Bounded latent | 9.91 | 10.99 | 82.14% | 91.52% |
| Global rank-trained | Not calibrated | Not calibrated | 82.47% | 91.49% |
| Task rank-trained | Not calibrated | Not calibrated | 82.64% | 91.71% |

## Paired comparisons

Negative differences favor the new model. The descriptive 95% percentile intervals resample 17 policy-group proxies for MAE or informative tasks for ranking, with 2,000 draws. They do not capture all shared-training dependence, unknown ancestry, noisy source means or publication selection. No multiple-comparison-adjusted significance claim is made.

| Condition | New model | Endpoint | Reference | Difference (pp) | Descriptive 95% interval |
|---|---|---|---|---:|---:|
| clean | LinearWide | mae | OriginalLatent | +0.00 | [+0.00, +0.00] |
| clean | LogitLatent | mae | OriginalLatent | -0.19 | [-0.61, +0.25] |
| randomized | LinearWide | mae | OriginalLatent | +0.00 | [+0.00, +0.00] |
| randomized | LogitLatent | mae | OriginalLatent | -0.58 | [-0.95, -0.23] |
| clean | LinearWide | ranking_error | OriginalLatent | +0.00 | [+0.00, +0.00] |
| clean | LogitLatent | ranking_error | OriginalLatent | -0.01 | [-0.42, +0.40] |
| clean | RankGlobal | ranking_error | AnchorMean | +0.07 | [-0.06, +0.20] |
| clean | RankTask | ranking_error | AnchorMean | -0.10 | [-0.50, +0.21] |
| randomized | LinearWide | ranking_error | OriginalLatent | +0.00 | [+0.00, +0.00] |
| randomized | LogitLatent | ranking_error | OriginalLatent | +0.00 | [-0.30, +0.30] |
| randomized | RankGlobal | ranking_error | AnchorMean | +0.03 | [-0.10, +0.15] |
| randomized | RankTask | ranking_error | AnchorMean | -0.20 | [-0.48, +0.11] |

## Anchor-count sensitivity on a fixed target set

Full training coverage below; all budgets evaluate the same 30 tasks in each draw. This table cannot be directly compared with the primary 40-task table without accounting for its different task mix.

| Condition | Anchors | Original latent MAE | Bounded latent MAE | Anchor-average agreement | Global rank-trained agreement | Task rank-trained agreement |
|---|---:|---:|---:|---:|---:|---:|
| clean | 5 | 10.30 | 9.97 | 81.85% | 82.14% | 81.88% |
| clean | 10 | 10.17 | 9.93 | 82.35% | 82.26% | 82.45% |
| clean | 20 | 9.43 | 9.18 | 84.58% | 84.35% | 84.80% |
| randomized | 5 | 12.25 | 11.60 | 91.15% | 91.12% | 91.03% |
| randomized | 10 | 11.60 | 11.06 | 91.50% | 91.48% | 91.65% |
| randomized | 20 | 10.94 | 10.75 | 92.36% | 92.30% | 92.21% |

Paired changes from 10 to 20 anchors, evaluated on the common 30-task target set (negative error differences favor 20 anchors):

| Condition | Method | Endpoint | Change (pp) | Descriptive 95% interval |
|---|---|---|---:|---:|
| clean | LogitLatent | mae | -0.75 | [-1.29, -0.31] |
| randomized | LogitLatent | mae | -0.31 | [-0.56, -0.05] |
| clean | AnchorMean | ranking_error | -2.22 | [-3.01, -1.52] |
| randomized | AnchorMean | ranking_error | -0.86 | [-1.29, -0.42] |

The training-data effect is also conditional on model choice. On the common target set with ten anchors, bounded-minus-original MAE is +0.09 clean / +0.75 randomized at the smallest budget (~157 cells), versus -0.23 / -0.54 at the full budget (~833 cells). The bounded model benefits from the larger sample in this comparison, but this limited range cannot establish continued scaling. The intermediate-budget task ranker can underperform the anchor baseline; its improvements are not monotonic with training coverage.

## Selection diagnostics

Counts below use full training coverage and one copy of each repeated subset result: 45 outer fits per condition and anchor budget. Selection percentages are descriptive, not independent observations.

| Condition | Anchors | Method | Selected settings |
|---|---:|---|---|
| clean | 5 | LinearWide | rank 1.0: 44, rank 2.0: 1 |
| clean | 5 | LogitLatent | rank 1.0: 43, rank 2.0: 2 |
| clean | 5 | RankGlobal | ridge 0.1: 12, ridge 1.0: 5, ridge 10.0: 3, ridge 100.0: 7, ridge infinity: 18 |
| clean | 5 | RankTask | ridge 0.1: 1, ridge 1.0: 7, ridge 10.0: 11, ridge 100.0: 7, ridge infinity: 19 |
| clean | 10 | LinearWide | rank 1.0: 43, rank 2.0: 2 |
| clean | 10 | LogitLatent | rank 1.0: 37, rank 2.0: 8 |
| clean | 10 | RankGlobal | ridge 0.1: 10, ridge 1.0: 5, ridge 10.0: 1, ridge infinity: 29 |
| clean | 10 | RankTask | ridge 0.1: 1, ridge 1.0: 4, ridge 10.0: 17, ridge infinity: 23 |
| clean | 20 | LinearWide | rank 1.0: 32, rank 2.0: 11, rank 4.0: 2 |
| clean | 20 | LogitLatent | rank 1.0: 31, rank 2.0: 9, rank 4.0: 5 |
| clean | 20 | RankGlobal | ridge 0.1: 9, ridge 1.0: 7, ridge 100.0: 2, ridge infinity: 27 |
| clean | 20 | RankTask | ridge 0.1: 4, ridge 1.0: 6, ridge 10.0: 15, ridge 100.0: 2, ridge infinity: 18 |
| randomized | 5 | LinearWide | rank 1.0: 34, rank 2.0: 11 |
| randomized | 5 | LogitLatent | rank 1.0: 44, rank 2.0: 1 |
| randomized | 5 | RankGlobal | ridge 0.1: 13, ridge 1.0: 2, ridge 10.0: 1, ridge 100.0: 3, ridge infinity: 26 |
| randomized | 5 | RankTask | ridge 0.1: 7, ridge 1.0: 14, ridge 10.0: 3, ridge 100.0: 2, ridge infinity: 19 |
| randomized | 10 | LinearWide | rank 1.0: 35, rank 2.0: 10 |
| randomized | 10 | LogitLatent | rank 1.0: 44, rank 2.0: 1 |
| randomized | 10 | RankGlobal | ridge 0.1: 12, ridge 1.0: 9, ridge infinity: 24 |
| randomized | 10 | RankTask | ridge 0.1: 3, ridge 1.0: 21, ridge 10.0: 7, ridge infinity: 14 |
| randomized | 20 | LinearWide | rank 1.0: 15, rank 2.0: 23, rank 4.0: 7 |
| randomized | 20 | LogitLatent | rank 1.0: 21, rank 2.0: 21, rank 4.0: 3 |
| randomized | 20 | RankGlobal | ridge 0.1: 9, ridge 1.0: 5, ridge infinity: 31 |
| randomized | 20 | RankTask | ridge 0.1: 7, ridge 1.0: 15, ridge 10.0: 12, ridge 100.0: 2, ridge infinity: 9 |

## Close-gap diagnostics

Primary ten-anchor predictions, restricting to nonzero observed gaps below ten points. These are retrospective gap strata; the actual hidden gap is unavailable at deployment.

| Condition | Method | Close-gap agreement |
|---|---|---:|
| clean | AnchorMean | 62.15% |
| clean | LogitLatent | 62.45% |
| clean | OriginalLatent | 61.91% |
| clean | RankGlobal | 61.85% |
| clean | RankTask | 62.13% |
| randomized | AnchorMean | 63.59% |
| randomized | LogitLatent | 64.55% |
| randomized | OriginalLatent | 63.67% |
| randomized | RankGlobal | 63.72% |
| randomized | RankTask | 63.91% |

## Interpretation and limits

Increasing model capacity is useful only when the available policies and anchors identify the added structure. The linear experiment favors low-dimensional models. The bounded transformation tests a different score scale and saturation behavior; its score-error effect must be distinguished from policy-ranking effects. The direct ranking models test whether nonuniform anchor weights or task-specific corrections improve model selection. Each is a specific model class, not a general test of all nonlinear or ranking approaches.

All three model classes and all budgets are reported, including null or adverse results. These exploratory comparisons reuse the same public policy cohort after earlier analyses. Any chosen winner or targeted follow-up needs a fresh policy cohort or independent benchmark for confirmation. No claim is made about physical robot performance, unseen tasks, zero-observation policies, or scaling beyond the available training coverage.

## Reproduce

Run `python -m src.robotwin_model_sweep --stage 1`, then stage 2, then stage 3. Run `python -m src.robotwin_model_sweep_report` and `python -m unittest discover -s tests -v`. The full reproduction entry point includes all stages. No network access or robot inference is used.

Outputs include robotwin_models_summary.csv, robotwin_models_paired_deltas.csv, robotwin_models_selections.csv, per-stage fold audits and predictions, and figures/10_robotwin_model_variants.png and figures/11_robotwin_model_scaling.png. Utility outputs are explicitly labeled. The frozen plan and source hashes are bundled.

Pinned source: https://github.com/RoboTwin-Platform/RoboTwin-Platform.github.io/blob/0abb5a03455073fa74047a1c4015b0f88cedc254/data/robotwin_leaderboard.json . Source protocol and ancestry caveats are documented in robotwin_findings.md.
