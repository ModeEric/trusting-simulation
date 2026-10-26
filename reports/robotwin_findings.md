# RoboTwin: predicting simulated performance from a small task panel

24 October 2026. Independent retrospective research using published evaluation means. No new robot trials, simulator rollouts, robot-policy training/inference, or paid API calls. Not commissioned or endorsed by RoboTwin or Robocurve.

**Result:** the frozen prediction procedure works on this larger benchmark, but the result is narrower than “a rich model learns reliable task-specific rankings.” With ten anchor scores, the anchor average achieves 82.5% tie-adjusted ranking agreement in clean scenes and 91.5% in randomized scenes for observed gaps of at least ten percentage points. The latent model improves success-rate estimates over scalar calibration, but does not establish a ranking improvement. Close-gap agreement is only about 62–64%.

## Data and effective sample size

The pinned official leaderboard contains 25 jointly trained configurations × 50 tasks × 2 conditions = 2,500 published success-rate means. Five single-task SFT entries are excluded from prediction. Clean and randomized are separate evaluation tracks; both use policies trained on clean demonstrations. Two configurations use depth, so the frozen sensitivity excludes them.

| Cohort | Policies | Group proxies | Tasks | Means | Zero scores |
|---|---:|---:|---:|---:|---:|
| clean | 25 | 17 | 50 | 1250 | 0.16% |
| randomized | 25 | 17 | 50 | 1250 | 9.76% |
| clean_rgb | 23 | 17 | 50 | 1150 | 0.17% |
| randomized_rgb | 23 | 17 | 50 | 1150 | 10.61% |

The RGB cohorts are overlapping sensitivities, not extra data. The 17 groups are conservative ancestry proxies, not 17 verified independent architectures. Two connected groups combine named related methods and documented shared initializations; remaining entries have provisional singleton IDs. Undocumented shared pretraining may remain. The Wan-related proxy deliberately overgroups 4D-WAM, whose clean-to-random variant uses Lingbot-VA, because the same named method also studies FastWAM variants. See robotwin_plan.json and robotwin_policy_metadata.csv for the frozen membership and rationale.

The board describes 100 evaluation trials per task. The source contains 1 noninteger percentage cell, so exact per-cell trial denominators are not inferred; n_trials remains blank and the nominal protocol is retained separately. Recomputed means differ from displayed means by at most 0.004 percentage points, consistent with display rounding. These aggregates do not supply independent episode labels or sampling intervals.

## Held-out prediction design

We reused src.sim_prediction.py unchanged from the RoboDojo experiment: three family-grouped folds, three seeded partitions, five score-independent anchor draws, and nested budgets of 5/10 tasks. Every test policy reveals only those anchor scores; its remaining 45/40 targets are withheld from fitting and hyperparameter selection. Both members of a ranked pair are held out together in the same fold and belong to distinct groups. No comparisons mix predictions across folds.

The training policies supply scores on all 50 task identities. This therefore predicts additional known-task scores for a policy with a small observed task panel. It does not predict a completely new task, or a model with no observations. Randomized predictions use randomized anchors; they do not forecast randomized performance from clean anchors. Coefficients are refit within RoboTwin training folds: this transports a method, not coefficients learned on RoboDojo.

The source, grouping plan, predictor implementation and settings were hashed before inspecting RoboTwin predictor results. This is a local audit trail, not an external preregistration. Mutation tests alter held-out target labels and every label in the other condition and verify identical selected hyperparameters and saved predictions.

## Predictors

Let A be the anchor set, s_mA its observed scores for policy m, and mu_t the training-policy mean on task t. All predictions are clipped to [0,1].

- Zero: always zero.
- TaskMean: mu_t, identical across test policies.
- AnchorMean: mean(s_mA), identical across target tasks for a policy.
- Additive: mu_t + mean(s_mA - mu_A).
- Scale: c_m * mu_t, where c_m = (mu_A dot s_mA + 0.01) / (mu_A dot mu_A + 0.01).
- Neighbor3: average full profiles of the three training policies closest on anchor scores.
- Latent: mu_t + v_t dot z_m, where z_m = (V_A.T V_A + lambda I)^(-1) V_A.T (s_mA - mu_A). Training-only centered SVD learns V. Inner grouped CV chooses rank 1/2/4 and ridge 0.01/0.1/1 for MAE.

Scale for score prediction and AnchorMean for ranking were chosen as references from the earlier RoboDojo experiment. All seven methods are reported; the outer test results did not select predictor settings. Latent is a learned task-profile model, not a hand-coded task-taxonomy calibrator.

## Absolute results

MAE averages repeats within policy, policies within group, and groups equally. Ranking error pools disjoint-fold eligible pairs within each design, averages repeated designs within task, then tasks equally. Actual ties and gaps below 0.10 are excluded from the primary ranking endpoint; predicted ties cost 0.5. Agreement is 100% minus this error, not an ordinary pooled accuracy.

| Anchors | Predictor | Clean MAE (pp) | Clean ranking error | Randomized MAE (pp) | Randomized ranking error |
|---:|---|---:|---:|---:|---:|
| 5 | Zero | 75.45 | 50.00% | 44.13 | 50.00% |
| 5 | TaskMean | 14.46 | 50.00% | 24.69 | 50.00% |
| 5 | AnchorMean | 18.36 | 17.65% | 19.33 | 8.70% |
| 5 | Additive | 11.26 | 17.90% | 12.26 | 8.75% |
| 5 | Scale | 11.96 | 17.78% | 14.43 | 8.87% |
| 5 | Neighbor3 | 10.62 | 22.20% | 12.31 | 10.70% |
| 5 | Latent | 10.35 | 17.86% | 12.21 | 8.93% |
| 10 | Zero | 75.36 | 50.00% | 43.88 | 50.00% |
| 10 | TaskMean | 14.39 | 50.00% | 24.66 | 50.00% |
| 10 | AnchorMean | 17.25 | 17.46% | 17.78 | 8.49% |
| 10 | Additive | 10.80 | 18.02% | 11.71 | 8.51% |
| 10 | Scale | 11.46 | 17.82% | 13.53 | 8.39% |
| 10 | Neighbor3 | 10.09 | 21.45% | 11.92 | 10.38% |
| 10 | Latent | 10.10 | 17.87% | 11.57 | 8.48% |

Ten-anchor latent MAE is 10.10 pp clean / 11.57 pp randomized, versus 11.46 / 13.53 for Scale. This is a clearer score-error improvement than in the earlier RoboDojo experiment. However, the stronger Additive baseline is already at 10.80 / 11.71 pp; the incremental latent advantage is small, especially in randomized scenes. Neighbor3 has similar clean MAE but worse rankings. More expressive score prediction has not automatically produced better model selection.

## Paired uncertainty and sensitivity

The following differences are Latent minus reference; negative favors Latent. Descriptive 95% percentile intervals resample 17 group proxies for MAE or informative tasks for ranking, with 2,000 draws. Repeated folds and anchors are averaged before resampling. These intervals do not capture all shared-training dependencies, unknown ancestry, task selection or noise in published means.

| Condition | Endpoint | Reference | Difference (pp) | Descriptive 95% interval (pp) |
|---|---|---|---:|---:|
| clean | mae | AnchorMean | -7.14 | [-8.66, -5.72] |
| clean | mae | Scale | -1.36 | [-2.02, -0.70] |
| clean | mae | Additive | -0.70 | [-1.14, -0.27] |
| randomized | mae | AnchorMean | -6.20 | [-7.37, -5.07] |
| randomized | mae | Scale | -1.96 | [-2.83, -1.10] |
| randomized | mae | Additive | -0.14 | [-0.46, +0.24] |
| clean | ranking_error | AnchorMean | +0.42 | [-0.23, +1.00] |
| clean | ranking_error | Scale | +0.05 | [-0.56, +0.64] |
| clean | ranking_error | Additive | -0.15 | [-1.57, +0.83] |
| randomized | ranking_error | AnchorMean | -0.01 | [-0.32, +0.28] |
| randomized | ranking_error | Scale | +0.09 | [-0.28, +0.40] |
| randomized | ranking_error | Additive | -0.04 | [-0.37, +0.29] |

All latent-versus-AnchorMean ranking intervals include zero. We do not claim a ranking improvement.

| Cohort, ten anchors | AnchorMean agreement | Latent agreement | Scale MAE (pp) | Latent MAE (pp) |
|---|---:|---:|---:|---:|
| clean | 82.54% | 82.13% | 11.46 | 10.10 |
| clean_rgb | 80.93% | 80.59% | 11.87 | 10.64 |
| randomized | 91.51% | 91.52% | 13.53 | 11.57 |
| randomized_rgb | 91.97% | 91.42% | 13.80 | 11.65 |

Excluding depth policies preserves the central result. It changes group sizes, training/test membership and task comparisons, so the numeric difference is not a causal estimate of depth’s value.

| Condition, ten anchors | Latent MAE across five anchor sets (pp) | AnchorMean ranking error across sets | Latent rank choices (45 outer fits) |
|---|---:|---:|---|
| clean | 9.54–11.37 | 14.88–18.19% | rank 1: 43, rank 2: 2 |
| randomized | 11.06–12.33 | 7.97–8.45% | rank 1: 35, rank 2: 10 |

Each anchor draw also changes the held-out task mix, so the across-draw range need not contain the primary task-balanced average. Rank one is selected in most folds. That is consistent with a strong global ability component; it does not establish that richer task mechanisms are absent. Five to ten anchors produces only modest gains on these fixed draws. No learning-curve experiment varying the number of training policies was run, so this does not prove that more data will keep improving prediction.

## Close gaps, coverage and failure cases

| Condition | Gap stratum | AnchorMean agreement | Latent agreement | Tasks | Unique eligible pair-task comparisons |
|---|---|---:|---:|---:|---:|
| clean | ≥10 pp | 82.54% | 82.13% | 48 | 4284 |
| clean | 0–10 pp, excluding ties | 62.15% | 61.91% | 50 | 2898 |
| randomized | ≥10 pp | 91.51% | 91.52% | 50 | 6116 |
| randomized | 0–10 pp, excluding ties | 63.59% | 63.67% | 50 | 1502 |

Unique pair-task counts measure coverage, not independent sample size. They include only pairs co-held-out in at least one design. Actual-gap filtering is a retrospective diagnostic requiring the hidden outcome; it is not a deployable confidence rule. Higher randomized ranking agreement does not mean randomized scores are easier to predict: the policies separate more strongly, and the eligible pair population changes.

Worst ten-anchor tasks for the prespecified AnchorMean ranking reference (selected descriptively after the run):

| Condition | Task | AnchorMean ranking error | Latent ranking error | Unique material-gap pairs |
|---|---|---:|---:|---:|
| clean | Adjust Bottle | 65.4% | 57.4% | 51 |
| clean | Shake Bottle | 56.3% | 50.8% | 17 |
| clean | Open Microwave | 39.4% | 40.5% | 130 |
| clean | Handover Mic | 36.4% | 39.7% | 51 |
| clean | Rotate QRcode | 30.7% | 33.1% | 97 |
| randomized | Open Microwave | 25.8% | 26.0% | 111 |
| randomized | Dump Bin Bigbin | 20.0% | 21.9% | 121 |
| randomized | Grab Roller | 17.0% | 15.8% | 117 |
| randomized | Place Cans Plasticbox | 15.6% | 14.7% | 127 |
| randomized | Shake Bottle | 13.3% | 9.2% | 116 |

## What to improve next

The most useful next experiment is to target residual ranking reversals while retaining this global-ability baseline. Predeclare a pairwise ranking objective and a small residual model, choose its complexity using inner grouped folds, and compare paired outer-fold errors against AnchorMean. Any model designed after seeing these failures is exploratory on RoboTwin and needs an additional untouched dataset or later model cohort for confirmation.

A second experiment can choose the anchor panel using training policies only, then compare it with the same random panels. Task families and near-duplicate tasks should be grouped for any new-task generalization claim. A training-family learning curve is needed to test the user’s “more data helps” thesis; 2,500 means still represent only 25 configurations, 50 task identities, and two related conditions.

## Boundaries of the result

- No physical outcomes are validated here. Accurate simulation prediction still needs a separate matched sim-to-real test before it can guide real-robot selection.
- This is meta-predictor holdout, not a claim that the original robot policies were unexposed to benchmark tasks during training.
- Original policy training recipes, pretraining, modalities and reported evaluation procedures differ. Public leaderboard entries can contain benchmark-selection and publication bias.
- Group proxies and task bootstraps do not eliminate correlated models, related tasks or shared evaluation infrastructure.
- Published means are treated as fixed targets; raw evaluation episodes and exact denominators are unavailable here. Close-gap rankings may partly reflect measurement noise.
- The earlier RoboDojo result had far more zero outcomes. Its small MAE is not directly comparable with RoboTwin’s denser success distribution.
- Neither benchmark establishes that simulation is a reliable physical proxy for these particular policy comparisons.

## Reproduce and inspect

From the bundled repository, run `python -m src.reproduce` for all studies and tests, or `python -m src.robotwin_prediction` then `python -m src.robotwin_report` for this replication. Reproduction is offline and verifies pinned hashes.

Outputs: `robotwin_summary.csv`, `robotwin_reference_deltas.csv`, `robotwin_predictions.csv.gz`, `robotwin_fold_audit.json`, `robotwin_gap_diagnostics.csv`, `robotwin_gap_task_metrics.csv`, `robotwin_anchor_variation.csv`, and `figures/08_robotwin_prediction.png` under results/. Normalized source tables and cohort matrices are under data/processed/. The original SIMPLER PDF is a separate earlier report.

Sources:

- Official leaderboard and protocol: https://robotwin-platform.github.io/leaderboard
- Pinned raw data: https://github.com/RoboTwin-Platform/RoboTwin-Platform.github.io/blob/0abb5a03455073fa74047a1c4015b0f88cedc254/data/robotwin_leaderboard.json
- PrismWAM ancestry evidence: https://arxiv.org/html/2610.26564v1
- 4D-WAM variant evidence: https://arxiv.org/html/2608.08023v1
- Inherited grouping evidence: pinned RoboDojo website and sim_prediction_family_edges.csv in this repository.

Access date: 24 October 2026 (America/Los_Angeles). Source revision, URLs, sizes and SHA-256 checksums are retained in data/download_audit.json and data/manifest.json. No videos or trajectory archives were downloaded.
