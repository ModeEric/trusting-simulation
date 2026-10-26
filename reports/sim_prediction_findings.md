# Predicting simulated robot performance from a small task panel

24 October 2026. Exploratory, retrospective analysis of pinned public RoboDojo results. Independent research; not commissioned or endorsed by RoboDojo or Robocurve. No new trials, simulator rollouts, robot-policy inference/training, or paid API calls.

**Finding:** ten existing task scores predict rankings on the remaining known benchmark tasks well when the observed policy gap is at least 10 percentage points. A simple anchor-task average beats the richer latent model on ranking. The latent model reduces absolute score error, but its advantage over a one-scalar calibration is small and uncertain. This does not establish prediction of physical performance or unseen tasks.

## Data and effective sample size

The primary cohort includes only non-agent, non-single-task-trained entries with simulation training configurations in the pinned website export. The broader sensitivity includes entries lacking that documentation. Neither is a standardized training-recipe experiment. Family groups conservatively combine named related methods and shared published initialization checkpoints. They are grouping proxies, not proof of independent ancestry.

| Cohort | Policy configurations | Family groups | Base tasks | Observed means | Zero scores |
|---|---:|---:|---:|---:|---:|
| primary | 22 | 13 | 42 | 924 | 67.6% |
| broad | 52 | 31 | 42 | 2184 | 54.9% |

All selected configurations have all 42 base-task scores. Twelve randomized variants are excluded to avoid learning from near-duplicate tasks across the anchor/target split. The primary and broad cohorts overlap; the broad result is a sensitivity analysis, not external replication. Repeated folds, pairs, and anchors do not create additional independent policies or tasks. Published means are the targets; no episode counts, binomial intervals, or synthetic trials are inferred.

## What is held out

Three outer folds hold out whole family groups, repeated for three seeded partitions. Training policies supply all 42 known-task scores. Each held-out policy reveals only 5 or 10 anchor scores; its remaining 37 or 32 scores are excluded from fitting and hyperparameter selection. Five uniformly sampled anchor orders are fixed independently of scores, shared across compared policies, with the five-task set nested inside the ten-task set.
Both policies in every ranking comparison are held out together in the same fold and belong to different family groups. Cross-fold predictions are never compared: doing so could let each predictor train on the other policy’s target scores. Inner family-grouped CV selects the latent rank and regularization using outer-training policies only. Mutation tests replace every outer-test nonanchor label and confirm identical selected hyperparameters and predictions.
The 42 task identities are known through training policies. This tests prediction for a held-out policy family with observed anchors, not prediction for an entirely new task or a policy with no observations. It also does not remove any benchmark exposure during the underlying robot policies’ original training.

## Predictors

Let s_ma be policy m’s observed anchor score, A the anchor set, and mu_t the mean score on task t among outer-training policies. All predictions are clipped to [0,1].

- **Zero:** predict zero everywhere; necessary because many scores are zero.
- **TaskMean:** predict mu_t for every held-out policy; all policy rankings are ties.
- **AnchorMean:** predict mean(s_ma over a in A) for every target task; one global ability number per policy.
- **Additive:** predict mu_t + mean(s_ma - mu_a over anchors).
- **Scale:** predict c_m * mu_t, with c_m = (sum(mu_a * s_ma) + 0.01) / (sum(mu_a^2) + 0.01). This fits one scalar from anchors, regularized toward 1.
- **Neighbor3:** average the full profiles of the three training policies closest on anchor scores.
- **Latent:** predict mu_t + v_t dot z_m. Training-only centered SVD learns task vectors v_t; z_m = (V_A.T V_A + lambda I)^(-1) V_A.T (s_mA - mu_A). Inner grouped CV chooses rank 1/2/4 and lambda 0.01/0.1/1.0 for MAE.

Latent factors are a capability-profile hypothesis, not hand-labeled task mechanics. We did not optimize a separate rank-loss model or tune on outer test results. The implementation and design hashes were frozen before inspecting these predictor results; this is a local audit trail, not external preregistration.

## Absolute results

MAE is in percentage points, averaging repeated designs within policy, then policies within family, then families equally. Ranking error first pools jointly held-out pairs across disjoint folds within a design, averages designs within task, then tasks equally. Only observed gaps >=10 points are eligible; predicted ties cost 0.5. Thus 100% minus error is tie-adjusted agreement, not accuracy on all pairs. Nonzero-target MAE applies the same family aggregation after restricting targets to observed scores above zero.

| Cohort | Anchors | Predictor | MAE (pp) | Nonzero MAE (pp) | Rank error |
|---|---:|---|---:|---:|---:|
| primary | 5 | Zero | 3.43 | 8.39 | 50.00% |
| primary | 5 | TaskMean | 3.44 | 8.59 | 50.00% |
| primary | 5 | AnchorMean | 5.16 | 7.68 | 8.62% |
| primary | 5 | Additive | 3.65 | 7.56 | 9.10% |
| primary | 5 | Scale | 2.70 | 6.29 | 11.23% |
| primary | 5 | Neighbor3 | 2.64 | 7.08 | 20.49% |
| primary | 5 | Latent | 2.62 | 6.48 | 24.06% |
| primary | 10 | Zero | 3.50 | 8.57 | 50.00% |
| primary | 10 | TaskMean | 3.48 | 8.80 | 50.00% |
| primary | 10 | AnchorMean | 4.78 | 7.32 | 5.29% |
| primary | 10 | Additive | 3.52 | 7.57 | 5.29% |
| primary | 10 | Scale | 2.17 | 4.84 | 6.61% |
| primary | 10 | Neighbor3 | 2.32 | 6.42 | 11.65% |
| primary | 10 | Latent | 2.04 | 5.00 | 11.56% |
| broad | 5 | Zero | 10.40 | 16.61 | 50.00% |
| broad | 5 | TaskMean | 10.27 | 16.70 | 50.00% |
| broad | 5 | AnchorMean | 12.26 | 14.36 | 8.36% |
| broad | 5 | Additive | 9.11 | 13.02 | 8.57% |
| broad | 5 | Scale | 6.76 | 10.03 | 11.04% |
| broad | 5 | Neighbor3 | 5.64 | 8.81 | 15.29% |
| broad | 5 | Latent | 6.63 | 10.12 | 12.84% |
| broad | 10 | Zero | 10.41 | 16.58 | 50.00% |
| broad | 10 | TaskMean | 10.17 | 16.67 | 50.00% |
| broad | 10 | AnchorMean | 11.53 | 13.28 | 7.22% |
| broad | 10 | Additive | 8.44 | 12.26 | 7.40% |
| broad | 10 | Scale | 5.57 | 8.03 | 10.22% |
| broad | 10 | Neighbor3 | 5.35 | 8.40 | 14.21% |
| broad | 10 | Latent | 5.55 | 8.11 | 12.33% |

**Interpretation:** a global ability estimate is strong for material-gap rankings on these data. Learning task-specific latent profiles improves score prediction over TaskMean, but it does not improve ranking over AnchorMean. The primary ten-anchor latent MAE is 2.04 points versus 2.17 for Scale; the paired family-bootstrap difference is -0.13 points, with a descriptive 95% interval of approximately [-0.27, +0.01]. That is insufficient evidence that the more complex predictor is better than scalar calibration. Its ranking error is 11.56%, compared with 5.29% for AnchorMean.
The low primary MAE needs context: 67.6% of observed cells are zero, and the zero predictor already achieves 3.50-point MAE at ten anchors. Latent MAE on positive-score targets is 5.00 points; Scale is 4.84. The broader cohort has higher scores and larger errors: ten-anchor Scale MAE 5.57 points and AnchorMean ranking error 7.22%.

## Pair coverage, close comparisons, and ties

The following counts are unique (policy pair, target task) comparisons observed across repeated held-out designs. They are correlated, not independent sample sizes. The observed 10-point gap filter is a retrospective evaluation stratum: a deployable system cannot know the true target gap without evaluation.

| Cohort | Anchors | Gap | Unique pair-task comparisons | Informative tasks | AnchorMean error | Latent error | AnchorMean tie fraction* |
|---|---:|---|---:|---:|---:|---:|---:|
| primary | 5 | material | 574 | 18 | 8.62% | 24.06% | 3.94% |
| primary | 5 | close | 1347 | 32 | 21.88% | 34.39% | 9.29% |
| primary | 10 | material | 574 | 18 | 5.29% | 11.56% | 0.45% |
| primary | 10 | close | 1347 | 32 | 18.13% | 25.36% | 1.40% |
| broad | 5 | material | 10294 | 34 | 8.36% | 12.84% | 1.53% |
| broad | 5 | close | 8973 | 37 | 22.93% | 25.63% | 4.15% |
| broad | 10 | material | 10294 | 34 | 7.22% | 12.33% | 0.21% |
| broad | 10 | close | 8973 | 37 | 21.14% | 23.78% | 0.70% |

*Tie fraction pools repeated eligible comparisons and is descriptive; ranking errors retain task-macro weighting. Close means strictly nonzero observed gaps below 10 points; exact observed ties are excluded. No uncertainty in the published target means is reconstructed. Only 18/42 primary tasks have any eligible material-gap comparison across the tested designs; the corresponding broad count is 34/42. Good large-gap ranking does not establish reliable ordering of near-tied policies.

## Uncertainty and anchor sensitivity

Intervals resample 13 primary/31 broad family groups for MAE and informative task clusters for ranking (2,000 draws). They are descriptive conditional intervals: shared training sets, task selection, uncertain ancestry, and noisy published means are not fully modeled. All baselines and richer methods are reported; selecting a winner here requires fresh external validation.

| Cohort | Ten-anchor method | MAE 95% interval (pp) | Ranking error 95% interval | MAE across five anchor sets (pp) |
|---|---|---:|---:|---:|
| primary | AnchorMean | [2.61, 7.26] | [2.88%, 8.18%] | 4.25–5.67 |
| primary | Scale | [1.66, 2.72] | [2.95%, 10.99%] | 1.56–2.78 |
| primary | Latent | [1.45, 2.64] | [4.17%, 23.93%] | 1.57–2.48 |
| broad | AnchorMean | [8.38, 14.75] | [5.62%, 9.00%] | 10.87–12.86 |
| broad | Scale | [4.30, 6.87] | [6.81%, 14.21%] | 4.89–6.52 |
| broad | Latent | [4.29, 6.80] | [8.13%, 17.59%] | 4.67–6.45 |

## Largest primary ranking failures

Ten-anchor material-gap error by task, averaged over repeated designs. Sparse per-task comparisons make these descriptive failure examples, not established categories.

| Task | Unique eligible policy pairs | AnchorMean error | Scale error | Latent error |
|---|---:|---:|---:|---:|
| press_by_number | 13 | 0.00% | 0.00% | 100.00% |
| match_and_pick_from_conveyor | 31 | 23.85% | 24.22% | 23.89% |
| swap_blocks | 5 | 8.33% | 33.33% | 16.67% |
| put_bottles_into_dustbin | 77 | 11.91% | 12.47% | 12.84% |
| build_tower | 39 | 9.71% | 11.86% | 12.00% |
| stack_bowls | 73 | 10.12% | 8.41% | 8.35% |

## What to try next

1. Freeze Scale for score prediction and AnchorMean for ranking as candidate reference methods, then replicate on an independently audited benchmark with per-policy, per-task results. RoboTwin is a candidate; its training tracks must be separated before combining entries. This run has not performed that replication.
2. Test training-only anchor selection and budget curves against these uniformly sampled anchors. Do not select the anchor panel using held-out policies’ target scores. Ranking-specific fitting is a separate prospective extension, since the current latent objective is MAE.
3. Evaluate policy families with comparable overall strength and tasks with nontrivial success rates. AnchorMean’s strong performance is consistent with global ability explaining much of the current ranking signal; close-gap diagnostics are harder. More diverse independent policy families and better-documented evaluation repetitions are more valuable than counting more correlated episodes as examples. This experiment does not yet test a training-data scaling law.
4. Keep physical validation separate. Predicting a simulated score introduces one error; using simulation as a physical proxy introduces another. Neither the present simulation-only results nor SIMPLER’s small matched table bounds both errors on RoboDojo/Robocurve. A physical claim needs held-out, matched policy-task-hardware evidence.

No additional compute or paid access is needed to reproduce this run. More public, compatible evaluation tables are needed for external validation, not larger models trained on these same test folds.

## Provenance and reproduction

- Source: [RoboDojo simulation leaderboard](https://robodojo-benchmark.com/leaderboard?bench=sim), pinned website JavaScript snapshot dated 2026-10-24. Exact URLs and SHA-256 checksums are in data/download_audit.json and data/manifest.json; the bundled data never refresh silently.
- Protocol: [RoboDojo paper v3](https://arxiv.org/html/2607.04434v3), including separate single-task-trained entries and simulation training setup. Website and paper versions are distinct; model inclusion uses the pinned website metadata.
- Run `python -m src.sim_prediction` then `python -m src.sim_prediction_report`, or `python -m src.reproduce` for the whole project and tests.
- Design: sim_prediction_plan.json and data/sim_prediction_plan_freeze.json. Inputs: data/processed/sim_prediction_policy_metadata.csv, family_edges.csv (full prefix sim_prediction_), and cohort matrices. Audit: results/sim_prediction_fold_audit.json.
- Full outputs: sim_prediction_predictions.csv.gz, summary.csv, deltas.csv, family_metrics.csv, task_metrics.csv, gap_diagnostics.csv, and anchor_variation.csv under results/ with the sim_prediction_ prefix.
- The earlier PDF remains the original sim-to-real study; this report documents the separate simulation-only pivot.
