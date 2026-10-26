# RoboTwin learning curves: scaling the score predictor’s training data

24 October 2026. Exploratory follow-up using the same pinned public evaluations. No new robot trials, simulation rollouts, robot-policy inference/training, or paid API calls.

**Finding:** more evaluated training policies improve the latent predictor over the measured range. Mean training size rises from 183 to 833 policy-task score cells per fit. Latent score MAE falls from 13.89 to 10.10 percentage points in clean scenes and from 17.87 to 11.57 in randomized scenes. Its ranking error falls too, but ends approximately level with the much simpler anchor-average ranking baseline. Gains diminish over the observed budgets; no scaling law or extrapolation is established.

## What “training data” means

One cell is one published success-rate mean for a policy on a task. It is not one robot episode or training demonstration. We increase the number of evaluated policy groups supplied to the meta-predictor; the robot policies themselves are fixed. Every selected training policy contributes scores for all 50 known tasks in one condition. Ten revealed scores for each held-out policy are fixed and are excluded from the horizontal-axis count.

Clean and randomized are fit separately. The full-data point uses only 16–17 of the 25 policies in each fold because the remaining policies are held out. It therefore uses about 833 training cells, not all 2,500 source cells. The two conditions share task and policy identities and are not independent replications.

## Controlled design

The original three family-grouped folds × three seeded partitions and five ten-task anchor draws are unchanged. Within each outer training pool, we sample five score-independent random group orders. At each budget, we retain the first ceil(fraction × available groups), with at least two groups. The subsets are nested: a larger budget adds groups and never replaces the smaller set. All configurations of a chosen group are retained. Test policies, anchors, target tasks and eligible ranking pairs are fixed across budgets.

The latent rank/ridge selection is rerun using inner grouped CV within each selected subset. The original seven predictors are unchanged. Model selection never sees hidden outer-test labels or omitted training rows. The learning-curve plan and code were frozen before curve results were inspected; prior full-data results were already known. This is exploratory analysis, not external preregistration.

| Fraction of available groups | Mean groups | Mean policies | Mean training cells | Training-cell range |
|---:|---:|---:|---:|---:|
| 20% | 2.67 | 3.67 | 183.33 | 100–500 |
| 40% | 5.00 | 7.71 | 385.56 | 200–600 |
| 60% | 7.33 | 11.11 | 555.56 | 300–700 |
| 80% | 9.67 | 14.27 | 713.33 | 500–800 |
| 100% | 11.33 | 16.67 | 833.33 | 800–850 |

These counts are identical across conditions. Group sizes differ, so a fraction of groups is not an exact fraction of policies or cells. The chart uses mean cell counts to show actual data volume; the ranges above preserve that variation. Group proxies, not cells, are the meaningful unit of policy diversity.

## Latent-model results

| Mean training cells | Clean MAE (pp) | Randomized MAE (pp) | Clean ranking agreement | Randomized ranking agreement |
|---:|---:|---:|---:|---:|
| 183 | 13.88 | 17.91 | 72.57% | 87.03% |
| 386 | 11.42 | 12.59 | 78.75% | 90.97% |
| 556 | 10.81 | 11.92 | 80.48% | 91.25% |
| 713 | 10.33 | 11.67 | 81.48% | 91.33% |
| 833 | 10.10 | 11.57 | 82.13% | 91.52% |

MAE gives policy groups equal weight after averaging within policy and group. Ranking error pools disjoint folds within each design, averages repeats within task, then tasks equally. Only observed gaps ≥0.10 enter ranking; predicted ties cost 0.5. This retrospective margin filter does not provide a deployable confidence rule. There are 17 group proxies for MAE, 48 informative clean tasks and 50 randomized tasks for ranking. Repeated subsets, folds and pairs are not independent samples.

The plotted blue bands resample the same group/task clusters jointly across budgets (2,000 draws). They are descriptive intervals conditional on these task and policy cohorts. They do not fully model shared training sets, uncertain ancestry, evaluation noise or publication selection.

| Condition | Endpoint | Full minus smallest budget (pp) | Paired descriptive 95% interval |
|---|---|---:|---:|
| clean | mae | -3.77 | [-4.68, -2.86] |
| clean | ranking_error | -9.56 | [-10.94, -8.19] |
| randomized | mae | -6.33 | [-7.98, -4.82] |
| randomized | ranking_error | -4.49 | [-5.25, -3.83] |

## Baselines change the interpretation

The anchor average has no training-policy parameters: its ranking agreement stays exactly at 82.54% clean / 91.51% randomized at every budget. It uses only the fixed ten observed scores of each held-out policy. The latent model starts substantially worse at 72.51% / 87.03%, and reaches 82.13% / 91.52%. More data helps the latent model recover the baseline’s strong global-ability signal; it has not established a better ranking rule.

The additive baseline is also data-efficient. Its clean MAE goes from 12.03 to 10.80 pp, and randomized MAE from 12.82 to 11.71 pp. Latent overtakes it only with larger budgets, and the final randomized score advantage remains small. The scalar baseline ends at 11.46 / 13.53 pp. All seven methods and intervals are retained in robotwin_learning_curve_summary.csv; the figure shows four prespecified reference methods.

## What is supported

Within this fixed task catalog, adding evaluated policy groups improves the latent predictor’s held-out score and ranking estimates. The result supports collecting broader policy coverage for this meta-prediction problem. It does not show that arbitrary additional cells, repetitions of the same policy, or more robot-training demonstrations yield the same gain. The finite range cannot determine whether errors will continue to fall, saturate, or reverse on substantially different policy families.

No physical outcomes or completely unseen tasks are tested. Most groups remain provisional ancestry proxies. The clean/randomized performance difference is not a causal effect of randomization: their distributions and eligible policy gaps differ.

## Reproduce

Run `python -m src.robotwin_learning_curve` followed by `python -m src.robotwin_learning_curve_report`, then `python -m unittest discover -s tests -v`. The full `python -m src.reproduce` command also includes these stages. No network access is needed.

The plan, audit and output names use the robotwin_learning_curve prefix. The audit records every selected training group, unused pool, test policy, anchor, inner CV candidate and selected setting. Tests verify nested subsets, fixed test comparisons, invariance to forbidden labels, a constant anchor-only baseline, and exact reproduction of the previous full-data endpoint.

Source: the pinned official RoboTwin leaderboard at https://github.com/RoboTwin-Platform/RoboTwin-Platform.github.io/blob/0abb5a03455073fa74047a1c4015b0f88cedc254/data/robotwin_leaderboard.json . See robotwin_findings.md and the data manifest for provenance and cohort details.
