# Exploratory blend experiment: one ranking fix, no task-aware ranking gain

24 October 2026. Existing SIMPLER aggregates only. This follow-up was designed after inspecting the initial analysis; it is not independent confirmatory evidence. The original frozen C1 experiment is unchanged.

## Model and selection

`prediction = (1 - w) * matched_sim + w * global_sim`. Global ability averages the same policy’s available simulated task scores within embodiment. The primary version includes the target simulated task, as permitted by the original B0 definition; a sensitivity excludes it.
GlobalBlend chooses one weight from 0, 0.1, ..., 1. TaskBlend chooses separate weights for precision_contact=0 and 1 (121 candidates). No new task labels are introduced. The global model is nested in the task-aware model. The primary selection objective is task-average material-gap ranking error. A separately reported secondary analysis selects for MAE.
Outer LOTO excludes every physical outcome of the test task. Remaining tasks provide inner task-held-out validation scores for candidate selection. Because each candidate is a fixed blend, there are no coefficients to fit inside an inner fold; the inner loop evaluates each fixed rule on each remaining task and averages task losses. It does not add another layer of independent evidence. All global abilities use simulation only.
Tied validation loss (within 1e-12) favors equal weights between categories, then the smallest average weight. Real gaps below 0.10 are excluded from ranking evaluation; predicted ties cost 0.5. Hyperparameters never use outer physical outcomes. The full candidate losses and fold memberships are recorded in blend_fold_audit.json.

## Primary ranking objective

| Embodiment | Method | Task-average rank error | Correct / eligible pairs | MAE (pp) |
|---|---|---:|---:|---:|
| google_robot | B1 | 5.71% | 61/65 | 13.91 |
| google_robot | GlobalBlend_ranking | 4.29% | 62/65 | 13.88 |
| google_robot | TaskBlend_ranking | 7.92% | 60/65 | 16.96 |
| widowx | B1 | 12.50% | 8/9 | 7.66 |
| widowx | GlobalBlend_ranking | 12.50% | 8/9 | 7.66 |
| widowx | TaskBlend_ranking | 12.50% | 8/9 | 7.66 |

**What changed:** GlobalBlend fixes Octo-Base versus RT-1-Begin on Move Near. Their simulated gap is -0.8 points; a training-selected weight of 0.3 makes the predicted gap +1.3 points. The observed physical gap is +33.3 points. No other eligible pair changes. Google weights are 0.2 for the held-out Close Drawer fold and 0.3 for the other four folds. All ranking-selected WidowX weights are zero.
TaskBlend fixes the same Move Near pair but introduces two errors on Apple into Closed Drawer: RT-1-X versus RT-1-15pct and versus RT-1-Converged. Each physical difference is 22.2 points in favor of RT-1-X. In that fold, inner validation chooses a weight of 1.0 for precision tasks, using global ability alone. The other drawer training tasks favor global rankings, but that rule does not hold for the apple task. This is a concrete failure of the coarse category to generalize.

## Uncertainty and robustness

The global Google improvement is -1.43 ranking-error percentage points. Its descriptive paired task-bootstrap 95% interval is [-4.29, 0.00] points. It is entirely one task and one comparison; dropping Move Near eliminates the gain. Four or five task clusters cannot support a strong generalization claim. Overlapping policy pairs are not independent.
Excluding each target simulated task from global ability preserves the primary ranking results: GlobalBlend 62/65 Google comparisons, TaskBlend 60/65; both 8/9 WidowX. The secondary MAE-selected global blend then also reaches 62/65 Google comparisons, illustrating sensitivity of selected weights to feature construction.

| Embodiment | Shuffled labels no worse than actual TaskBlend | Shuffled labels better than B1 |
|---|---:|---:|
| google_robot | 76/100 | 32/100 |
| widowx | 100/100 | 0/100 |

These seeded permutations are a diagnostic, not a p-value: there are few distinct task-label assignments, and assignments repeat. The meaningful labels do not establish a ranking advantage.

## Secondary MAE objective

| Embodiment | Matched simulation MAE | Global blend MAE | Task-aware blend MAE |
|---|---:|---:|---:|
| google_robot | 13.91 pp | 13.76 pp | 16.43 pp |
| widowx | 7.66 pp | 7.20 pp | 6.39 pp |

WidowX task-aware MAE improves by 1.27 points versus matched simulation, but the descriptive paired task-bootstrap interval for the difference is [-4.01, +0.52] points. It does not correct the carrot ranking. Google task-aware MAE worsens, with an additional ranking error on Pick Coke Can. The MAE objective is a separate experiment, not a post-hoc replacement for the ranking endpoint.

## Decision

Retain the simple global blend as a small exploratory candidate. Do not claim task-attribute ranking improvement or a validated trust rule. The remaining question is replication on additional compatible tasks, not how many more category features can be tuned on these nine tasks. A useful future extension would test independently available policy capability profiles; model selection on the same outer folds would require fresh validation.

Run `python -m src.reproduce`, or run `python -m src.blend_experiment` followed by `python -m src.blend_report`. The original frozen plan remains intact; blend_plan.json and data/blend_plan_freeze.json record this extension. Raw inputs were not changed. No trials, rollouts, robot-policy inference or paid API calls were used.
