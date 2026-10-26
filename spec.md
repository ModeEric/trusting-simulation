# When Can Simulation Predict Real-World Robot Rankings?

Implementation brief derived from the specification supplied by the user on
2026-10-24. The supplied specification governs; this file condenses it for the
reproducible repository. See analysis_plan.json for exact implementation choices
and analysis_amendments.md for clarifications made before residual analysis.

## North-star question

Which parts of a cheap simulation leaderboard are safe to trust, and which
physical experiments are essential for selecting the better robot policy?

## Hard constraints

Use existing published results, public metadata and recorded traces only. No
new robot trials, simulation rollouts, model inference, training, or paid API
calls. Do not download multi-GB trajectory archives. Preserve benchmark versions,
source URLs, checksums, checkpoints, embodiments, score types, protocols and
sampling weights. Do not synthesize trials from published averages. Positive
results or statistical significance are not necessary; do not add complexity
merely to obtain a favorable finding. Do not imply Robocurve endorsement.

## Stages and gates

1. SIMPLER proof of concept. Extract REAL_PERF and SIMPLER_PERF from pinned
   source with a narrow AST parser. Join exact task-policy IDs; reproduce
   authors' per-task Pearson and MMRV, independently test Spearman and pairwise
   orderings. Publish missingness and effective task/policy-family counts.
2. RoboDojo portability audit. Find same-model task-level simulated and physical
   outcomes. Verify exact checkpoint, task, robot, controller, rubric and layout
   compatibility. Proceed to task-level external validation only when multiple
   shared policies and at least two task families have defensible matches.
   Aggregate-only results permit descriptive aggregate concordance, not
   held-out task prediction. Task-name similarity is insufficient.
3. Robocurve physical stress test. Analyze the published three-model, six-task,
   20-trials-per-cell table. Report progress differences, completion, conditional
   intervals and ranking sensitivity to task mix. Match simulations only if
   policy/scaffold/hardware/scoring correspondence is defensible. Otherwise
   report a data gap and the smallest next validating experiment.

## Data model

Observation: dataset/revision/domain/embodiment/checkpoint/task/metric. Keep
success rate and partial-credit progress separate. Normalize percentages to
[0,1]. Retain RT-1 checkpoint identities and an RT-1 family ID. Exact matches
are eligible for unqualified paired analysis; adapted/category-only/unmatched
comparisons remain explicitly weaker. Unknown n_trials stays missing.

Files: task_scores.csv-type normalized tables, task_metadata.csv-type frozen
annotations, and episodes.csv-type records only where observed trials exist.

## Taxonomy

Hand-auditable grasp, placement, articulation, precision/contact, multistep and
deformable indicators, plus official category and confidence. Only one or two
prespecified attributes enter fitted models. Freeze before residual inspection;
report alternate reasonable labels as sensitivity rather than replacing labels
after seeing errors. Preserve embodiment strata and avoid rich feature search.

## Prediction models

- B0: average simulated performance across available tasks, with a sensitivity
  excluding the target simulated task.
- B1: direct matched-task simulated success; essential strong baseline.
- B2: global sim-to-real calibration fitted to training tasks only.
- C1: B2 plus at most two frozen task attributes with strong fixed shrinkage.
- C2: selective trust/defer procedure based only on out-of-fold predicted error
  or uncertainty; no threshold selection using held-out physical labels.

No free intercepts for unseen tasks. At prediction time task metadata and all
simulated outcomes are available, but the target task's real outcomes are not.

## Validation and measurements

Primary: leave-one-task-out, entirely within robot embodiment. Secondary:
coarse-task-family and policy-family holdout when identifiable. Never random
split task-policy cells. Use fixed regularization or inner grouped CV.

Primary decision outcome: task-conditional pairwise ranking error. Exclude real
ties and mark real gaps smaller than 0.10 as indeterminate; report coverage and
eligible comparison counts. Add MMRV, score MAE and descriptive correlations.
Summarize tasks before pooling; preserve raw task numbers, paired method-minus-
baseline deltas and leave-one-task sensitivity. Task-cluster bootstrap intervals
with so few tasks are descriptive, not precise population guarantees.

Falsification: shuffled task attributes, removal of RT-1 family, embodiment
stratification, direct B1 comparison, close/large gap strata, alternate labels.
Do not claim category validity from one held-out task. Interpret negative or
underpowered results explicitly.

## Deliverables

Reproducible source with one-command offline analysis, pinned small public
inputs, provenance/coverage manifest, grouped validation tests, 2-3 central
figures, short research note, and prioritized recommendation. Central figures:
held-out predicted vs observed physical scores; task-level ranking/calibration;
selective risk-coverage only if defensible, otherwise a coverage/matching matrix.
Physical-only Robocurve results may be supplemental.

## Limitations to retain

Large episode totals do not create many independent task-policy observations.
Tasks are not a random robotics sample; checkpoints are correlated. Unknown
trial counts, initial conditions, scene resets, score definitions, robot and
policy scaffolds can dominate measured differences. High aggregate correlation
does not guarantee correct pair rankings. Six Robocurve tasks were selected
using Astra outcomes. Another model's simulated traces do not validate the
Astra/Opus comparison. A current simulation failure is not proof of an
irreducible transfer gap or a commercial moat.

## Primary sources

- https://proceedings.mlr.press/v270/li25c.html
- https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/utils/metrics.py
- https://github.com/simpler-env/SimplerEnv/blob/main/tools/calc_metrics.py
- https://github.com/RoboDojo-Benchmark/RoboDojo
- https://robodojo-benchmark.com/leaderboard?bench=sim
- https://robodojo-benchmark.com/leaderboard?bench=realworld
- https://robodojo-benchmark.com/doc/sim-tasks/
- https://robocurve.org/opus-5-5-robodojo-rc-tier-1/
- https://robocurve.org/opus-5-5-robodojo-rc-tier-1/data/trials.csv
