# Trusting Simulation: an audit of robot policy selection

Eric Modesitt · October 2026 · Independent public-data research

## Research question and contribution

Can simulated evaluations replace some physical testing when choosing a robot
policy? A useful answer must go beyond correlation: it must specify the decision,
compare credible baselines, examine consequential ranking errors, and establish
that proxy and target evaluations concern compatible tasks and checkpoints.

This report audits those requirements across five public sources. The physical
evidence comes from SIMPLER and Robocurve; RoboDojo exposes a protocol-compatibility
problem. Separate RoboTwin and LIBERO-Plus experiments ask whether a small observed
panel predicts a held-out policy's remaining **simulated** results. They test
prediction methodology, not physical transfer.

All outcomes were published by the source authors. This work performs statistical
analysis and fits small predictors locally. It runs no new robot-policy inference,
robot-policy training, simulator rollouts, or physical trials. Frozen analysis plans,
source hashes, grouped folds and target-label mutation tests provide an audit trail;
they are not an external preregistration.

## 1. SIMPLER: matched simulation is useful, with local reversals

The pinned SIMPLER repository contains 42 paired task-policy score cells: 30 Google
Robot cells and 12 WidowX cells across nine task entries and two embodiments.
Matched simulation correctly orders 61 of 65 eligible Google comparisons and eight
of nine WidowX comparisons. Eligibility requires at least a ten-point observed
physical success gap. Overlapping pairs share policies and tasks; these fractions
are descriptive counts rather than independent Bernoulli trials.

The leave-one-task-out analysis withholds every physical label for the test task
and fits within embodiment. Baselines include a global simulated average, matched
simulation, affine calibration, and a fixed-penalty calibration with two frozen
task attributes. The task-aware calibrator improves neither ranking nor score MAE
relative to matched simulation. Because its fitted simulation slopes are positive,
task-only additive attributes cannot change policy ordering within a task. This
model therefore tests absolute calibration; it does not rule out richer task-policy
interactions.

Closing drawers contributes three material ranking errors. Move Near and Carrot
on Plate contribute one each. On Move Near, simulation slightly favors RT-1-Begin
over Octo-Base, whereas physical success favors Octo-Base by 33.3 points. Useful
aggregate agreement does not eliminate important individual decisions.

The authors' Pearson/MMRV functions are reproduced. The paper and repository have
different policy coverage and two different eggplant values, which are reported
explicitly in sensitivity analyses. Unknown episode denominators remain unknown.
Task-cluster intervals use only five Google/four WidowX task entries and are
unstable. Holding out calibrator labels does not hold out SIMPLER's design around
these real robot setups.

An exploratory global blend corrects one Google ordering, from 61/65 to 62/65;
the task-aware blend falls to 60/65. This small follow-up does not establish a
reliable improvement or a calibrated rule for deferring physical tests.

Source: [SIMPLER paper](https://proceedings.mlr.press/v270/li25c.html) and
[pinned metrics/constants](https://github.com/simpler-env/SimplerEnv/blob/eddb5690749a1e34267807c0f7b5eede323918d7/simpler_env/utils/metrics.py).
See the [original note](../reports/Trusting_Simulation_Research_Note.pdf) and
[blend follow-up](../reports/blend_findings.md).

## 2. Robocurve: task mix can reverse the physical benchmark leader

The independent physical audit checks all 360 published trials, including every
source task-model mean and completion count. Astra averages 36.69% progress and
Opus 5.5 averages 36.01%, a 0.68-point difference. The conditional bootstrap interval
for the difference spans approximately -4.51 to +5.88 points.

Astra leads Stack Bowls by 22.5 progress points; Opus 5.5 leads Cap Pen by 16.
Omitting Stack Bowls or Store In Safe reverses the overall leader. Completion
is separate: five of Astra's 120 trials and one of Opus 5.5's 120 trials fully
complete the task. Infrastructure `status=success` is not task success.

The six tasks were selected using Astra outcomes. Human grading was not blinded,
resets were manual, and hardware/scaffolds differ from the original RoboDojo
protocol. The bootstrap conditions on these six tasks and does not model rig/time
dependence. This analysis evaluates robustness of a physical comparison; it is
not a paired test of simulated versus physical scores.

Source: [Robocurve report and source CSV links](https://robocurve.org/opus-5-5-robodojo-rc-tier-1/).

## 3. RoboDojo: physical outcomes exist, but the pairing gate fails

The first website-only inspection missed the paper's physical table. The corrected
audit recovers 180 method-task physical score cells: ten methods across 18 tasks,
with progress and completion metrics and documented ten-trial denominators.
Paper v1/v3 differences are retained in the audit.

However, simulation and physical tasks are deliberately unpaired and policies are
fine-tuned separately by domain. Shared method names do not establish identical
checkpoints. Cross-task correlations are descriptive method-level associations;
they cannot validate matched-task leave-one-task-out transport. The useful next
evidence is existing results with explicit checkpoint, task, embodiment, rubric,
controller and layout identifiers.

Source: [RoboDojo paper v3](https://arxiv.org/html/2607.04434v3).
See [combined feasibility audit](../reports/robodojo_feasibility.md).

## 4. RoboTwin: small panels predict held-out policies on known simulated tasks

The replication uses 25 jointly trained policy configurations across 50 tasks in
clean and randomized conditions: 2,500 published means. Conservative ancestry
proxy groups are held out together. Ten same-condition anchor scores predict the
remaining 40 known task scores; training policies may supply outcomes on those
target tasks. This is neither unseen-task prediction nor physical validation.

AnchorMean achieves 82.5% clean and 91.5% randomized ranking agreement for observed
gaps of at least ten points. Close-gap agreement falls to roughly 62–64%. Latent
prediction improves score MAE over scalar calibration, from 11.46 to 10.10 points
clean and 13.53 to 11.57 randomized, without a convincing ranking improvement.

Training-policy coverage and anchor coverage are varied separately. Expanded linear
capacity is not selected at full coverage. A bounded logit-space model modestly
improves score error. Direct ranking training improves agreement by only 0.10/0.20
points over AnchorMean, with paired descriptive intervals including zero. Revealing
more anchors has a larger observed ranking effect on an unchanged target panel.
Ancestry is incompletely documented and episode denominators are not reconstructed.

Source: [pinned RoboTwin leaderboard](https://github.com/RoboTwin-Platform/RoboTwin-Platform.github.io/blob/0abb5a03455073fa74047a1c4015b0f88cedc254/data/robotwin_leaderboard.json).
Details: [replication](../reports/robotwin_findings.md),
[learning curve](../reports/robotwin_learning_curve_findings.md),
[model sweep](../reports/robotwin_model_sweep_findings.md).

## 5. LIBERO-Plus: better probability prediction does not imply better ranking

All 361,080 binary outcomes in the 36 published files are checked against compressed
and uncompressed hashes, case alignment and published seven-axis means. The primary
cohort is 24 distinct, unmodified-prompt pi0.5 checkpoints evaluated on 10,030 cases
derived from 40 base tasks: 240,720 binary observations. Other protocols, changed
prompts and repeat passes do not enter primary prediction.

Nine frozen intervention groups keep related ablations/seeds together; they are
not independent architecture or ancestry families. For a held-out checkpoint,
all variants of a target base task remain hidden. Three repeated three-fold
partitions and three nested anchor orders vary training-checkpoint coverage and
observed anchor coverage separately, with 16 target tasks fixed per anchor order.
Inner grouped validation selects latent rank and ridge using Brier score.

At eight anchors, increasing the average training cohort from 5.6 to 16 checkpoints
reduces latent Brier from 0.0650 to 0.0561. Its final task-axis MAE is 4.31 points,
versus 5.01 for a simple additive offset. Yet its ranking agreement is 65.7%, versus
84.1% for that offset. Its Brier advantage over an axis-specific offset is small
and uncertain, and disappears in the stricter camera/noncamera holdout.

The study shows an empirical checkpoint-coverage trend, not a fitted scaling law,
physical-transfer result, or broad architectural generalization. Repeated splits,
related checkpoints and perturbed versions of base tasks do not create independent
evidence. Details and uncertainty intervals are in the
[LIBERO-Plus report](../reports/libero_plus_findings.md).

Source: [pinned Targeted Data outcome release](https://github.com/Alakazam-studios/targeted-data-libero-plus/tree/554a0d06924f64f15551119e393373a9bc008771).

## Implications and reproducibility

A model-selection audit should distinguish absolute prediction quality from
ranking, match task/checkpoint protocols before estimating transfer, separate
progress from completion, compare against simple baselines, and report sensitivity
to task composition. These data do not establish when physical testing can safely
be skipped. Stronger claims need representative paired evaluations and an
independently tested decision rule.

The [reproduction record](reproduction.md) distinguishes the complete clean run
using the recovered archive from the publicly rerunnable licensed-data subset.
Original source files with uncertain redistribution rights remain excluded.
The repository's documentation makes the remaining data-availability gap explicit.
