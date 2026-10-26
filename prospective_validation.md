# Prioritized next validation

No new experiments were run or authorized by this implementation.

## Priority 1: establish whether compatible existing evaluations exist

The continuation recovered RoboDojo's July physical table: 180 method-task
cells, ten trials each. The remaining blocker is not simply a missing score
export. Its simulated and physical tasks were deliberately unpaired, and
policies were separately trained. Ask whether the same evaluated checkpoints
have existing results on explicitly paired tasks. Each record should include:

source_revision, model_id, checkpoint_id, policy_scaffold_version, task_id,
embodiment, controller_version, layout_distribution_version, rubric_version,
metric, score, successes, n_trials, seed_or_reset_id, evaluation_timestamp,
aggregation_weight, log_url, redistribution_terms.

Use exact-policy, exact-task and same-embodiment records only after reviewing the
objective, rubric and distribution. Cover Blocks and Insert Tubes on ARX X5
are first candidates because names and embodiment labels overlap. These two
tasks are enough for an initial compatibility check, not a task-family
prediction claim. The 1,910 video entries cannot be converted to measured
outcomes without grading or score metadata, and are not guaranteed to be a
complete trial inventory. Paper-level trial counts do not make the video list
complete. Keep July and October revisions separate.

For a narrower method-level proxy study, request July per-task simulation scores,
current physical tables for InternW0-Δ and OpenWAM-α, exact submitted artifacts,
and all three physical round results for π0.5, GalaxeaVLA and InternVLA-A1.
These would improve descriptive comparisons but would not create matched
sim-to-real validation.

Prioritize evidence by purpose, rather than claiming a learned trust rule:

| Purpose | First tasks | Existing evidence | Next observation |
|---|---|---|---|
| Check possible protocol alignment | Cover Blocks; Insert Tubes, ARX X5 | Same names/hardware labels; original tasks still unpaired | Explicit objectives, geometry, controller, layouts, rubrics and common checkpoint identity |
| Quantify repeated-run ranking stability | Store in Safe; Stack Bowls | Safe success SD up to 23.1 pp; Galaxea bowl progress SD 16.6 pp across three runs | Raw model × task × round scores and reset metadata |
| Avoid an all-zero completion benchmark | Piper X tasks | Only two of six tasks distinguish methods by completion in July table | Keep progress and completion separate; include attainable tasks without choosing them from one model's wins |

The published π0.5 10-trial timing is 10.8 minutes for Cover Blocks, 9.0 for
Insert Tubes, 9.8 for Store in Safe and 7.3 for Stack Bowls, including resets,
local RTX4090 inference and execution. This supports a rough within-setup cost
comparison only, not a transferable Robocurve budget. Sources: RoboDojo paper
Tables 2, 5 and 9, https://arxiv.org/html/2607.04434v3.

For Robocurve, request matched simulations of the exact Astra/Opus models,
Inspect Robots 0.58.0 scaffold/effort, YAM hardware/controller, task layouts and
progress rubrics if such recordings already exist. Original RoboDojo ARX X5
simulations are not substitutes. A shared display name is insufficient.

## Priority 2: a separately planned pilot, only if needed

A useful pilot would cover 12 tasks, three families with four tasks each, three
exact model configurations, and 20 physical trials per task-model cell: 720
trials. Freeze a representative task list without consulting one model's
outcomes. Within each family, include easier and difficult cases. Balance rigs,
trial order and reset layouts across models; use blinded grading and record
both binary completion and partial progress independently. Use existing matched
simulation results where possible. A new simulator or new rollout data would
require a later stage and is outside this sprint.

This budget is for feasibility and large reversals. At success rates near 0.5,
two independent estimates with 20 trials each give a standard error for their
difference of about 0.158; a normal-approximation 95% half-width is about 0.31.
Even 192 trials per model only bring that approximate half-width to 0.10. This
is a precision illustration, not a power calculation or universal sample-size
recommendation. Pairing common randomized layouts can reduce variance only if
measured covariance supports it. Sequential testing requires a predeclared
stopping rule rather than repeatedly checking ordinary confidence intervals.

The main falsifiable outcome is held-out, task-averaged ranking error at a
predeclared coverage (for example 75%), compared with matched simulation B1.
Select the rule and any threshold only on training tasks; hold out whole
families as an additional stress test. Report changes in evaluated task mix,
indeterminate physical gaps, failure counts, and unfiltered outcomes. Do not
choose a safety/trust probability from the current tiny residual sample.

## Useful hypotheses, not established rules

- Contact/terminal-state tasks may require more physical checking. Current
  close-drawer and carrot failures motivate testing; open-drawer performance
  shows that a broad articulation label is not enough.
- A small simulated gap may deserve deferral. The Move Near error is a stark
  example, but near-zero gaps can also accompany correct physical rankings.
- Task-only additive calibration cannot repair individual policy orderings.
  Policy-by-task interactions or direct pair-error prediction would be needed
  for that, and require a much broader matched dataset.

No author outreach or external messages were sent.
