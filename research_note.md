# When can simulation predict real-world robot rankings?

**Independent research audit | 24 October 2026 | Existing public data only**

Matched simulation is useful on SIMPLER, but these public samples do not establish
a reliable rule for when to skip physical testing. The prespecified task-aware
calibrator does not improve the strong matched-task baseline. The more useful
application finding is a concrete map of ranking failures and missing validation
data. This work was not commissioned or endorsed by Robocurve.

## Data and evaluation

The pinned SIMPLER constants contain **42 paired policy-task cells across nine
task entries**, split into five Google Robot tasks with six policies and four
WidowX tasks with three policies. These are aggregate success rates, not 42
independent tasks or thousands of independent policy comparisons. RT-1
checkpoints and Octo sizes share families. Open/close drawer are separate entries
but one of the paper's eight named families.

All fitted predictors hold out the entire physical test task and train only
within the same embodiment. B0 is global simulated ability; B1 is the matched
simulation score; B2 is affine calibration; C1 adds frozen articulation and
precision features with a fixed ridge penalty. A shrinkage-only B2 control is
also included. The primary comparison retains observed real gaps of at least
10 percentage points. Ties, close gaps, MMRV and score MAE are separate outputs.
Metrics are averaged by task before embodiment-level summaries.

## Absolute results

| Method | Google ranking error | Google MAE | WidowX ranking error | WidowX MAE |
|---|---:|---:|---:|---:|
| B0: global simulation | 9.7% | 20.8 pp | 12.5% | 9.7 pp |
| B1: matched simulation | **5.7%** | 13.9 pp | **12.5%** | **7.7 pp** |
| B2: affine calibration | 5.7% | **11.1 pp** | 12.5% | 9.6 pp |
| C1: task attributes | 5.7% | 19.6 pp | 12.5% | 11.1 pp |

B1 correctly orders **61/65 Google** and **8/9 WidowX** material-gap comparisons.
These fractions weight pairs and differ from the task-averaged errors above.
Across both strata, close non-tied pairs are correct in 7/10 cases. The counts
are descriptive; overlapping pairs and checkpoints are not independent.

All nine Pearson/MMRV outputs reproduce the pinned author implementations to
numerical precision. Per-task Pearson ranges from 0.571 to approximately 1.000;
MMRV ranges from 0 to 0.123. The strongest failures are concrete:

- **Close Drawer:** three wrong orderings among 14 material-gap pairs. Simulation
  ranks RT-1-X above RT-1-Converged, while physical success is 74.1% vs 92.6%.
- **Move Near:** simulation marginally favors RT-1-Begin over Octo-Base (5.0% vs
  4.2%), while real success favors Octo-Base by 33.3 points (35.0% vs 1.7%).
- **Carrot on Plate:** simulation favors Octo-Small over Octo-Base (9.7% vs
  8.3%), while physical success favors Octo-Base by 16.7 points.

Task-aware calibration adds no ranking benefit: with a positive simulation
slope, additive task features cannot change within-task ordering. Every fitted
slope is positive. Its poor MAE partly reflects the fixed shrinkage; against the
same-penalty B2 control, attributes change MAE by +0.7 points on Google and -2.1
on WidowX, still worse than B1. This does not rule out other task-aware methods.
Articulation and precision are collinear in Google; fine mechanism claims are
not identifiable.

The exploratory held-out trust score also fails to establish useful deferral.
At its fixed 0.90 confidence threshold, retained-comparison task-average error
rises from 5.7% to 7.1% on Google and from 12.5% to 16.7% on WidowX. Evaluated task
subsets change with coverage. These are uncalibrated scores, not probability
guarantees; no operating threshold is recommended. The third central figure is
therefore a data-coverage matrix rather than a persuasive-looking trust curve.

## External audit and physical stress test

**Correction to the first pass:** RoboDojo paper Table 2 provides **180 physical
method-task cells: 10 methods × 18 tasks**, with progress and success rate and
10 documented trials per cell. All 80 task-to-aggregate checks pass within
rounding tolerance. The website-only audit missed this source. The paper's
July tables are kept separate from the October website's 3,320 simulation cells
and 12 shared leaderboard names.

The stronger limitation is compatibility. Paper §3.2.1 explicitly describes
unpaired simulation/physical tasks; Appendix K reports separate domain
fine-tuning, often with different training budgets. Shared method names are not
the same evaluated policy. Formal matched-task external validation therefore
still stops. Even same-name Cover Blocks and Insert Tubes on ARX X5 require a
protocol audit and common checkpoints.

As a **descriptive cross-task method proxy**, the same-paper overall simulation
ranking has physical success-rate task-average errors of 10.0% on ARX X5 and
26.8% on Piper (five informative tasks each). Piper X has 0% error but only two
informative tasks; four tasks have all methods tied at zero success. These
counts use observed gaps ≥10 points, just one success out of ten trials, so they
are not confident estimates of true ordering. Overall progress correlation is
Pearson 0.700 / Spearman 0.576 over ten methods. No external task-aware model is
fitted, and category proxy results are not used to select a winner.

Published repeatability also depends on task. Across three physical runs,
π0.5's overall success-rate SD is 1.0 point, versus 23.1 points for Store in Safe.
Only SD summaries are available here, not raw round outcomes. Aggregate
stability does not establish task-level ranking stability.

All **360 Robocurve trials** reproduce the published CSV cell means and
completion counts. Astra averages 36.69% progress, Opus 5.5 36.01%, and Opus 5
19.88%. Astra minus Opus 5.5 is **+0.68 progress points**, with a conditional
95% within-cell bootstrap interval **[-4.51, +5.88]**. Astra leads by 22.5 points
on Stack Bowls; Opus 5.5 leads by 16 points on Cap Pen. Removing Stack Bowls or
Store In Safe reverses the aggregate leader. Full completions are only 5/120,
1/120 and 2/120 respectively. This is physical-only evidence about task mix,
not a sim-to-real result.

## Recommendation and smallest validating next step

**First establish whether compatible existing evaluations exist:** request
results for the **same evaluated checkpoint** on explicitly paired tasks,
including rubric/layout/controller versions and sampling weights. More
leaderboard averages or downloadable training checkpoints cannot fill that gap.
Cover Blocks and Insert Tubes are first protocol-inspection candidates; Store
in Safe and Stack Bowls motivate requesting raw repeated-round scores. The
paper reports 10-trial costs of 10.8, 9.0, 9.8 and 7.3 minutes respectively for
its π0.5 setup; these are not Robocurve cost estimates.

If existing exports cannot resolve this, a prospective pilot could use **12
prespecified tasks across three skill families, three exact policies and 20
physical trials per cell (720 trials)**, with matched simulation protocols and
blinded/randomized physical evaluation. Reuse simulation outcomes only when
versions match; any new execution belongs to a separately authorized stage.
Twenty trials per cell are a feasibility pilot, not adequate precision for most
10-point policy gaps. Measure variance and increase counts before making close
ranking decisions. The falsifiable test is whether a frozen simulation-based
trust rule lowers held-out task ranking error at a predeclared retained coverage
relative to B1. Keep several tasks per family for grouped validation and include
failure-prone and ordinary tasks, rather than selecting only apparent successes.

## Limits and reproducibility

LOTO holds out calibrator labels, not the original simulator-design process;
these published simulations were engineered for their real robot setups.
Only five/four SIMPLER tasks support each stratum. Task-cluster intervals are
unstable; task distributions are not a random robotics sample. The paper adds
OpenVLA to its Google table and reports different real eggplant scores for two
WidowX cells. The pinned repository is primary; substituting those paper values
does not change B1 orderings. Unknown denominators remain blank, with no
pseudotrials. Robocurve selected its six tasks using Astra, uses non-blinded
grading/manual resets, and differs from RoboDojo in hardware, agents and rubrics.
Bootstrap intervals omit selection and rig/time dependence. None of this proves
an irreducible simulation gap or a commercial moat.

Run `python -m src.reproduce` for the offline pipeline, figures, source audit and
tests. See `README.md`, `data/manifest.json`, `results/fold_audit.json`,
`results/source_audit.md` and the full task-level tables. The bundle includes
source code; no external repository has been published.

Sources: [SIMPLER paper](https://proceedings.mlr.press/v270/li25c.html),
[pinned metrics/data](https://github.com/simpler-env/SimplerEnv/blob/eddb5690749a1e34267807c0f7b5eede323918d7/simpler_env/utils/metrics.py),
[RoboDojo paper v3](https://arxiv.org/html/2607.04434v3),
[RoboDojo simulation](https://robodojo-benchmark.com/leaderboard?bench=sim),
[physical leaderboard](https://robodojo-benchmark.com/leaderboard?bench=realworld),
[rollout metadata](https://robodojo-benchmark.com/data/rolloutManifest.generated.json),
[Robocurve report](https://robocurve.org/opus-5-5-robodojo-rc-tier-1/),
[trial CSV](https://robocurve.org/opus-5-5-robodojo-rc-tier-1/data/trials.csv).
