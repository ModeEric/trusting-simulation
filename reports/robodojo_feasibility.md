# RoboDojo public-data feasibility audit — corrected continuation

**Physical task scores are public. Formal matched sim-to-real validation still fails the compatibility gate.**

The first pass inspected the website bundle and video manifest, but missed paper Table 2. This continuation recovers 180 physical method-task cells (10 methods × 18 tasks), each with progress and success rate. The paper documents 10 trials per cell; 1,800 underlying trials are not 1,800 independent tasks and no raw episodes have been reconstructed.

## Provenance and extraction

Primary source: [arxiv:2607.04434v3; leaderboard 2026-07-03](https://arxiv.org/html/2607.04434v3); Tables 1, 2, 5 and 9. Tables 1, 5 and 9 match v1, but Table 2 corrects Spirit v1.5 on Piper X: Cap Pen progress changes from 0 to 4 points, and the embodiment average changes from 4 to 0.7. The v3 row is internally consistent. All 80 v3 physical aggregate checks agree within 0.1 percentage point (rounding tolerance). Task order comes from the explicit note under Table 2. Human teleoperation is excluded. See robodojo_paper_version_changes.csv.
Table 1 supplies simulation category/overall scores for 30 methods, including all 10 physical methods. This July 3 paper snapshot is kept separate from the October website snapshot (62 simulated and 12 physical names). October simulation task scores are not silently joined to July physical scores.

## Why the gate remains closed

Paper §3.2.1 explicitly says the simulation and real tasks were not designed as one-to-one pairs and that the real benchmark does not measure direct matched-task transfer. Appendix K specifies separate simulation/physical fine-tuning, often with different numbers of optimization steps; §5.2 trains a separate seed per physical embodiment. Shared method names therefore do not identify the same evaluated checkpoint. Even Cover Blocks and Insert Tubes need a protocol audit despite matching names and ARX X5 labels.
The normalized tables use domain/embodiment-specific policy identifiers and a separate method_id for exploratory method comparisons. There are zero verified exact pairs. No B1 matched-task baseline, C1 external validation or LOTO sim-to-real claim is made.

## Descriptive cross-task proxy analysis

Use the same-paper overall simulation ranking to order separately trained methods on each physical task. No coefficients are fitted. Progress and success are separate tracks. The 10-point observed physical-gap threshold is descriptive, not a confidence criterion: with 10 trials, one success changes the observed success rate by 10 points. Predicted ties receive 0.5 loss, as in SIMPLER; the overall-score proxy has no eligible predicted ties. Overlapping policy pairs and related methods are dependent. No significance claim or task-population interval is made.

| Metric | Embodiment | Informative tasks / 6 | Wrong / eligible pairs | Task-average error |
|---|---|---:|---:|---:|
| progress | arx_x5 | 5/6 | 22/107 | 17.5% |
| progress | piper | 5/6 | 28/121 | 18.8% |
| progress | piper_x | 3/6 | 2/21 | 22.2% |
| success_rate | arx_x5 | 5/6 | 12/84 | 10.0% |
| success_rate | piper | 5/6 | 31/105 | 26.8% |
| success_rate | piper_x | 2/6 | 0/18 | 0.0% |

Tasks with no eligible pairs stay in the coverage count and have undefined error, not zero error. All five category proxies are also exported for inspection; none is selected as a winning predictor. These are cross-task method associations, not measured simulation failures.

| Metric | Overall Pearson | Overall Spearman | Shared methods |
|---|---:|---:|---:|
| progress | 0.700 | 0.576 | 10 |
| success_rate | 0.632 | 0.632 | 10 |

## Repeated physical evaluation and measurement cost

Table 9 reports standard deviations across three runs for π0.5, GalaxeaVLA and InternVLA-A1. It does not release the three run-level outcomes. Overall success-rate SDs are 1.0, 1.2 and 1.3 points, while Store in Safe SDs are 23.1, 11.5 and 17.3 points. Stack Bowls has Galaxea progress SD 16.6 points. Aggregate stability can mask substantial task variation; these are run SDs, not errors on a population mean.
Table 5 reports 202 minutes for 180 π0.5 physical trials, including resets, local RTX4090 inference and execution. For a compatibility inspection, Cover Blocks costs 10.8 minutes per 10 trials and Insert Tubes 9.0 minutes; Store in Safe costs 9.8 and Stack Bowls 7.3. These costs apply only to the published RoboDojo setup, not Robocurve or a future model.

## Resource needed next

Request existing **same evaluated checkpoint** results on explicitly paired tasks with compatible embodiment, controller, layouts, horizon and rubric. Ask whether such evaluations exist; recovering more current leaderboard averages or downloading training checkpoints does not fill this gap. For a narrower cross-task method study, request the current two missing methods’ task tables, frozen submission artifacts, simulation per-task scores from the same revision, and repeated-run scores. Raw physical round data would help assess task-level ranking stability.

No videos, checkpoints or large trajectory archives were downloaded. Repository code licensing does not automatically grant redistribution rights for all paper/site/video assets. Preserve source attribution; check terms before a public mirror.

Files: `robodojo_paper_real_task_scores.csv`, `robodojo_paper_sim_categories.csv`, `robodojo_paper_repeated_run_sd.csv`, `robodojo_paper_evaluation_cost.csv` in data/processed; `robodojo_paper_proxy_*` and aggregate checks in results. `robodojo_coverage_matrix.csv` marks July physical availability separately from October simulation proxies.

Original website-only audit is retained as `robodojo_website_feasibility.md`; its scoped absence finding is superseded by this combined report.

Source: https://arxiv.org/html/2607.04434v3
Website: https://robodojo-benchmark.com/leaderboard
