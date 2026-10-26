# Report assets

Start with the [technical overview](../docs/technical-report.md) and the original
[two-page note](Trusting_Simulation_Research_Note.pdf), which focuses on the
physical evidence. The later studies below concern prediction in simulation.

| Report | Scope |
| --- | --- |
| [Blend follow-up](blend_findings.md) | Exploratory SIMPLER calibration/ranking follow-up |
| [RoboDojo feasibility](robodojo_feasibility.md) | Physical results recovered, protocol compatibility still fails |
| [Simulation task-panel prediction](sim_prediction_findings.md) | Known-task, held-out policy-family prediction |
| [RoboTwin replication](robotwin_findings.md) | Independent simulated score matrix |
| [RoboTwin learning curve](robotwin_learning_curve_findings.md) | Training-policy evaluation coverage |
| [RoboTwin model sweep](robotwin_model_sweep_findings.md) | Expanded linear, bounded and ranking objectives |
| [LIBERO-Plus expansion](libero_plus_findings.md) | Binary outcomes; checkpoint and anchor coverage |
| [Source/uncertainty audit](source_audit.md) | Paper/code discrepancies and uncertainty limits |

CSV files contain selected aggregate summaries. Four original analysis figures
are included. References in detailed reports to other CSVs or figures identify
outputs generated under `results/` by reproduction; they are not all release assets.

`verification.json`, `full-validation.json`, `public-validation.json`, and test logs
record the successful clean runs and their scopes. `requirements-verified.txt`
records all installed package versions used for the release checks.

Run `python scripts/verify_release.py` to compare the released code and bundled
inputs with the tested snapshot, without installing numerical dependencies.
