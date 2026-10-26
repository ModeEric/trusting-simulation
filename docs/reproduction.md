# Reproduction and data availability

## Publicly rerunnable studies: SIMPLER and LIBERO-Plus

The public release includes the licensed numerical source files required for
these two studies, with original MIT/Apache-2.0 notices. This command requires
no recovered archive and performs no network downloads:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.reproduce_public
```

It regenerates the primary SIMPLER evaluation and the complete LIBERO-Plus
experiment/report, then runs the applicable metric, leakage, source-validation
and acquisition tests. `results/public_validation.json` records its own status;
it never claims that the full multi-benchmark suite ran. GitHub Actions runs this
public subset on Python 3.11.

The release uses the exact top-level versions in `requirements-lock.txt`.
Transitive packages are not locked by that file. The verified environment is
recorded in `reports/verification.json` and its accompanying package inventory.

## Complete audit using the recovered archive

From a clean release checkout, with the same dependencies installed:

```bash
python scripts/acquire_inputs.py --archive /path/to/Trusting_Simulation_Analysis.zip
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.reproduce
```

The import reads only the eight excluded computational inputs from the ZIP,
validates byte counts and SHA-256 hashes, and leaves the original archive intact.
It does not import archived experiment outputs or execute archived code. The ZIP
is a private recovery artifact and is not hosted as a public release asset.

The complete offline pipeline rebuilds normalized data, SIMPLER calibration and
blend experiments, corrected RoboDojo paper audits, simulated task-panel prediction,
RoboTwin prediction/learning curves/all three model sweeps, LIBERO-Plus ingestion/
prediction/report, physical trial analysis, source audits, figures and the two-page
note. It then runs all unit tests. No `data/processed/` or `results/` files are
required at startup. `results/validation.json` is reset at each run, including on
input-verification failure, so an old passed status cannot certify a new run.

Selected read-only report assets live in `reports/`. After a successful full run,
`python scripts/package_reports.py` refreshes them. Large generated CSV/gzip/NPZ
outputs remain untracked.

## Public download audit, October 26, 2026

`python scripts/acquire_inputs.py` attempts direct upstream acquisition of the
excluded inputs. All bytes must match `data/release_inputs.json`; it never accepts
a changed leaderboard or HTML rendering as the archived source.

| Excluded input | Fresh acquisition result |
| --- | --- |
| Robocurve report HTML | Exact archived SHA-256 reproduced |
| Robocurve trials CSV | Exact archived SHA-256 reproduced |
| Robocurve cells CSV | Exact archived SHA-256 reproduced |
| RoboTwin pinned leaderboard JSON | Exact archived SHA-256 reproduced |
| RoboDojo hashed JavaScript bundle | HTTP 404 |
| RoboDojo rollout manifest | Returned bytes differ from the archived snapshot |
| RoboDojo paper v1 HTML | Returned bytes differ from the archived snapshot |
| RoboDojo paper v3 HTML | Returned bytes differ from the archived snapshot |

Differences in returned HTML do not establish that paper scores changed; page
rendering alone can change bytes. The source checks intentionally make no such
equivalence assumption. Full public reproduction remains limited by four exact
snapshots and by redistribution permission for excluded inputs. Closing that gap
requires permission to host the original snapshots, a stable upstream archive,
or a separately versioned and reviewed ingestion revision. None is silently
assumed in this release.

## Verification record and scope

`reports/verification.json` identifies the tested code/input fingerprint, full
stage status, test result, clean-start conditions and public-subset result. The
clean runs use the same released code, plans and licensed files; only excluded
inputs are restored from the recovery archive for the full run. This separates
local end-to-end verification from public source availability.

### Numerical comparison with the recovered outputs

The released primary SIMPLER, RoboTwin, RoboTwin model-sweep and LIBERO-Plus
summary tables match the archived numerical results within 1e-9. The reduced-
coverage RoboTwin latent learning curve has small differences: at most 0.063
percentage points in a point estimate, with 45 of 2,250 tuning choices differing
and no split-membership changes. Full-coverage primary results are unchanged.
The archive used Python 3.12.14; this release was verified on 3.11.15 with the same
top-level numerical package versions. The original SVD predictor retains
rank-deficient directions, so numerical sensitivity is a plausible contributor;
its cause was not isolated. No frozen model code was changed to force agreement.
Released reports use the clean-run values. See
[`archive-comparison.json`](../reports/archive-comparison.json) for exact deltas.

No new robot trials, simulation rollouts, policy inference or paid APIs occur in
either reproduction path. Internet access is needed only to install dependencies
or deliberately acquire excluded inputs.
