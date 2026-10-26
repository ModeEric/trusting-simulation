"""Explicit paper/code discrepancy check; main constants remain unchanged."""
import json
import pandas as pd
from .common import RAW,PROCESSED,RESULTS,write_json
from .evaluate import grouped_predictions,score_predictions
from .metrics import author_pearson

def audit():
    data=pd.read_csv(PROCESSED/'simpler_paired.csv')
    paper=data.copy()
    task='widowx_put_eggplant_in_basket'
    # Direct transcription of Table 3 (PDF p.19). No inferred trials.
    adjustments={'octo-base':.233,'octo-small':.433}
    records=[]
    for policy,value in adjustments.items():
        ix=(data.task_id==task)&(data.policy_id==policy)
        records.append(dict(task_id=task,policy_id=policy,repository_real=float(data.loc[ix,'real'].iloc[0]),
                            paper_table3_real=value,source='PMLR PDF Table 3, p.19'))
        paper.loc[ix,'real']=value
    pd.DataFrame(records).to_csv(RESULTS/'paper_code_discrepancies.csv',index=False)
    p,_=grouped_predictions(paper);met,_=score_predictions(p)
    met.to_csv(RESULTS/'paper_values_sensitivity.csv',index=False)
    text='''# Source and uncertainty audit

The primary input is the repository's pinned REAL_PERF / SIMPLER_PERF constants,
not a reconstructed episode dataset. Nine dictionary task entries correspond to
eight named task families in the paper because open/close drawer are separated.
There are 30 Google Robot cells and 12 WidowX cells; RT-1 checkpoint variants and
the two Octo model sizes are not independent policy families.

## Paper and repository are not identical

- Table 2 (PMLR PDF p.18) includes OpenVLA-7B, which is absent from the constants.
  Paper correlations using that larger policy set should not equal the six-policy
  Google correlations reproduced by tools/calc_metrics.py.
- Table 3 (p.19) reports WidowX eggplant real success of 0.233 for Octo-Base and
  0.433 for Octo-Small; the constants use 0.250 and 0.400. This is a genuine
  source difference, not an assumed rounding correction. The reason is unknown.
- paper_values_sensitivity.csv substitutes only these two published values.
  Its B1 ordering and 10-point-margin eligible comparisons are unchanged.
  The paper's reported correlation values also use precision beyond some
  rounded displayed inputs, so tiny numerical differences are expected.

## Counts and intervals

Appendix A (p.15) describes 75 Google pick trials, 60 move-near trials, 54 trials
across open/close drawers (27 per direction), 27 apple-drawer trials, and 24
trials per WidowX task. Octo simulation averages three seeds; Google simulation
also varies arm/gripper colors (p.17). The paper's eggplant rates, including grasp
rates consistent with a denominator of 30, do not straightforwardly match a
denominator of 24; the rates do not establish the actual denominator.
The constants contain no count metadata and differ from Table 3. Therefore the
normalized primary table leaves n_trials blank, and no per-cell Bernoulli
standard errors or synthetic episodes are generated.

SIMPLER intervals resample tasks within each embodiment, with only five/four
tasks. They are descriptive, unstable summaries, not guarantees about new robot
tasks. The close-gap cutoff uses rounded observed real means and cannot certify
the true gap exceeds 10 points. Checkpoints remain correlated within each task.

Robocurve intervals independently resample 20 observed trials within each
model-task cell, conditional on the six selected tasks and recorded rubrics.
Trials with the same ordinal index are not treated as matched. Completion uses
score == 1, never the infrastructure status field. Zero-completion cells get
Wilson intervals rather than a misleading zero-width bootstrap interval.
Human graders knew the model, resets were manual, and rig/time dependencies may
remain. The CSV task Classify Objects is displayed as Sort Objects in the report;
the normalized task ID preserves the CSV name and metadata provides the alias.

## Data-use provenance

SIMPLER source is MIT licensed (license retained). RoboDojo repository is MIT;
website/data/video rights are not inferred from the code license. The public
scores and metadata are retained locally with attribution. The public release
excludes unlicensed website/paper snapshots and per-trial derived exports;
see THIRD_PARTY_NOTICES.md and docs/reproduction.md for release boundaries.
No video archives, models, or trajectories were downloaded. No model training,
robot trials, rollouts, inference, or paid API calls were performed.
'''
    (RESULTS/'source_audit.md').write_text(text)
    return records
if __name__=='__main__':audit()
