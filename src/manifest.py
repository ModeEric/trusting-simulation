"""Create the human-reviewable provenance manifest from frozen acquisition logs."""
import sys,json,importlib.metadata
from .common import ROOT,RAW,RESULTS,write_json,verify_raw,sha256

def main():
    records=verify_raw()
    dependencies={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','reportlab']}
    sources=[]
    for path,r in sorted(records.items()):
        row=r.copy()
        if 'robodojo_app' in path:row['source_revision']='website-2026-10-24; not repository SHA'
        elif 'robodojo_rollout' in path:row['source_revision']='2026-10-23T00:00:00.000Z'
        elif 'robodojo_tree' in path:row['source_revision']=json.loads((ROOT/path).read_text())['sha']
        elif 'robocurve' in path:row['source_revision']='report-2026-09-23; content-addressed snapshot'
        sources.append(row)
    write_json(ROOT/'data/manifest.json',dict(analysis_date='2026-10-24',
       python=sys.version,dependencies=dependencies,sources=sources,
       derived_source_notes=['simpler_paper.txt is a pypdf text extraction, not an original file',
         'Website snapshots and repository commits are versioned independently',
         'No raw videos, policy inference or simulation rollouts were acquired'],
       plan_hashes=json.loads((ROOT/'data/plan_freeze.json').read_text()),
       exploratory_blend_plan=json.loads((ROOT/'data/blend_plan_freeze.json').read_text()),
       simulation_prediction_plan=json.loads((ROOT/'data/sim_prediction_plan_freeze.json').read_text()),
       simulation_prediction_coverage=json.loads((RESULTS/'sim_prediction_coverage.json').read_text()),
       robotwin_prediction_plan=json.loads((ROOT/'data/robotwin_plan_freeze.json').read_text()),
       robotwin_prediction_coverage=json.loads((RESULTS/'robotwin_coverage.json').read_text()),
       robotwin_learning_curve_plan=json.loads((ROOT/'data/robotwin_learning_curve_freeze.json').read_text()),
       robotwin_model_sweep_plan=json.loads((ROOT/'data/robotwin_model_sweep_freeze.json').read_text()),
       libero_plus_plan=json.loads((ROOT/'data/libero_plus_freeze.json').read_text()),
       libero_plus_provenance=json.loads((RAW/'libero_plus/provenance.json').read_text()),
       libero_plus_coverage=json.loads((RESULTS/'libero_plus_run.json').read_text()),
       coverage=dict(simpler_paired_cells=42,simpler_tasks=9,robocurve_trials=360,
                     robodojo=json.loads((RESULTS/'robodojo_feasibility.json').read_text())),
       uncertainty='See results/source_audit.md; no inferred SIMPLER episode counts'))
    write_json(RESULTS/'software_versions.json',dict(python=sys.version,dependencies=dependencies))
if __name__=='__main__':main()
