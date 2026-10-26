#!/usr/bin/env python3
"""Refresh the small public report selection after a successful complete run."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / 'results'
    state = json.loads((results / 'validation.json').read_text())
    if state['status'] != 'passed' or not any('src.libero_plus_experiment' in x['command'] for x in state['stages']):
        raise SystemExit('A complete successful reproduction, including LIBERO-Plus, is required.')
    output = ROOT / 'reports'
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    names = ['blend_findings.md', 'sim_prediction_findings.md', 'robotwin_findings.md',
             'robotwin_learning_curve_findings.md', 'robotwin_model_sweep_findings.md',
             'libero_plus_findings.md', 'robodojo_feasibility.md', 'source_audit.md',
             'simpler_summary.csv', 'robotwin_summary.csv', 'libero_plus_summary.csv',
             'libero_plus_paired_deltas.csv']
    for name in names:
        shutil.copy2(results / name, output / name)
    for name in ['01_simpler_predictions.png', '04_robocurve_physical_stress_test.png',
                 '08_robotwin_prediction.png', '12_libero_plus_scaling.png']:
        shutil.copy2(results / 'figures' / name, output / 'figures' / name)
    shutil.copy2(ROOT / 'output/pdf/Trusting_Simulation_Research_Note.pdf', output)
    print('Refreshed selected reports, tables, four figures and the two-page note.')


if __name__ == '__main__':
    main()
