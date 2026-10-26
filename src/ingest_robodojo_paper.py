"""Recover published physical scores; retain the unmatched-domain boundary.

Table 2 contains physical task outcomes absent from the website's static
leaderboard table. Table 1 is a SAME-PAPER snapshot of simulation categories.
Joining by method is only a cross-task, separately-trained-method proxy.
No checkpoint or exact task equality is inferred from this join.
"""
import json
import re
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from .common import RAW, PROCESSED, RESULTS, validate_scores, verify_raw, write_json
from .paper_tables import read_tables

URL = 'https://arxiv.org/html/2607.04434v3'
REVISION = 'arxiv:2607.04434v3; leaderboard 2026-07-03'
TABLE_IDS = ('S5.T1.6.1', 'S5.T2.6.1', 'S6.T5.6.1', 'A3.T9.6.1')
# Explicit task order printed immediately below paper Table 2. Not inferred
# from website ordering, score patterns, or task descriptions.
TASK_ORDER = {
    'ARX X5': ['cover_blocks', 'make_bread', 'make_food',
               'pack_and_pour_fruit', 'store_in_safe', 'insert_tubes'],
    'Piper': ['stack_and_cover_blocks', 'fill_pen_holder',
              'put_objects_into_basket', 'insert_charger', 'stack_bowls',
              'stand_up_bottles'],
    'Piper X': ['classify_objects', 'disassemble_LEGO', 'hang_mugs',
               'pack_objects_into_backpack', 'sweep_blocks', 'cap_pen'],
}
EMBODIMENT = {'ARX X5': 'arx_x5', 'Piper': 'piper', 'Piper X': 'piper_x'}
METRICS = ('progress', 'success_rate')
METHODS = {'Pi-05', 'Pi-0', 'GROOT-N1.7', 'InternVLA-A1', 'GalaxeaVLA (G0)',
           'Xiaomi-Robotics-0', 'X-VLA', 'StarVLA', 'Spirit v1.5', 'DM0'}


def method_id(text):
    if text.startswith('π0.5'): return 'Pi-05'
    if text.startswith('π0'): return 'Pi-0'
    if text.startswith('GR00T-N1.7'): return 'GROOT-N1.7'
    if text.startswith('StarVLA-'): return 'StarVLA'
    if text.startswith('GalaxeaVLA'): return 'GalaxeaVLA (G0)'
    # Only strip a terminal author citation; keep method variant parentheses.
    return re.sub(r' \([^()]*et al\.,[^()]*\)$', '', text)


def score_pair(text):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)%?', text)
    if not match:
        raise ValueError(f'Unexpected score cell: {text!r}')
    values = tuple(float(x) / 100 for x in match.groups())
    if not all(0 <= x <= 1 for x in values):
        raise ValueError('Score outside [0,1]')
    return values


def extract(tables):
    sim, physical, aggregates = [], [], []
    rows = tables[TABLE_IDS[0]]
    categories = rows[0][1:]
    if categories != ['Generalization', 'Precision', 'Long-Horizon', 'Memory', 'Open', 'Average']:
        raise ValueError('Simulation header changed')
    for row in rows[1:]:
        method = method_id(row[0])
        if method == 'Human Expert (Teleop)': continue
        if len(row) != 7: raise ValueError('Unexpected sim row')
        for category, cell in zip(categories, row[1:]):
            for metric, score in zip(METRICS, score_pair(cell)):
                sim.append(dict(source='robodojo_paper', source_revision=REVISION,
                    domain='sim', embodiment='arx_x5', method_id=method,
                    policy_id=method+'::sim_finetune', category=category,
                    metric=metric, score=score, raw_cell=cell, n_trials=None,
                    source_url=URL+'#S5.T1'))
    current = None
    for row in tables[TABLE_IDS[1]][1:]:
        if len(row) == 10:
            current = method_id(row[0]); raw_method = row[0]
            robot = row[1]; cells = row[2:8]; avg = row[8]; overall = row[9]
            if robot != 'ARX X5': raise ValueError('Unexpected row-span order')
            if current != 'Human Expert (Teleop)':
                for metric, score in zip(METRICS, score_pair(overall)):
                    aggregates.append(dict(method_id=current, embodiment='overall', metric=metric, score=score))
        elif len(row) == 8 and current:
            robot = row[0]; cells = row[1:7]; avg = row[7]
        else: raise ValueError(f'Unexpected real row: {row}')
        if current == 'Human Expert (Teleop)': continue
        if current not in METHODS: raise ValueError(f'Unknown physical method: {current}')
        for metric, score in zip(METRICS, score_pair(avg)):
            aggregates.append(dict(method_id=current, embodiment=EMBODIMENT[robot], metric=metric, score=score))
        for position, (task, cell) in enumerate(zip(TASK_ORDER[robot], cells), 1):
            for metric, score in zip(METRICS, score_pair(cell)):
                physical.append(dict(source='robodojo_paper', source_revision=REVISION,
                    domain='real', embodiment=EMBODIMENT[robot], method_id=current,
                    policy_id=current+'::real_finetune::'+EMBODIMENT[robot],
                    policy_family=current, raw_method=raw_method, task_id=task,
                    task_column=position, metric=metric, score=score, raw_cell=cell,
                    n_trials=10, n_trials_source='paper sections 3.2.3 and 5.2',
                    protocol_id='RoboDojo-RealEval-paper-2026-07-03',
                    matched_task_id='', match_quality='unmatched',
                    checkpoint_identity='separate domain training; artifact hashes unverified',
                    source_url=URL+'#S5.T2'))
    sim, physical, aggregates = map(pd.DataFrame, (sim, physical, aggregates))
    validate_scores(sim, ['source_revision', 'policy_id', 'category', 'metric'])
    validate_scores(physical, ['source_revision', 'policy_id', 'task_id', 'metric'])
    if len(physical) != 360 or physical.method_id.nunique() != 10:
        raise ValueError('Unexpected physical table coverage')
    checks = []
    for r in aggregates.itertuples():
        d = physical[(physical.method_id == r.method_id) & (physical.metric == r.metric)]
        if r.embodiment != 'overall': d = d[d.embodiment == r.embodiment]
        mean = float(d.score.mean())
        checks.append(dict(method_id=r.method_id, embodiment=r.embodiment,
            metric=r.metric, recomputed=mean, published=r.score,
            absolute_difference=abs(mean-r.score),
            passes_rounding_tolerance=bool(abs(mean-r.score) <= .001 + 1e-12)))
    checks = pd.DataFrame(checks)
    if not checks.passes_rounding_tolerance.all():
        raise ValueError('Physical task means fail aggregate consistency check')
    return sim, physical, aggregates, checks


def stability_and_cost(tables):
    stability = []
    for row in tables[TABLE_IDS[3]][2:]:
        if len(row) != 7: raise ValueError('Unexpected stability row')
        # Table 9 uses connect_charger; Table 2 uses insert_charger.
        # Preserve source label and explicit within-physical-table alias.
        raw_task = row[0]
        task = 'insert_charger' if raw_task == 'connect_charger' else raw_task
        for j, method in enumerate(['Pi-05', 'GalaxeaVLA (G0)', 'InternVLA-A1']):
            for k, metric in enumerate(['success_rate', 'progress']):
                stability.append(dict(source_revision=REVISION, task_id=task,
                    source_task_label=raw_task, method_id=method, metric=metric,
                    across_run_sd=float(row[1+2*j+k])/100, n_runs=3,
                    statistic='published standard deviation; not standard error',
                    raw_round_scores_available=False, source_url=URL+'#A3.T9'))
    cost = []
    for row in tables[TABLE_IDS[2]][1:-1]:
        if len(row) != 6: raise ValueError('Unexpected cost row')
        for j in range(0,6,2):
            label = row[j]
            cost.append(dict(source_revision=REVISION,
                task_id='insert_charger' if label == 'connect_charger' else label,
                source_task_label=label, method_id='Pi-05', n_trials=10,
                total_minutes=float(row[j+1]), includes='reset + inference + execution',
                scope='original RoboDojo local RTX4090; not Robocurve', source_url=URL+'#S6.T5'))
    if abs(sum(r['total_minutes'] for r in cost)-202) > .01:
        raise ValueError('Cost total differs from published total')
    return pd.DataFrame(stability), pd.DataFrame(cost)


def describe_proxy(sim, real, published):
    """No learned coefficients, task matching, or independent-pair inference."""
    task_results, pair_results, aggregate_results = [], [], []
    for metric in METRICS:
        for category in sim.category.drop_duplicates():
            s = sim[(sim.metric == metric)&(sim.category == category)].set_index('method_id').score
            for (robot, task), d in real[real.metric == metric].groupby(['embodiment','task_id']):
                d = d.sort_values('method_id'); y = d.set_index('method_id').score
                x = s.reindex(y.index)
                if x.isna().any(): raise ValueError('Missing same-paper proxy')
                eligible = wrong = pred_ties = nonzero = real_ties = 0
                for a,b in combinations(y.index,2):
                    gap = float(y[a]-y[b]); pred_gap = float(x[a]-x[b])
                    tied = abs(gap) <= 1e-12
                    material = not tied and abs(gap) >= .10 - 1e-12
                    nonzero += not tied; real_ties += tied
                    correct = bool(np.sign(gap) == np.sign(pred_gap))
                    if material:
                        predicted_tie = abs(pred_gap) <= 1e-12
                        eligible += 1; wrong += not correct and not predicted_tie
                        pred_ties += predicted_tie
                    pair_results.append(dict(metric=metric, category=category,
                        embodiment=robot, task_id=task, method_a=a, method_b=b,
                        real_gap=gap, sim_proxy_gap=pred_gap, material_gap=material,
                        real_tie=tied, correct=correct))
                rho = float(spearmanr(x,y).statistic) if y.nunique()>1 and x.nunique()>1 else None
                task_results.append(dict(metric=metric, category=category,
                    embodiment=robot, task_id=task, n_methods=len(y),
                    n_pairs=len(y)*(len(y)-1)//2, n_real_ties=real_ties,
                    n_material=eligible, n_wrong=wrong, n_predicted_ties=pred_ties,
                    ranking_error=(wrong+.5*pred_ties)/eligible if eligible else None,
                    spearman=rho, max_physical_score=float(y.max()),
                    n_nonzero_methods=int((y>0).sum()),
                    interpretation='descriptive cross-task method proxy; separate domain training'))
            for robot in ['overall','arx_x5','piper','piper_x']:
                y = published[(published.metric==metric)&(published.embodiment==robot)].set_index('method_id').score
                x = s.reindex(y.index)
                aggregate_results.append(dict(metric=metric, category=category,
                    real_embodiment=robot, n_methods=len(y),
                    pearson=float(pearsonr(x,y).statistic),
                    spearman=float(spearmanr(x,y).statistic)))
    task_results, pair_results, aggregate_results = map(pd.DataFrame,
                                                       (task_results,pair_results,aggregate_results))
    summary = task_results.groupby(['metric','category','embodiment']).agg(
        n_tasks=('task_id','size'), n_informative_tasks=('ranking_error','count'),
        task_macro_error=('ranking_error','mean'), n_material=('n_material','sum'),
        n_wrong=('n_wrong','sum'), n_predicted_ties=('n_predicted_ties','sum'),
        n_real_ties=('n_real_ties','sum')).reset_index()
    # Preserve all categories; do not choose the best from these results.
    task_results.to_csv(RESULTS/'robodojo_paper_proxy_task_metrics.csv',index=False)
    pair_results.to_csv(RESULTS/'robodojo_paper_proxy_pairs.csv',index=False)
    aggregate_results.to_csv(RESULTS/'robodojo_paper_proxy_correlations.csv',index=False)
    summary.to_csv(RESULTS/'robodojo_paper_proxy_summary.csv',index=False)
    return task_results, aggregate_results, summary


def main():
    verify_raw()
    tables = read_tables(RAW/'continuation/robodojo_paper_v3.html')
    earlier = read_tables(RAW/'continuation/robodojo_paper_v1.html')
    version_equal = {t: tables[t] == earlier[t] for t in TABLE_IDS}
    differences=[]
    for table in TABLE_IDS:
        for row_index,(old,new) in enumerate(zip(earlier[table],tables[table])):
            if len(old)!=len(new):raise ValueError('Version comparison requires identical table shape')
            for column_index,(a,b) in enumerate(zip(old,new)):
                if a!=b:differences.append(dict(table_id=table,row_index=row_index,
                    column_index=column_index,v1=a,v3=b,
                    interpretation='Spirit v1.5 Piper X: cap_pen score and embodiment average corrected'))
    pd.DataFrame(differences).to_csv(RESULTS/'robodojo_paper_version_changes.csv',index=False)
    sim, real, ag, checks = extract(tables)
    stability, cost = stability_and_cost(tables)
    for frame, name in [(sim,'robodojo_paper_sim_categories'),(real,'robodojo_paper_real_task_scores'),
                        (ag,'robodojo_paper_real_aggregates'),(stability,'robodojo_paper_repeated_run_sd'),
                        (cost,'robodojo_paper_evaluation_cost')]:
        frame.to_csv(PROCESSED/(name+'.csv'),index=False)
    checks.to_csv(RESULTS/'robodojo_paper_aggregate_checks.csv',index=False)
    task_metrics, correlations, summary = describe_proxy(sim,real,ag)
    facts = json.loads((RESULTS/'robodojo_website_feasibility.json').read_text())
    facts.update(real_task_policy_scores_found=180, real_task_score_revision=REVISION,
        paper_real_methods=10, paper_real_tasks=18, paper_trials_per_cell=10,
        paper_documented_trials=1800, paper_raw_episodes_recovered=0,
        paper_sim_methods=int(sim.method_id.nunique()), paper_shared_method_names=10,
        paper_sim_task_scores_available=False, paper_verified_checkpoint_matches=0,
        paper_tables_identical_v1_v3=version_equal, go=False,
        reason='Tasks deliberately unpaired and policies separately fine-tuned by domain; same-paper sim task scores unavailable',
        correction='Physical task scores recovered from paper Table 2; website-only audit was incomplete')
    write_json(RESULTS/'robodojo_feasibility.json',facts)
    cov = pd.read_csv(RESULTS/'robodojo_website_coverage_matrix.csv')
    cov['real_task_score_available'] = cov.policy_id.isin(real.method_id.unique())
    cov['real_score_revision'] = np.where(cov.real_task_score_available,REVISION,'')
    cov['n_trials'] = np.where(cov.real_task_score_available,10,np.nan)
    cov['checkpoint_identity'] = np.where(cov.real_task_score_available,'separate_domain_training','unverified')
    cov['score_revisions_aligned'] = False
    cov['match_note'] = 'Candidate only; original tasks deliberately unpaired; October sim vs July physical snapshot'
    cov.to_csv(RESULTS/'robodojo_coverage_matrix.csv',index=False)
    write_report(facts,checks,correlations,summary,stability,cost,task_metrics)
    print(json.dumps({k:facts[k] for k in ['real_task_policy_scores_found','paper_shared_method_names','go','reason']},indent=2))
    return facts


def write_report(facts, checks, correlations, summary, stability, cost, task_metrics):
    lines = ['# RoboDojo public-data feasibility audit — corrected continuation','',
      '**Physical task scores are public. Formal matched sim-to-real validation still fails the compatibility gate.**', '',
      'The first pass inspected the website bundle and video manifest, but missed paper Table 2. This continuation recovers 180 physical method-task cells (10 methods × 18 tasks), each with progress and success rate. The paper documents 10 trials per cell; 1,800 underlying trials are not 1,800 independent tasks and no raw episodes have been reconstructed.', '',
      '## Provenance and extraction', '',
      f'Primary source: [{REVISION}]({URL}); Tables 1, 2, 5 and 9. Tables 1, 5 and 9 match v1, but Table 2 corrects Spirit v1.5 on Piper X: Cap Pen progress changes from 0 to 4 points, and the embodiment average changes from 4 to 0.7. The v3 row is internally consistent. All {len(checks)} v3 physical aggregate checks agree within 0.1 percentage point (rounding tolerance). Task order comes from the explicit note under Table 2. Human teleoperation is excluded. See robodojo_paper_version_changes.csv.',
      'Table 1 supplies simulation category/overall scores for 30 methods, including all 10 physical methods. This July 3 paper snapshot is kept separate from the October website snapshot (62 simulated and 12 physical names). October simulation task scores are not silently joined to July physical scores.', '',
      '## Why the gate remains closed', '',
      'Paper §3.2.1 explicitly says the simulation and real tasks were not designed as one-to-one pairs and that the real benchmark does not measure direct matched-task transfer. Appendix K specifies separate simulation/physical fine-tuning, often with different numbers of optimization steps; §5.2 trains a separate seed per physical embodiment. Shared method names therefore do not identify the same evaluated checkpoint. Even Cover Blocks and Insert Tubes need a protocol audit despite matching names and ARX X5 labels.',
      'The normalized tables use domain/embodiment-specific policy identifiers and a separate method_id for exploratory method comparisons. There are zero verified exact pairs. No B1 matched-task baseline, C1 external validation or LOTO sim-to-real claim is made.', '',
      '## Descriptive cross-task proxy analysis', '',
      'Use the same-paper overall simulation ranking to order separately trained methods on each physical task. No coefficients are fitted. Progress and success are separate tracks. The 10-point observed physical-gap threshold is descriptive, not a confidence criterion: with 10 trials, one success changes the observed success rate by 10 points. Predicted ties receive 0.5 loss, as in SIMPLER; the overall-score proxy has no eligible predicted ties. Overlapping policy pairs and related methods are dependent. No significance claim or task-population interval is made.', '',
      '| Metric | Embodiment | Informative tasks / 6 | Wrong / eligible pairs | Task-average error |',
      '|---|---|---:|---:|---:|']
    for r in summary[summary.category=='Average'].itertuples():
        lines.append(f'| {r.metric} | {r.embodiment} | {r.n_informative_tasks}/6 | {r.n_wrong}/{r.n_material} | {r.task_macro_error:.1%} |')
    lines += ['', 'Tasks with no eligible pairs stay in the coverage count and have undefined error, not zero error. All five category proxies are also exported for inspection; none is selected as a winning predictor. These are cross-task method associations, not measured simulation failures.', '',
      '| Metric | Overall Pearson | Overall Spearman | Shared methods |','|---|---:|---:|---:|']
    for r in correlations[(correlations.category=='Average')&(correlations.real_embodiment=='overall')].itertuples():
        lines.append(f'| {r.metric} | {r.pearson:.3f} | {r.spearman:.3f} | {r.n_methods} |')
    lines += ['', '## Repeated physical evaluation and measurement cost', '',
      'Table 9 reports standard deviations across three runs for π0.5, GalaxeaVLA and InternVLA-A1. It does not release the three run-level outcomes. Overall success-rate SDs are 1.0, 1.2 and 1.3 points, while Store in Safe SDs are 23.1, 11.5 and 17.3 points. Stack Bowls has Galaxea progress SD 16.6 points. Aggregate stability can mask substantial task variation; these are run SDs, not errors on a population mean.',
      'Table 5 reports 202 minutes for 180 π0.5 physical trials, including resets, local RTX4090 inference and execution. For a compatibility inspection, Cover Blocks costs 10.8 minutes per 10 trials and Insert Tubes 9.0 minutes; Store in Safe costs 9.8 and Stack Bowls 7.3. These costs apply only to the published RoboDojo setup, not Robocurve or a future model.', '',
      '## Resource needed next', '',
      'Request existing **same evaluated checkpoint** results on explicitly paired tasks with compatible embodiment, controller, layouts, horizon and rubric. Ask whether such evaluations exist; recovering more current leaderboard averages or downloading training checkpoints does not fill this gap. For a narrower cross-task method study, request the current two missing methods’ task tables, frozen submission artifacts, simulation per-task scores from the same revision, and repeated-run scores. Raw physical round data would help assess task-level ranking stability.', '',
      'No videos, checkpoints or large trajectory archives were downloaded. Repository code licensing does not automatically grant redistribution rights for all paper/site/video assets. Preserve source attribution; check terms before a public mirror.', '',
      'Files: `robodojo_paper_real_task_scores.csv`, `robodojo_paper_sim_categories.csv`, `robodojo_paper_repeated_run_sd.csv`, `robodojo_paper_evaluation_cost.csv` in data/processed; `robodojo_paper_proxy_*` and aggregate checks in results. `robodojo_coverage_matrix.csv` marks July physical availability separately from October simulation proxies.', '',
      'Original website-only audit is retained as `robodojo_website_feasibility.md`; its scoped absence finding is superseded by this combined report.', '',
      f'Source: {URL}', 'Website: https://robodojo-benchmark.com/leaderboard']
    (RESULTS/'robodojo_feasibility.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
