"""RoboTwin replication using the unchanged, previously frozen predictors."""
import json
import numpy as np
import pandas as pd
from .common import ROOT, RAW, PROCESSED, RESULTS, sha256, verify_raw, write_json
from .sim_prediction import run_cohort, summarize

COHORTS=['clean','randomized','clean_rgb','randomized_rgb']


def verify_plan():
    verify_raw()
    for name in ['sim_prediction_plan_freeze.json','robotwin_plan_freeze.json']:
        for path,digest in json.loads((ROOT/'data'/name).read_text()).items():
            if sha256(ROOT/path)!=digest:raise ValueError('Frozen prediction input or code changed: '+path)


def prepare():
    plan=json.loads((ROOT/'robotwin_plan.json').read_text())
    source=json.loads((RAW/'robotwin_leaderboard.json').read_text())
    if source['schema_version']!='1.0':raise ValueError('Unknown source schema')
    tasks=source['tasks'];entries=source['entries']
    if len(tasks)!=50 or len(set(tasks))!=50:raise ValueError('Unexpected or duplicate tasks')
    if len({e['id'] for e in entries})!=len(entries):raise ValueError('Duplicate policy ID')
    families={m:g['name'] for g in plan['groups'] for m in g['members']}
    meta=[];scores=[];mean_audit=[]
    for e in entries:
        if e['setting'] not in ['cotrain','sft']:raise ValueError('Unknown training track')
        slug=e['id'].split(':',1)[1]
        meta.append(dict(policy_id=e['id'],display_name=e['name'],setting=e['setting'],depth=e['depth'],
                         family=families.get(slug,slug),family_status='conservative related group' if slug in families else 'provisional singleton; ancestry incomplete',
                         broad_included=e['setting']=='cotrain',contributor=e.get('contributor'),listed_at=e.get('listed_at')))
        for condition,key in [('clean','easy'),('randomized','hard')]:
            values=e[key]
            if len(values)!=50 or any(not isinstance(x,(int,float)) for x in values):raise ValueError('Incomplete task scores')
            if not np.isfinite(values).all() or not all(0<=x<=100 for x in values):raise ValueError('Score outside [0,100]')
            mean_audit.append(dict(policy_id=e['id'],condition=condition,published_mean=e[key+'_mean'],
                recomputed_mean=float(np.mean(values)),difference=float(np.mean(values)-e[key+'_mean']),
                noninteger_percentage_cells=sum(abs(x-round(x))>1e-9 for x in values)))
            for task,value in zip(tasks,values):scores.append(dict(source='robotwin',source_revision=plan['source_revision'],domain='simulation',
                embodiment=source['board']['robot'],policy_id=e['id'],policy_family=families.get(slug,slug),task_id=task,condition=condition,
                training_track=e['setting'],metric='success_rate',score=value/100,n_trials=None,nominal_protocol_trials=100,
                protocol_id='robotwin2-50-clean-cotrain-'+condition if e['setting']=='cotrain' else 'robotwin2-single-task-'+condition,
                source_url='https://github.com/RoboTwin-Platform/RoboTwin-Platform.github.io/blob/'+plan['source_revision']+'/data/robotwin_leaderboard.json'))
    meta=pd.DataFrame(meta).sort_values('policy_id');scores=pd.DataFrame(scores)
    meta.to_csv(PROCESSED/'robotwin_policy_metadata.csv',index=False)
    scores.to_csv(PROCESSED/'robotwin_task_scores.csv',index=False)
    pd.DataFrame(mean_audit).to_csv(RESULTS/'robotwin_source_mean_audit.csv',index=False)
    selected=meta[meta.broad_included]
    if len(selected)!=25:raise ValueError('Unexpected cotrain cohort')
    matrices={};coverage=[]
    for cohort in COHORTS:
        condition=cohort.split('_')[0];md=selected[~selected.depth] if cohort.endswith('_rgb') else selected
        matrix=scores[(scores.condition==condition)&scores.policy_id.isin(md.policy_id)].pivot(index='policy_id',columns='task_id',values='score').sort_index(axis=1)
        if matrix.shape!=(len(md),50) or matrix.isna().any().any():raise ValueError('Missing primary outcomes')
        matrix.to_csv(PROCESSED/f'robotwin_matrix_{cohort}.csv')
        matrices[cohort]=(md,matrix)
        coverage.append(dict(cohort=cohort,condition=condition,policies=len(md),family_groups=md.family.nunique(),tasks=50,cells=matrix.size,
            depth_policies=int(md.depth.sum()),zero_fraction=float((matrix==0).to_numpy().mean()),
            note='group proxies; same-condition anchors; not physical or unseen-task validation'))
    write_json(RESULTS/'robotwin_coverage.json',coverage)
    return matrices


def main():
    verify_plan();matrices=prepare();outputs=[];models=[];tasks=[];audits=[]
    for cohort,(md,matrix) in matrices.items():
        out,mo,ta,au=run_cohort(md,matrix,cohort)
        outputs.append(out);models.append(mo);tasks.append(ta);audits+=au
        print('Finished RoboTwin '+cohort,flush=True)
    predictions=pd.concat(outputs,ignore_index=True);model_rows=pd.concat(models,ignore_index=True);task_rows=pd.concat(tasks,ignore_index=True)
    predictions.to_csv(RESULTS/'robotwin_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary,family,task_mean,deltas=summarize(model_rows,task_rows)
    for frame,name in [(summary,'summary'),(model_rows,'policy_metrics'),(task_rows,'ranking_metrics'),(family,'family_metrics'),(task_mean,'task_metrics'),(deltas,'deltas')]:
        frame.to_csv(RESULTS/f'robotwin_{name}.csv',index=False)
    write_json(RESULTS/'robotwin_fold_audit.json',audits)
    print(summary[['cohort','budget','method','mae','ranking_error']].to_string(index=False))


if __name__=='__main__':main()
