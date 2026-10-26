"""Extract only literal dictionaries; never import the simulator."""
import ast, json
import pandas as pd
from .common import ROOT,RAW,PROCESSED,RESULTS,validate_scores,verify_raw,write_json
from .taxonomy import metadata

def read_constants(path=RAW/'simpler_metrics.py'):
    values={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in {'REAL_PERF','SIMPLER_PERF'}:
                    values[target.id]=ast.literal_eval(node.value)
    if set(values)!={'REAL_PERF','SIMPLER_PERF'}: raise ValueError('Missing source constants')
    return values

def family(policy):
    if policy.startswith('rt-1'): return 'rt-1'
    if policy.startswith('rt-2'): return 'rt-2'
    if policy.startswith('octo'): return 'octo'
    raise ValueError(f'Unmapped policy: {policy}')

def ingest():
    verified=verify_raw(names=['data/raw/simpler_metrics.py','data/raw/simpler_calc_metrics.py','data/raw/simpler_LICENSE']);data=read_constants()
    real,sim=data['REAL_PERF'],data['SIMPLER_PERF']
    tasks=set(real)|set(sim);meta=metadata(tasks);meta.to_csv(PROCESSED/'task_metadata.csv',index=False)
    revision=verified['data/raw/simpler_metrics.py']['source_revision']
    url=f'https://github.com/simpler-env/SimplerEnv/blob/{revision}/simpler_env/utils/metrics.py'
    rows=[];coverage=[]
    for task in sorted(tasks):
        rp=set(real.get(task,{}));sp=set(sim.get(task,{}));paired=rp&sp
        coverage.append(dict(task_id=task,n_real=len(rp),n_sim=len(sp),n_matched=len(paired),
                             real_only=';'.join(sorted(rp-sp)),sim_only=';'.join(sorted(sp-rp))))
        for domain,source in [('real',real),('sim',sim)]:
            for policy,score in source.get(task,{}).items():
                rows.append(dict(source='simpler',source_revision=revision,domain=domain,
                    embodiment='google_robot' if task.startswith('google') else 'widowx',
                    policy_id=policy,policy_family=family(policy),task_id=task,
                    task_family=meta.set_index('task_id').loc[task,'task_family'],metric='success_rate',
                    score=score,n_trials=None,protocol_id='simpler_visual_matching_public_constants',
                    matched_task_id=task,match_quality='exact' if policy in paired else 'unmatched',source_url=url))
    scores=pd.DataFrame(rows);validate_scores(scores,['source','domain','embodiment','policy_id','task_id','metric'])
    scores.to_csv(PROCESSED/'simpler_task_scores.csv',index=False)
    pd.DataFrame(coverage).to_csv(RESULTS/'simpler_coverage.csv',index=False)
    ids=['task_id','policy_id','embodiment','policy_family']
    pair=scores.pivot(index=ids,columns='domain',values='score').dropna().reset_index()
    pair=pair.merge(meta.drop(columns=['source','embodiment']),on='task_id',validate='many_to_one')
    pair.to_csv(PROCESSED/'simpler_paired.csv',index=False)
    print(f'SIMPLER: {len(pair)} paired cells, {pair.task_id.nunique()} tasks, {pair.embodiment.nunique()} embodiments')
    return pair
if __name__=='__main__': ingest()
