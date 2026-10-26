"""Run the three frozen modeling experiments sequentially on public scores."""
import argparse
import gzip
import json
import time
import numpy as np
import pandas as pd
from .common import ROOT,PROCESSED,RESULTS,sha256,write_json
from .robotwin_prediction import verify_plan
from .robotwin_learning_curve import selected_groups
from .sim_prediction import SEED,folds
from .robotwin_models import fit_predict_stage,pair_counts


def verify():
    verify_plan()
    for path,digest in json.loads((ROOT/'data/robotwin_model_sweep_freeze.json').read_text()).items():
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen model-sweep input changed: '+path)


def run(stage):
    verify();start=time.perf_counter()
    plan=json.loads((ROOT/'robotwin_model_sweep_plan.json').read_text())
    meta=pd.read_csv(PROCESSED/'robotwin_policy_metadata.csv');meta=meta[meta.broad_included].set_index('policy_id')
    ids=meta.index.to_numpy();families=meta.family.to_numpy();audits=[];primary=[]
    prefix=f'robotwin_models_stage{stage}'
    write_json(RESULTS/f'{prefix}_status.json',dict(status='running',stage=stage))
    with gzip.open(RESULTS/f'{prefix}_policy_metrics.csv.gz','wt') as model_file,gzip.open(RESULTS/f'{prefix}_ranking_metrics.csv.gz','wt') as task_file:
        first=True
        for cohort in plan['cohorts']:
            matrix=pd.read_csv(PROCESSED/f'robotwin_matrix_{cohort}.csv',index_col=0)
            tasks=matrix.columns.to_numpy();y=matrix.loc[ids].to_numpy()
            for partition in range(3):
                for fold,test in enumerate(folds(families,SEED+partition)):
                    available=np.setdiff1d(np.arange(len(ids)),test);pool=sorted(set(families[available]));cache={}
                    model_rows=[];task_rows=[]
                    for draw in range(5):
                        order=np.random.default_rng(SEED+1000+draw).permutation(len(tasks));common=np.sort(order[20:])
                        for budget in plan['anchor_budgets']:
                            anchors=order[:budget];standard=np.setdiff1d(np.arange(len(tasks)),anchors)
                            for subset_draw in range(plan['subset_draws']):
                                for fraction in plan['training_fractions']:
                                    groups=selected_groups(pool,fraction,partition,fold,subset_draw)
                                    train=available[np.isin(families[available],groups)]
                                    key=(draw,budget,tuple(train))
                                    if key not in cache:
                                        cache[key]=fit_predict_stage(stage,y[train],families[train],y[test][:,anchors],anchors,SEED+partition+500)
                                    preds,audit=cache[key]
                                    design=dict(stage=stage,cohort=cohort,partition=partition,fold=fold,anchor_draw=draw,budget=budget,
                                                subset_draw=subset_draw,training_fraction=fraction)
                                    audits.append(dict(**design,train_policies=ids[train].tolist(),test_policies=ids[test].tolist(),
                                        train_families=groups,test_families=sorted(set(families[test])),available_train_families=pool,
                                        n_train_cells=len(train)*50,anchor_tasks=tasks[anchors].tolist(),
                                        standard_targets=tasks[standard].tolist(),common_targets=tasks[common].tolist(),**audit))
                                    for method,p in preds.items():
                                        for track,targets in [('standard',standard),('common30',common)]:
                                            actual=y[test][:,targets];prediction=p[:,targets]
                                            if not method.startswith('Rank'):
                                                error=np.abs(prediction-actual)
                                                for j,idx in enumerate(test):model_rows.append(dict(**design,target_track=track,method=method,
                                                    policy_id=ids[idx],family=families[idx],mae=float(error[j].mean())))
                                            loss,count=pair_counts(actual,prediction,families[test])
                                            for j,t in enumerate(targets):task_rows.append(dict(**design,target_track=track,method=method,
                                                task_id=tasks[t],loss_sum=float(loss[j]),n_eligible=int(count[j])))
                                        if budget==10 and fraction==1 and subset_draw==0:
                                            for j,idx in enumerate(test):
                                                for t in standard:primary.append(dict(**design,method=method,policy_id=ids[idx],family=families[idx],
                                                    task_id=tasks[t],actual=float(y[idx,t]),prediction=float(p[j,t]),
                                                    output_kind='ranking_utility' if method.startswith('Rank') else 'success_rate'))
                    if model_rows:pd.DataFrame(model_rows).to_csv(model_file,index=False,header=first)
                    pd.DataFrame(task_rows).to_csv(task_file,index=False,header=first)
                    first=False
                    print(f'Stage {stage}: {cohort}, partition {partition}, fold {fold} complete',flush=True)
    write_json(RESULTS/f'{prefix}_audit.json',audits)
    pd.DataFrame(primary).to_csv(RESULTS/f'{prefix}_primary_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summarize(stage,audits)
    write_json(RESULTS/f'{prefix}_status.json',dict(status='passed',stage=stage,elapsed_seconds=round(time.perf_counter()-start,3),
        designs=len(audits),new_simulation_rollouts=0,robot_policy_inference_calls=0,paid_api_calls=0))


def summarize(stage,audits):
    prefix=f'robotwin_models_stage{stage}';keys=['cohort','training_fraction','budget','target_track','method']
    models=pd.read_csv(RESULTS/f'{prefix}_policy_metrics.csv.gz') if stage!=3 else pd.DataFrame()
    tasks=pd.read_csv(RESULTS/f'{prefix}_ranking_metrics.csv.gz')
    if len(models):
        family=models.groupby(keys+['family','policy_id']).mae.mean().groupby(keys+['family']).mean().reset_index()
        family.to_csv(RESULTS/f'{prefix}_family_metrics.csv',index=False)
    pooled=tasks.groupby(keys+['partition','anchor_draw','subset_draw','task_id'])[['loss_sum','n_eligible']].sum().reset_index()
    pooled['ranking_error']=pooled.loss_sum/pooled.n_eligible.replace(0,np.nan)
    task=pooled.groupby(keys+['task_id']).ranking_error.mean().reset_index()
    task.to_csv(RESULTS/f'{prefix}_task_metrics.csv',index=False)
    a=pd.DataFrame(audits)
    counts=a.drop_duplicates(['cohort','training_fraction','partition','fold','subset_draw']).groupby(['cohort','training_fraction']).n_train_cells.mean().reset_index(name='training_cells_mean')
    rows=[]
    for endpoint,frame in ([('mae',family)] if len(models) else [])+[('ranking_error',task)]:
        for key,g in frame.groupby(keys):
            values=g[endpoint].dropna().to_numpy();rng=np.random.default_rng(SEED)
            boot=rng.choice(values,size=(2000,len(values)),replace=True).mean(axis=1)
            rows.append(dict(zip(keys,key),endpoint=endpoint,value=float(values.mean()),low=float(np.quantile(boot,.025)),
                             high=float(np.quantile(boot,.975)),n_clusters=len(values)))
    summary=pd.DataFrame(rows).merge(counts,on=['cohort','training_fraction'])
    summary.to_csv(RESULTS/f'{prefix}_summary.csv',index=False)
    print(summary[(summary.training_fraction==1)&(summary.budget==10)&(summary.target_track=='standard')][['cohort','method','endpoint','value']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,choices=[1,2,3],required=True)
    run(parser.parse_args().stage)
