"""Nested training-data ablation with the original RoboTwin test sets fixed."""
import json
import math
import time
import numpy as np
import pandas as pd
from .common import ROOT, PROCESSED, RESULTS, sha256, write_json
from .robotwin_prediction import verify_plan
from .sim_prediction import SEED, METHODS, folds, fit_training, tune_latent, predict


def selected_groups(pool, fraction, partition, fold, draw):
    order=np.random.default_rng(SEED+2000+100*partition+10*fold+draw).permutation(sorted(pool))
    return sorted(order[:max(2,math.ceil(fraction*len(order)))].tolist())


def fit_subset(matrix,meta,train_ids,test_ids,anchor_tasks,partition):
    """Only selected training rows and test anchors are accessed for fitting."""
    anchors=matrix.columns.get_indexer(anchor_tasks)
    train=matrix.loc[train_ids].to_numpy()
    option,inner=tune_latent(train,meta.loc[train_ids,'family'].to_numpy(),anchors,SEED+partition+500)
    predictions=predict(fit_training(train),matrix.loc[test_ids,anchor_tasks].to_numpy(),anchors,option)
    return predictions,option,inner


def run():
    verify_plan()
    for path,digest in json.loads((ROOT/'data/robotwin_learning_curve_freeze.json').read_text()).items():
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen learning-curve input changed: '+path)
    plan=json.loads((ROOT/'robotwin_learning_curve_plan.json').read_text())
    meta=pd.read_csv(PROCESSED/'robotwin_policy_metadata.csv')
    meta=meta[meta.broad_included].set_index('policy_id')
    ids=meta.index.to_numpy();families=meta.family.to_numpy()
    model_rows=[];task_rows=[];audits=[]
    for cohort in plan['cohorts']:
        matrix=pd.read_csv(PROCESSED/f'robotwin_matrix_{cohort}.csv',index_col=0)
        tasks=matrix.columns.to_numpy();y=matrix.loc[ids].to_numpy()
        for partition in range(3):
            for fold,test in enumerate(folds(families,SEED+partition)):
                available=np.setdiff1d(np.arange(len(ids)),test);pool=sorted(set(families[available]))
                cache={}
                for draw in range(5):
                    order=np.random.default_rng(SEED+1000+draw).permutation(len(tasks))
                    anchors=order[:10];targets=np.setdiff1d(np.arange(len(tasks)),anchors)
                    actual=y[test][:,targets]
                    a,b=np.triu_indices(len(test),1);keep=families[test][a]!=families[test][b];a=a[keep];b=b[keep]
                    dy=actual[a]-actual[b];eligible=np.abs(dy)>=.10-1e-12
                    for subset_draw in range(5):
                        for fraction in plan['training_fractions']:
                            groups=selected_groups(pool,fraction,partition,fold,subset_draw)
                            train=available[np.isin(families[available],groups)]
                            key=(draw,tuple(train))
                            if key not in cache:
                                cache[key]=fit_subset(matrix,meta,ids[train].tolist(),ids[test].tolist(),tasks[anchors].tolist(),partition)
                            preds,option,inner=cache[key]
                            design=dict(cohort=cohort,partition=partition,fold=fold,anchor_draw=draw,subset_draw=subset_draw,training_fraction=fraction)
                            audits.append(dict(**design,train_policies=ids[train].tolist(),test_policies=ids[test].tolist(),
                                train_families=groups,test_families=sorted(set(families[test])),available_train_families=pool,
                                anchor_tasks=tasks[anchors].tolist(),target_tasks=tasks[targets].tolist(),
                                n_train_groups=len(groups),n_train_policies=len(train),n_train_cells=len(train)*len(tasks),
                                latent_rank=option[0],latent_ridge=option[1],**inner))
                            for method,p in preds.items():
                                pred=p[:,targets];error=np.abs(pred-actual)
                                for j,idx in enumerate(test):model_rows.append(dict(**design,method=method,
                                    policy_id=ids[idx],family=families[idx],mae=float(error[j].mean())))
                                dp=pred[a]-pred[b]
                                loss=np.where(np.abs(dp)<1e-12,.5,(np.sign(dp)!=np.sign(dy)).astype(float))
                                for j,t in enumerate(targets):task_rows.append(dict(**design,method=method,task_id=tasks[t],
                                    loss_sum=float(loss[eligible[:,j],j].sum()),n_eligible=int(eligible[:,j].sum())))
                print(f'Finished learning curve {cohort} partition {partition} fold {fold}',flush=True)
    models=pd.DataFrame(model_rows);tasks=pd.DataFrame(task_rows)
    models.to_csv(RESULTS/'robotwin_learning_curve_policy_metrics.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    tasks.to_csv(RESULTS/'robotwin_learning_curve_ranking_metrics.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_json(RESULTS/'robotwin_learning_curve_audit.json',audits)
    summarize(models,tasks,audits)


def summarize(models,tasks,audits):
    keys=['cohort','training_fraction','method']
    family=models.groupby(keys+['family','policy_id']).mae.mean().groupby(keys+['family']).mean().reset_index()
    pooled=tasks.groupby(keys+['partition','anchor_draw','subset_draw','task_id'])[['loss_sum','n_eligible']].sum().reset_index()
    pooled['ranking_error']=pooled.loss_sum/pooled.n_eligible.replace(0,np.nan)
    task=pooled.groupby(keys+['task_id']).ranking_error.mean().reset_index()
    family.to_csv(RESULTS/'robotwin_learning_curve_family_metrics.csv',index=False)
    task.to_csv(RESULTS/'robotwin_learning_curve_task_metrics.csv',index=False)
    a=pd.DataFrame(audits)
    counts=a.drop_duplicates(['cohort','training_fraction','partition','fold','subset_draw']).groupby(['cohort','training_fraction']).agg(
        training_groups_mean=('n_train_groups','mean'),training_groups_min=('n_train_groups','min'),training_groups_max=('n_train_groups','max'),
        training_policies_mean=('n_train_policies','mean'),training_policies_min=('n_train_policies','min'),training_policies_max=('n_train_policies','max'),
        training_cells_mean=('n_train_cells','mean'),training_cells_min=('n_train_cells','min'),training_cells_max=('n_train_cells','max')).reset_index()
    counts.to_csv(RESULTS/'robotwin_learning_curve_training_counts.csv',index=False)
    rows=[];deltas=[]
    for cohort in ['clean','randomized']:
        for endpoint,frame,index in [('mae',family,'family'),('ranking_error',task,'task_id')]:
            for method in METHODS:
                wide=frame[(frame.cohort==cohort)&(frame.method==method)].pivot(index=index,columns='training_fraction',values=endpoint).dropna()
                rng=np.random.default_rng(SEED);draws=rng.integers(len(wide),size=(2000,len(wide)))
                boot=wide.to_numpy()[draws].mean(axis=1)
                for j,fraction in enumerate(wide.columns):
                    rows.append(dict(cohort=cohort,method=method,endpoint=endpoint,training_fraction=fraction,
                        value=float(wide.iloc[:,j].mean()),low=float(np.quantile(boot[:,j],.025)),high=float(np.quantile(boot[:,j],.975)),n_clusters=len(wide)))
                delta=wide.iloc[:,-1]-wide.iloc[:,0];db=delta.to_numpy()[draws].mean(axis=1)
                deltas.append(dict(cohort=cohort,method=method,endpoint=endpoint,delta_full_minus_small=float(delta.mean()),
                    low=float(np.quantile(db,.025)),high=float(np.quantile(db,.975)),n_clusters=len(wide)))
    summary=pd.DataFrame(rows).merge(counts,on=['cohort','training_fraction'])
    summary.to_csv(RESULTS/'robotwin_learning_curve_summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(RESULTS/'robotwin_learning_curve_deltas.csv',index=False)
    print(summary[summary.method=='Latent'][['cohort','training_fraction','training_cells_mean','endpoint','value']].to_string(index=False))


if __name__=='__main__':
    start=time.perf_counter();run()
    write_json(RESULTS/'robotwin_learning_curve_run_metadata.json',dict(elapsed_seconds=round(time.perf_counter()-start,3),
        source='pinned public aggregates',new_simulation_rollouts=0,robot_policy_inference_calls=0,paid_api_calls=0))
