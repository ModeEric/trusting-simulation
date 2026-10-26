"""Exploratory, nested task-held-out blending of local and global simulation."""
import json
from itertools import combinations, product
import numpy as np
import pandas as pd
from .common import ROOT, RESULTS, PROCESSED, SEED, verify_raw, write_json, sha256
from .evaluate import score_predictions, summary_with_intervals

ATTRIBUTE='precision_contact'
GRID=np.linspace(0,1,11)

def global_scores(sim_only, rows, exclude_target=False):
    if 'real' in sim_only: raise ValueError('Global ability must not receive physical labels')
    out=[]
    for r in rows.itertuples():
        available=sim_only[sim_only.policy_id==r.policy_id]
        if exclude_target:available=available[available.task_id!=r.task_id]
        if available.empty:raise ValueError('No simulated history for policy')
        out.append(float(available.sim.mean()))
    return np.asarray(out)

def candidate_predictions(rows, ability, candidates):
    z=rows[ATTRIBUTE].to_numpy().astype(int)
    if not np.isin(z,[0,1]).all():raise ValueError('Expected binary attribute')
    weights=np.asarray(candidates)[None,:,:]
    w=np.where(z[:,None]==0,weights[0,:,0],weights[0,:,1])
    s=rows.sim.to_numpy()[:,None]
    return s+w*(ability[:,None]-s)

def task_loss(rows, predictions, objective):
    y=rows.real.to_numpy()
    if objective=='mae':return np.abs(predictions-y[:,None]).mean(axis=0)
    a,b=np.triu_indices(len(rows),1)
    dy=y[a]-y[b];eligible=np.abs(dy)>=.10-1e-12
    if not eligible.any():return np.full(predictions.shape[1],np.nan)
    dp=predictions[a[eligible]]-predictions[b[eligible]]
    return np.where(np.abs(dp)<1e-12,.5,(np.sign(dp)!=np.sign(dy[eligible,None])).astype(float)).mean(axis=0)

def select_weights(train, sim_only, task_aware, objective, exclude_target=False):
    candidates=list(product(GRID,GRID)) if task_aware else [(w,w) for w in GRID]
    losses=[];inner=[]
    for task,validation in train.groupby('task_id',sort=True):
        features=validation.drop(columns='real')
        ability=global_scores(sim_only,features,exclude_target)
        loss=task_loss(validation,candidate_predictions(features,ability,candidates),objective)
        losses.append(loss)
        inner.append(dict(validation_task=task,
            other_training_tasks=sorted(set(train.task_id)-{task}),
            candidate_losses=[None if not np.isfinite(v) else float(v) for v in loss]))
    means=np.nanmean(np.asarray(losses),axis=0)
    if not np.isfinite(means).any():chosen=0
    else:
        best=np.nanmin(means)
        tied=np.flatnonzero(np.isfinite(means)&(means<=best+1e-12))
        chosen=min(tied,key=lambda i:(abs(candidates[i][1]-candidates[i][0]),sum(candidates[i]),candidates[i][0]))
    return candidates[chosen],dict(candidates=candidates,inner_validation=inner,
        selected_index=int(chosen),selected_loss=float(means[chosen]) if np.isfinite(means[chosen]) else None)

def predict_outer(train, test_features, sim_only, task_aware, objective, exclude_target=False):
    if 'real' in test_features:raise ValueError('Held-out physical labels passed to predictor')
    if set(train.task_id)&set(test_features.task_id):raise ValueError('Task split overlap')
    weights,audit=select_weights(train,sim_only,task_aware,objective,exclude_target)
    ability=global_scores(sim_only,test_features,exclude_target)
    pred=candidate_predictions(test_features,ability,[weights])[:,0]
    return pred,weights,audit,ability

def run(data,exclude_target=False,objectives=('ranking','mae'),audit_details=True):
    predictions=[];audits=[]
    for emb,stratum in data.groupby('embodiment',sort=True):
        sim_only=stratum[['task_id','policy_id','sim']].copy()
        for held,testing in stratum.groupby('task_id',sort=True):
            train=stratum[stratum.task_id!=held]
            features=testing.drop(columns='real')
            for objective in objectives:
                for aware in [False,True]:
                    name=('TaskBlend' if aware else 'GlobalBlend')+'_'+objective
                    pred,ws,audit,ability=predict_outer(train,features,sim_only,aware,objective,exclude_target)
                    row=testing.copy();row['prediction']=pred;row['method']=name
                    row['global_sim']=ability;row['weight']=np.where(features[ATTRIBUTE]==0,ws[0],ws[1])
                    predictions.append(row)
                    audits.append(dict(embodiment=emb,test_task=held,method=name,
                        train_tasks=sorted(train.task_id.unique()),w0=float(ws[0]),w1=float(ws[1]),
                        **(audit if audit_details else {})))
    return pd.concat(predictions,ignore_index=True),audits

def deltas(metrics):
    rng=np.random.default_rng(SEED);rows=[]
    for emb,g in metrics.groupby('embodiment'):
        for objective in ['ranking','mae']:
            for method,baseline in [('GlobalBlend_'+objective,'B1'),('TaskBlend_'+objective,'B1'),
                                    ('TaskBlend_'+objective,'GlobalBlend_'+objective)]:
                for metric in ['ranking_error','mae','mmrv']:
                    wide=g.pivot(index='task_id',columns='method',values=metric)
                    delta=(wide[method]-wide[baseline]).dropna().to_numpy()
                    boot=rng.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
                    omitted=[np.delete(delta,i).mean() for i in range(len(delta))]
                    rows.append(dict(embodiment=emb,method=method,baseline=baseline,metric=metric,
                        delta=float(delta.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                        leave_one_task_min=float(min(omitted)),leave_one_task_max=float(max(omitted))))
    return pd.DataFrame(rows)

def main():
    verify_raw()
    frozen=json.loads((ROOT/'data/blend_plan_freeze.json').read_text())
    if sha256(ROOT/'blend_plan.json')!=frozen['sha256']:raise ValueError('Blend plan changed')
    data=pd.read_csv(PROCESSED/'simpler_paired.csv')
    pred,audits=run(data)
    baseline=pd.read_csv(RESULTS/'simpler_predictions.csv')
    baseline=baseline[baseline.method.isin(['B0','B1'])]
    combined=pd.concat([baseline,pred],ignore_index=True)
    met,pairs=score_predictions(combined)
    summary=summary_with_intervals(met)
    for frame,name in [(combined,'predictions'),(met,'task_metrics'),(pairs,'pairs'),(summary,'summary'),(deltas(met),'deltas')]:
        frame.to_csv(RESULTS/f'blend_{name}.csv',index=False)
    write_json(RESULTS/'blend_fold_audit.json',audits)
    pd.DataFrame([{k:a[k] for k in ['embodiment','test_task','method','w0','w1']} for a in audits]).to_csv(RESULTS/'blend_weights.csv',index=False)
    # Changed decisions are paired to the same B1 observation, not counted as new tasks.
    key=['embodiment','task_id','policy_a','policy_b']
    base=pairs[pairs.method=='B1'][key+['tie_aware_loss','predicted_gap']].rename(columns={'tie_aware_loss':'baseline_loss','predicted_gap':'baseline_gap'})
    changed=pairs[pairs.method.str.contains('Blend')].merge(base,on=key,validate='many_to_one')
    changed=changed[changed.eligible & (changed.tie_aware_loss!=changed.baseline_loss)].copy()
    changed['effect']=np.where(changed.tie_aware_loss<changed.baseline_loss,'fixed','introduced')
    changed.to_csv(RESULTS/'blend_changed_pairs.csv',index=False)
    sensitivity,_=run(data,exclude_target=True)
    sm,_=score_predictions(pd.concat([baseline,sensitivity],ignore_index=True))
    # B0 here remains the primary comparator, explicitly named; blends exclude target.
    summary_with_intervals(sm).to_csv(RESULTS/'blend_exclude_target_summary.csv',index=False)
    rng=np.random.default_rng(SEED);controls=[]
    for rep in range(100):
        shuffled=data.copy()
        for emb,g in data.groupby('embodiment'):
            meta=g[['task_id',ATTRIBUTE]].drop_duplicates().sort_values('task_id')
            mapping=dict(zip(meta.task_id,rng.permutation(meta[ATTRIBUTE].to_numpy())))
            shuffled.loc[g.index,ATTRIBUTE]=g.task_id.map(mapping)
        p,_=run(shuffled,objectives=('ranking',),audit_details=False)
        m,_=score_predictions(p)
        for emb,g in m[m.method=='TaskBlend_ranking'].groupby('embodiment'):
            controls.append(dict(permutation=rep,embodiment=emb,ranking_error=float(g.ranking_error.mean())))
    pd.DataFrame(controls).to_csv(RESULTS/'blend_permutations.csv',index=False)
    print(summary[['embodiment','method','ranking_error','mae','n_correct','n_decided']].to_string(index=False))

if __name__=='__main__':main()
