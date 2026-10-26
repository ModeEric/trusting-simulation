"""Predict a held-out policy's unobserved simulation scores from anchor scores.

No physical data, rollouts, policy inference, or test-target data enter fitting.
Ranking compares jointly held-out policies only, never cross-fold predictions.
"""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from .common import ROOT, RAW, RESULTS, PROCESSED, verify_raw, write_json, sha256
from .ingest_robodojo import extract
from .js_literals import Parser, assigned_literal

METHODS=['Zero','TaskMean','AnchorMean','Additive','Scale','Neighbor3','Latent']
SEED=20261024
# Conservative grouping hypotheses, made before prediction residual inspection.
# Cross-links (StarVLA PI and GR00T) intentionally merge their connected families.
LINEAGES=[['Pi_0','Pi_05','Pi_05_SF','starVLA-PI_v3'],
 ['starVLA','starVLA-OFT','starVLA-GR00T','starVLA-PI_v3','vlact'],
 ['GR00T_N17','starVLA-GR00T'],['InternVLA_A1','InternVLA_A1_5','InternW0_delta'],
 ['Xiaomi_Robotics_0','Xiaomi_Robotics_1'],['GalaxeaVLA','G05'],
 ['Dexbotic_DM0','OpenDM05'],['VPP2','VPP2-Preview'],
 ['Liber_agent_wam','Liber_wam'],['ME_Brain_1','MachEmbodied-U0'],['H_RDT','RDT_1B']]

def prepare():
    sim,_,aliases,scores,date=extract()
    txt=(RAW/'robodojo_app.js').read_text()
    types=assigned_literal(txt,'Tr');configs=Parser(txt[txt.index('$l={')+3:]).value()
    ids=sorted(scores);parent={x:x for x in ids}
    def root(x):
        while parent[x]!=x:x=parent[x]
        return x
    edges=[]
    def union(a,b,reason):
        if a in parent and b in parent:
            parent[root(b)]=root(a);edges.append(dict(policy_a=a,policy_b=b,reason=reason))
    for group in LINEAGES:
        for x in group[1:]:union(group[0],x,'manually grouped named lineage; conservative, not identity proof')
    bycheckpoint={}
    for policy,config in configs.items():
        checkpoint=config.get('sim',{}).get('checkpoint',{}).get('label')
        if checkpoint and checkpoint!='from scratch':bycheckpoint.setdefault(checkpoint,[]).append(policy)
    for checkpoint,group in bycheckpoint.items():
        for x in group[1:]:union(group[0],x,'shared published initialization: '+checkpoint)
    components={}
    for x in ids:components.setdefault(root(x),[]).append(x)
    family={x:min(g) for g in components.values() for x in g}
    meta=[]
    for e in sim:
        model=e['model'];policy=aliases[model];config=configs.get(policy,{}).get('sim',{})
        reason='agent track' if types[model].startswith('agent') else 'single-task training' if 'Single Task' in model else ''
        meta.append(dict(policy_id=policy,display_name=model,policy_type=types[model],family=family[policy],
            documented_training=bool(config),primary_included=bool(config) and not reason,
            broad_included=not reason,exclusion=reason,initialization=config.get('checkpoint',{}).get('label','unverified'),
            training_steps=config.get('steps'),training_epochs=config.get('epochs'),source_revision=date,
            family_caveat='audited grouping proxy; unknown cross-method ancestry may remain'))
    meta=pd.DataFrame(meta).sort_values('policy_id')
    meta.to_csv(PROCESSED/'sim_prediction_policy_metadata.csv',index=False)
    pd.DataFrame(edges).to_csv(PROCESSED/'sim_prediction_family_edges.csv',index=False)
    tasks=sorted({t for row in scores.values() for t in row if not t.endswith('_random')})
    matrix=pd.DataFrame({p:{t:row[t]['successRate']/100 for t in tasks if t in row} for p,row in scores.items()}).T.reindex(columns=tasks)
    if len(tasks)!=42:raise ValueError('Unexpected base task catalog')
    for cohort,flag in [('primary','primary_included'),('broad','broad_included')]:
        chosen=meta.loc[meta[flag],'policy_id'].tolist();m=matrix.loc[chosen]
        if m.isna().any().any():raise ValueError('Missing base-task outcomes; do not impute test targets')
        m.to_csv(PROCESSED/f'sim_prediction_matrix_{cohort}.csv',index_label='policy_id')
    return meta,matrix

def folds(families,seed,k=3):
    """Assign complete related-policy groups; tie/order randomization uses no scores."""
    families=np.asarray(families);rng=np.random.default_rng(seed)
    groups=list(np.unique(families));rng.shuffle(groups)
    # Descending size improves balance; stable sorting retains random tie order.
    groups.sort(key=lambda f:-int((families==f).sum()))
    bins=[[] for _ in range(min(k,len(groups)))];sizes=[0]*len(bins)
    for f in groups:
        j=int(np.argmin(sizes));bins[j].append(f);sizes[j]+=int((families==f).sum())
    return [np.flatnonzero(np.isin(families,b)) for b in bins]

def fit_training(y):
    if not np.isfinite(y).all():raise ValueError('Training matrix must be observed')
    mean=y.mean(axis=0)
    _,_,vt=np.linalg.svd(y-mean,full_matrices=False)
    return dict(mean=mean,values=y.copy(),basis=vt[:4].T)

def latent_predict(fit,anchor_values,anchors,rank,ridge):
    basis=fit['basis'][:,:min(rank,fit['basis'].shape[1])]
    a=basis[anchors]
    weights=np.linalg.solve(a.T@a+ridge*np.eye(a.shape[1]),a.T@(anchor_values-fit['mean'][anchors]).T)
    return np.clip(fit['mean'][None,:]+weights.T@basis.T,0,1)

def predict(fit,anchor_values,anchors,latent=(2,.1)):
    """Only the test policy's revealed anchor values enter this function."""
    m=fit['mean'];n=len(anchor_values)
    if anchor_values.shape[1]!=len(anchors):raise ValueError('Anchor/value mismatch')
    offset=(anchor_values-m[anchors]).mean(axis=1)
    scale=np.maximum(0,(anchor_values@m[anchors]+.01)/(m[anchors]@m[anchors]+.01))
    distance=((anchor_values[:,None,:]-fit['values'][None,:,anchors])**2).mean(axis=2)
    nearest=np.argsort(distance,axis=1,kind='stable')[:,:min(3,len(fit['values']))]
    return dict(Zero=np.zeros((n,len(m))),TaskMean=np.tile(m,(n,1)),
      AnchorMean=np.tile(anchor_values.mean(axis=1)[:,None],(1,len(m))),
      Additive=np.clip(m[None,:]+offset[:,None],0,1),
      Scale=np.clip(scale[:,None]*m[None,:],0,1),
      Neighbor3=fit['values'][nearest].mean(axis=1),
      Latent=latent_predict(fit,anchor_values,anchors,*latent))

def family_mean(values,families):
    return float(np.mean([np.nanmean(values[np.asarray(families)==f]) for f in np.unique(families)]))

def tune_latent(y,families,anchors,seed):
    targets=np.setdiff1d(np.arange(y.shape[1]),anchors)
    options=[(r,l) for r in [1,2,4] for l in [1.,.1,.01]]
    losses={o:np.full(len(y),np.nan) for o in options};audit=[]
    for validation in folds(families,seed):
        train=np.setdiff1d(np.arange(len(y)),validation)
        if not len(train):raise ValueError('Insufficient training families')
        fit=fit_training(y[train])
        for option in options:
            p=latent_predict(fit,y[validation][:,anchors],anchors,*option)
            losses[option][validation]=np.abs(p[:,targets]-y[validation][:,targets]).mean(axis=1)
        audit.append(dict(train_families=sorted(set(np.asarray(families)[train])),validation_families=sorted(set(np.asarray(families)[validation]))))
    results=[dict(rank=o[0],ridge=o[1],mae=family_mean(v,families)) for o,v in losses.items()]
    best=min(r['mae'] for r in results)
    # Candidate order gives the frozen conservative tie break.
    chosen=next(r for r in results if r['mae']<=best+1e-12)
    return (chosen['rank'],chosen['ridge']),dict(inner_folds=audit,candidates=results)

def run_cohort(meta,matrix,cohort):
    flag='primary_included' if cohort=='primary' else 'broad_included'
    md=meta[meta[flag]].set_index('policy_id');ids=md.index.to_numpy();families=md.family.to_numpy()
    tasks=matrix.columns.to_numpy();y=matrix.loc[ids].to_numpy();outputs=[];model_rows=[];task_rows=[];audits=[]
    for partition in range(3):
        for fold,test in enumerate(folds(families,SEED+partition)):
            train=np.setdiff1d(np.arange(len(y)),test);fit=fit_training(y[train])
            if set(families[train])&set(families[test]):raise ValueError('Family leakage')
            for draw in range(5):
                order=np.random.default_rng(SEED+1000+draw).permutation(len(tasks))
                for budget in [5,10]:
                    anchors=order[:budget];targets=np.setdiff1d(np.arange(len(tasks)),anchors)
                    option,inner=tune_latent(y[train],families[train],anchors,SEED+partition+500)
                    preds=predict(fit,y[test][:,anchors],anchors,option)
                    design=dict(cohort=cohort,partition=partition,fold=fold,anchor_draw=draw,budget=budget)
                    audits.append(dict(**design,train_policies=ids[train].tolist(),test_policies=ids[test].tolist(),
                        train_families=sorted(set(families[train])),test_families=sorted(set(families[test])),
                        anchor_tasks=tasks[anchors].tolist(),target_tasks=tasks[targets].tolist(),
                        latent_rank=option[0],latent_ridge=option[1],**inner))
                    for method,p in preds.items():
                        actual=y[test][:,targets];prediction=p[:,targets]
                        abs_err=np.abs(prediction-actual)
                        for j,idx in enumerate(test):
                            nz=actual[j]>0;zero=~nz
                            model_rows.append(dict(**design,method=method,policy_id=ids[idx],family=families[idx],
                                n_targets=len(targets),mae=float(abs_err[j].mean()),mse=float((abs_err[j]**2).mean()),
                                nonzero_mae=float(abs_err[j,nz].mean()) if nz.any() else np.nan,
                                zero_mae=float(abs_err[j,zero].mean()) if zero.any() else np.nan))
                            for q,t in enumerate(targets):outputs.append(dict(**design,method=method,policy_id=ids[idx],family=families[idx],task_id=tasks[t],actual=actual[j,q],prediction=prediction[j,q]))
                        a,b=np.triu_indices(len(test),1);cross=families[test][a]!=families[test][b];a=a[cross];b=b[cross]
                        for q,t in enumerate(targets):
                            dy=actual[a,q]-actual[b,q];dp=prediction[a,q]-prediction[b,q]
                            eligible=np.abs(dy)>=.10-1e-12
                            loss=np.where(np.abs(dp)<1e-12,.5,(np.sign(dp)!=np.sign(dy)).astype(float))
                            task_rows.append(dict(**design,method=method,task_id=tasks[t],n_cross_family_pairs=len(a),
                                n_eligible=int(eligible.sum()),n_predicted_ties=int((eligible&(np.abs(dp)<1e-12)).sum()),
                                loss_sum=float(loss[eligible].sum()),ranking_error=float(loss[eligible].mean()) if eligible.any() else np.nan,
                                close_pairs=int(((np.abs(dy)>1e-12)&~eligible).sum())))
    return pd.DataFrame(outputs),pd.DataFrame(model_rows),pd.DataFrame(task_rows),audits

def summarize(models,tasks):
    # Average repeated observations before clustering: they are not new policies/tasks.
    families=models.groupby(['cohort','budget','method','family','policy_id'])[['mae','mse','nonzero_mae','zero_mae']].mean().groupby(['cohort','budget','method','family']).mean().reset_index()
    # First pool disjoint outer-fold pairs within each partition/task/anchor draw;
    # then average repeated designs within task, finally give tasks equal weight.
    tm=tasks.groupby(['cohort','budget','method','partition','anchor_draw','task_id'])[['loss_sum','n_eligible','n_predicted_ties']].sum().reset_index()
    tm['ranking_error']=tm.loss_sum/tm.n_eligible.replace(0,np.nan)
    task_mean=tm.groupby(['cohort','budget','method','task_id']).ranking_error.mean().reset_index()
    rng=np.random.default_rng(SEED);summary=[];delta_rows=[]
    for (cohort,budget,method),g in families.groupby(['cohort','budget','method']):
        t=task_mean[(task_mean.cohort==cohort)&(task_mean.budget==budget)&(task_mean.method==method)].ranking_error.dropna()
        boots=rng.choice(g.mae,size=(2000,len(g)),replace=True).mean(axis=1)
        tb=rng.choice(t,size=(2000,len(t)),replace=True).mean(axis=1) if len(t) else np.full(2000,np.nan)
        summary.append(dict(cohort=cohort,budget=budget,method=method,n_families=len(g),
            mae=float(g.mae.mean()),mae_low=float(np.quantile(boots,.025)),mae_high=float(np.quantile(boots,.975)),
            rmse=float(np.sqrt(g.mse.mean())),nonzero_mae=float(g.nonzero_mae.mean()),zero_mae=float(g.zero_mae.mean()),
            ranking_error=float(t.mean()),ranking_error_low=float(np.quantile(tb,.025)),ranking_error_high=float(np.quantile(tb,.975)),
            n_informative_tasks=len(t)))
    for cohort in families.cohort.unique():
        for budget in [5,10]:
            for endpoint,frame,index in [('mae',families,'family'),('ranking_error',task_mean,'task_id')]:
                wide=frame[(frame.cohort==cohort)&(frame.budget==budget)].pivot(index=index,columns='method',values=endpoint)
                for method in ['Neighbor3','Latent']:
                    for baseline in ['TaskMean','Additive','Scale']:
                        d=(wide[method]-wide[baseline]).dropna().to_numpy()
                        boot=rng.choice(d,size=(2000,len(d)),replace=True).mean(axis=1)
                        delta_rows.append(dict(cohort=cohort,budget=budget,endpoint=endpoint,method=method,baseline=baseline,
                            delta=float(d.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    return pd.DataFrame(summary),families,task_mean,pd.DataFrame(delta_rows)

def main():
    verify_raw()
    freeze=json.loads((ROOT/'data/sim_prediction_plan_freeze.json').read_text())
    for path,digest in freeze.items():
        if sha256(ROOT/path)!=digest:raise ValueError('Simulation prediction plan/mapping changed')
    meta,matrix=prepare();all_outputs=[];all_models=[];all_tasks=[];audits=[]
    for cohort in ['primary','broad']:
        out,models,tasks,audit=run_cohort(meta,matrix,cohort)
        all_outputs.append(out);all_models.append(models);all_tasks.append(tasks);audits+=audit
        print('Finished '+cohort,flush=True)
    out=pd.concat(all_outputs,ignore_index=True);models=pd.concat(all_models,ignore_index=True);tasks=pd.concat(all_tasks,ignore_index=True)
    out.to_csv(RESULTS/'sim_prediction_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary,family,task_mean,deltas=summarize(models,tasks)
    for frame,name in [(summary,'summary'),(models,'policy_metrics'),(tasks,'ranking_metrics'),(family,'family_metrics'),(task_mean,'task_metrics'),(deltas,'deltas')]:
        frame.to_csv(RESULTS/f'sim_prediction_{name}.csv',index=False)
    write_json(RESULTS/'sim_prediction_fold_audit.json',audits)
    coverage=[]
    for cohort,flag in [('primary','primary_included'),('broad','broad_included')]:
        g=meta[meta[flag]];values=matrix.loc[g.policy_id].to_numpy()
        coverage.append(dict(cohort=cohort,policies=len(g),family_groups=g.family.nunique(),base_tasks=42,cells=int(values.size),
            fraction_zero=float((values==0).mean()),known_task_prediction=True,new_task_prediction=False,
            physical_validation=False,trial_denominators='not reconstructed',
            caveat='Family groups are conservative audited proxies; training recipes differ; broad includes incomplete provenance'))
    write_json(RESULTS/'sim_prediction_coverage.json',coverage)
    print(summary[['cohort','budget','method','mae','ranking_error','nonzero_mae']].to_string(index=False))

if __name__=='__main__':main()
