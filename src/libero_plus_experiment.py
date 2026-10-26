"""Grouped missing-cell prediction on pinned LIBERO-Plus binary records."""
import gzip
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .ingest_libero_plus import ROOT, RAW, verify, sha
from .sim_prediction import folds

GROUPS={
 'base':['pi05-libero'],
 'base_recipe':['run-a-pass0','run-b-nolang','run-b-control','run-c-nolang','run-c-cleaned-lang'],
 'no_recovery':['noclips','noclips-nolang'],
 'camera_adapter':['main-model','main-model-noviews','all-views','main-model-seed2','main-model-seed3','cross-view-adapter'],
 'stage_order':['adapter-then-fullft','fullft-then-adapter'],
 'camera_full_weights':['views-all-weights','cross-view-all-weights'],
 'teacher_relabel':['teacher-rollouts','teacher-failures'],
 'layouts':['views-and-layouts','layouts-moved-cameras','layout-rollouts'],
 'language_mix':['rewordings-in-mix']}
CAMERA={'camera_adapter','stage_order','camera_full_weights','teacher_relabel','layouts'}
METHODS=['TaskMean','AnchorMean','Additive','AxisOffset','Neighbor3','Latent']
RESULTS=ROOT/'results'; PROCESSED=ROOT/'data/processed'

def base_task(case_id):
    # The moved-layout suffix occurs only on the ten goal-suite tasks.
    x=re.split(r'_view_|_language_|_light_|_table_|_tb_|_add_|_noise_|_level[0-9]+_',case_id)[0]
    return x.removesuffix('_moved')

def prepare():
    manifest,_=verify()
    names={n:g for g,ns in GROUPS.items() for n in ns}
    files=[f for f in manifest['files'] if f['file'].removesuffix('.jsonl.gz') in names]
    assert len(files)==24 and len({f['weights_sha256'] for f in files})==24
    assert all(f['protocol_id']==manifest['protocol_id'] and ';' not in f['protocol'] for f in files)
    arrays=[];metadata=[];cases=None
    for f in files:
        rows=[json.loads(r) for r in gzip.open(RAW/'results'/f['file'],'rt')]
        arrays.append([r['success'] for r in rows])
        name=f['file'].removesuffix('.jsonl.gz')
        metadata.append(dict(policy_id=name,group=names[name],weights_sha256=f['weights_sha256'],model=f['model']))
        if cases is None:
            cases=pd.DataFrame(rows).drop(columns=['success','steps'])
            cases['base_task']=cases.case_id.map(base_task)
    assert cases.base_task.nunique()==40
    assert cases.groupby('suite').base_task.nunique().eq(10).all()
    assert cases.groupby('base_task')['axis'].nunique().eq(7).all()
    meta=pd.DataFrame(metadata)
    y=np.asarray(arrays,dtype=float)
    cases.to_csv(PROCESSED/'libero_plus_cases.csv',index=False)
    meta.to_csv(PROCESSED/'libero_plus_policies.csv',index=False)
    np.savez_compressed(PROCESSED/'libero_plus_outcomes.npz',y=y)
    return y,meta,cases

def fit_basis(y):
    mean=y.mean(axis=0);center=y-mean
    eig,u=np.linalg.eigh(center@center.T)
    order=np.argsort(eig)[::-1];eig=eig[order];u=u[:,order]
    valid=eig>max(1,float(eig[0]))*1e-12
    rank=min(8,int(valid.sum()))
    basis=center.T@u[:,:rank]/np.sqrt(eig[:rank]) if rank else np.zeros((y.shape[1],0))
    return dict(mean=mean,basis=basis*np.sqrt(y.shape[1]),values=y)

def latent_predict(fit,observed,anchors,targets,rank,ridge):
    basis=fit['basis'][:,:rank];a=basis[anchors];k=a.shape[1]
    z=np.linalg.solve(a.T@a/len(anchors)+ridge*np.eye(k),a.T@(observed-fit['mean'][anchors]).T/len(anchors))
    return np.clip(fit['mean'][targets]+z.T@basis[targets].T,0,1)

def balanced_loss(actual,pred,groups,task_codes):
    loss=(actual-pred)**2
    return float(np.mean([np.mean([loss[groups==g][:,task_codes==t].mean() for t in np.unique(task_codes)]) for g in np.unique(groups)]))

def tune(y,groups,anchors,targets,task_codes,seed):
    opts=[(r,l) for r in [1,2,4,8] for l in [10.,1.,.1]]
    losses={o:[] for o in opts};audit=[]
    for val in folds(groups,seed):
        train=np.setdiff1d(np.arange(len(y)),val)
        if not len(train):continue
        fit=fit_basis(y[train])
        audit.append(dict(train_groups=sorted(set(groups[train])),validation_groups=sorted(set(groups[val]))))
        for o in opts:
            p=latent_predict(fit,y[val][:,anchors],anchors,targets,*o)
            losses[o].append(balanced_loss(y[val][:,targets],p,groups[val],task_codes))
    values=[float(np.mean(losses[o])) if losses[o] else float('inf') for o in opts]
    best=opts[int(np.argmin(values))]
    return best,dict(inner_folds=audit,choices=[dict(rank=r,ridge=l,brier=v) for (r,l),v in zip(opts,values)])

def predict(y,observed,anchors,targets,axes,option,fit=None):
    """Held-out checkpoint outcomes outside anchors are deliberately not arguments."""
    fit=fit_basis(y) if fit is None else fit;m=fit['mean'];n=len(observed)
    offset=(observed-m[anchors]).mean(axis=1)
    p=dict(TaskMean=np.tile(m[targets],(n,1)),
        AnchorMean=np.broadcast_to(observed.mean(axis=1)[:,None],(n,len(targets))).copy(),
        Additive=np.clip(m[targets]+offset[:,None],0,1))
    axis_p=p['Additive'].copy()
    for axis in np.unique(axes):
        a=axes[anchors]==axis;t=axes[targets]==axis
        if a.any():
            delta=(observed[:,a]-m[anchors[a]]).mean(axis=1)
            axis_p[:,t]=np.clip(m[targets[t]]+delta[:,None],0,1)
    p['AxisOffset']=axis_p
    distances=((observed[:,None,:]-y[None,:,anchors])**2).mean(axis=2)
    nearest=np.argsort(distances,axis=1,kind='stable')[:,:min(3,len(y))]
    p['Neighbor3']=y[nearest][:,:,targets].mean(axis=1)
    p['Latent']=latent_predict(fit,observed,anchors,targets,*option)
    return p

def make_task_split(cases,seed,budget):
    order=np.random.default_rng(seed).permutation(sorted(cases.base_task.unique()))
    anchors=np.flatnonzero(cases.base_task.isin(order[:budget]))
    targets=np.flatnonzero(cases.base_task.isin(order[24:]))
    assert not set(cases.base_task.iloc[anchors])&set(cases.base_task.iloc[targets])
    return anchors,targets,order

def select_training(pool,groups,fraction,seed):
    order=np.random.default_rng(seed).permutation(sorted(set(groups[pool])))
    n=max(2,int(np.ceil(len(order)*fraction)))
    return pool[np.isin(groups[pool],order[:n])]

def score(actual,preds,meta,cases,design):
    rows=[];ranking=[];aggregates=[]
    groups=meta.group.to_numpy()
    for task in sorted(cases.base_task.unique()):
        ti=np.flatnonzero(cases.base_task.to_numpy()==task)
        for method,p in preds.items():
            for group in np.unique(groups):
                gi=np.flatnonzero(groups==group)
                brier=float(((actual[np.ix_(gi,ti)]-p[np.ix_(gi,ti)])**2).mean())
                errs=[]
                for axis in sorted(cases.axis.unique()):
                    ci=np.flatnonzero((cases.base_task.to_numpy()==task)&(cases.axis.to_numpy()==axis))
                    if len(ci):errs.extend(np.abs(actual[np.ix_(gi,ci)].mean(axis=1)-p[np.ix_(gi,ci)].mean(axis=1)))
                rows.append(dict(**design,method=method,base_task=task,group=group,brier=brier,rate_mae=float(np.mean(errs))))
            a,b=np.triu_indices(len(actual),1)
            for axis in sorted(cases.axis.unique()):
                ci=np.flatnonzero((cases.base_task.to_numpy()==task)&(cases.axis.to_numpy()==axis))
                if not len(ci):continue
                av=actual[:,ci].mean(axis=1);pv=p[:,ci].mean(axis=1)
                dy=av[a]-av[b];dp=pv[a]-pv[b];eligible=np.abs(dy)>=.1-1e-12
                loss=np.where(np.abs(dp)<1e-12,.5,(np.sign(dp)!=np.sign(dy)).astype(float))
                ranking.append(dict(**design,method=method,base_task=task,axis=axis,n_cases=len(ci),eligible=int(eligible.sum()),
                    error=float(loss[eligible].mean()) if eligible.any() else np.nan,loss_sum=float(loss[eligible].sum())))
                if design['fraction']==1 and design['budget']==8 and design['draw']==0:
                    for i,policy in enumerate(meta.policy_id):
                        aggregates.append(dict(**design,method=method,policy_id=policy,base_task=task,axis=axis,n_cases=len(ci),actual=av[i],prediction=pv[i]))
    return rows,ranking,aggregates

def freeze():
    paths=['libero_plus_plan.json','src/ingest_libero_plus.py','src/libero_plus_experiment.py',
        'data/raw/libero_plus/provenance.json','data/raw/libero_plus/results/MANIFEST.json']
    record={p:sha((ROOT/p).read_bytes()) for p in paths}
    dest=ROOT/'data/libero_plus_freeze.json'
    if dest.exists():assert json.loads(dest.read_text())==record,'Frozen inputs changed'
    else:dest.write_text(json.dumps(record,indent=2)+'\n')

def export_frames(rows,ranks,aggs):
    for name,records in [('task_metrics',rows),('ranking_metrics',ranks),('predictions',aggs)]:
        text=pd.DataFrame(records).to_csv(index=False).encode()
        dest=RESULTS/f'libero_plus_{name}.csv.gz';temp=dest.with_suffix('.tmp')
        temp.write_bytes(gzip.compress(text,mtime=0));temp.replace(dest)
        assert gzip.decompress(dest.read_bytes())==text,'Incomplete output file: '+str(dest)

def replay():
    """Deterministically regenerate exports using already saved split/tuning choices."""
    freeze();y,meta,cases=prepare();axes=cases['axis'].to_numpy()
    audits=json.loads((RESULTS/'libero_plus_audit.json').read_text());rows=[];ranks=[];aggs=[]
    for a in audits:
        train=meta.set_index('policy_id').index.get_indexer(a['train_ids']);test=meta.set_index('policy_id').index.get_indexer(a['test_ids'])
        anchors=np.flatnonzero(cases.base_task.isin(a['anchor_tasks']));targets=np.flatnonzero(cases.base_task.isin(a['target_tasks']))
        option=(a['selected']['rank'],a['selected']['ridge'])
        p=predict(y[train],y[test][:,anchors],anchors,targets,axes,option)
        keys=['study','partition','fold','draw','fraction','budget','n_train','n_train_groups','n_train_cells','n_anchor_cases']
        rr,kk,aa=score(y[test][:,targets],p,meta.iloc[test],cases.iloc[targets],{k:a[k] for k in keys})
        rows.extend(rr);ranks.extend(kk);aggs.extend(aa)
    export_frames(rows,ranks,aggs)
    print('Replayed all saved choices; all three compressed output files verified.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay-audit',action='store_true');args=parser.parse_args()
    if args.replay_audit:return replay()
    start=time.perf_counter();freeze();y,meta,cases=prepare()
    plan=json.loads((ROOT/'libero_plus_plan.json').read_text());seed=plan['seed']
    groups=meta.group.to_numpy();axes=cases.axis.to_numpy()
    rows=[];ranks=[];aggs=[];audits=[]
    settings=[(1.,b) for b in plan['anchor_budgets_base_tasks']]+[(f,8) for f in plan['training_group_fractions'] if f!=1]
    for partition in range(3):
        for fold,test in enumerate(folds(groups,seed+partition)):
            pool=np.setdiff1d(np.arange(len(y)),test)
            for draw in range(3):
                for fraction,budget in settings:
                    train=select_training(pool,groups,fraction,seed+partition*100+fold*10+draw)
                    anchors,targets,order=make_task_split(cases,seed+1000+draw,budget)
                    option,tuning=tune(y[train],groups[train],anchors,targets,cases.base_task.to_numpy()[targets],seed+partition)
                    p=predict(y[train],y[test][:,anchors],anchors,targets,axes,option)
                    design=dict(study='primary',partition=partition,fold=fold,draw=draw,fraction=fraction,budget=budget,
                        n_train=len(train),n_train_groups=len(set(groups[train])),n_train_cells=len(train)*len(cases),n_anchor_cases=len(anchors))
                    rr,kk,aa=score(y[test][:,targets],p,meta.iloc[test],cases.iloc[targets],design)
                    rows.extend(rr);ranks.extend(kk);aggs.extend(aa)
                    audits.append(dict(**design,train_ids=meta.policy_id.iloc[train].tolist(),test_ids=meta.policy_id.iloc[test].tolist(),
                        train_groups=sorted(set(groups[train])),test_groups=sorted(set(groups[test])),
                        anchor_tasks=order[:budget].tolist(),target_tasks=order[24:].tolist(),selected=dict(rank=option[0],ridge=option[1]),**tuning))
            print(f'Finished primary partition {partition+1}/3, fold {fold+1}/3',flush=True)
    # More conservative sensitivity: all camera-stage blocks stay on one side.
    broad=np.array(['camera' if g in CAMERA else 'noncamera' for g in groups])
    for fold,label in enumerate(['camera','noncamera']):
        test=np.flatnonzero(broad==label);train=np.flatnonzero(broad!=label)
        for draw in range(3):
            anchors,targets,order=make_task_split(cases,seed+1000+draw,8)
            option,tuning=tune(y[train],groups[train],anchors,targets,cases.base_task.to_numpy()[targets],seed)
            p=predict(y[train],y[test][:,anchors],anchors,targets,axes,option)
            design=dict(study='camera_block',partition=0,fold=fold,draw=draw,fraction=1.,budget=8,n_train=len(train),
                n_train_groups=len(set(groups[train])),n_train_cells=len(train)*len(cases),n_anchor_cases=len(anchors))
            rr,kk,aa=score(y[test][:,targets],p,meta.iloc[test],cases.iloc[targets],design)
            rows.extend(rr);ranks.extend(kk);aggs.extend(aa)
            audits.append(dict(**design,train_ids=meta.policy_id.iloc[train].tolist(),test_ids=meta.policy_id.iloc[test].tolist(),
                train_groups=sorted(set(groups[train])),test_groups=sorted(set(groups[test])),anchor_tasks=order[:8].tolist(),
                target_tasks=order[24:].tolist(),selected=dict(rank=option[0],ridge=option[1]),**tuning))
    export_frames(rows,ranks,aggs)
    (RESULTS/'libero_plus_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    (RESULTS/'libero_plus_run.json').write_text(json.dumps(dict(status='passed',seconds=time.perf_counter()-start,
        checkpoints=len(y),cases=len(cases),base_tasks=40,groups=len(GROUPS),fits=len(audits),new_rollouts=0,paid_calls=0),indent=2)+'\n')
    print('Finished all prediction experiments.',flush=True)

if __name__=='__main__':main()
