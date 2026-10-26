"""LOTO first; pair comparisons are summarized per task, not treated as IID."""
import argparse,json
import numpy as np
import pandas as pd
from .common import ROOT,PROCESSED,RESULTS,SEED,write_json
from .baselines import METHODS,predict_fold,fit_calibrator,design
from .taxonomy import FEATURES
from .metrics import summarize_task,task_pairs,mmrv,author_pearson

def grouped_predictions(data,methods=METHODS,group='task_id'):
    out=[];audit=[]
    for embodiment,stratum in data.groupby('embodiment',sort=True):
        sim_only=stratum[['task_id','policy_id','sim']].copy()
        for held,testing in stratum.groupby(group,sort=True):
            train=stratum[stratum[group]!=held].copy()
            if train.empty:continue
            test=testing.drop(columns='real')
            for method in methods:
                pred,coef=predict_fold(train,test,sim_only,method)
                result=testing.copy();result['prediction']=pred;result['method']=method;result['fold']=held
                out.append(result)
                audit.append(dict(embodiment=embodiment,held_out_group=held,group=group,method=method,
                    train_tasks=sorted(train.task_id.unique()),test_tasks=sorted(test.task_id.unique()),
                    train_physical_cells=len(train),test_physical_cells=len(test),
                    coefficients=None if coef is None else coef.tolist()))
    return pd.concat(out,ignore_index=True),audit

def score_predictions(pred):
    rows=[];pairs=[]
    for (emb,task,method),frame in pred.groupby(['embodiment','task_id','method'],sort=True):
        rows.append(dict(embodiment=emb,task_id=task,method=method,**summarize_task(frame)))
        for row in task_pairs(frame):pairs.append(dict(method=method,**row))
    return pd.DataFrame(rows),pd.DataFrame(pairs)

def summary_with_intervals(metrics):
    rows=[];rng=np.random.default_rng(SEED)
    for (emb,method),g in metrics.groupby(['embodiment','method'],sort=True):
        row=dict(embodiment=emb,method=method,n_tasks=len(g),n_eligible=int(g.n_eligible.sum()),
                 n_decided=int(g.n_decided.sum()),n_correct=int(g.n_correct.sum()))
        for col in ['ranking_error','mae','mmrv','sign_accuracy']:
            values=g[col].dropna().to_numpy()
            boot=rng.choice(values,size=(2000,len(values)),replace=True).mean(axis=1)
            row[col]=float(values.mean());row[col+'_lo']=float(np.quantile(boot,.025));row[col+'_hi']=float(np.quantile(boot,.975))
        rows.append(row)
    return pd.DataFrame(rows)

def paired_deltas(metrics):
    rows=[];rng=np.random.default_rng(SEED)
    for emb,g in metrics.groupby('embodiment'):
        for endpoint in ['mae','ranking_error','mmrv']:
            wide=g.pivot(index='task_id',columns='method',values=endpoint)
            for baseline in ['B1','B2','B2_ridge']:
                for method in ['B2','C1']:
                    if method==baseline:continue
                    vals=(wide[method]-wide[baseline]).dropna().to_numpy()
                    boot=rng.choice(vals,size=(2000,len(vals)),replace=True).mean(axis=1)
                    loo=np.array([np.delete(vals,i).mean() for i in range(len(vals))])
                    rows.append(dict(embodiment=emb,endpoint=endpoint,method=method,baseline=baseline,
                        delta=float(vals.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                        leave_one_task_min=float(loo.min()),leave_one_task_max=float(loo.max())))
    return pd.DataFrame(rows)

def selective_predictions(data):
    # This is not calibrated epistemic uncertainty. It is an exploratory
    # task-balanced ridge prediction of pair loss with fixed shrinkage.
    confidence=[];folds=[]
    for emb,stratum in data.groupby('embodiment'):
        for task,testing in stratum.groupby('task_id'):
            train=stratum[stratum.task_id!=task]
            training_pairs=[]
            for _,g in train.groupby('task_id'):
                pairs=pd.DataFrame(task_pairs(g.assign(prediction=g.sim)))
                pairs=pairs[pairs.eligible & ~pairs.pred_tie]
                for f in FEATURES:pairs[f]=g.iloc[0][f]
                training_pairs.append(pairs)
            p=pd.concat(training_pairs,ignore_index=True)
            test_pairs=pd.DataFrame(task_pairs(testing.assign(prediction=testing.sim)))
            for f in FEATURES:test_pairs[f]=testing.iloc[0][f]
            # Build prediction features without consulting held-out outcomes.
            x=np.column_stack([np.ones(len(p)),p.sim_gap.abs(),*[p[f] for f in FEATURES]])
            xt=np.column_stack([np.ones(len(test_pairs)),test_pairs.sim_gap.abs(),*[test_pairs[f] for f in FEATURES]])
            w=1/p.groupby('task_id').task_id.transform('size').to_numpy();w=w/w.sum()
            coef=np.linalg.pinv(x.T@(w[:,None]*x)+np.diag([0,.1,.1,.1]))@(x.T@(w*p.tie_aware_loss.to_numpy()))
            test_pairs['predicted_error']=np.clip(xt@coef,0,1)
            test_pairs['confidence']=1-test_pairs.predicted_error
            test_pairs['n_train_errors']=int((p.tie_aware_loss>0).sum())
            confidence.append(test_pairs)
            folds.append(dict(embodiment=emb,task_id=task,train_tasks=sorted(train.task_id.unique()),
                              train_pairs=len(p),train_errors=int((p.tie_aware_loss>0).sum()),coefficients=coef.tolist()))
    pairs=pd.concat(confidence,ignore_index=True);curves=[]
    for emb,g in pairs.groupby('embodiment'):
        # Confidence thresholds operate on ALL simulated non-ties at deployment.
        allpairs=g[~g.pred_tie].copy()
        eligible=allpairs[allpairs.eligible]
        for threshold in np.linspace(0,1,21):
            selected=allpairs[allpairs.confidence>=threshold-1e-12]
            evaluated=selected[selected.eligible]
            task_risk=evaluated.groupby('task_id').tie_aware_loss.mean()
            curves.append(dict(embodiment=emb,threshold=threshold,
                coverage=len(selected)/len(allpairs),eligible_coverage=len(evaluated)/len(eligible),
                ranking_error=float(task_risk.mean()) if len(task_risk) else np.nan,
                n_selected=len(selected),n_eligible_selected=len(evaluated),n_tasks_evaluated=len(task_risk)))
    return pairs,pd.DataFrame(curves),folds

def negative_controls(data,metrics):
    rng=np.random.default_rng(SEED);rows=[]
    for rep in range(100):
        shuffled=data.copy()
        for emb,g in shuffled.groupby('embodiment'):
            meta=g[['task_id',*FEATURES]].drop_duplicates().sort_values('task_id')
            perm=rng.permutation(len(meta));values=meta[FEATURES].to_numpy()[perm]
            mapping=dict(zip(meta.task_id,values))
            for idx,row in g.iterrows():shuffled.loc[idx,FEATURES]=mapping[row.task_id]
        pred,_=grouped_predictions(shuffled,methods=['C1'])
        met,_=score_predictions(pred)
        for emb,g in met.groupby('embodiment'):
            rows.append(dict(permutation=rep,embodiment=emb,mae=g.mae.mean(),mmrv=g.mmrv.mean()))
    return pd.DataFrame(rows)

def policy_family_cv(data):
    out=[]
    for emb,g in data.groupby('embodiment'):
        for fam,test in g.groupby('policy_family'):
            train=g[g.policy_family!=fam]
            nfamilies=train.policy_family.nunique()
            if train.empty:continue
            for method in ['B1','B2','B2_ridge','C1']:
                features=FEATURES if method=='C1' else []
                pred=test.sim.to_numpy() if method=='B1' else design(test,features)@fit_calibrator(train,features,.1 if method in ['C1','B2_ridge'] else 0)
                for (_,row),v in zip(test.iterrows(),pred):
                    out.append(dict(embodiment=emb,held_family=fam,train_families=nfamilies,
                        task_id=row.task_id,policy_id=row.policy_id,method=method,real=row.real,prediction=v,
                        absolute_error=abs(np.clip(v,0,1)-row.real),
                        caveat='MAE only; singleton held families cannot support within-task ranks; training physical labels include these tasks'))
    return pd.DataFrame(out)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['simpler'],default='simpler')
    parser.add_argument('--cv',choices=['leave-one-task-out'],default='leave-one-task-out');parser.parse_args()
    data=pd.read_csv(PROCESSED/'simpler_paired.csv')
    pred,audit=grouped_predictions(data);met,pairs=score_predictions(pred)
    pred.to_csv(RESULTS/'simpler_predictions.csv',index=False)
    met.to_csv(RESULTS/'simpler_metrics.csv',index=False);pairs.to_csv(RESULTS/'simpler_pairs.csv',index=False)
    write_json(RESULTS/'fold_audit.json',audit)
    summary=summary_with_intervals(met);summary.to_csv(RESULTS/'simpler_summary.csv',index=False)
    paired_deltas(met).to_csv(RESULTS/'paired_deltas.csv',index=False)
    reference=met[met.method=='B1'];reference.to_csv(RESULTS/'simpler_source_metrics.csv',index=False)
    selective,curves,folds=selective_predictions(data)
    selective.to_csv(RESULTS/'selective_pair_predictions.csv',index=False);curves.to_csv(RESULTS/'risk_coverage_exploratory.csv',index=False)
    write_json(RESULTS/'selective_fold_audit.json',folds)
    negative_controls(data,met).to_csv(RESULTS/'taxonomy_permutations.csv',index=False)
    sensitivities=[]
    for label,sub in [('remove_rt1',data[data.policy_family!='rt-1'].copy()),('alternate_precision',data.copy())]:
        if label=='alternate_precision':sub.loc[sub.task_id.str.contains('drawer|carrot'),'precision_contact']=0
        p,_=grouped_predictions(sub);m,_=score_predictions(p);m['sensitivity']=label;sensitivities.append(m)
    pd.concat(sensitivities).to_csv(RESULTS/'sensitivities.csv',index=False)
    p,a=grouped_predictions(data,group='task_family');m,_=score_predictions(p)
    m.to_csv(RESULTS/'family_holdout_metrics.csv',index=False);write_json(RESULTS/'family_fold_audit.json',a)
    policy_family_cv(data).to_csv(RESULTS/'policy_family_holdout.csv',index=False)
    print(summary[['embodiment','method','mae','ranking_error','mmrv','n_correct','n_decided']].to_string(index=False))
if __name__=='__main__':main()
