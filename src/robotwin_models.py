"""Small score predictors and pairwise ranking objectives for public aggregates."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from .sim_prediction import folds, family_mean, fit_training, predict as original_predict

RANKS=[1,2,4,8,12]
RIDGES=[10.,1.,.1,.01]
RANKING_RIDGES=[np.inf,100.,10.,1.,.1]


def factor_fit(y,bounded=False):
    transformed=logit(np.clip(y,.01,.99)) if bounded else np.asarray(y)
    mean=transformed.mean(axis=0)
    _,singular,vt=np.linalg.svd(transformed-mean,full_matrices=False)
    tolerance=max(transformed.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0)
    effective=int((singular>tolerance).sum())
    return dict(mean=mean,basis=vt[:min(12,effective)].T,bounded=bounded,effective_rank=effective)


def factor_predict(fit,anchor_values,anchors,rank,ridge):
    basis=fit['basis'][:,:rank];a=basis[anchors]
    observed=logit(np.clip(anchor_values,.01,.99)) if fit['bounded'] else anchor_values
    z=np.linalg.solve(a.T@a+ridge*np.eye(a.shape[1]),a.T@(observed-fit['mean'][anchors]).T)
    values=fit['mean'][None,:]+z.T@basis.T
    return expit(values) if fit['bounded'] else np.clip(values,0,1)


def select_factors(y,families,anchors,seed,bounded=False):
    options=[(r,l) for r in RANKS for l in RIDGES]
    losses={o:np.full(len(y),np.nan) for o in options};inner=[]
    targets=np.setdiff1d(np.arange(y.shape[1]),anchors)
    for validation in folds(families,seed):
        train=np.setdiff1d(np.arange(len(y)),validation)
        fit=factor_fit(y[train],bounded)
        for option in options:
            p=factor_predict(fit,y[validation][:,anchors],anchors,*option)
            losses[option][validation]=np.abs(p[:,targets]-y[validation][:,targets]).mean(axis=1)
        inner.append(dict(train_families=sorted(set(families[train])),validation_families=sorted(set(families[validation])),
            matrix_rank=fit['effective_rank']))
    candidates=[dict(rank=r,ridge=l,mae=family_mean(losses[(r,l)],families)) for r,l in options]
    choices={}
    for name,ranks,ridges in ([('LogitLatent',RANKS,RIDGES)] if bounded else
        [('OriginalLatent',[1,2,4],[1.,.1,.01]),('LinearRidge',[1,2,4],RIDGES),('LinearWide',RANKS,RIDGES)]):
        selected=[c for c in candidates if c['rank'] in ranks and c['ridge'] in ridges]
        best=min(c['mae'] for c in selected)
        c=next(c for c in selected if c['mae']<=best+1e-12)
        choices[name]=(c['rank'],c['ridge'])
    return choices,dict(inner_folds=inner,candidates=candidates)


def pair_data(y,families,anchors):
    families=np.asarray(families);a,b=np.triu_indices(len(y),1)
    keep=families[a]!=families[b];a=a[keep];b=b[keep]
    differences=y[a][:,anchors]-y[b][:,anchors]
    dy=y[a]-y[b];labels=np.sign(dy);eligible=np.abs(dy)>=.10-1e-12
    sizes={f:int((families==f).sum()) for f in set(families)}
    weights=np.array([1/(sizes[families[i]]*sizes[families[j]]) for i,j in zip(a,b)])[:,None]*eligible
    denom=weights.sum(axis=0);weights=weights/np.where(denom>0,denom,1)
    return differences,labels,weights


def pair_objective(flat,differences,labels,weights,ridge,global_model=False):
    """Analytic pairwise logistic objective; no synthetic episode likelihood."""
    width=differences.shape[1];n_tasks=labels.shape[1]
    delta=flat.reshape(width,1 if global_model else n_tasks)
    utility=differences.mean(axis=1)[:,None]+differences@delta
    margin=labels*utility/.1
    penalty_multiplier=max(1,int((weights.sum(axis=0)>0).sum())) if global_model else 1
    value=float((weights*np.logaddexp(0,-margin)).sum()+.5*ridge*penalty_multiplier*(delta**2).sum())
    derivative=-weights*labels*expit(-margin)/.1
    if global_model:derivative=derivative.sum(axis=1,keepdims=True)
    gradient=differences.T@derivative+ridge*penalty_multiplier*delta
    return value,gradient.ravel()


def fit_ranker(y,families,anchors,ridge,global_model=False):
    differences,labels,weights=pair_data(y,families,anchors)
    shape=(len(anchors),1 if global_model else y.shape[1])
    start=np.zeros(np.prod(shape));before=None
    if np.isinf(ridge) or not weights.any():
        delta=start.reshape(shape);status=dict(success=True,iterations=0,gradient_max=0.,fallback='anchor-only' if np.isinf(ridge) else 'no eligible training pairs')
    else:
        args=(differences,labels,weights,ridge,global_model)
        before=pair_objective(start,*args)[0]
        result=minimize(pair_objective,start,args=args,jac=True,method='L-BFGS-B',
            options={'maxiter':300,'ftol':1e-11,'gtol':1e-7,'maxls':30})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError('Pairwise optimizer did not converge: '+str(result.message))
        delta=result.x.reshape(shape)
        status=dict(success=True,iterations=int(result.nit),gradient_max=float(np.max(np.abs(result.jac))),
            objective_before=before,objective_after=float(result.fun))
    return dict(delta=delta,mean_anchor=y[:,anchors].mean(axis=0),n_tasks=y.shape[1],status=status)


def predict_ranker(fit,anchor_values):
    utility=anchor_values.mean(axis=1)[:,None]+(anchor_values-fit['mean_anchor'])@fit['delta']
    return np.broadcast_to(utility,(len(anchor_values),fit['n_tasks'])).copy()


def pair_counts(actual,prediction,families):
    a,b=np.triu_indices(len(actual),1);families=np.asarray(families)
    keep=families[a]!=families[b];a=a[keep];b=b[keep]
    dy=actual[a]-actual[b];dp=prediction[a]-prediction[b]
    eligible=np.abs(dy)>=.10-1e-12
    loss=np.where(np.abs(dp)<1e-12,.5,(np.sign(dp)!=np.sign(dy)).astype(float))
    return (loss*eligible).sum(axis=0),eligible.sum(axis=0)


def select_rankers(y,families,anchors,seed):
    families=np.asarray(families);targets=np.setdiff1d(np.arange(y.shape[1]),anchors)
    totals={(name,ridge):[np.zeros(len(targets)),np.zeros(len(targets))] for name in ['RankGlobal','RankTask'] for ridge in RANKING_RIDGES}
    inner=[];optimizers=[]
    for validation in folds(families,seed):
        train=np.setdiff1d(np.arange(len(y)),validation)
        inner.append(dict(train_families=sorted(set(families[train])),validation_families=sorted(set(families[validation]))))
        # Comparisons within a single held-out family are excluded from the protocol.
        if len(set(families[validation]))<2:continue
        for name in ['RankGlobal','RankTask']:
            for ridge in RANKING_RIDGES:
                fit=fit_ranker(y[train],families[train],anchors,ridge,name=='RankGlobal')
                p=predict_ranker(fit,y[validation][:,anchors])
                loss,count=pair_counts(y[validation][:,targets],p[:,targets],families[validation])
                totals[(name,ridge)][0]+=loss;totals[(name,ridge)][1]+=count
                optimizers.append(fit['status'])
    candidates=[];choices={};available={}
    for name in ['RankGlobal','RankTask']:
        usable=[]
        for ridge in RANKING_RIDGES:
            loss,count=totals[(name,ridge)];valid=count>0
            error=float(np.mean(loss[valid]/count[valid])) if valid.any() else None
            candidates.append(dict(method=name,ridge='infinity' if np.isinf(ridge) else ridge,error=error,n_tasks=int(valid.sum())))
            if error is not None:usable.append((ridge,error))
        available[name]=bool(usable)
        best=min(v for _,v in usable) if usable else None
        choices[name]=next((ridge for ridge,v in usable if v<=best+1e-12),np.inf)
    return choices,dict(inner_folds=inner,candidates=candidates,selection_available=available,
        inner_optimizer_fits=len(optimizers),max_inner_optimizer_iterations=max([x['iterations'] for x in optimizers],default=0))


def fit_predict_stage(stage,y,families,anchor_values,anchors,seed):
    """Only training labels and held-out anchor values cross this interface."""
    if stage in [1,2]:
        choices,audit=select_factors(y,np.asarray(families),anchors,seed,stage==2)
        fit=factor_fit(y,stage==2)
        out={name:factor_predict(fit,anchor_values,anchors,*option) for name,option in choices.items()}
        if stage==1:
            base=original_predict(fit_training(y),anchor_values,anchors,choices['OriginalLatent'])
            out.update({k:base[k] for k in ['AnchorMean','Additive','Scale']})
            # Preserve the original implementation exactly for its reference predictions.
            out['OriginalLatent']=base['Latent']
        audit['selected']={name:dict(rank=option[0],ridge=option[1],effective_rank=min(option[0],fit['effective_rank'])) for name,option in choices.items()}
    else:
        choices,audit=select_rankers(y,np.asarray(families),anchors,seed);out={};selected={}
        for name,ridge in choices.items():
            fit=fit_ranker(y,np.asarray(families),anchors,ridge,name=='RankGlobal')
            out[name]=predict_ranker(fit,anchor_values)
            selected[name]=dict(ridge='infinity' if np.isinf(ridge) else ridge,optimizer=fit['status'])
        audit['selected']=selected
    return out,audit
