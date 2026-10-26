"""Independent implementations; tests compare them to pinned author functions."""
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr

def author_pearson(sim,real):
    x=np.asarray(sim,dtype=float);y=np.asarray(real,dtype=float)
    x=x-x.mean();y=y-y.mean()
    if np.all(x==y):return 1.0
    return float(np.dot(x,y)/(np.sqrt(np.dot(x,x)*np.dot(y,y))+1e-8))

def mmrv(sim,real):
    """Author convention: a strict > comparison disagreement, including ties.

    A predicted tie against a true ordering penalizes the truly higher model
    only. This differs from both strict inversion loss and 0.5 tie penalties.
    Argument order matters because violation magnitudes use REAL outcomes.
    """
    s=np.asarray(sim,dtype=float);y=np.asarray(real,dtype=float)
    if s.shape!=y.shape:raise ValueError('shape mismatch')
    disagreement=(s[:,None]>s[None,:]) != (y[:,None]>y[None,:])
    return float(np.where(disagreement,np.abs(y[:,None]-y[None,:]),0).max(axis=1).mean())

def task_pairs(frame,prediction='prediction',margin=0.10):
    rows=[]
    for i,j in combinations(range(len(frame)),2):
        a,b=frame.iloc[i],frame.iloc[j]
        dy=float(a.real-b.real);dp=float(a[prediction]-b[prediction])
        real_tie=abs(dy)<1e-12;pred_tie=abs(dp)<1e-12
        loss=0.5 if pred_tie else float(np.sign(dy)!=np.sign(dp))
        rows.append(dict(task_id=a.task_id,embodiment=a.embodiment,
            policy_a=a.policy_id,policy_b=b.policy_id,real_gap=dy,predicted_gap=dp,
            sim_gap=float(a.sim-b.sim),real_tie=real_tie,pred_tie=pred_tie,
            eligible=not real_tie and abs(dy)>=margin-1e-12,
            close=not real_tie and abs(dy)<margin-1e-12,
            tie_aware_loss=loss if not real_tie else np.nan,
            correct=not pred_tie and np.sign(dy)==np.sign(dp)))
    return rows

def summarize_task(frame,prediction='prediction',margin=0.10):
    import pandas as pd
    pairs=pd.DataFrame(task_pairs(frame,prediction,margin))
    y=frame.real.to_numpy();pred=frame[prediction].to_numpy()
    eligible=pairs[pairs.eligible] if len(pairs) else pairs
    decided=eligible[~eligible.pred_tie] if len(eligible) else eligible
    raw=pairs[~pairs.real_tie & ~pairs.pred_tie] if len(pairs) else pairs
    close=pairs[pairs.close & ~pairs.pred_tie] if len(pairs) else pairs
    return dict(n_policies=len(frame),n_pairs=len(pairs),n_real_ties=int(pairs.real_tie.sum()) if len(pairs) else 0,
        n_eligible=len(eligible),n_decided=len(decided),n_abstain=len(eligible)-len(decided),
        n_correct=int(decided.correct.sum()) if len(decided) else 0,
        sign_accuracy=float(decided.correct.mean()) if len(decided) else np.nan,
        ranking_error=float(eligible.tie_aware_loss.mean()) if len(eligible) else np.nan,
        raw_sign_accuracy=float(raw.correct.mean()) if len(raw) else np.nan,
        close_sign_accuracy=float(close.correct.mean()) if len(close) else np.nan,
        close_n=len(close),mmrv=mmrv(pred,y),mae=float(np.abs(np.clip(pred,0,1)-y).mean()),
        pearson=author_pearson(pred,y),
        spearman=float(spearmanr(pred,y).statistic) if np.ptp(pred)>0 and np.ptp(y)>0 else np.nan,
        mean_bias=float((np.clip(pred,0,1)-y).mean()))
