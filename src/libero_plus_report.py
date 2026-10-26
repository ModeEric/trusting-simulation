"""Summaries and task-cluster intervals for the frozen binary-outcome study."""
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ingest_libero_plus import ROOT, RAW
from .libero_plus_experiment import base_task

R=ROOT/'results';F=R/'figures'
KEY=['study','fraction','budget','method']
COLORS={'TaskMean':'#6b7280','AnchorMean':'#a16207','Additive':'#2563eb','AxisOffset':'#059669','Neighbor3':'#a855f7','Latent':'#dc2626'}
LABELS={'TaskMean':'Other-checkpoint mean','Additive':'Global offset','AxisOffset':'Axis-specific offset','Latent':'Latent model'}

def interval(values,seed=73):
    x=np.asarray(values);x=x[np.isfinite(x)]
    if not len(x):return [np.nan,np.nan]
    rng=np.random.default_rng(seed)
    b=x[rng.integers(0,len(x),(2000,len(x)))].mean(axis=1)
    return np.quantile(b,[.025,.975])

def md_table(frame):
    # Avoid a non-core tabulate dependency.
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])

def main():
    data=pd.read_csv(R/'libero_plus_task_metrics.csv.gz')
    ranking=pd.read_csv(R/'libero_plus_ranking_metrics.csv.gz')
    audit=json.loads((R/'libero_plus_audit.json').read_text())
    designs=pd.DataFrame([{k:a[k] for k in ['study','fraction','budget','n_train','n_train_groups','n_train_cells','n_anchor_cases']} for a in audit])
    # First collapse resampling repetitions, then weight intervention blocks equally.
    bytask=data.groupby(KEY+['base_task','group'])[['brier','rate_mae']].mean().groupby(KEY+['base_task']).mean().reset_index()
    rk=ranking.groupby(KEY+['base_task','axis']).error.mean().groupby(KEY+['base_task']).mean().reset_index()
    rk['agreement']=1-rk.pop('error')
    large_cells=ranking[ranking.n_cases>=20].groupby(KEY+['base_task','axis']).error.mean().groupby(KEY+['base_task']).mean().groupby(KEY).mean().reset_index()
    large_cells['agreement']=1-large_cells.pop('error')
    large_cells.to_csv(R/'libero_plus_ranking_min20_sensitivity.csv',index=False)
    tasks=bytask.merge(rk,on=KEY+['base_task'],how='outer')
    tasks.to_csv(R/'libero_plus_by_task.csv',index=False)
    summaries=[]
    for key,part in tasks.groupby(KEY):
        d=dict(zip(KEY,key));ds=designs[(designs.study==d['study'])&(designs.fraction==d['fraction'])&(designs.budget==d['budget'])]
        d.update({c:float(ds[c].mean()) for c in ['n_train','n_train_groups','n_train_cells','n_anchor_cases']})
        for metric in ['brier','rate_mae','agreement']:
            vals=part[metric].dropna();lo,hi=interval(vals)
            summaries.append(dict(**d,metric=metric,value=vals.mean(),lower=lo,upper=hi,n_base_tasks=len(vals)))
    summary=pd.DataFrame(summaries);summary.to_csv(R/'libero_plus_summary.csv',index=False)
    deltas=[]
    for (study,f,b),part in tasks.groupby(['study','fraction','budget']):
        for metric in ['brier','rate_mae','agreement']:
            pivot=part.pivot(index='base_task',columns='method',values=metric)
            for ref in ['TaskMean','Additive','AxisOffset']:
                delta=(pivot.Latent-pivot[ref]).dropna();lo,hi=interval(delta)
                deltas.append(dict(study=study,fraction=f,budget=b,metric=metric,comparison='Latent minus '+ref,
                    delta=delta.mean(),lower=lo,upper=hi,n_base_tasks=len(delta)))
    pd.DataFrame(deltas).to_csv(R/'libero_plus_paired_deltas.csv',index=False)
    # A repeat pass is an observed reproducibility check, not a new policy.
    repeat=[]
    for name in ['run-a-pass0','run-a-pass1']:
        repeat.append(pd.DataFrame([json.loads(x) for x in gzip.open(RAW/'results'/f'{name}.jsonl.gz','rt')]).assign(base_task=lambda x:x.case_id.map(base_task)))
    a,b=repeat
    assert a.case_id.equals(b.case_id)
    rates=[x.groupby(['base_task','axis']).success.mean() for x in repeat]
    noise=dict(case_disagreement=float((a.success!=b.success).mean()),
        base_task_axis_rate_mae=float((rates[0]-rates[1]).abs().mean()),
        task_axis_cells=len(rates[0]),fraction_cells_gap_ge_10pp=float(((rates[0]-rates[1]).abs()>=.1-1e-12).mean()),
        caution='One checkpoint, two passes; not a universal noise floor or binomial independent-trial estimate.')
    (R/'libero_plus_repeat_noise.json').write_text(json.dumps(noise,indent=2)+'\n')
    # Count unique eligible comparisons rather than treating repeated folds as new data.
    cases=pd.read_csv(ROOT/'data/processed/libero_plus_cases.csv');meta=pd.read_csv(ROOT/'data/processed/libero_plus_policies.csv')
    y=np.load(ROOT/'data/processed/libero_plus_outcomes.npz')['y'];unique=set();target_tasks=set()
    variation=dict(mean_success=float(y.mean()),all_success_cases=int((y.sum(axis=0)==len(y)).sum()),
        all_failure_cases=int((y.sum(axis=0)==0).sum()),varying_cases=int(((y.sum(axis=0)>0)&(y.sum(axis=0)<len(y))).sum()))
    (R/'libero_plus_case_variation.json').write_text(json.dumps(variation,indent=2)+'\n')
    for design in audit:
        if design['study']!='primary' or design['fraction']!=1 or design['budget']!=8:continue
        ix=meta.set_index('policy_id').index.get_indexer(design['test_ids'])
        for task in design['target_tasks']:
            target_tasks.add(task)
            for axis in cases['axis'].unique():
                mask=(cases.base_task==task)&(cases['axis']==axis)
                av=y[ix][:,mask].mean(axis=1);i,j=np.triu_indices(len(ix),1)
                for aa,bb in zip(i,j):
                    if abs(av[aa]-av[bb])>=.1-1e-12:
                        pair=tuple(sorted([design['test_ids'][aa],design['test_ids'][bb]]));unique.add((task,axis,*pair))
    coverage=dict(unique_eligible_comparisons=len(unique),represented_target_base_tasks=len(target_tasks),
        warning='These comparisons share checkpoints and cases and are not independent samples.')
    (R/'libero_plus_ranking_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    for col,kind in enumerate(['checkpoint','anchor']):
        for row,metric in enumerate(['brier','agreement']):
            ax=axs[row,col]
            mask=(summary.study=='primary')&(summary.metric==metric)
            mask&=summary.budget.eq(8) if kind=='checkpoint' else summary.fraction.eq(1)
            part=summary[mask]
            xcol='n_train' if kind=='checkpoint' else 'n_anchor_cases'
            for method in LABELS:
                p=part[part.method==method].sort_values(xcol)
                mul=100 if metric=='agreement' else 1
                ax.plot(p[xcol],p.value*mul,'o-',color=COLORS[method],label=LABELS[method],lw=2,ms=4)
                if method=='Latent':ax.fill_between(p[xcol],p.lower*mul,p.upper*mul,color=COLORS[method],alpha=.12)
            ax.set_xlabel('Training checkpoints (mean across grouped folds)' if kind=='checkpoint' else 'Observed anchor cases per held-out checkpoint')
            ax.set_ylabel('Brier score ↓' if metric=='brier' else 'Pairwise agreement, gap ≥10 pp (%) ↑')
            ax.grid(alpha=.18)
            if kind=='anchor':ax.set_xscale('log');ax.set_xticks(part.n_anchor_cases.unique(),[f'{v:,.0f}' for v in part.n_anchor_cases.unique()],rotation=25)
            if row==0:ax.set_title('More training checkpoints; 8 anchor base tasks' if kind=='checkpoint' else 'More anchors; all available training checkpoints',loc='left',fontweight='bold')
    axs[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('LIBERO-Plus: does more observed data improve prediction?',fontsize=17,fontweight='bold')
    fig.supxlabel('24 related π0.5 checkpoints · 40 base tasks · fixed target tasks within each draw\nShading: descriptive 95% base-task bootstrap for latent model; repeats are not independent samples.',fontsize=10)
    for ext in ['png','svg']:fig.savefig(F/f'12_libero_plus_scaling.{ext}',dpi=180)
    plt.close(fig)
    endpoint=summary[(summary.study=='primary')&(summary.fraction==1)&(summary.budget==8)]
    table=endpoint.pivot(index='method',columns='metric',values='value').reindex(['TaskMean','AnchorMean','Additive','AxisOffset','Neighbor3','Latent'])
    display=pd.DataFrame({'Method':table.index,'Brier':table.brier.map(lambda v:f'{v:.4f}'),
        'Task × axis MAE (pp)':table.rate_mae.map(lambda v:f'{100*v:.2f}'),
        'Ranking agreement (%)':table.agreement.map(lambda v:f'{100*v:.2f}')})
    train_curve=summary[(summary.study=='primary')&(summary.budget==8)&(summary.metric=='brier')]
    anc_curve=summary[(summary.study=='primary')&(summary.fraction==1)&(summary.metric=='brier')]
    def curve_table(frame,x):
        pv=frame.pivot(index=x,columns='method',values='value')[list(LABELS)].reset_index()
        for c in pv:
            pv[c]=pv[c].map(lambda v:f'{v:.4f}' if c!=x else f'{v:,.1f}')
        return md_table(pv)
    sensitivity=summary[(summary.study=='camera_block')&(summary.metric.isin(['brier','agreement']))].pivot(index='method',columns='metric',values='value')
    sens=pd.DataFrame({'Method':sensitivity.index,'Brier':sensitivity.brier.map(lambda v:f'{v:.4f}'),'Ranking agreement (%)':sensitivity.agreement.map(lambda v:f'{100*v:.2f}')})
    # Diagnose target axes using saved predictions (draw 0, 8-anchor endpoint).
    pred=pd.read_csv(R/'libero_plus_predictions.csv.gz');pred=pred[pred.study=='primary'].copy()
    pred['ae']=abs(pred.prediction-pred.actual)
    axes=pred.groupby(['method','base_task','axis']).ae.mean().groupby(['method','axis']).mean().unstack(0)*100
    axes.to_csv(R/'libero_plus_axis_errors.csv')
    choices=pd.DataFrame([dict(study=a['study'],fraction=a['fraction'],budget=a['budget'],**a['selected']) for a in audit])
    choices.to_csv(R/'libero_plus_selected_models.csv',index=False)
    scaling=[]
    for kind,low,high in [('checkpoints',.25,1.),('anchors',1,24)]:
        for metric in ['brier','rate_mae','agreement']:
            q=tasks[(tasks.study=='primary')&(tasks.method=='Latent')]
            q=q[q.budget==8] if kind=='checkpoints' else q[q.fraction==1]
            pivot=q.pivot(index='base_task',columns='fraction' if kind=='checkpoints' else 'budget',values=metric)
            delta=(pivot[high]-pivot[low]).dropna();lo,hi=interval(delta)
            scaling.append(dict(resource=kind,metric=metric,delta=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(scaling).to_csv(R/'libero_plus_scaling_deltas.csv',index=False)
    axis_rank=1-ranking[(ranking.study=='primary')&(ranking.fraction==1)&(ranking.budget==8)].groupby(['method','base_task','axis']).error.mean().groupby(['method','axis']).mean().unstack(0)
    axis_rank.to_csv(R/'libero_plus_axis_ranking.csv')
    failures=axis_rank[['Additive','AxisOffset','Latent']].mul(100).round(1).reset_index()
    text=f'''# LIBERO-Plus: scaling prediction from public binary outcomes

This is an independent missing-result prediction study, not a new robot-policy evaluation or a sim-to-real validation.

**Main finding:** More training checkpoints improve the latent model's binary probability predictions: Brier falls from 0.0650 to 0.0561 as the average training set grows from 5.6 to 16 checkpoints, a 13.8% reduction. The paired base-task bootstrap interval for the change is [-0.01095, -0.00713], conditional on this checkpoint cohort. But at eight anchor tasks, latent ranking agreement is only 65.7%, versus 84.1% for the simple global-offset model. The latent predictor's better success-rate MAE (4.31 versus 5.01 points) does not translate into better model selection.

Compared with the stronger axis-specific offset baseline, the latent Brier advantage is small: -0.00086, with descriptive paired interval [-0.00180, +0.00007]. Under the stricter camera/noncamera holdout, its Brier is 0.06392 versus 0.06340 for the axis-specific baseline. These results support an empirical data-scaling trend, but not a robust claim that latent modeling is the best predictor or ranking method.

## Data and effective sample size

Source: [Targeted Data Sets for π0.5 on LIBERO-Plus](https://github.com/Alakazam-studios/targeted-data-libero-plus), pinned commit `554a0d06924f64f15551119e393373a9bc008771`. Apache-2.0 code and results; original README, NOTICE, license, manifest, and all 36 result files are preserved. The records originate from the paper authors, not an independent benchmark rerun. We checked compressed and uncompressed SHA-256 hashes, case alignment, binary outcomes, and reproduced all 36 published seven-axis means.

The public release has 361,080 outcomes, 36 passes and 29 weight files. The primary cohort has **24 distinct checkpoints × 10,030 cases = 240,720 binary observations**. We exclude changed-prompt/rewriter configurations and other evaluation protocols, and use only run A's first pass in prediction. The extra pass is reserved for the noise diagnostic. These cases derive from **40 base tasks**, four suites, seven perturbation axes, and five nominal difficulty levels. Difficulty labels are retained as source metadata; we do not assume equal numerical difficulty across axes.

All primary checkpoints share π0.5. Nine frozen intervention groups keep related seed replicates and ablations together; they are **not nine independent model families**, and cross-group parents remain. A stricter camera versus noncamera sensitivity further restricts training. Broader model-family generalization cannot be tested here.

The cohort's mean binary success is {100*variation['mean_success']:.2f}%. Of 10,030 cases, {variation['all_success_cases']:,} succeed for every checkpoint, {variation['all_failure_cases']:,} fail for every checkpoint, and {variation['varying_cases']:,} vary between checkpoints. An always-success classifier would therefore already have high accuracy, which is why accuracy is not our primary metric.

## Prediction setup

For a held-out checkpoint, reveal the outcomes on some anchor base tasks and predict outcomes on 16 other base tasks. All variants of a base task stay on the same side for that held-out checkpoint. Training checkpoints supply outcomes on the target cases: this is **missing-cell prediction on known benchmark tasks**, not prediction on a task unseen by every checkpoint. No held-out checkpoint's target labels enter training or hyperparameter selection.

We use three repeated three-fold intervention-group partitions and three nested anchor orders. Each anchor order holds target tasks fixed while revealing 1, 2, 4, 8, 16 or 24 base tasks. Checkpoint scaling uses 25%, 50%, 75% or 100% of available training intervention groups with eight anchor tasks. Larger fractions retain the smaller training sets. The two horizontal axes therefore mean different resources: training checkpoint evaluations versus additional observations of the held-out checkpoint. They are not robot training data or robot training compute.

TaskMean averages other checkpoints' binary outcomes on each case. AnchorMean predicts the held-out checkpoint's average anchor success everywhere. Additive adds one anchor-estimated offset to TaskMean. AxisOffset estimates seven separate offsets. Neighbor3 averages the three training checkpoints closest on anchors. Latent learns a rank-limited basis of checkpoint-by-case variation; inner grouped validation selects rank and ridge using Brier score. All fitted probabilities are bounded to [0,1].

With centered training outcome matrix C, latent case factors V are the leading right singular vectors rescaled by sqrt(number of cases). For observed anchor vector s_A and training mean mu: z = (V_Aᵀ V_A / |A| + lambda I)^(-1) V_Aᵀ(s_A - mu_A) / |A|, and p_T = clip(mu_T + V_T z, 0, 1). No target observations of this checkpoint appear in this equation.

## Results at eight anchor base tasks and all available training checkpoints

{md_table(display)}

Brier is mean squared error on binary outcomes and is the primary proper scoring rule. MAE measures predicted versus observed success fractions within base-task × axis groups, rather than pretending each binary case is a measured success probability. Ranking compares jointly held-out checkpoints on these groups; observed differences below 10 percentage points are excluded and predicted ties count half. Repeated splits/pairs are not new independent samples. The endpoint covers {coverage['represented_target_base_tasks']} unique target base tasks across draws and {coverage['unique_eligible_comparisons']:,} unique eligible (checkpoint pair, base task, axis) comparisons; these still share data. Task-axis groups have 4–50 cases. An exploratory reporting-only sensitivity restricts ranking to groups with at least 20 cases (`libero_plus_ranking_min20_sensitivity.csv`); it does not change model selection.

### Scaling the number of training checkpoints

{curve_table(train_curve,'n_train')}

### Scaling observed anchor cases of the held-out checkpoint

{curve_table(anc_curve,'n_anchor_cases')}

These are empirical learning curves over the observed range. We have not fitted or established an asymptotic power-law scaling law. Detailed paired changes and descriptive base-task bootstrap intervals are in `libero_plus_paired_deltas.csv`; task-level values are in `libero_plus_by_task.csv`.

## Stricter camera versus noncamera intervention holdout

{md_table(sens)}

This sensitivity puts all 15 camera-related checkpoints on one side and all nine other checkpoints on the other, with eight anchor tasks. It is still within π0.5, and it does not eliminate shared pretraining or common ancestors.

## Where ranking fails

The following table aggregates eligible comparisons by base task within each perturbation axis, at eight anchors and full training data. Different axes can have different eligible pairs, so this is descriptive error localization, not a controlled causal effect of the perturbation.

{md_table(failures)}

The latent model is tuned for Brier score, not the sign of between-checkpoint performance differences. One plausible explanation is that shrinkage and case-level residual fitting improve probability estimates while distorting smaller checkpoint differences. We have not tested that mechanism. A concrete next experiment is a global ability term plus strongly regularized task corrections, selected using jointly held-out pairwise ranking loss. Keep these results fixed as the baseline rather than retuning against this test set.

## Repeat-pass diagnostic and limitations

Run A's two public passes disagree on **{100*noise['case_disagreement']:.2f}% of binary case outcomes**. Their observed base-task × axis success rates differ by **{100*noise['base_task_axis_rate_mae']:.2f} percentage points MAE**; **{100*noise['fraction_cells_gap_ge_10pp']:.1f}%** of these 280 groups differ by at least 10 points. This is one checkpoint's repeatability, not an estimated universal noise floor. Neither the cases nor checkpoint variants should be treated as independent robot systems.

Intervals resample base tasks after averaging repeated splits, so they are conditional on these checkpoints and on this benchmark. The nominal 10-point ranking margin does not imply a statistically resolved difference; several task-axis groups contain few cases. We do not synthesize trials, estimate unknown binomial denominators, or use model-selection results from held-out checkpoints.

Because these are closely related ablations on the same benchmark, correlated failures and common training data can make missing-cell prediction easier than prediction across architectures. High binary accuracy can also be trivial on a high-success dataset; we report Brier and compare against case-wise TaskMean. No finding here establishes real-world ranking validity.

## Reproduction

From the project root, run `python -m src.ingest_libero_plus`, then `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.libero_plus_experiment`, then `python -m src.libero_plus_report`, and `python -m unittest discover -s tests -v`. The supplied inputs are pinned; reproduction is offline. `--download` on ingestion is optional for the original pinned URLs. No new rollouts, robot-policy inference, physical trials, or paid API calls were used.

![Learning curves](figures/12_libero_plus_scaling.png)
'''
    (R/'libero_plus_findings.md').write_text(text)
    print(display.to_string(index=False));print('\nRepeat diagnostic:',noise)

if __name__=='__main__':main()
