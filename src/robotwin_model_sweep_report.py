"""Compare all prespecified model variants, including null findings."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import RESULTS
from .sim_prediction import SEED

METHODS=['AnchorMean','Additive','Scale','OriginalLatent','LinearRidge','LinearWide','LogitLatent','RankGlobal','RankTask']
LABELS={'AnchorMean':'Anchor average','Additive':'Additive','Scale':'Scalar','OriginalLatent':'Original latent',
        'LinearRidge':'Wider ridge grid','LinearWide':'Higher-dimensional','LogitLatent':'Bounded latent','RankGlobal':'Global rank-trained','RankTask':'Task rank-trained'}


def collect():
    summaries=[];family=[];task=[];selected=[];predictions=[]
    for stage in [1,2,3]:
        prefix=f'robotwin_models_stage{stage}'
        if json.loads((RESULTS/f'{prefix}_status.json').read_text())['status']!='passed':raise ValueError('Unfinished stage')
        summaries.append(pd.read_csv(RESULTS/f'{prefix}_summary.csv'))
        if stage!=3:family.append(pd.read_csv(RESULTS/f'{prefix}_family_metrics.csv'))
        task.append(pd.read_csv(RESULTS/f'{prefix}_task_metrics.csv'))
        predictions.append(pd.read_csv(RESULTS/f'{prefix}_primary_predictions.csv.gz'))
        for a in json.loads((RESULTS/f'{prefix}_audit.json').read_text()):
            for name,option in a['selected'].items():
                selected.append(dict(stage=stage,cohort=a['cohort'],partition=a['partition'],fold=a['fold'],anchor_draw=a['anchor_draw'],
                    budget=a['budget'],subset_draw=a['subset_draw'],training_fraction=a['training_fraction'],method=name,
                    rank=option.get('rank'),effective_rank=option.get('effective_rank'),ridge=option['ridge'],
                    selection_available=a.get('selection_available',{}).get(name,True),
                    optimizer_iterations=option.get('optimizer',{}).get('iterations')))
    s=pd.concat(summaries,ignore_index=True);f=pd.concat(family,ignore_index=True);t=pd.concat(task,ignore_index=True)
    selection=pd.DataFrame(selected);selection.to_csv(RESULTS/'robotwin_models_selections.csv',index=False)
    s.to_csv(RESULTS/'robotwin_models_summary.csv',index=False)
    deltas=[]
    for endpoint,frame,index in [('mae',f,'family'),('ranking_error',t,'task_id')]:
        for key,g in frame.groupby(['cohort','training_fraction','budget','target_track']):
            wide=g.pivot(index=index,columns='method',values=endpoint)
            for method in ['LinearRidge','LinearWide','LogitLatent','RankGlobal','RankTask']:
                if method not in wide:continue
                for baseline in ['OriginalLatent','AnchorMean','Additive']:
                    values=(wide[method]-wide[baseline]).dropna().to_numpy();rng=np.random.default_rng(SEED)
                    boot=rng.choice(values,size=(2000,len(values)),replace=True).mean(axis=1)
                    deltas.append(dict(zip(['cohort','training_fraction','budget','target_track'],key),endpoint=endpoint,method=method,baseline=baseline,
                        delta=float(values.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),n_clusters=len(values)))
    d=pd.DataFrame(deltas);d.to_csv(RESULTS/'robotwin_models_paired_deltas.csv',index=False)
    anchor_deltas=[]
    for endpoint,frame,index in [('mae',f,'family'),('ranking_error',t,'task_id')]:
        for (cohort,method),g in frame[(frame.training_fraction==1)&(frame.target_track=='common30')].groupby(['cohort','method']):
            wide=g.pivot(index=index,columns='budget',values=endpoint)
            values=(wide[20]-wide[10]).dropna().to_numpy();rng=np.random.default_rng(SEED)
            boot=rng.choice(values,size=(2000,len(values)),replace=True).mean(axis=1)
            anchor_deltas.append(dict(cohort=cohort,method=method,endpoint=endpoint,delta_20_minus_10=float(values.mean()),
                low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),n_clusters=len(values)))
    pd.DataFrame(anchor_deltas).to_csv(RESULTS/'robotwin_models_anchor_deltas.csv',index=False)
    close=gap_diagnostics(pd.concat(predictions,ignore_index=True))
    return s,d,selection,close


def gap_diagnostics(predictions):
    rows=[]
    for key,g in predictions.groupby(['cohort','method','partition','fold','anchor_draw']):
        y=g.pivot(index='policy_id',columns='task_id',values='actual');p=g.pivot(index='policy_id',columns='task_id',values='prediction').reindex(index=y.index,columns=y.columns)
        family=g.drop_duplicates('policy_id').set_index('policy_id').loc[y.index,'family'].to_numpy()
        a,b=np.triu_indices(len(y),1);keep=family[a]!=family[b];a=a[keep];b=b[keep]
        dy=y.to_numpy()[a]-y.to_numpy()[b];dp=p.to_numpy()[a]-p.to_numpy()[b]
        loss=np.where(np.abs(dp)<1e-12,.5,(np.sign(dp)!=np.sign(dy)).astype(float))
        for kind,mask in [('material',np.abs(dy)>=.10-1e-12),('close',(np.abs(dy)>1e-12)&(np.abs(dy)<.10-1e-12))]:
            for j,task in enumerate(y.columns):rows.append(dict(zip(['cohort','method','partition','fold','anchor_draw'],key),gap_group=kind,
                task_id=task,loss_sum=float(loss[mask[:,j],j].sum()),n_eligible=int(mask[:,j].sum())))
    pooled=pd.DataFrame(rows).groupby(['cohort','method','partition','anchor_draw','gap_group','task_id'])[['loss_sum','n_eligible']].sum().reset_index()
    pooled['error']=pooled.loss_sum/pooled.n_eligible.replace(0,np.nan)
    task=pooled.groupby(['cohort','method','gap_group','task_id']).error.mean().reset_index()
    task.to_csv(RESULTS/'robotwin_models_gap_task_metrics.csv',index=False)
    out=task.groupby(['cohort','method','gap_group']).agg(error=('error','mean'),n_tasks=('error','count')).reset_index()
    out.to_csv(RESULTS/'robotwin_models_gap_summary.csv',index=False)
    return out


def plot_primary(summary):
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.6))
    colors={'AnchorMean':'#8290a2','Additive':'#bf915f','OriginalLatent':'#3c66b4','LinearWide':'#7095d4',
            'LogitLatent':'#248e81','RankGlobal':'#b8862c','RankTask':'#855cb6'}
    for row,cohort in enumerate(['clean','randomized']):
        for col,endpoint in enumerate(['mae','ranking_error']):
            methods=['Additive','OriginalLatent','LinearWide','LogitLatent'] if endpoint=='mae' else ['AnchorMean','OriginalLatent','LinearWide','LogitLatent','RankGlobal','RankTask']
            s=summary[(summary.cohort==cohort)&(summary.training_fraction==1)&(summary.budget==10)&(summary.target_track=='standard')&(summary.endpoint==endpoint)].set_index('method').loc[methods]
            value=s.value.to_numpy()*100;err=np.array([value-s.low.to_numpy()*100,s.high.to_numpy()*100-value])
            ax=axes[row,col];names=[LABELS[m] for m in methods]
            ax.barh(names,value,color=[colors[m] for m in methods],xerr=err,height=.65,error_kw={'ecolor':'#354251','lw':.9,'capsize':2})
            ax.invert_yaxis();ax.set_title(cohort.capitalize()+' scenes',loc='left',fontweight='bold')
            for i,v in enumerate(value):ax.text(s.iloc[i].high*100+.25,i,f'{v:.2f}',va='center',fontsize=9)
            ax.set_xlim(0,17 if col==0 else max(27,float(s.high.max()*100+5)))
            ax.set_xlabel('Score MAE (percentage points)' if col==0 else 'Ranking error (%) · observed gaps ≥10 points')
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('RoboTwin: higher dimensions, bounded scores, and direct ranking training',x=.025,ha='left',fontsize=15,fontweight='bold')
    fig.text(.025,.929,'Full training pool · 10 observed anchors → 40 held-out task scores · fixed family-grouped validation',fontsize=10.5,color='#465468')
    fig.text(.025,.025,'Lower is better. Whiskers are descriptive 95% cluster-bootstrap intervals. Ranking ties cost 0.5.\nRank-trained outputs are utilities, so success-rate MAE is not reported for them. No physical-transfer validation.',fontsize=9,color='#465468')
    fig.tight_layout(rect=[.01,.08,.99,.9],h_pad=2,w_pad=2)
    for ext in ['png','svg']:fig.savefig(RESULTS/f'figures/10_robotwin_model_variants.{ext}',dpi=180)
    plt.close(fig)


def plot_scaling(summary):
    fig,axes=plt.subplots(2,2,figsize=(11.5,8))
    panels=[('mae','LogitLatent','OriginalLatent','Bounded minus original latent · MAE'),
            ('ranking_error','RankTask','AnchorMean','Task rank-trained minus anchor average · ranking error')]
    for col,(endpoint,method,baseline,title) in enumerate(panels):
        matrices=[]
        for cohort in ['clean','randomized']:
            s=summary[(summary.cohort==cohort)&(summary.target_track=='common30')&(summary.endpoint==endpoint)]
            wide=s.pivot(index=['training_fraction','budget'],columns='method',values='value')
            matrices.append(((wide[method]-wide[baseline])*100).unstack('budget'))
        scale=max(1,max(float(np.abs(m.to_numpy()).max()) for m in matrices))
        for row,(cohort,m) in enumerate(zip(['clean','randomized'],matrices)):
            ax=axes[row,col];im=ax.imshow(m.to_numpy(),cmap='RdBu_r',vmin=-scale,vmax=scale,aspect='auto')
            ax.set_xticks(range(3),['5','10','20']);ax.set_xlabel('Observed anchor tasks')
            counts=summary[(summary.cohort==cohort)&(summary.target_track=='common30')].groupby('training_fraction').training_cells_mean.first()
            ax.set_yticks(range(3),[f'{counts.loc[f]:.0f}' for f in m.index]);ax.set_ylabel('Training score cells (mean per fit)')
            ax.set_title(cohort.capitalize()+' scenes',loc='left',fontweight='bold')
            for i in range(3):
                for j in range(3):
                    v=m.iloc[i,j];ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=12,color='white' if abs(v)>.65*scale else '#263341')
            fig.colorbar(im,ax=ax,label='Difference (percentage points)',fraction=.046,pad=.04)
            if row==0:ax.text(0,1.23,title,transform=ax.transAxes,fontsize=10,fontweight='bold')
    fig.suptitle('Separating training-policy coverage from anchor coverage',x=.04,ha='left',fontsize=15,fontweight='bold')
    fig.text(.04,.035,'Blue / negative favors the new model. Each comparison uses exactly the same 30 target tasks at every anchor budget.\nSmall-budget ranking selection may fall back to the anchor baseline when inner folds lack cross-family pairs.',fontsize=9,color='#465468')
    fig.tight_layout(rect=[.02,.09,.99,.9],h_pad=2.8,w_pad=2)
    for ext in ['png','svg']:fig.savefig(RESULTS/f'figures/11_robotwin_model_scaling.{ext}',dpi=180)
    plt.close(fig)


def main():
    summary,deltas,selection,close=collect();plot_primary(summary);plot_scaling(summary)
    p=summary[(summary.training_fraction==1)&(summary.budget==10)&(summary.target_track=='standard')]
    lines=['# RoboTwin: three sequential modeling experiments','',
      '24 October 2026. Independent exploratory analysis of pinned public evaluation means. All three experiments were specified before inspecting their results. No robot trials, simulation rollouts, robot-policy training/inference, or paid API calls.','',
      '**Finding:** expanded linear capacity does not improve the primary result because inner validation still selects low ranks. The bounded model modestly improves success-rate MAE, with clearer evidence in randomized scenes. Direct ranking training improves primary agreement by only 0.10 points clean / 0.20 randomized over the anchor average, with both descriptive paired intervals including zero. Increasing anchor coverage produces a larger observed ranking gain than changing model class.','',
      '## Setup and scope','',
      'We retain the 25 jointly trained RoboTwin configurations, 50 known task identities and 17 conservative ancestry proxies. Clean and randomized conditions remain separate. The same three grouped outer folds, three seeded partitions and five random anchor orders are used. Both policies in each ranking comparison are held out together and belong to different groups.','',
      'Training coverage uses 20%, 60% and 100% of available groups, with the first three nested subset orders from the preceding learning-curve study. Anchor budgets are 5, 10 and 20. The primary result is full training coverage and ten anchors evaluated on the remaining 40 tasks, preserving the original endpoint. The controlled scaling comparison always evaluates the last 30 tasks of the seeded order, so changing anchor count does not change its target set.','',
      'An observation is a published policy-task mean. Trial denominators remain unverified; no episode labels or binomial likelihood are reconstructed. More training cells here means more evaluated policies supplied to the meta-predictor. The underlying robot policies are fixed.','',
      '## 1. Higher-dimensional linear models','',
      'OriginalLatent retains rank caps 1/2/4 and ridge 1/0.1/0.01. LinearRidge adds ridge 10 as a control. LinearWide also allows rank caps 8 and 12. Centered SVD learns task loadings; ridge regression infers a held-out policy’s coordinates from anchors. Inner grouped CV selects family-macro MAE. Requested rank is capped by the available numerical rank, which can be smaller in inner folds.','',
      'At full training coverage, neither rank 8 nor 12 is selected at any of the three anchor budgets. The wider linear model and wider-ridge control reproduce the original model’s primary 10-anchor results. These data do not establish that greater latent dimension helps; low-dimensional choices winning in this small sample do not prove higher-dimensional structure is absent.','',
      '## 2. Bounded nonlinear model','',
      'LogitLatent clips published proportions to [0.01,0.99] for numerical stability, transforms them to logits, fits centered SVD factors in that space, infers coordinates by ridge regression on logit anchors, and returns sigmoid outputs. It uses the same expanded rank/ridge grid and selects probability-space MAE in inner grouped CV.','',
      'This is a bounded nonlinear prediction rule, with squared-error factor fitting in logit space. It does not jointly optimize a probability-space matrix-factorization likelihood and does not add interactions between latent coordinates. The clipping threshold is a fixed numerical choice, not an inferred trial-count correction.','',
      '## 3. Direct pairwise ranking training','',
      'RankGlobal learns one correction to the anchor average; RankTask learns a separate correction for each known task. Their utility is mean(anchor scores) plus a linear function of centered anchors. Training minimizes pairwise logistic loss for distinct-group policy pairs whose observed success gap is at least 0.10, with fixed temperature 0.10. Pair weights are inverse products of the groups’ policy counts, normalized within task. L2 regularization shrinks the correction toward zero.','',
      'Inner grouped validation selects shrinkage from infinity, 100, 10, 1 and 0.1 using task-macro material-gap ranking error. Infinity exactly recovers the anchor-only ranking baseline. Inner ranking comparisons never cross folds. If no eligible cross-family validation pair exists, the selection is flagged unavailable and falls back to infinity. This happens at the smallest training budgets and must not be interpreted as evidence of absent task structure.','',
      'The outputs are uncalibrated ranking utilities. We do not report their numeric difference from success rates as MAE. Training includes the known tasks’ available training-policy labels; only held-out policy anchors enter prediction. Optimizer failures raise errors rather than being silently included.','',
      '## Primary absolute results','',
      'MAE averages repeats within policy, policies within group and groups equally. Ranking error pools disjoint folds within a design, averages designs within task and then tasks equally. Observed gaps must be at least ten percentage points; predicted ties cost 0.5.','',
      '| Method | Clean MAE (pp) | Randomized MAE (pp) | Clean ranking agreement | Randomized ranking agreement |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        s=p[p.method==method].set_index(['cohort','endpoint'])
        vals=[]
        for cohort in ['clean','randomized']:vals.append(f"{100*s.loc[(cohort,'mae'),'value']:.2f}" if (cohort,'mae') in s.index else 'Not calibrated')
        for cohort in ['clean','randomized']:vals.append(f"{100*(1-s.loc[(cohort,'ranking_error'),'value']):.2f}%")
        lines.append('| '+LABELS[method]+' | '+' | '.join(vals)+' |')
    lines += ['', '## Paired comparisons','',
      'Negative differences favor the new model. The descriptive 95% percentile intervals resample 17 policy-group proxies for MAE or informative tasks for ranking, with 2,000 draws. They do not capture all shared-training dependence, unknown ancestry, noisy source means or publication selection. No multiple-comparison-adjusted significance claim is made.','',
      '| Condition | New model | Endpoint | Reference | Difference (pp) | Descriptive 95% interval |','|---|---|---|---|---:|---:|']
    d=deltas[(deltas.training_fraction==1)&(deltas.budget==10)&(deltas.target_track=='standard')]
    for r in d[((d.method.isin(['LinearWide','LogitLatent']))&(d.baseline=='OriginalLatent'))|((d.method.isin(['RankGlobal','RankTask']))&(d.baseline=='AnchorMean'))].itertuples():
        lines.append(f'| {r.cohort} | {r.method} | {r.endpoint} | {r.baseline} | {100*r.delta:+.2f} | [{100*r.low:+.2f}, {100*r.high:+.2f}] |')
    lines += ['', '## Anchor-count sensitivity on a fixed target set','',
      'Full training coverage below; all budgets evaluate the same 30 tasks in each draw. This table cannot be directly compared with the primary 40-task table without accounting for its different task mix.','',
      '| Condition | Anchors | Original latent MAE | Bounded latent MAE | Anchor-average agreement | Global rank-trained agreement | Task rank-trained agreement |','|---|---:|---:|---:|---:|---:|---:|']
    for cohort in ['clean','randomized']:
        for budget in [5,10,20]:
            s=summary[(summary.cohort==cohort)&(summary.training_fraction==1)&(summary.target_track=='common30')&(summary.budget==budget)].set_index(['method','endpoint'])
            vals=[f"{100*s.loc[(m,'mae'),'value']:.2f}" for m in ['OriginalLatent','LogitLatent']]
            vals += [f"{100*(1-s.loc[(m,'ranking_error'),'value']):.2f}%" for m in ['AnchorMean','RankGlobal','RankTask']]
            lines.append(f'| {cohort} | {budget} | '+' | '.join(vals)+' |')
    lines += ['', 'Paired changes from 10 to 20 anchors, evaluated on the common 30-task target set (negative error differences favor 20 anchors):','',
      '| Condition | Method | Endpoint | Change (pp) | Descriptive 95% interval |','|---|---|---|---:|---:|']
    anchor_delta=pd.read_csv(RESULTS/'robotwin_models_anchor_deltas.csv')
    for r in anchor_delta[((anchor_delta.method=='AnchorMean')&(anchor_delta.endpoint=='ranking_error'))|((anchor_delta.method=='LogitLatent')&(anchor_delta.endpoint=='mae'))].itertuples():
        lines.append(f'| {r.cohort} | {r.method} | {r.endpoint} | {100*r.delta_20_minus_10:+.2f} | [{100*r.low:+.2f}, {100*r.high:+.2f}] |')
    lines += ['', 'The training-data effect is also conditional on model choice. On the common target set with ten anchors, bounded-minus-original MAE is +0.09 clean / +0.75 randomized at the smallest budget (~157 cells), versus -0.23 / -0.54 at the full budget (~833 cells). The bounded model benefits from the larger sample in this comparison, but this limited range cannot establish continued scaling. The intermediate-budget task ranker can underperform the anchor baseline; its improvements are not monotonic with training coverage.','',
      '## Selection diagnostics','',
      'Counts below use full training coverage and one copy of each repeated subset result: 45 outer fits per condition and anchor budget. Selection percentages are descriptive, not independent observations.','',
      '| Condition | Anchors | Method | Selected settings |','|---|---:|---|---|']
    sel=selection[(selection.training_fraction==1)&(selection.subset_draw==0)]
    for (cohort,budget,method),g in sel[sel.method.isin(['LinearWide','LogitLatent','RankGlobal','RankTask'])].groupby(['cohort','budget','method']):
        col='ridge' if method.startswith('Rank') else 'rank'
        counts=g[col].astype(str).value_counts().sort_index()
        text=', '.join(f'{col} {k}: {v}' for k,v in counts.items())
        lines.append(f'| {cohort} | {budget} | {method} | {text} |')
    lines += ['', '## Close-gap diagnostics','',
      'Primary ten-anchor predictions, restricting to nonzero observed gaps below ten points. These are retrospective gap strata; the actual hidden gap is unavailable at deployment.','',
      '| Condition | Method | Close-gap agreement |','|---|---|---:|']
    for r in close[(close.gap_group=='close')&close.method.isin(['AnchorMean','OriginalLatent','LogitLatent','RankGlobal','RankTask'])].itertuples():
        lines.append(f'| {r.cohort} | {r.method} | {100*(1-r.error):.2f}% |')
    lines += ['', '## Interpretation and limits','',
      'Increasing model capacity is useful only when the available policies and anchors identify the added structure. The linear experiment favors low-dimensional models. The bounded transformation tests a different score scale and saturation behavior; its score-error effect must be distinguished from policy-ranking effects. The direct ranking models test whether nonuniform anchor weights or task-specific corrections improve model selection. Each is a specific model class, not a general test of all nonlinear or ranking approaches.','',
      'All three model classes and all budgets are reported, including null or adverse results. These exploratory comparisons reuse the same public policy cohort after earlier analyses. Any chosen winner or targeted follow-up needs a fresh policy cohort or independent benchmark for confirmation. No claim is made about physical robot performance, unseen tasks, zero-observation policies, or scaling beyond the available training coverage.','',
      '## Reproduce','',
      'Run `python -m src.robotwin_model_sweep --stage 1`, then stage 2, then stage 3. Run `python -m src.robotwin_model_sweep_report` and `python -m unittest discover -s tests -v`. The full reproduction entry point includes all stages. No network access or robot inference is used.','',
      'Outputs include robotwin_models_summary.csv, robotwin_models_paired_deltas.csv, robotwin_models_selections.csv, per-stage fold audits and predictions, and figures/10_robotwin_model_variants.png and figures/11_robotwin_model_scaling.png. Utility outputs are explicitly labeled. The frozen plan and source hashes are bundled.','',
      'Pinned source: https://github.com/RoboTwin-Platform/RoboTwin-Platform.github.io/blob/0abb5a03455073fa74047a1c4015b0f88cedc254/data/robotwin_leaderboard.json . Source protocol and ancestry caveats are documented in robotwin_findings.md.','']
    (RESULTS/'robotwin_model_sweep_findings.md').write_text('\n'.join(lines))
    print('Wrote all three model comparisons, paired diagnostics and two figures.')


if __name__=='__main__':main()
