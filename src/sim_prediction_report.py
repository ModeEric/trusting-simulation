"""Read-only diagnostics and figures for the frozen simulation predictor run."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import RESULTS, PROCESSED, write_json


def diagnostics(prefix='sim_prediction'):
    predictions=pd.read_csv(RESULTS/f'{prefix}_predictions.csv.gz')
    meta=pd.read_csv(PROCESSED/f'{prefix}_policy_metadata.csv').set_index('policy_id')
    rows=[];unique={}
    keys=['cohort','budget','method','partition','fold','anchor_draw']
    for key,g in predictions[predictions.method.isin(['AnchorMean','Scale','Latent'])].groupby(keys):
        y=g.pivot(index='policy_id',columns='task_id',values='actual')
        p=g.pivot(index='policy_id',columns='task_id',values='prediction').reindex(index=y.index,columns=y.columns)
        fam=meta.loc[y.index,'family'].to_numpy();a,b=np.triu_indices(len(y),1)
        keep=fam[a]!=fam[b];a=a[keep];b=b[keep]
        dy=y.to_numpy()[a]-y.to_numpy()[b];dp=p.to_numpy()[a]-p.to_numpy()[b]
        tie=np.abs(dp)<1e-12;loss=np.where(tie,.5,(np.sign(dp)!=np.sign(dy)).astype(float))
        for kind,mask in [('material',np.abs(dy)>=.10-1e-12),('close',(np.abs(dy)>1e-12)&(np.abs(dy)<.10-1e-12))]:
            for j,task in enumerate(y.columns):
                eligible=mask[:,j]
                rows.append(dict(zip(keys,key),task_id=task,gap_group=kind,n_eligible=int(eligible.sum()),
                                 loss_sum=float(loss[eligible,j].sum()),n_ties=int(tie[eligible,j].sum())))
                if key[2]=='AnchorMean':
                    cells=unique.setdefault((key[0],key[1],kind),set())
                    cells.update((str(y.index[a[i]]),str(y.index[b[i]]),task) for i in np.flatnonzero(eligible))
    detail=pd.DataFrame(rows)
    # Match the primary ranking aggregation: pool folds, average repeats per task, then tasks.
    pooled=detail.groupby(['cohort','budget','method','partition','anchor_draw','task_id','gap_group'])[['n_eligible','loss_sum','n_ties']].sum().reset_index()
    pooled['error']=pooled.loss_sum/pooled.n_eligible.replace(0,np.nan)
    per_task=pooled.groupby(['cohort','budget','method','task_id','gap_group']).error.mean().reset_index()
    unique_counts=[]
    for (cohort,budget,kind),cells in unique.items():
        counts={}
        for _,_,task in cells:counts[task]=counts.get(task,0)+1
        for task,count in counts.items():unique_counts.append(dict(cohort=cohort,budget=budget,gap_group=kind,task_id=task,unique_pairs=count))
    per_task=per_task.merge(pd.DataFrame(unique_counts),how='left').fillna({'unique_pairs':0})
    summary=per_task.groupby(['cohort','budget','method','gap_group']).agg(ranking_error=('error','mean'),n_tasks=('error','count')).reset_index()
    counts=detail.groupby(['cohort','budget','method','gap_group'])[['n_eligible','n_ties']].sum().reset_index()
    summary=summary.merge(counts);summary['pooled_tie_fraction']=summary.n_ties/summary.n_eligible
    summary['unique_policy_pair_task_comparisons']=[len(unique[(r.cohort,r.budget,r.gap_group)]) for r in summary.itertuples()]
    summary.to_csv(RESULTS/f'{prefix}_gap_diagnostics.csv',index=False)
    per_task.to_csv(RESULTS/f'{prefix}_gap_task_metrics.csv',index=False)
    # Each row is one anchor set, with folds/partitions averaged before family macro averaging.
    models=pd.read_csv(RESULTS/f'{prefix}_policy_metrics.csv')
    variation=models.groupby(['cohort','budget','method','anchor_draw','family','policy_id']).mae.mean().groupby(['cohort','budget','method','anchor_draw','family']).mean().groupby(['cohort','budget','method','anchor_draw']).mean().reset_index()
    rankvar=pooled[pooled.gap_group=='material'].groupby(['cohort','budget','method','anchor_draw','task_id']).error.mean().groupby(['cohort','budget','method','anchor_draw']).mean().reset_index(name='ranking_error')
    variation=variation.merge(rankvar,how='left')
    variation.to_csv(RESULTS/f'{prefix}_anchor_variation.csv',index=False)
    return summary,per_task,variation


def main():
    summary=pd.read_csv(RESULTS/'sim_prediction_summary.csv')
    coverage=json.loads((RESULTS/'sim_prediction_coverage.json').read_text())
    deltas=pd.read_csv(RESULTS/'sim_prediction_deltas.csv')
    diag,task_diag,variation=diagnostics()
    lines=['# Predicting simulated robot performance from a small task panel','',
      '24 October 2026. Exploratory, retrospective analysis of pinned public RoboDojo results. Independent research; not commissioned or endorsed by RoboDojo or Robocurve. No new trials, simulator rollouts, robot-policy inference/training, or paid API calls.', '',
      '**Finding:** ten existing task scores predict rankings on the remaining known benchmark tasks well when the observed policy gap is at least 10 percentage points. A simple anchor-task average beats the richer latent model on ranking. The latent model reduces absolute score error, but its advantage over a one-scalar calibration is small and uncertain. This does not establish prediction of physical performance or unseen tasks.', '',
      '## Data and effective sample size', '',
      'The primary cohort includes only non-agent, non-single-task-trained entries with simulation training configurations in the pinned website export. The broader sensitivity includes entries lacking that documentation. Neither is a standardized training-recipe experiment. Family groups conservatively combine named related methods and shared published initialization checkpoints. They are grouping proxies, not proof of independent ancestry.', '',
      '| Cohort | Policy configurations | Family groups | Base tasks | Observed means | Zero scores |',
      '|---|---:|---:|---:|---:|---:|']
    for c in coverage:lines.append(f"| {c['cohort']} | {c['policies']} | {c['family_groups']} | {c['base_tasks']} | {c['cells']} | {100*c['fraction_zero']:.1f}% |")
    lines += ['', 'All selected configurations have all 42 base-task scores. Twelve randomized variants are excluded to avoid learning from near-duplicate tasks across the anchor/target split. The primary and broad cohorts overlap; the broad result is a sensitivity analysis, not external replication. Repeated folds, pairs, and anchors do not create additional independent policies or tasks. Published means are the targets; no episode counts, binomial intervals, or synthetic trials are inferred.', '',
      '## What is held out', '',
      'Three outer folds hold out whole family groups, repeated for three seeded partitions. Training policies supply all 42 known-task scores. Each held-out policy reveals only 5 or 10 anchor scores; its remaining 37 or 32 scores are excluded from fitting and hyperparameter selection. Five uniformly sampled anchor orders are fixed independently of scores, shared across compared policies, with the five-task set nested inside the ten-task set.',
      'Both policies in every ranking comparison are held out together in the same fold and belong to different family groups. Cross-fold predictions are never compared: doing so could let each predictor train on the other policy’s target scores. Inner family-grouped CV selects the latent rank and regularization using outer-training policies only. Mutation tests replace every outer-test nonanchor label and confirm identical selected hyperparameters and predictions.',
      'The 42 task identities are known through training policies. This tests prediction for a held-out policy family with observed anchors, not prediction for an entirely new task or a policy with no observations. It also does not remove any benchmark exposure during the underlying robot policies’ original training.', '',
      '## Predictors', '',
      'Let s_ma be policy m’s observed anchor score, A the anchor set, and mu_t the mean score on task t among outer-training policies. All predictions are clipped to [0,1].', '',
      '- **Zero:** predict zero everywhere; necessary because many scores are zero.',
      '- **TaskMean:** predict mu_t for every held-out policy; all policy rankings are ties.',
      '- **AnchorMean:** predict mean(s_ma over a in A) for every target task; one global ability number per policy.',
      '- **Additive:** predict mu_t + mean(s_ma - mu_a over anchors).',
      '- **Scale:** predict c_m * mu_t, with c_m = (sum(mu_a * s_ma) + 0.01) / (sum(mu_a^2) + 0.01). This fits one scalar from anchors, regularized toward 1.',
      '- **Neighbor3:** average the full profiles of the three training policies closest on anchor scores.',
      '- **Latent:** predict mu_t + v_t dot z_m. Training-only centered SVD learns task vectors v_t; z_m = (V_A.T V_A + lambda I)^(-1) V_A.T (s_mA - mu_A). Inner grouped CV chooses rank 1/2/4 and lambda 0.01/0.1/1.0 for MAE.', '',
      'Latent factors are a capability-profile hypothesis, not hand-labeled task mechanics. We did not optimize a separate rank-loss model or tune on outer test results. The implementation and design hashes were frozen before inspecting these predictor results; this is a local audit trail, not external preregistration.', '',
      '## Absolute results', '',
      'MAE is in percentage points, averaging repeated designs within policy, then policies within family, then families equally. Ranking error first pools jointly held-out pairs across disjoint folds within a design, averages designs within task, then tasks equally. Only observed gaps >=10 points are eligible; predicted ties cost 0.5. Thus 100% minus error is tie-adjusted agreement, not accuracy on all pairs. Nonzero-target MAE applies the same family aggregation after restricting targets to observed scores above zero.', '',
      '| Cohort | Anchors | Predictor | MAE (pp) | Nonzero MAE (pp) | Rank error |',
      '|---|---:|---|---:|---:|---:|']
    for cohort in ['primary','broad']:
        for budget in [5,10]:
            for method in ['Zero','TaskMean','AnchorMean','Additive','Scale','Neighbor3','Latent']:
                r=summary[(summary.cohort==cohort)&(summary.budget==budget)&(summary.method==method)].iloc[0]
                lines.append(f'| {cohort} | {budget} | {method} | {100*r.mae:.2f} | {100*r.nonzero_mae:.2f} | {100*r.ranking_error:.2f}% |')
    lines += ['', '**Interpretation:** a global ability estimate is strong for material-gap rankings on these data. Learning task-specific latent profiles improves score prediction over TaskMean, but it does not improve ranking over AnchorMean. The primary ten-anchor latent MAE is 2.04 points versus 2.17 for Scale; the paired family-bootstrap difference is -0.13 points, with a descriptive 95% interval of approximately [-0.27, +0.01]. That is insufficient evidence that the more complex predictor is better than scalar calibration. Its ranking error is 11.56%, compared with 5.29% for AnchorMean.',
      'The low primary MAE needs context: 67.6% of observed cells are zero, and the zero predictor already achieves 3.50-point MAE at ten anchors. Latent MAE on positive-score targets is 5.00 points; Scale is 4.84. The broader cohort has higher scores and larger errors: ten-anchor Scale MAE 5.57 points and AnchorMean ranking error 7.22%.', '',
      '## Pair coverage, close comparisons, and ties', '',
      'The following counts are unique (policy pair, target task) comparisons observed across repeated held-out designs. They are correlated, not independent sample sizes. The observed 10-point gap filter is a retrospective evaluation stratum: a deployable system cannot know the true target gap without evaluation.', '',
      '| Cohort | Anchors | Gap | Unique pair-task comparisons | Informative tasks | AnchorMean error | Latent error | AnchorMean tie fraction* |',
      '|---|---:|---|---:|---:|---:|---:|---:|']
    for cohort in ['primary','broad']:
        for budget in [5,10]:
            for kind in ['material','close']:
                a=diag[(diag.cohort==cohort)&(diag.budget==budget)&(diag.gap_group==kind)&(diag.method=='AnchorMean')].iloc[0]
                l=diag[(diag.cohort==cohort)&(diag.budget==budget)&(diag.gap_group==kind)&(diag.method=='Latent')].iloc[0]
                lines.append(f'| {cohort} | {budget} | {kind} | {a.unique_policy_pair_task_comparisons:.0f} | {a.n_tasks:.0f} | {100*a.ranking_error:.2f}% | {100*l.ranking_error:.2f}% | {100*a.pooled_tie_fraction:.2f}% |')
    lines += ['', '*Tie fraction pools repeated eligible comparisons and is descriptive; ranking errors retain task-macro weighting. Close means strictly nonzero observed gaps below 10 points; exact observed ties are excluded. No uncertainty in the published target means is reconstructed. Only 18/42 primary tasks have any eligible material-gap comparison across the tested designs; the corresponding broad count is 34/42. Good large-gap ranking does not establish reliable ordering of near-tied policies.', '',
      '## Uncertainty and anchor sensitivity', '',
      'Intervals resample 13 primary/31 broad family groups for MAE and informative task clusters for ranking (2,000 draws). They are descriptive conditional intervals: shared training sets, task selection, uncertain ancestry, and noisy published means are not fully modeled. All baselines and richer methods are reported; selecting a winner here requires fresh external validation.', '',
      '| Cohort | Ten-anchor method | MAE 95% interval (pp) | Ranking error 95% interval | MAE across five anchor sets (pp) |',
      '|---|---|---:|---:|---:|']
    for cohort in ['primary','broad']:
        for method in ['AnchorMean','Scale','Latent']:
            r=summary[(summary.cohort==cohort)&(summary.budget==10)&(summary.method==method)].iloc[0]
            v=variation[(variation.cohort==cohort)&(variation.budget==10)&(variation.method==method)]
            lines.append(f'| {cohort} | {method} | [{100*r.mae_low:.2f}, {100*r.mae_high:.2f}] | [{100*r.ranking_error_low:.2f}%, {100*r.ranking_error_high:.2f}%] | {100*v.mae.min():.2f}–{100*v.mae.max():.2f} |')
    lines += ['', '## Largest primary ranking failures', '',
      'Ten-anchor material-gap error by task, averaged over repeated designs. Sparse per-task comparisons make these descriptive failure examples, not established categories.', '',
      '| Task | Unique eligible policy pairs | AnchorMean error | Scale error | Latent error |', '|---|---:|---:|---:|---:|']
    failures=task_diag[(task_diag.cohort=='primary')&(task_diag.budget==10)&(task_diag.gap_group=='material')].pivot(index='task_id',columns='method',values='error').dropna().sort_values('Latent',ascending=False).head(6)
    pair_counts=task_diag[(task_diag.cohort=='primary')&(task_diag.budget==10)&(task_diag.gap_group=='material')&(task_diag.method=='AnchorMean')].set_index('task_id').unique_pairs
    for task,r in failures.iterrows():lines.append(f'| {task} | {pair_counts[task]:.0f} | {100*r.AnchorMean:.2f}% | {100*r.Scale:.2f}% | {100*r.Latent:.2f}% |')
    lines += ['', '## What to try next', '',
      '1. Freeze Scale for score prediction and AnchorMean for ranking as candidate reference methods, then replicate on an independently audited benchmark with per-policy, per-task results. RoboTwin is a candidate; its training tracks must be separated before combining entries. This run has not performed that replication.',
      '2. Test training-only anchor selection and budget curves against these uniformly sampled anchors. Do not select the anchor panel using held-out policies’ target scores. Ranking-specific fitting is a separate prospective extension, since the current latent objective is MAE.',
      '3. Evaluate policy families with comparable overall strength and tasks with nontrivial success rates. AnchorMean’s strong performance is consistent with global ability explaining much of the current ranking signal; close-gap diagnostics are harder. More diverse independent policy families and better-documented evaluation repetitions are more valuable than counting more correlated episodes as examples. This experiment does not yet test a training-data scaling law.',
      '4. Keep physical validation separate. Predicting a simulated score introduces one error; using simulation as a physical proxy introduces another. Neither the present simulation-only results nor SIMPLER’s small matched table bounds both errors on RoboDojo/Robocurve. A physical claim needs held-out, matched policy-task-hardware evidence.', '',
      'No additional compute or paid access is needed to reproduce this run. More public, compatible evaluation tables are needed for external validation, not larger models trained on these same test folds.', '',
      '## Provenance and reproduction', '',
      '- Source: [RoboDojo simulation leaderboard](https://robodojo-benchmark.com/leaderboard?bench=sim), pinned website JavaScript snapshot dated 2026-10-24. Exact URLs and SHA-256 checksums are in data/download_audit.json and data/manifest.json; the bundled data never refresh silently.',
      '- Protocol: [RoboDojo paper v3](https://arxiv.org/html/2607.04434v3), including separate single-task-trained entries and simulation training setup. Website and paper versions are distinct; model inclusion uses the pinned website metadata.',
      '- Run `python -m src.sim_prediction` then `python -m src.sim_prediction_report`, or `python -m src.reproduce` for the whole project and tests.',
      '- Design: sim_prediction_plan.json and data/sim_prediction_plan_freeze.json. Inputs: data/processed/sim_prediction_policy_metadata.csv, family_edges.csv (full prefix sim_prediction_), and cohort matrices. Audit: results/sim_prediction_fold_audit.json.',
      '- Full outputs: sim_prediction_predictions.csv.gz, summary.csv, deltas.csv, family_metrics.csv, task_metrics.csv, gap_diagnostics.csv, and anchor_variation.csv under results/ with the sim_prediction_ prefix.',
      '- The earlier PDF remains the original sim-to-real study; this report documents the separate simulation-only pivot.']
    (RESULTS/'sim_prediction_findings.md').write_text('\n'.join(lines)+'\n')
    plot(summary)
    print('Wrote simulation prediction report, diagnostics, and figure 07.')


def plot(summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    methods=['Zero','TaskMean','AnchorMean','Scale','Neighbor3','Latent']
    labels=['Always zero','Task mean','Anchor average','Scalar calibration','3 neighbors','Latent profile']
    colors=['#93999F','#8B9FB0','#007F7A','#2467A3','#C58B32','#8C5BA3']
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    for row,cohort in enumerate(['primary','broad']):
        frame=summary[(summary.cohort==cohort)&(summary.budget==10)].set_index('method').loc[methods]
        for col,metric,title in [(0,'mae','Absolute score error (percentage points)'),(1,'ranking_error','Ranking error on gaps ≥10 points (%)')]:
            ax=axes[row,col];values=frame[metric].to_numpy()*100
            low=frame[metric+'_low'].to_numpy()*100;high=frame[metric+'_high'].to_numpy()*100
            ax.barh(np.arange(len(methods)),values,color=colors,alpha=.92)
            ax.errorbar(values,np.arange(len(methods)),xerr=np.maximum(0,np.vstack([values-low,high-values])),fmt='none',ecolor='#343B42',capsize=2,lw=1)
            for j,(v,h) in enumerate(zip(values,high)):ax.text(h+.025*max(high),j,f'{v:.2f}',va='center',fontsize=9)
            ax.set_yticks(np.arange(len(methods)),labels if col==0 else ['']*len(methods));ax.invert_yaxis()
            ax.set_xlim(0,max(high)*1.24);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            ax.set_title(('Primary: 22 policies / 13 groups' if cohort=='primary' else 'Broader sensitivity: 52 policies / 31 groups')+'\n'+title,loc='left',fontsize=11)
    fig.suptitle('Ten anchor scores predict large-gap rankings; simpler methods remain strong',x=.02,ha='left',fontsize=15,y=.99)
    fig.text(.02,.94,'42 known tasks • 32 scores hidden per policy • whole family groups held out • published simulation results only',fontsize=10)
    fig.text(.02,.015,'Bars: repeated grouped validation. Whiskers: descriptive 95% cluster-bootstrap intervals.\nRanking ties cost 0.5; only 18/42 primary and 34/42 broader tasks have material-gap comparisons. No real-world validation.',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.92))
    for ext in ['png','svg']:fig.savefig(RESULTS/f'figures/07_sim_prediction.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
