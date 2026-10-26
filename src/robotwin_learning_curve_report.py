"""Render the measured training-data learning curves without extrapolation."""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import RESULTS


def plot(summary):
    colors={'Latent':'#2c5fbd','Additive':'#c47b31','Scale':'#278c80','AnchorMean':'#687687'}
    labels={'Latent':'Latent model','Additive':'Additive baseline','Scale':'Scalar baseline','AnchorMean':'Anchor average (no training)'}
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.6))
    for row,cohort in enumerate(['clean','randomized']):
        for col,(endpoint,label,limits) in enumerate([('mae','Score MAE (percentage points)',(8,22)),('ranking_error','Ranking error (%) · gaps ≥10 points',(0,35))]):
            ax=axes[row,col]
            for method in ['AnchorMean','Scale','Additive','Latent']:
                s=summary[(summary.cohort==cohort)&(summary.method==method)&(summary.endpoint==endpoint)].sort_values('training_fraction')
                x=s.training_cells_mean.to_numpy();y=100*s.value.to_numpy()
                ax.plot(x,y,label=labels[method],color=colors[method],lw=2.3 if method=='Latent' else 1.7,
                        marker='o' if method=='Latent' else None,ms=5,ls='--' if method=='AnchorMean' else '-')
                if method=='Latent':
                    ax.fill_between(x,100*s.low,100*s.high,color=colors[method],alpha=.12,lw=0)
                    ax.annotate(f'{y[-1]:.1f}',(x[-1],y[-1]),xytext=(7,-9 if endpoint=='ranking_error' else -13),
                                textcoords='offset points',color=colors[method],fontsize=10,fontweight='bold')
            ax.set_title(cohort.capitalize()+' scenes',loc='left',fontweight='bold',fontsize=12)
            ax.set_xlabel('Training evaluation scores (mean per fit)');ax.set_ylabel(label)
            ax.set_ylim(*limits);ax.set_xlim(140,900);ax.set_xticks(x,[str(round(v)) for v in x])
            ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    handles,labels_=axes[0,0].get_legend_handles_labels()
    fig.suptitle('RoboTwin learning curves: how much does more training data help?',x=.035,ha='left',fontsize=16,fontweight='bold')
    fig.text(.035,.931,'Fixed held-out policies and 10 anchor tasks · 50 known tasks · same-condition prediction',fontsize=11,color='#465468')
    order=[3,2,1,0]
    fig.legend([handles[i] for i in order],[labels_[i] for i in order],loc='upper left',bbox_to_anchor=(.026,.91),
               ncol=4,frameon=False,fontsize=10,handlelength=2.4,columnspacing=1.6)
    fig.text(.035,.035,'Lower is better. Each point averages nested group subsets; training size grows from ~3.7 to ~16.7 policies per fit.\nBlue shading: descriptive 95% cluster-bootstrap interval for Latent. Scores are published evaluation means, not robot demonstrations.\nRanking ties cost 0.5; physical transfer and extrapolation beyond the observed data range are not tested.',fontsize=9,color='#465468')
    fig.tight_layout(rect=[.02,.115,.99,.86],h_pad=2,w_pad=2)
    for ext in ['png','svg']:fig.savefig(RESULTS/f'figures/09_robotwin_learning_curve.{ext}',dpi=180)
    plt.close(fig)


def main():
    summary=pd.read_csv(RESULTS/'robotwin_learning_curve_summary.csv')
    counts=pd.read_csv(RESULTS/'robotwin_learning_curve_training_counts.csv')
    deltas=pd.read_csv(RESULTS/'robotwin_learning_curve_deltas.csv')
    plot(summary)
    lines=['# RoboTwin learning curves: scaling the score predictor’s training data','',
      '24 October 2026. Exploratory follow-up using the same pinned public evaluations. No new robot trials, simulation rollouts, robot-policy inference/training, or paid API calls.','',
      '**Finding:** more evaluated training policies improve the latent predictor over the measured range. Mean training size rises from 183 to 833 policy-task score cells per fit. Latent score MAE falls from 13.89 to 10.10 percentage points in clean scenes and from 17.87 to 11.57 in randomized scenes. Its ranking error falls too, but ends approximately level with the much simpler anchor-average ranking baseline. Gains diminish over the observed budgets; no scaling law or extrapolation is established.','',
      '## What “training data” means','',
      'One cell is one published success-rate mean for a policy on a task. It is not one robot episode or training demonstration. We increase the number of evaluated policy groups supplied to the meta-predictor; the robot policies themselves are fixed. Every selected training policy contributes scores for all 50 known tasks in one condition. Ten revealed scores for each held-out policy are fixed and are excluded from the horizontal-axis count.','',
      'Clean and randomized are fit separately. The full-data point uses only 16–17 of the 25 policies in each fold because the remaining policies are held out. It therefore uses about 833 training cells, not all 2,500 source cells. The two conditions share task and policy identities and are not independent replications.','',
      '## Controlled design','',
      'The original three family-grouped folds × three seeded partitions and five ten-task anchor draws are unchanged. Within each outer training pool, we sample five score-independent random group orders. At each budget, we retain the first ceil(fraction × available groups), with at least two groups. The subsets are nested: a larger budget adds groups and never replaces the smaller set. All configurations of a chosen group are retained. Test policies, anchors, target tasks and eligible ranking pairs are fixed across budgets.','',
      'The latent rank/ridge selection is rerun using inner grouped CV within each selected subset. The original seven predictors are unchanged. Model selection never sees hidden outer-test labels or omitted training rows. The learning-curve plan and code were frozen before curve results were inspected; prior full-data results were already known. This is exploratory analysis, not external preregistration.','',
      '| Fraction of available groups | Mean groups | Mean policies | Mean training cells | Training-cell range |','|---:|---:|---:|---:|---:|']
    for r in counts[counts.cohort=='clean'].itertuples():
        lines.append(f'| {100*r.training_fraction:.0f}% | {r.training_groups_mean:.2f} | {r.training_policies_mean:.2f} | {r.training_cells_mean:.2f} | {r.training_cells_min}–{r.training_cells_max} |')
    lines += ['',
      'These counts are identical across conditions. Group sizes differ, so a fraction of groups is not an exact fraction of policies or cells. The chart uses mean cell counts to show actual data volume; the ranges above preserve that variation. Group proxies, not cells, are the meaningful unit of policy diversity.','',
      '## Latent-model results','',
      '| Mean training cells | Clean MAE (pp) | Randomized MAE (pp) | Clean ranking agreement | Randomized ranking agreement |','|---:|---:|---:|---:|---:|']
    for fraction in sorted(summary.training_fraction.unique()):
        s=summary[(summary.training_fraction==fraction)&(summary.method=='Latent')].set_index(['cohort','endpoint'])
        lines.append(f"| {s.training_cells_mean.iloc[0]:.0f} | {100*s.loc[('clean','mae'),'value']:.2f} | {100*s.loc[('randomized','mae'),'value']:.2f} | {100*(1-s.loc[('clean','ranking_error'),'value']):.2f}% | {100*(1-s.loc[('randomized','ranking_error'),'value']):.2f}% |")
    lines += ['',
      'MAE gives policy groups equal weight after averaging within policy and group. Ranking error pools disjoint folds within each design, averages repeats within task, then tasks equally. Only observed gaps ≥0.10 enter ranking; predicted ties cost 0.5. This retrospective margin filter does not provide a deployable confidence rule. There are 17 group proxies for MAE, 48 informative clean tasks and 50 randomized tasks for ranking. Repeated subsets, folds and pairs are not independent samples.','',
      'The plotted blue bands resample the same group/task clusters jointly across budgets (2,000 draws). They are descriptive intervals conditional on these task and policy cohorts. They do not fully model shared training sets, uncertain ancestry, evaluation noise or publication selection.','',
      '| Condition | Endpoint | Full minus smallest budget (pp) | Paired descriptive 95% interval |','|---|---|---:|---:|']
    for r in deltas[deltas.method=='Latent'].itertuples():
        lines.append(f'| {r.cohort} | {r.endpoint} | {100*r.delta_full_minus_small:+.2f} | [{100*r.low:+.2f}, {100*r.high:+.2f}] |')
    lines += ['',
      '## Baselines change the interpretation','',
      'The anchor average has no training-policy parameters: its ranking agreement stays exactly at 82.54% clean / 91.51% randomized at every budget. It uses only the fixed ten observed scores of each held-out policy. The latent model starts substantially worse at 72.51% / 87.03%, and reaches 82.13% / 91.52%. More data helps the latent model recover the baseline’s strong global-ability signal; it has not established a better ranking rule.','',
      'The additive baseline is also data-efficient. Its clean MAE goes from 12.03 to 10.80 pp, and randomized MAE from 12.82 to 11.71 pp. Latent overtakes it only with larger budgets, and the final randomized score advantage remains small. The scalar baseline ends at 11.46 / 13.53 pp. All seven methods and intervals are retained in robotwin_learning_curve_summary.csv; the figure shows four prespecified reference methods.','',
      '## What is supported','',
      'Within this fixed task catalog, adding evaluated policy groups improves the latent predictor’s held-out score and ranking estimates. The result supports collecting broader policy coverage for this meta-prediction problem. It does not show that arbitrary additional cells, repetitions of the same policy, or more robot-training demonstrations yield the same gain. The finite range cannot determine whether errors will continue to fall, saturate, or reverse on substantially different policy families.','',
      'No physical outcomes or completely unseen tasks are tested. Most groups remain provisional ancestry proxies. The clean/randomized performance difference is not a causal effect of randomization: their distributions and eligible policy gaps differ.','',
      '## Reproduce','',
      'Run `python -m src.robotwin_learning_curve` followed by `python -m src.robotwin_learning_curve_report`, then `python -m unittest discover -s tests -v`. The full `python -m src.reproduce` command also includes these stages. No network access is needed.','',
      'The plan, audit and output names use the robotwin_learning_curve prefix. The audit records every selected training group, unused pool, test policy, anchor, inner CV candidate and selected setting. Tests verify nested subsets, fixed test comparisons, invariance to forbidden labels, a constant anchor-only baseline, and exact reproduction of the previous full-data endpoint.','',
      'Source: the pinned official RoboTwin leaderboard at https://github.com/RoboTwin-Platform/RoboTwin-Platform.github.io/blob/0abb5a03455073fa74047a1c4015b0f88cedc254/data/robotwin_leaderboard.json . See robotwin_findings.md and the data manifest for provenance and cohort details.','']
    (RESULTS/'robotwin_learning_curve_findings.md').write_text('\n'.join(lines))
    print('Wrote measured learning-curve figure and report.')


if __name__=='__main__':main()
