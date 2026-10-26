"""Read-only RoboTwin replication diagnostics, report and figure."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, RESULTS
from .sim_prediction_report import diagnostics


def paired_references():
    rng=np.random.default_rng(20261024);rows=[]
    for endpoint,filename,index in [('mae','family_metrics','family'),('ranking_error','task_metrics','task_id')]:
        frame=pd.read_csv(RESULTS/f'robotwin_{filename}.csv')
        for (cohort,budget),g in frame.groupby(['cohort','budget']):
            wide=g.pivot(index=index,columns='method',values=endpoint)
            for baseline in ['AnchorMean','Scale','Additive']:
                d=(wide.Latent-wide[baseline]).dropna().to_numpy()
                boot=rng.choice(d,size=(2000,len(d)),replace=True).mean(axis=1)
                rows.append(dict(cohort=cohort,budget=budget,endpoint=endpoint,method='Latent',baseline=baseline,
                    delta=d.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),n_clusters=len(d)))
    out=pd.DataFrame(rows);out.to_csv(RESULTS/'robotwin_reference_deltas.csv',index=False)
    return out


def plot(summary):
    methods=['TaskMean','AnchorMean','Additive','Scale','Neighbor3','Latent']
    colors=['#b8c4cc','#7286a0','#95b5ae','#399d8c','#bc9871','#365eb4']
    fig,axes=plt.subplots(2,2,figsize=(12,8.3))
    for row,cohort in enumerate(['clean','randomized']):
        s=summary[(summary.cohort==cohort)&(summary.budget==10)].set_index('method').loc[methods]
        for col,(metric,label) in enumerate([('mae','Success-rate MAE (percentage points)'),('ranking_error','Ranking error (%)')]):
            ax=axes[row,col];values=s[metric].to_numpy()*100
            errors=np.array([values-s[metric+'_low'].to_numpy()*100,s[metric+'_high'].to_numpy()*100-values])
            ax.barh(methods,values,color=colors,height=.66,xerr=errors,error_kw={'ecolor':'#2d3748','capsize':2,'lw':.9})
            ax.invert_yaxis();ax.set_xlabel(label);ax.set_title(cohort.capitalize()+(' scenes' if col==0 else ' scenes · observed gaps ≥10 pp'),loc='left',fontweight='bold')
            ax.set_xlim(0,33 if col==0 else 60)
            for i,value in enumerate(values):ax.text(s.iloc[i][metric+'_high']*100+.5,i,f'{value:.1f}',va='center',fontsize=9)
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
            ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('RoboTwin: 10 observed tasks → predict 40 remaining task scores',x=.03,ha='left',fontsize=16,fontweight='bold')
    fig.text(.03,.925,'25 jointly trained policy configurations · 50 known tasks · whole family groups held out',fontsize=11,color='#455467')
    fig.text(.03,.025,'Same-condition anchors; clean and randomized evaluated separately. Lower is better. Ties cost 0.5.\nWhiskers: descriptive 95% cluster-bootstrap intervals. Ranking covers 48 clean / 50 randomized tasks. No physical validation.',fontsize=9,color='#455467')
    fig.tight_layout(rect=[.02,.08,.99,.9],h_pad=2.5,w_pad=2)
    for ext in ['png','svg']:fig.savefig(RESULTS/f'figures/08_robotwin_prediction.{ext}',dpi=180)
    plt.close(fig)


def main():
    summary=pd.read_csv(RESULTS/'robotwin_summary.csv')
    coverage=json.loads((RESULTS/'robotwin_coverage.json').read_text())
    diag,task_diag,variation=diagnostics('robotwin');deltas=paired_references()
    audit=pd.DataFrame(json.loads((RESULTS/'robotwin_fold_audit.json').read_text()))
    means=pd.read_csv(RESULTS/'robotwin_source_mean_audit.csv')
    plan=json.loads((ROOT/'robotwin_plan.json').read_text())
    lines=['# RoboTwin: predicting simulated performance from a small task panel','',
      '24 October 2026. Independent retrospective research using published evaluation means. No new robot trials, simulator rollouts, robot-policy training/inference, or paid API calls. Not commissioned or endorsed by RoboTwin or Robocurve.','',
      '**Result:** the frozen prediction procedure works on this larger benchmark, but the result is narrower than “a rich model learns reliable task-specific rankings.” With ten anchor scores, the anchor average achieves 82.5% tie-adjusted ranking agreement in clean scenes and 91.5% in randomized scenes for observed gaps of at least ten percentage points. The latent model improves success-rate estimates over scalar calibration, but does not establish a ranking improvement. Close-gap agreement is only about 62–64%.','',
      '## Data and effective sample size','',
      'The pinned official leaderboard contains 25 jointly trained configurations × 50 tasks × 2 conditions = 2,500 published success-rate means. Five single-task SFT entries are excluded from prediction. Clean and randomized are separate evaluation tracks; both use policies trained on clean demonstrations. Two configurations use depth, so the frozen sensitivity excludes them.','',
      '| Cohort | Policies | Group proxies | Tasks | Means | Zero scores |','|---|---:|---:|---:|---:|---:|']
    for c in coverage:lines.append(f"| {c['cohort']} | {c['policies']} | {c['family_groups']} | {c['tasks']} | {c['cells']} | {100*c['zero_fraction']:.2f}% |")
    lines += ['',
      'The RGB cohorts are overlapping sensitivities, not extra data. The 17 groups are conservative ancestry proxies, not 17 verified independent architectures. Two connected groups combine named related methods and documented shared initializations; remaining entries have provisional singleton IDs. Undocumented shared pretraining may remain. The Wan-related proxy deliberately overgroups 4D-WAM, whose clean-to-random variant uses Lingbot-VA, because the same named method also studies FastWAM variants. See robotwin_plan.json and robotwin_policy_metadata.csv for the frozen membership and rationale.','',
      f"The board describes 100 evaluation trials per task. The source contains {int(means.noninteger_percentage_cells.sum())} noninteger percentage cell, so exact per-cell trial denominators are not inferred; n_trials remains blank and the nominal protocol is retained separately. Recomputed means differ from displayed means by at most {means.difference.abs().max():.3f} percentage points, consistent with display rounding. These aggregates do not supply independent episode labels or sampling intervals.",'',
      '## Held-out prediction design','',
      'We reused src.sim_prediction.py unchanged from the RoboDojo experiment: three family-grouped folds, three seeded partitions, five score-independent anchor draws, and nested budgets of 5/10 tasks. Every test policy reveals only those anchor scores; its remaining 45/40 targets are withheld from fitting and hyperparameter selection. Both members of a ranked pair are held out together in the same fold and belong to distinct groups. No comparisons mix predictions across folds.','',
      'The training policies supply scores on all 50 task identities. This therefore predicts additional known-task scores for a policy with a small observed task panel. It does not predict a completely new task, or a model with no observations. Randomized predictions use randomized anchors; they do not forecast randomized performance from clean anchors. Coefficients are refit within RoboTwin training folds: this transports a method, not coefficients learned on RoboDojo.','',
      'The source, grouping plan, predictor implementation and settings were hashed before inspecting RoboTwin predictor results. This is a local audit trail, not an external preregistration. Mutation tests alter held-out target labels and every label in the other condition and verify identical selected hyperparameters and saved predictions.','',
      '## Predictors','',
      'Let A be the anchor set, s_mA its observed scores for policy m, and mu_t the training-policy mean on task t. All predictions are clipped to [0,1].','',
      '- Zero: always zero.',
      '- TaskMean: mu_t, identical across test policies.',
      '- AnchorMean: mean(s_mA), identical across target tasks for a policy.',
      '- Additive: mu_t + mean(s_mA - mu_A).',
      '- Scale: c_m * mu_t, where c_m = (mu_A dot s_mA + 0.01) / (mu_A dot mu_A + 0.01).',
      '- Neighbor3: average full profiles of the three training policies closest on anchor scores.',
      '- Latent: mu_t + v_t dot z_m, where z_m = (V_A.T V_A + lambda I)^(-1) V_A.T (s_mA - mu_A). Training-only centered SVD learns V. Inner grouped CV chooses rank 1/2/4 and ridge 0.01/0.1/1 for MAE.','',
      'Scale for score prediction and AnchorMean for ranking were chosen as references from the earlier RoboDojo experiment. All seven methods are reported; the outer test results did not select predictor settings. Latent is a learned task-profile model, not a hand-coded task-taxonomy calibrator.','',
      '## Absolute results','',
      'MAE averages repeats within policy, policies within group, and groups equally. Ranking error pools disjoint-fold eligible pairs within each design, averages repeated designs within task, then tasks equally. Actual ties and gaps below 0.10 are excluded from the primary ranking endpoint; predicted ties cost 0.5. Agreement is 100% minus this error, not an ordinary pooled accuracy.','',
      '| Anchors | Predictor | Clean MAE (pp) | Clean ranking error | Randomized MAE (pp) | Randomized ranking error |','|---:|---|---:|---:|---:|---:|']
    for budget in [5,10]:
        for method in ['Zero','TaskMean','AnchorMean','Additive','Scale','Neighbor3','Latent']:
            c=summary[(summary.cohort=='clean')&(summary.budget==budget)&(summary.method==method)].iloc[0]
            r=summary[(summary.cohort=='randomized')&(summary.budget==budget)&(summary.method==method)].iloc[0]
            lines.append(f'| {budget} | {method} | {100*c.mae:.2f} | {100*c.ranking_error:.2f}% | {100*r.mae:.2f} | {100*r.ranking_error:.2f}% |')
    lines += ['',
      'Ten-anchor latent MAE is 10.10 pp clean / 11.57 pp randomized, versus 11.46 / 13.53 for Scale. This is a clearer score-error improvement than in the earlier RoboDojo experiment. However, the stronger Additive baseline is already at 10.80 / 11.71 pp; the incremental latent advantage is small, especially in randomized scenes. Neighbor3 has similar clean MAE but worse rankings. More expressive score prediction has not automatically produced better model selection.','',
      '## Paired uncertainty and sensitivity','',
      'The following differences are Latent minus reference; negative favors Latent. Descriptive 95% percentile intervals resample 17 group proxies for MAE or informative tasks for ranking, with 2,000 draws. Repeated folds and anchors are averaged before resampling. These intervals do not capture all shared-training dependencies, unknown ancestry, task selection or noise in published means.','',
      '| Condition | Endpoint | Reference | Difference (pp) | Descriptive 95% interval (pp) |','|---|---|---|---:|---:|']
    for r in deltas[(deltas.budget==10)&deltas.cohort.isin(['clean','randomized'])].itertuples():
        lines.append(f'| {r.cohort} | {r.endpoint} | {r.baseline} | {100*r.delta:+.2f} | [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}] |')
    lines += ['', 'All latent-versus-AnchorMean ranking intervals include zero. We do not claim a ranking improvement.','',
      '| Cohort, ten anchors | AnchorMean agreement | Latent agreement | Scale MAE (pp) | Latent MAE (pp) |','|---|---:|---:|---:|---:|']
    for cohort in ['clean','clean_rgb','randomized','randomized_rgb']:
        s=summary[(summary.cohort==cohort)&(summary.budget==10)].set_index('method')
        lines.append(f"| {cohort} | {100*(1-s.loc['AnchorMean','ranking_error']):.2f}% | {100*(1-s.loc['Latent','ranking_error']):.2f}% | {100*s.loc['Scale','mae']:.2f} | {100*s.loc['Latent','mae']:.2f} |")
    lines += ['', 'Excluding depth policies preserves the central result. It changes group sizes, training/test membership and task comparisons, so the numeric difference is not a causal estimate of depth’s value.','',
      '| Condition, ten anchors | Latent MAE across five anchor sets (pp) | AnchorMean ranking error across sets | Latent rank choices (45 outer fits) |','|---|---:|---:|---|']
    for cohort in ['clean','randomized']:
        v=variation[(variation.cohort==cohort)&(variation.budget==10)]
        m=v[v.method=='Latent'].mae;r=v[v.method=='AnchorMean'].ranking_error
        counts=audit[(audit.cohort==cohort)&(audit.budget==10)].latent_rank.value_counts().sort_index()
        choices=', '.join(f'rank {int(k)}: {int(n)}' for k,n in counts.items())
        lines.append(f'| {cohort} | {100*m.min():.2f}–{100*m.max():.2f} | {100*r.min():.2f}–{100*r.max():.2f}% | {choices} |')
    lines += ['', 'Each anchor draw also changes the held-out task mix, so the across-draw range need not contain the primary task-balanced average. Rank one is selected in most folds. That is consistent with a strong global ability component; it does not establish that richer task mechanisms are absent. Five to ten anchors produces only modest gains on these fixed draws. No learning-curve experiment varying the number of training policies was run, so this does not prove that more data will keep improving prediction.','',
      '## Close gaps, coverage and failure cases','',
      '| Condition | Gap stratum | AnchorMean agreement | Latent agreement | Tasks | Unique eligible pair-task comparisons |','|---|---|---:|---:|---:|---:|']
    for cohort in ['clean','randomized']:
        for kind in ['material','close']:
            g=diag[(diag.cohort==cohort)&(diag.budget==10)&(diag.gap_group==kind)].set_index('method')
            a=g.loc['AnchorMean'];l=g.loc['Latent']
            lines.append(f"| {cohort} | {'≥10 pp' if kind=='material' else '0–10 pp, excluding ties'} | {100*(1-a.ranking_error):.2f}% | {100*(1-l.ranking_error):.2f}% | {int(a.n_tasks)} | {int(a.unique_policy_pair_task_comparisons)} |")
    lines += ['',
      'Unique pair-task counts measure coverage, not independent sample size. They include only pairs co-held-out in at least one design. Actual-gap filtering is a retrospective diagnostic requiring the hidden outcome; it is not a deployable confidence rule. Higher randomized ranking agreement does not mean randomized scores are easier to predict: the policies separate more strongly, and the eligible pair population changes.','',
      'Worst ten-anchor tasks for the prespecified AnchorMean ranking reference (selected descriptively after the run):','',
      '| Condition | Task | AnchorMean ranking error | Latent ranking error | Unique material-gap pairs |','|---|---|---:|---:|---:|']
    td=task_diag[(task_diag.budget==10)&(task_diag.gap_group=='material')]
    for cohort in ['clean','randomized']:
        worst=td[(td.cohort==cohort)&(td.method=='AnchorMean')].sort_values('error',ascending=False).head(5)
        latent=td[(td.cohort==cohort)&(td.method=='Latent')].set_index('task_id')
        for r in worst.itertuples():lines.append(f'| {cohort} | {r.task_id} | {100*r.error:.1f}% | {100*latent.loc[r.task_id,"error"]:.1f}% | {int(r.unique_pairs)} |')
    lines += ['',
      '## What to improve next','',
      'The most useful next experiment is to target residual ranking reversals while retaining this global-ability baseline. Predeclare a pairwise ranking objective and a small residual model, choose its complexity using inner grouped folds, and compare paired outer-fold errors against AnchorMean. Any model designed after seeing these failures is exploratory on RoboTwin and needs an additional untouched dataset or later model cohort for confirmation.','',
      'A second experiment can choose the anchor panel using training policies only, then compare it with the same random panels. Task families and near-duplicate tasks should be grouped for any new-task generalization claim. A training-family learning curve is needed to test the user’s “more data helps” thesis; 2,500 means still represent only 25 configurations, 50 task identities, and two related conditions.','',
      '## Boundaries of the result','',
      '- No physical outcomes are validated here. Accurate simulation prediction still needs a separate matched sim-to-real test before it can guide real-robot selection.',
      '- This is meta-predictor holdout, not a claim that the original robot policies were unexposed to benchmark tasks during training.',
      '- Original policy training recipes, pretraining, modalities and reported evaluation procedures differ. Public leaderboard entries can contain benchmark-selection and publication bias.',
      '- Group proxies and task bootstraps do not eliminate correlated models, related tasks or shared evaluation infrastructure.',
      '- Published means are treated as fixed targets; raw evaluation episodes and exact denominators are unavailable here. Close-gap rankings may partly reflect measurement noise.',
      '- The earlier RoboDojo result had far more zero outcomes. Its small MAE is not directly comparable with RoboTwin’s denser success distribution.',
      '- Neither benchmark establishes that simulation is a reliable physical proxy for these particular policy comparisons.','',
      '## Reproduce and inspect','',
      'From the bundled repository, run `python -m src.reproduce` for all studies and tests, or `python -m src.robotwin_prediction` then `python -m src.robotwin_report` for this replication. Reproduction is offline and verifies pinned hashes.','',
      'Outputs: `robotwin_summary.csv`, `robotwin_reference_deltas.csv`, `robotwin_predictions.csv.gz`, `robotwin_fold_audit.json`, `robotwin_gap_diagnostics.csv`, `robotwin_gap_task_metrics.csv`, `robotwin_anchor_variation.csv`, and `figures/08_robotwin_prediction.png` under results/. Normalized source tables and cohort matrices are under data/processed/. The original SIMPLER PDF is a separate earlier report.','',
      'Sources:', '',
      '- Official leaderboard and protocol: https://robotwin-platform.github.io/leaderboard',
      '- Pinned raw data: https://github.com/RoboTwin-Platform/RoboTwin-Platform.github.io/blob/'+plan['source_revision']+'/data/robotwin_leaderboard.json',
      '- PrismWAM ancestry evidence: https://arxiv.org/html/2610.26564v1',
      '- 4D-WAM variant evidence: https://arxiv.org/html/2608.08023v1',
      '- Inherited grouping evidence: pinned RoboDojo website and sim_prediction_family_edges.csv in this repository.', '',
      'Access date: 24 October 2026 (America/Los_Angeles). Source revision, URLs, sizes and SHA-256 checksums are retained in data/download_audit.json and data/manifest.json. No videos or trajectory archives were downloaded.','']
    (RESULTS/'robotwin_findings.md').write_text('\n'.join(lines));plot(summary)
    print('Wrote RoboTwin diagnostics, research note and figure.')


if __name__=='__main__':main()
