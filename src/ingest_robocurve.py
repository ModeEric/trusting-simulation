"""Physical-only analysis; bootstrap unpaired trials within fixed task cells."""
import json,re
from html import unescape
from urllib.parse import urljoin
import numpy as np
import pandas as pd
from .common import RAW,PROCESSED,RESULTS,SEED,write_json,validate_scores,verify_raw
BASE='https://robocurve.org/opus-5-5-robodojo-rc-tier-1/'
ASTRA='gpt-6-astra';OPUS='claude-opus-5-5'

def wilson(k,n,z=1.959963984540054):
    p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,mid-half),min(1,mid+half)

def run_links():
    links={}
    for attrs,body in re.findall(r'<tr\b([^>]*)>(.*?)</tr>',(RAW/'robocurve_report.html').read_text(),re.S):
        m=re.search(r'\bid="run-([^\"]+)"',attrs)
        if not m:continue
        rid=re.search(r'(adhoc_[a-z0-9]+)',m.group(1)).group(1)
        hrefs=[urljoin(BASE,unescape(x)) for x in re.findall(r'href="([^\"]+)"',body)]
        links[rid]=dict(log_url=next((x for x in hrefs if '/log/' in x),''),
            video_url=';'.join(x for x in hrefs if '.mp4' in x),
            metadata_url=next((x for x in hrefs if x.endswith('.json')),''),
            rig_id=re.sub(r'_adhoc_.*$','',m.group(1)))
    return links

def ingest():
    verify_raw();trials=pd.read_csv(RAW/'robocurve_trials.csv');sourcecells=pd.read_csv(RAW/'robocurve_cells.csv')
    if len(trials)!=360 or trials.run_id.duplicated().any():raise ValueError('Unexpected or duplicate run inventory')
    if not trials.score.between(0,1).all():raise ValueError('Invalid scores')
    if not (trials.groupby(['model','task']).size()==20).all():raise ValueError('Unexpected cell sizes')
    # status=success means the run infrastructure succeeded, not task completion.
    trials['completed']=np.isclose(trials.score,1.0)
    links=run_links()
    if set(trials.run_id)!=set(links):raise ValueError('Run link inventory mismatch')
    episodes=[]
    for row in trials.itertuples():
        episodes.append(dict(source='robocurve',domain='real',policy_id=row.model,task_id=row.task,
             episode_id=row.run_id,metric='progress',score=row.score,started_at=row.started_at,**links[row.run_id]))
    episodes=pd.DataFrame(episodes);episodes.to_csv(PROCESSED/'robocurve_episodes.csv',index=False)
    rng=np.random.default_rng(SEED);rows=[];boot={};completion_boot={}
    for (task,model),g in trials.groupby(['task','model'],sort=True):
        v=g.score.to_numpy();samples=rng.choice(v,size=(20000,len(v)),replace=True)
        means=samples.mean(axis=1);boot[task,model]=means
        completion_boot[task,model]=np.isclose(samples,1).mean(axis=1)
        lo,hi=np.quantile(means,[.025,.975]);clo,chi=wilson(int(g.completed.sum()),len(g))
        rows.append(dict(task_id=task,policy_id=model,n_trials=len(g),mean_progress=v.mean(),
             progress_ci_low=lo,progress_ci_high=hi,completed=int(g.completed.sum()),
             completion_rate=g.completed.mean(),completion_ci_low=clo,completion_ci_high=chi))
    cells=pd.DataFrame(rows)
    check=cells.merge(sourcecells,left_on=['task_id','policy_id'],right_on=['task','model'],validate='one_to_one',suffixes=('','_source'))
    if not np.allclose(check.mean_progress,check.mean_score,atol=1e-12):raise ValueError('Published cell means not reproduced')
    if not np.array_equal(check.completed,check.completed_source):raise ValueError('Published completion counts not reproduced')
    cells.to_csv(RESULTS/'robocurve_task_results.csv',index=False)
    scores=[]
    for r in cells.itertuples():
        for metric,value in [('progress',r.mean_progress),('success_rate',r.completion_rate)]:
            scores.append(dict(source='robocurve',source_revision='report-2026-09-23',domain='real',
              embodiment='bimanual_i2rt_yam',policy_id=r.policy_id,policy_family='claude-opus' if r.policy_id.startswith('claude') else 'gpt-6',
              task_id=r.task_id,metric=metric,score=value,n_trials=r.n_trials,
              protocol_id='inspect-robots-0.58.0-medium-40calls-900steps-speed25pct',
              matched_task_id='',match_quality='unmatched',source_url=BASE+'data/trials.csv'))
    pd.DataFrame(scores).to_csv(PROCESSED/'robocurve_task_scores.csv',index=False)
    differences=[]
    tasks=sorted(trials.task.unique());indexed=cells.set_index(['task_id','policy_id'])
    for task in tasks:
        a=indexed.loc[(task,ASTRA)];b=indexed.loc[(task,OPUS)]
        difference=boot[task,ASTRA]-boot[task,OPUS];lo,hi=np.quantile(difference,[.025,.975])
        differences.append(dict(task_id=task,astra_mean=a.mean_progress,opus55_mean=b.mean_progress,
          astra_minus_opus55=a.mean_progress-b.mean_progress,ci_low=lo,ci_high=hi,
          astra_completed=int(a.completed),opus55_completed=int(b.completed),n_each=20,
          completion_difference=a.completion_rate-b.completion_rate))
    dif=pd.DataFrame(differences);dif.to_csv(RESULTS/'robocurve_differences.csv',index=False)
    overall=[]
    for model,g in trials.groupby('model',sort=True):
        combined=np.mean([boot[task,model] for task in tasks],axis=0);lo,hi=np.quantile(combined,[.025,.975])
        overall.append(dict(policy_id=model,mean_progress=g.score.mean(),ci_low=lo,ci_high=hi,
                            n_trials=len(g),completed=int(g.completed.sum()),completion_rate=g.completed.mean()))
    pd.DataFrame(overall).to_csv(RESULTS/'robocurve_overall.csv',index=False)
    dboot=np.mean([boot[t,ASTRA]-boot[t,OPUS] for t in tasks],axis=0)
    write_json(RESULTS/'robocurve_aggregate_difference.json',dict(astra_minus_opus55=float(dif.astra_minus_opus55.mean()),
      ci_low=float(np.quantile(dboot,.025)),ci_high=float(np.quantile(dboot,.975)),bootstrap_replicates=20000,
      conditioning='six selected tasks fixed; independent resampling within model-task cells, not paired trials',
      uncertainty_excludes='task selection, rater bias, rig/time clustering, layout and protocol variation'))
    mix=[]
    for omit in [None,*tasks]:
        keep=trials if omit is None else trials[trials.task!=omit]
        means=keep.groupby('model').score.mean()
        mix.append(dict(omitted_task='none' if omit is None else omit,astra=means[ASTRA],opus55=means[OPUS],
                        astra_minus_opus55=means[ASTRA]-means[OPUS],leader=ASTRA if means[ASTRA]>means[OPUS] else OPUS))
    pd.DataFrame(mix).to_csv(RESULTS/'robocurve_task_mix.csv',index=False)
    # This is a descriptive sensitivity, not a randomized task-population model.
    weights=rng.dirichlet(np.ones(len(tasks)),size=20000)
    deltas=weights@dif.set_index('task_id').loc[tasks].astra_minus_opus55.to_numpy()
    write_json(RESULTS/'robocurve_task_weight_sensitivity.json',dict(weighting='illustrative Dirichlet(1) across the fixed six tasks',
        fraction_weights_favor_astra=float((deltas>0).mean()),not_a_probability_of_model_superiority=True))
    descriptions={
      'Pack And Pour Fruit':(0,0,1,'Sequential loading and pouring of rigid fruit props'),
      'Stack Bowls':(0,1,1,'Stable placement of multiple bowls; partial overlap earns progress'),
      'Store In Safe':(1,0,1,'Load three objects and close an articulated lid'),
      'Cap Pen':(0,1,1,'Bimanual grasp and alignment; contact can earn partial credit without alignment'),
      'Stand Up Bottles':(0,1,1,'Reorient three bottles into stable upright poses'),
      'Classify Objects':(0,0,1,'Classify and place objects in two bins')}
    meta=[]
    for t,(a,p,m,d) in descriptions.items():meta.append(dict(source='robocurve',task_id=t,embodiment='bimanual_i2rt_yam',official_category='Tier 1',grasp=1,placement=1,articulation=a,precision_contact=p,multistep=m,deformable=0,display_name='Sort Objects' if t=='Classify Objects' else t,description=d,taxonomy_confidence='rubric-based, descriptive only'))
    pd.DataFrame(meta).to_csv(PROCESSED/'robocurve_task_metadata.csv',index=False)
    print(dif[['task_id','astra_minus_opus55','ci_low','ci_high']].to_string(index=False))
    print('All published cell means and completion counts reproduced exactly.')
if __name__=='__main__':ingest()
