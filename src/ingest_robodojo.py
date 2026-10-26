"""Audit website exports, then recover the paper's physical task-score table."""
import argparse,json,re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr,kendalltau
from .common import RAW,PROCESSED,RESULTS,write_json,verify_raw,sha256
from .js_literals import assigned_literal

DOC='https://robodojo-benchmark.com'
REAL_TASKS=[('cap_pen','piper_x'),('classify_objects','piper_x'),('cover_blocks','arx_x5'),
 ('disassemble_LEGO','piper_x'),('fill_pen_holder','piper'),('hang_mugs','piper_x'),
 ('insert_charger','piper'),('insert_tubes','arx_x5'),('make_bread','arx_x5'),('make_food','arx_x5'),
 ('pack_and_pour_fruit','arx_x5'),('pack_objects_into_backpack','piper_x'),('put_objects_into_basket','piper'),
 ('stack_and_cover_blocks','piper'),('stack_bowls','piper'),('stand_up_bottles','piper'),
 ('store_in_safe','arx_x5'),('sweep_blocks','piper_x')]
# Cross-task proxies are explicitly weaker than task-name matches.
PROXIES={'insert_charger':'plug_in_charger','make_bread':'make_toast',
         'pack_objects_into_backpack':'pack_objects_into_box','stack_and_cover_blocks':'cover_blocks'}

def extract():
    verify_raw();text=(RAW/'robodojo_app.js').read_text()
    sim=assigned_literal(text,'wo');real=assigned_literal(text,'vr');aliases=assigned_literal(text,'xe')
    m=re.search(r"\bul=JSON\.parse\('((?:\\.|[^'\\])*)'\)",text)
    if not m:raise ValueError('Pinned bundle data changed: no simulation JSON')
    tasks=json.loads(m.group(1))
    date=assigned_literal(text,'ko')
    return sim,real,aliases,tasks,date

def audit():
    sim,real,aliases,tasks,date=extract()
    manifest=json.loads((RAW/'robodojo_rollout_manifest.json').read_text())
    sim_names={x['model'] for x in sim};real_names={x['model'] for x in real};overlap=sorted(sim_names&real_names)
    rows=[]
    for model,values in tasks.items():
        for task,scores in values.items():
            for metric,k in [('progress','score'),('success_rate','successRate')]:
                rows.append(dict(source='robodojo',source_revision=date,domain='sim',embodiment='arx_x5',
                  policy_id=model,policy_family='unverified',task_id=task,metric=metric,
                  score=scores[k]/100,n_trials=None,match_quality='unmatched',
                  checkpoint_verified=False,source_url=DOC+'/assets/index-DA5jHulR.js'))
    pd.DataFrame(rows).to_csv(PROCESSED/'robodojo_sim_task_scores.csv',index=False)
    # Aggregates retain their official weighting; never substitute a simple
    # unweighted mean over 54 base + random task entries for the headline score.
    aggregated=[]
    for domain,entries in [('sim',sim),('real',real)]:
        for e in entries:
            if domain=='sim':groups={k:v for k,v in e.items() if isinstance(v,dict)}
            else:groups={**{k:v for k,v in e.items() if isinstance(v,dict)},'average':{'score':e['score'],'successRate':e['successRate']}}
            for group,val in groups.items():
                for metric,key in [('progress','score'),('success_rate','successRate')]:
                    aggregated.append(dict(domain=domain,policy_id=e['model'],group=group,metric=metric,score=val[key]/100))
    ag=pd.DataFrame(aggregated);ag.to_csv(PROCESSED/'robodojo_aggregate_scores.csv',index=False)
    correlations=[]
    for metric in ['progress','success_rate']:
        for real_group in ['average','arxX5','piper','piperX']:
            x=ag[(ag.domain=='sim')&(ag.group=='average')&(ag.metric==metric)].set_index('policy_id').score
            y=ag[(ag.domain=='real')&(ag.group==real_group)&(ag.metric==metric)].set_index('policy_id').score
            xy=pd.concat([x.rename('sim'),y.rename('real')],axis=1,join='inner')
            correlations.append(dict(metric=metric,real_group=real_group,n_models=len(xy),
                pearson=float(pearsonr(xy.sim,xy.real).statistic),spearman=float(spearmanr(xy.sim,xy.real).statistic),
                kendall=float(kendalltau(xy.sim,xy.real).statistic),claim='cross-task aggregate association; checkpoint identity unverified'))
            if real_group=='average':xy.to_csv(RESULTS/f'robodojo_aggregate_pairs_{metric}.csv')
    pd.DataFrame(correlations).to_csv(RESULTS/'robodojo_aggregate_correlations.csv',index=False)
    coverage=[]
    for model in overlap:
        alias=aliases[model]
        clips={(x['task'],x['robot']):x for x in manifest.get('real',{}).get(alias,[])}
        for task,robot in REAL_TASKS:
            sim_task=task if task in tasks.get(alias,{}) else PROXIES.get(task)
            name_match=task in tasks.get(alias,{})
            sim_available=sim_task in tasks.get(alias,{}) if sim_task else False
            # No exact outcome match can be certified without physical scores,
            # checkpoint IDs and protocol mapping, even for same-name tasks.
            quality='adapted' if name_match else 'category_only' if sim_available else 'unmatched'
            video=clips.get((task,robot),{})
            coverage.append(dict(policy_id=model,simulation_model_id=alias,checkpoint_identity='unverified',
              real_task=task,real_embodiment=robot,sim_task=sim_task if sim_available else '',sim_embodiment='arx_x5',
              sim_task_score_available=sim_available,real_task_score_available=False,
              same_task_name=name_match,same_embodiment=robot=='arx_x5',match_quality=quality,
              n_real_video_clips=len(video.get('clips',[])),n_trials=None,
              match_note='candidate correspondence only; no verified score/protocol/checkpoint match'))
    cov=pd.DataFrame(coverage);cov.to_csv(RESULTS/'robodojo_website_coverage_matrix.csv',index=False)
    task_map=cov.drop_duplicates(['real_task','real_embodiment'])
    task_map[['real_task','real_embodiment','sim_task','sim_embodiment','same_task_name','same_embodiment','match_quality']].to_csv(RESULTS/'robodojo_task_correspondence.csv',index=False)
    facts=dict(snapshot=date,sim_models=len(sim),sim_per_task_models=len(tasks),real_models=len(real),
      shared_display_names=overlap,n_shared_display_names=len(overlap),verified_checkpoint_matches=0,
      sim_task_variants=len({t for x in tasks.values() for t in x}),sim_task_policy_cells=sum(len(x) for x in tasks.values()),
      real_task_policy_scores_found=0,real_video_models=len(manifest['real']),
      real_video_clips=sum(len(x['clips']) for e in manifest['real'].values() for x in e),
      real_clips_with_outcome=sum('outcome' in c for e in manifest['real'].values() for x in e for c in x['clips']),
      same_name_task_candidates=int(task_map.same_task_name.sum()),
      same_name_and_embodiment_candidates=int((task_map.same_task_name&task_map.same_embodiment).sum()),
      go=False,reason='No task-level real scores in audited public static sources; model checkpoint equality not established',
      bundle_sha256=sha256(RAW/'robodojo_app.js'),rollout_revision=manifest['generatedAt'])
    write_json(RESULTS/'robodojo_website_feasibility.json',facts)
    lines=['# RoboDojo website-only feasibility audit','',
      '**Scope note: see robodojo_feasibility.md for the combined audit. Paper Table 2 supplies physical task scores absent from these website exports.**','',
      f"Snapshot: {date}. Audited the public application bundle, task catalogs, repository tree and rollout metadata.",
      '',f"Found {len(sim)} simulated leaderboard names, {len(real)} physical leaderboard names, and {len(overlap)} shared names. Checkpoint equality is unverified.",
      f"The simulation JSON contains {facts['sim_task_policy_cells']} task-policy cells across {facts['sim_task_variants']} task variants (42 base plus 12 random-layout variants).",
      f"The physical leaderboard exposes overall and three embodiment aggregates. The video manifest has {facts['real_video_clips']} physical clip entries but no outcome fields; clip counts are not evaluation denominators.",
      '', '**Gate: STOP formal external task-level validation.** No matched physical score table was found in the audited sources. This is not a claim that none exists elsewhere.',
      '', 'Seven task-name candidates are retained; cover_blocks and insert_tubes also match the ARX X5 embodiment label. Identical names are not proof of identical initial conditions, controllers or rubrics.',
      'The full model × task × embodiment matrix is in robodojo_coverage_matrix.csv. Adapted means a candidate pairing requiring review, not a validated pairing.',
      '', 'Aggregate correlations are descriptive cross-task associations over shared display names. They cannot validate the held-out-task procedure, and simulated ARX X5 scores are not treated as paired measurements for Piper hardware.',
      '', 'Next data request: checkpoint/revision + task + embodiment + score definition + successes/trials (or graded progress) + protocol and layout version. Start with ARX X5 Cover Blocks and Insert Tubes, then add multiple tasks within each capability family.',
      '', 'Public code has an MIT license; no explicit data/video redistribution license was located in the audited material. Retain attribution and obtain terms before broader redistribution; this bundle records only public metadata and scores. No videos were downloaded.',
      '', 'Sources:', '- '+DOC+'/leaderboard?bench=sim','- '+DOC+'/leaderboard?bench=realworld','- '+DOC+'/data/rolloutManifest.generated.json','- https://github.com/RoboDojo-Benchmark/RoboDojo']
    (RESULTS/'robodojo_website_feasibility.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in facts.items() if k not in ['shared_display_names','bundle_sha256']},indent=2))
    from .ingest_robodojo_paper import main as paper_audit
    return paper_audit()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');p.parse_args();audit()
