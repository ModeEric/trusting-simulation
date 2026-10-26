"""Source, grouped validation and label-isolation checks on actual RoboTwin data."""
import json
import unittest
import numpy as np
import pandas as pd
from src.common import ROOT, RAW, PROCESSED, RESULTS
from src.robotwin_prediction import verify_plan
from src.sim_prediction import fit_training, predict, tune_latent, SEED


class RoboTwinTests(unittest.TestCase):
    def test_pinned_source_normalization_and_cohorts(self):
        verify_plan()
        source=json.loads((RAW/'robotwin_leaderboard.json').read_text())
        scores=pd.read_csv(PROCESSED/'robotwin_task_scores.csv')
        self.assertEqual(len(scores),3000)
        self.assertFalse(scores.duplicated(['policy_id','task_id','condition']).any())
        self.assertTrue(scores.n_trials.isna().all())
        lookup=scores.set_index(['policy_id','task_id','condition']).score
        for entry in source['entries']:
            for condition,key in [('clean','easy'),('randomized','hard')]:
                for task,value in zip(source['tasks'],entry[key]):
                    self.assertAlmostEqual(lookup.loc[(entry['id'],task,condition)],value/100)
        md=pd.read_csv(PROCESSED/'robotwin_policy_metadata.csv').set_index('policy_id')
        for cohort in ['clean','randomized','clean_rgb','randomized_rgb']:
            y=pd.read_csv(PROCESSED/f'robotwin_matrix_{cohort}.csv',index_col=0)
            self.assertEqual(y.shape,(23 if cohort.endswith('_rgb') else 25,50))
            self.assertTrue(np.isfinite(y).all().all())
            self.assertTrue(((y>=0)&(y<=1)).all().all())
            self.assertTrue((md.loc[y.index,'setting']=='cotrain').all())
            if cohort.endswith('_rgb'):self.assertFalse(md.loc[y.index,'depth'].any())
        self.assertEqual(md.loc['cotrain:PrismWAM','family'],md.loc['cotrain:PrismWAM_Star','family'])
        self.assertEqual(md.loc['cotrain:PrismWAM','family'],md.loc['cotrain:FastWAM','family'])

    def test_outer_inner_and_anchor_separation(self):
        audits=json.loads((RESULTS/'robotwin_fold_audit.json').read_text())
        self.assertEqual(len(audits),4*3*3*5*2)
        membership={}
        for a in audits:
            train=set(a['train_families']);test=set(a['test_families'])
            self.assertFalse(train & test)
            self.assertFalse(set(a['train_policies']) & set(a['test_policies']))
            self.assertFalse(set(a['anchor_tasks']) & set(a['target_tasks']))
            self.assertEqual(len(a['anchor_tasks']),a['budget'])
            self.assertEqual(len(a['target_tasks']),50-a['budget'])
            for inner in a['inner_folds']:
                it=set(inner['train_families']);iv=set(inner['validation_families'])
                self.assertFalse(it & iv);self.assertEqual(it | iv,train)
            key=(a['cohort'],a['partition'],a['anchor_draw'],a['budget'])
            membership.setdefault(key,[]).extend(a['test_policies'])
        for key,ids in membership.items():
            self.assertEqual(len(ids),23 if key[0].endswith('_rgb') else 25)
            self.assertEqual(len(ids),len(set(ids)))

    def test_actual_predictions_ignore_target_and_other_condition_labels(self):
        audits=json.loads((RESULTS/'robotwin_fold_audit.json').read_text())
        scores=pd.read_csv(PROCESSED/'robotwin_task_scores.csv')
        md=pd.read_csv(PROCESSED/'robotwin_policy_metadata.csv').set_index('policy_id')
        saved=pd.read_csv(RESULTS/'robotwin_predictions.csv.gz')
        for condition in ['clean','randomized']:
            a=next(x for x in audits if x['cohort']==condition and x['budget']==10)
            def run(table):
                y=table[table.condition==condition].pivot(index='policy_id',columns='task_id',values='score').sort_index(axis=1)
                anchors=y.columns.get_indexer(a['anchor_tasks'])
                train=y.loc[a['train_policies']].to_numpy()
                option,inner=tune_latent(train,md.loc[a['train_policies'],'family'].to_numpy(),anchors,SEED+500)
                p=predict(fit_training(train),y.loc[a['test_policies']].to_numpy()[:,anchors],anchors,option)
                return p,option,inner,y.columns
            p,opt,inner,columns=run(scores)
            mutated=scores.copy()
            forbidden=(mutated.condition!=condition)|((mutated.policy_id.isin(a['test_policies']))&(mutated.task_id.isin(a['target_tasks'])))
            mutated.loc[forbidden,'score']=1-mutated.loc[forbidden,'score']
            q,opt2,inner2,_=run(mutated)
            self.assertEqual(opt,opt2);self.assertEqual(inner,inner2)
            self.assertEqual(opt,(a['latent_rank'],a['latent_ridge']))
            for method in p:
                np.testing.assert_array_equal(p[method],q[method])
                g=saved[(saved.cohort==condition)&(saved.partition==0)&(saved.fold==0)&(saved.anchor_draw==0)&(saved.budget==10)&(saved.method==method)]
                actual=g.pivot(index='policy_id',columns='task_id',values='prediction').loc[a['test_policies'],a['target_tasks']]
                np.testing.assert_allclose(actual,p[method][:,columns.get_indexer(a['target_tasks'])],atol=1e-14)
            mutated=scores.copy()
            allowed=mutated.policy_id.isin(a['test_policies'])&mutated.task_id.isin(a['anchor_tasks'])&(mutated.condition==condition)
            mutated.loc[allowed,'score']=0
            q,_,_,_=run(mutated)
            self.assertGreater(np.max(np.abs(p['AnchorMean']-q['AnchorMean'])),.1)

    def test_ranking_counts_and_diagnostics(self):
        audits=json.loads((RESULTS/'robotwin_fold_audit.json').read_text())
        rows=pd.read_csv(RESULTS/'robotwin_ranking_metrics.csv')
        md=pd.read_csv(PROCESSED/'robotwin_policy_metadata.csv').set_index('policy_id')
        for cohort in ['clean','randomized','clean_rgb','randomized_rgb']:
            matrix=pd.read_csv(PROCESSED/f'robotwin_matrix_{cohort}.csv',index_col=0)
            for a in [x for x in audits if x['cohort']==cohort and x['partition']==0 and x['anchor_draw']==0 and x['budget']==10]:
                ids=a['test_policies']
                chosen=rows[(rows.cohort==cohort)&(rows.partition==0)&(rows.anchor_draw==0)&(rows.budget==10)&(rows.fold==a['fold'])&(rows.method=='TaskMean')].set_index('task_id')
                for task in a['target_tasks']:
                    eligible=sum(md.loc[x,'family']!=md.loc[z,'family'] and abs(matrix.loc[x,task]-matrix.loc[z,task])>=.10-1e-12 for j,x in enumerate(ids) for z in ids[j+1:])
                    self.assertEqual(chosen.loc[task,'n_eligible'],eligible)
                    self.assertEqual(chosen.loc[task,'loss_sum'],eligible*.5)
        summary=pd.read_csv(RESULTS/'robotwin_summary.csv').set_index(['cohort','budget','method'])
        diag=pd.read_csv(RESULTS/'robotwin_gap_diagnostics.csv')
        for r in diag[diag.gap_group=='material'].itertuples():
            self.assertAlmostEqual(r.ranking_error,summary.loc[(r.cohort,r.budget,r.method),'ranking_error'],places=12)
            self.assertEqual(r.n_tasks,summary.loc[(r.cohort,r.budget,r.method),'n_informative_tasks'])

