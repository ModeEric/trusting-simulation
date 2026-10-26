"""Objective correctness and isolation of held-out labels in the model sweep."""
import json
import unittest
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from src.common import ROOT,PROCESSED,RESULTS,sha256
from src.robotwin_models import factor_fit,factor_predict,pair_data,pair_objective,fit_ranker,predict_ranker,fit_predict_stage
from src.sim_prediction import SEED


class RoboTwinModelTests(unittest.TestCase):
    def test_pairwise_analytic_gradient_and_optimizer(self):
        rng=np.random.default_rng(104);y=rng.uniform(size=(10,9));families=np.repeat(np.arange(5),2);anchors=np.arange(4)
        d,labels,weights=pair_data(y,families,anchors)
        for global_model in [False,True]:
            x=rng.normal(0,.03,size=4*(1 if global_model else 9));args=(d,labels,weights,1.,global_model)
            error=check_grad(lambda z:pair_objective(z,*args)[0],lambda z:pair_objective(z,*args)[1],x)
            self.assertLess(error,1e-4)
            fit=fit_ranker(y,families,anchors,.1,global_model)
            self.assertTrue(fit['status']['success'])
            self.assertLessEqual(fit['status']['objective_after'],fit['status']['objective_before'])
            p=predict_ranker(fit,y[:,anchors])
            self.assertAlmostEqual(pair_objective(x,d,labels,weights,1.,global_model)[0],
                pair_objective(x,-d,-labels,weights,1.,global_model)[0],places=12)
            if global_model:np.testing.assert_allclose(p,p[:,0,None]+np.zeros_like(p))

    def test_factor_capacity_and_probability_bounds(self):
        rng=np.random.default_rng(3);y=rng.uniform(size=(20,30));anchors=np.arange(10)
        y[0,0]=0;y[1,1]=1
        for bounded in [False,True]:
            fit=factor_fit(y,bounded)
            self.assertEqual(fit['basis'].shape,(30,12))
            p=factor_predict(fit,y[:3,anchors],anchors,12,.01)
            self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())
            if bounded:self.assertTrue(((p>0)&(p<1)).all())
        fit=factor_fit(np.ones((3,10))*.5)
        p=factor_predict(fit,np.ones((2,4))*.5,np.arange(4),12,.1)
        np.testing.assert_allclose(p,.5)

    def test_stage_predictions_ignore_held_out_targets(self):
        md=pd.read_csv(PROCESSED/'robotwin_policy_metadata.csv').set_index('policy_id')
        y=pd.read_csv(PROCESSED/'robotwin_matrix_clean.csv',index_col=0)
        old=json.loads((RESULTS/'robotwin_fold_audit.json').read_text())
        a=next(x for x in old if x['cohort']=='clean' and x['budget']==10)
        anchors=y.columns.get_indexer(a['anchor_tasks'])
        def run(stage,table):
            return fit_predict_stage(stage,table.loc[a['train_policies']].to_numpy(),md.loc[a['train_policies'],'family'].to_numpy(),
                table.loc[a['test_policies'],a['anchor_tasks']].to_numpy(),anchors,SEED+500)
        changed=y.copy();changed.loc[a['test_policies'],a['target_tasks']]=1-changed.loc[a['test_policies'],a['target_tasks']]
        for stage in [1,2,3]:
            p,audit=run(stage,y);q,audit2=run(stage,changed)
            self.assertEqual(audit,audit2)
            for method in p:np.testing.assert_array_equal(p[method],q[method])

    def test_small_group_ranking_selection_falls_back_visibly(self):
        rng=np.random.default_rng(2);y=rng.uniform(size=(6,12));families=np.repeat(np.arange(3),2);anchors=np.arange(5)
        p,audit=fit_predict_stage(3,y,families,y[:2,anchors],anchors,SEED)
        for method in ['RankGlobal','RankTask']:
            self.assertFalse(audit['selection_available'][method])
            self.assertEqual(audit['selected'][method]['ridge'],'infinity')
            np.testing.assert_allclose(p[method],np.broadcast_to(y[:2,anchors].mean(axis=1)[:,None],(2,12)))

    def test_frozen_sweep_and_grouped_audit(self):
        for path,digest in json.loads((ROOT/'data/robotwin_model_sweep_freeze.json').read_text()).items():
            self.assertEqual(sha256(ROOT/path),digest)
        old=json.loads((RESULTS/'robotwin_fold_audit.json').read_text())
        reference={tuple(a[k] for k in ['cohort','partition','fold','anchor_draw']):a for a in old if a['budget']==10}
        for stage in [1,2,3]:
            audits=json.loads((RESULTS/f'robotwin_models_stage{stage}_audit.json').read_text())
            self.assertEqual(len(audits),2*3*3*5*3*3*3)
            common={}
            for a in audits:
                train=set(a['train_families']);test=set(a['test_families'])
                self.assertFalse(train&test)
                self.assertFalse(set(a['anchor_tasks'])&set(a['standard_targets']))
                self.assertFalse(set(a['anchor_tasks'])&set(a['common_targets']))
                self.assertEqual(len(a['common_targets']),30)
                self.assertEqual(len(a['anchor_tasks']),a['budget'])
                for inner in a['inner_folds']:
                    it=set(inner['train_families']);iv=set(inner['validation_families'])
                    self.assertFalse(it&iv);self.assertEqual(it|iv,train)
                key=tuple(a[k] for k in ['cohort','partition','fold','anchor_draw','subset_draw','training_fraction'])
                if key in common:self.assertEqual(common[key],a['common_targets'])
                common[key]=a['common_targets']
                if a['budget']==10:
                    ref=reference[tuple(a[k] for k in ['cohort','partition','fold','anchor_draw'])]
                    self.assertEqual(a['test_policies'],ref['test_policies'])
                    self.assertEqual(a['standard_targets'],ref['target_tasks'])

    def test_original_endpoint_and_primary_predictions_reproduce(self):
        summary=pd.read_csv(RESULTS/'robotwin_models_stage1_summary.csv')
        old=pd.read_csv(RESULTS/'robotwin_summary.csv');old=old[old.budget==10].set_index(['cohort','method'])
        selected=summary[(summary.budget==10)&(summary.training_fraction==1)&(summary.target_track=='standard')&summary.method.isin(['OriginalLatent','AnchorMean','Additive','Scale'])]
        for r in selected.itertuples():
            method='Latent' if r.method=='OriginalLatent' else r.method
            self.assertAlmostEqual(r.value,old.loc[(r.cohort,method),r.endpoint],places=12)
        scores=pd.read_csv(PROCESSED/'robotwin_task_scores.csv').set_index(['policy_id','task_id','condition']).score
        for stage in [1,2,3]:
            predictions=pd.read_csv(RESULTS/f'robotwin_models_stage{stage}_primary_predictions.csv.gz')
            self.assertTrue((predictions.budget==10).all());self.assertTrue((predictions.training_fraction==1).all())
            self.assertTrue((predictions.subset_draw==0).all())
            for r in predictions.drop_duplicates(['cohort','policy_id','task_id']).itertuples():
                self.assertAlmostEqual(r.actual,scores.loc[(r.policy_id,r.task_id,r.cohort)])
            if stage!=3:self.assertTrue(predictions.prediction.between(0,1).all())
            else:self.assertTrue((predictions.output_kind=='ranking_utility').all())

    def test_independent_gap_reconstruction_matches_primary_summary(self):
        summary=pd.read_csv(RESULTS/'robotwin_models_summary.csv')
        primary=summary[(summary.training_fraction==1)&(summary.budget==10)&(summary.target_track=='standard')&(summary.endpoint=='ranking_error')].set_index(['cohort','method'])
        gap=pd.read_csv(RESULTS/'robotwin_models_gap_summary.csv')
        for r in gap[gap.gap_group=='material'].itertuples():
            self.assertAlmostEqual(r.error,primary.loc[(r.cohort,r.method),'value'],places=12)

