import json
import unittest
import numpy as np
import pandas as pd
from src.common import ROOT, PROCESSED, RESULTS, sha256
from src.blend_experiment import predict_outer, candidate_predictions, task_loss
from src.metrics import summarize_task

class BlendTests(unittest.TestCase):
    def test_no_outer_label_leakage_and_inner_membership(self):
        data=pd.read_csv(PROCESSED/'simpler_paired.csv')
        for emb,g in data.groupby('embodiment'):
            sim=g[['task_id','policy_id','sim']]
            for task,test in g.groupby('task_id'):
                train=g[g.task_id!=task]
                for objective in ['ranking','mae']:
                    p,w,a,_=predict_outer(train,test.drop(columns='real'),sim,True,objective)
                    changed=g.copy();changed.loc[changed.task_id==task,'real']=1-changed.loc[changed.task_id==task,'real']
                    q,v,b,_=predict_outer(changed[changed.task_id!=task],changed[changed.task_id==task].drop(columns='real'),sim,True,objective)
                    np.testing.assert_allclose(p,q,rtol=0,atol=0);self.assertEqual(w,v)
                    for fold in a['inner_validation']:
                        self.assertNotEqual(task,fold['validation_task'])
                        self.assertNotIn(task,fold['other_training_tasks'])
                        self.assertNotIn(fold['validation_task'],fold['other_training_tasks'])
        with self.assertRaises(ValueError):predict_outer(train,test,sim,True,'ranking')

    def test_vectorized_losses_match_independent_metric(self):
        data=pd.read_csv(PROCESSED/'simpler_paired.csv')
        for _,g in data.groupby('task_id'):
            candidates=np.column_stack([g.sim.to_numpy(),np.ones(len(g))*.5,1-g.sim.to_numpy()])
            expected=[summarize_task(g.assign(prediction=candidates[:,i]))['ranking_error'] for i in range(3)]
            np.testing.assert_allclose(task_loss(g,candidates,'ranking'),expected,equal_nan=True)

    def test_zero_blend_exactly_recovers_simulation(self):
        data=pd.read_csv(PROCESSED/'simpler_paired.csv')
        p=candidate_predictions(data,np.ones(len(data)),[(0,0),(1,1)])
        np.testing.assert_allclose(p[:,0],data.sim,atol=0,rtol=0)
        np.testing.assert_allclose(p[:,1],1)
        self.assertEqual(sha256(ROOT/'blend_plan.json'),json.loads((ROOT/'data/blend_plan_freeze.json').read_text())['sha256'])

    def test_no_ranking_changes_hidden_by_aggregate(self):
        d=pd.read_csv(RESULTS/'blend_changed_pairs.csv')
        self.assertTrue(d.eligible.all())
        self.assertTrue((d.tie_aware_loss!=d.baseline_loss).all())
        self.assertTrue((d[d.effect=='fixed'].tie_aware_loss<d[d.effect=='fixed'].baseline_loss).all())
