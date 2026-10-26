import unittest
import numpy as np
import pandas as pd
from src.common import PROCESSED,RESULTS
from src.evaluate import grouped_predictions,selective_predictions
from src.baselines import predict_fold

class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=pd.read_csv(PROCESSED/'simpler_paired.csv')
    def test_test_task_mutation_cannot_change_its_predictions(self):
        # Mutate all target physical outcomes; every target-fold prediction,
        # fitted coefficient and selective confidence must remain unchanged.
        emb=self.data[self.data.embodiment=='google_robot'].copy()
        target=sorted(emb.task_id.unique())[0]
        original,audit=grouped_predictions(emb)
        changed=emb.copy();changed.loc[changed.task_id==target,'real']=np.linspace(.97,.03,(changed.task_id==target).sum())
        new,other_audit=grouped_predictions(changed)
        cols=['task_id','policy_id','method']
        o=original[original.task_id==target].sort_values(cols).prediction
        n=new[new.task_id==target].sort_values(cols).prediction
        np.testing.assert_allclose(o,n,atol=0,rtol=0)
        self.assertEqual([x for x in audit if x['held_out_group']==target],
                         [x for x in other_audit if x['held_out_group']==target])
        a,_,_=selective_predictions(emb);b,_,_=selective_predictions(changed)
        np.testing.assert_allclose(a[a.task_id==target].confidence,b[b.task_id==target].confidence,atol=0,rtol=0)
    def test_fold_memberships_are_disjoint(self):
        import json
        for filename in ['fold_audit.json','family_fold_audit.json']:
            for f in json.loads((RESULTS/filename).read_text()):
                self.assertFalse(set(f['train_tasks'])&set(f['test_tasks']))
    def test_reject_label_carrying_sim_table(self):
        d=self.data[self.data.embodiment=='google_robot'];task=d.task_id.iloc[0]
        with self.assertRaises(ValueError):predict_fold(d[d.task_id!=task],d[d.task_id==task],d,'B0')
    def test_no_embodiment_pooling(self):
        _,audit=grouped_predictions(self.data)
        for f in audit:
            expected=self.data[self.data.embodiment==f['embodiment']]
            self.assertTrue(set(f['train_tasks']).issubset(expected.task_id))
    def test_positive_affine_calibration_preserves_rank(self):
        import json
        audit=json.loads((RESULTS/'fold_audit.json').read_text())
        self.assertTrue(all(f['coefficients'][1]>0 for f in audit if f['coefficients'] is not None))
        pred=pd.read_csv(RESULTS/'simpler_predictions.csv')
        for (_,method),g in pred.groupby(['task_id','method']):
            if method in ['B2','B2_ridge','C1']:
                np.testing.assert_array_equal(np.sign(g.sim.to_numpy()[:,None]-g.sim.to_numpy()[None,:]),
                                              np.sign(g.prediction.to_numpy()[:,None]-g.prediction.to_numpy()[None,:]))
