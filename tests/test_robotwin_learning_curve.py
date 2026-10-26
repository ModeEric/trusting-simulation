"""Check the controlled data-size ablation and its unchanged full-data endpoint."""
import json
import unittest
import numpy as np
import pandas as pd
from src.common import ROOT,PROCESSED,RESULTS,sha256
from src.robotwin_learning_curve import fit_subset


class RoboTwinLearningCurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audits=json.loads((RESULTS/'robotwin_learning_curve_audit.json').read_text())

    def test_frozen_inputs_and_nested_training_subsets(self):
        for path,digest in json.loads((ROOT/'data/robotwin_learning_curve_freeze.json').read_text()).items():
            self.assertEqual(sha256(ROOT/path),digest)
        groups={}
        for a in self.audits:
            train=set(a['train_families']);test=set(a['test_families']);pool=set(a['available_train_families'])
            self.assertFalse(train & test);self.assertTrue(train<=pool)
            self.assertEqual(a['n_train_cells'],50*len(a['train_policies']))
            self.assertEqual(a['n_train_groups'],len(train))
            self.assertFalse(set(a['anchor_tasks']) & set(a['target_tasks']))
            self.assertEqual(len(a['anchor_tasks']),10);self.assertEqual(len(a['target_tasks']),40)
            for inner in a['inner_folds']:
                it=set(inner['train_families']);iv=set(inner['validation_families'])
                self.assertFalse(it & iv);self.assertEqual(it|iv,train)
            key=tuple(a[k] for k in ['cohort','partition','fold','anchor_draw','subset_draw'])
            groups.setdefault(key,{})[a['training_fraction']]=train
            if a['training_fraction']==1:self.assertEqual(train,pool)
        self.assertEqual(len(self.audits),2*3*3*5*5*5)
        for g in groups.values():
            values=[g[f] for f in sorted(g)]
            self.assertEqual(len(values),5)
            for x,y in zip(values,values[1:]):self.assertTrue(x<=y)

    def test_test_policies_anchors_targets_are_fixed_from_original_experiment(self):
        original=json.loads((RESULTS/'robotwin_fold_audit.json').read_text())
        reference={tuple(a[k] for k in ['cohort','partition','fold','anchor_draw']):a for a in original if a['budget']==10}
        for a in self.audits:
            ref=reference[tuple(a[k] for k in ['cohort','partition','fold','anchor_draw'])]
            for field in ['test_policies','test_families','anchor_tasks','target_tasks']:
                self.assertEqual(a[field],ref[field])
            self.assertTrue(set(a['train_policies'])<=set(ref['train_policies']))

    def test_full_data_endpoint_reproduces_original_results(self):
        summary=pd.read_csv(RESULTS/'robotwin_learning_curve_summary.csv')
        old=pd.read_csv(RESULTS/'robotwin_summary.csv')
        old=old[old.budget==10].set_index(['cohort','method'])
        for r in summary[summary.training_fraction==1].itertuples():
            self.assertAlmostEqual(r.value,old.loc[(r.cohort,r.method),r.endpoint],places=12)
            self.assertAlmostEqual(r.training_cells_mean,50*25*2/3,places=9)

    def test_anchor_baseline_and_pair_eligibility_do_not_change_with_data_budget(self):
        summary=pd.read_csv(RESULTS/'robotwin_learning_curve_summary.csv')
        for _,g in summary[summary.method=='AnchorMean'].groupby(['cohort','endpoint']):
            self.assertLess(g.value.max()-g.value.min(),1e-13)
        rows=pd.read_csv(RESULTS/'robotwin_learning_curve_ranking_metrics.csv.gz')
        groups=rows[rows.method=='Latent'].groupby(['cohort','partition','fold','anchor_draw','subset_draw','task_id']).n_eligible
        self.assertTrue((groups.max()==groups.min()).all())

    def test_subset_fit_ignores_unused_training_rows_and_hidden_targets(self):
        meta=pd.read_csv(PROCESSED/'robotwin_policy_metadata.csv').set_index('policy_id')
        for cohort in ['clean','randomized']:
            a=next(x for x in self.audits if x['cohort']==cohort and x['training_fraction']==.2)
            y=pd.read_csv(PROCESSED/f'robotwin_matrix_{cohort}.csv',index_col=0)
            args=(meta,a['train_policies'],a['test_policies'],a['anchor_tasks'],a['partition'])
            p,opt,inner=fit_subset(y,*args)
            changed=y.copy()
            unused=changed.index.difference(a['train_policies']+a['test_policies'])
            changed.loc[unused]=1-changed.loc[unused]
            changed.loc[a['test_policies'],a['target_tasks']]=1-changed.loc[a['test_policies'],a['target_tasks']]
            q,opt2,inner2=fit_subset(changed,*args)
            self.assertEqual(opt,opt2);self.assertEqual(inner,inner2)
            for method in p:np.testing.assert_array_equal(p[method],q[method])

