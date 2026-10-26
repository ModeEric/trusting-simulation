"""Validate grouping, public score reproduction, and target-label isolation."""
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.ingest_libero_plus import ROOT, RAW, sha, verify
from src.libero_plus_experiment import fit_basis, latent_predict, predict, make_task_split, select_training, tune

class LiberoPlusTests(unittest.TestCase):
    def test_changed_source_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'raw';shutil.copytree(RAW,dest)
            with (dest/'results/MANIFEST.json').open('a') as f:f.write(' ')
            with patch('src.ingest_libero_plus.RAW',dest):
                with self.assertRaises(AssertionError):verify()

    def test_source_scores_and_cohort(self):
        result=json.loads((ROOT/'results/libero_plus_ingestion.json').read_text())
        self.assertEqual(result['passes'],36);self.assertEqual(result['outcome_records'],361080)
        meta=pd.read_csv(ROOT/'data/processed/libero_plus_policies.csv')
        self.assertEqual(len(meta),24);self.assertEqual(meta.weights_sha256.nunique(),24)
        cases=pd.read_csv(ROOT/'data/processed/libero_plus_cases.csv')
        self.assertEqual(cases.base_task.nunique(),40);self.assertEqual(cases.case_id.nunique(),10030)
        self.assertFalse(cases.base_task.str.endswith('_moved').any())

    def test_base_task_holdout_and_nested_anchors(self):
        cases=pd.read_csv(ROOT/'data/processed/libero_plus_cases.csv')
        prev=set();target=None
        for n in [1,2,4,8,16,24]:
            a,t,_=make_task_split(cases,123,n)
            self.assertTrue(prev<=set(a));prev=set(a)
            self.assertFalse(set(cases.base_task.iloc[a])&set(cases.base_task.iloc[t]))
            if target is not None:np.testing.assert_array_equal(target,t)
            target=t
        groups=np.array(['a','a','b','c','d','d','e']);pool=np.arange(7);prev=set()
        for f in [.25,.5,.75,1]:
            train=select_training(pool,groups,f,3)
            self.assertTrue(prev<=set(train));prev=set(train)
            for g in set(groups[train]):self.assertEqual(set(np.flatnonzero(groups==g))&set(train),set(np.flatnonzero(groups==g)))

    def test_predictions_and_tuning_ignore_test_targets(self):
        rng=np.random.default_rng(1);table=rng.binomial(1,.7,(10,60)).astype(float)
        train=np.arange(8);test=np.array([8,9]);a=np.arange(20);t=np.arange(20,60)
        groups=np.repeat(np.arange(4),2);tasks=np.repeat(np.arange(4),10);axes=np.arange(60)%7
        def run(data):
            option,audit=tune(data[train],groups,a,t,tasks,6)
            return predict(data[train],data[test][:,a],a,t,axes,option),audit
        p,audit=run(table);changed=table.copy();changed[np.ix_(test,t)]=1-changed[np.ix_(test,t)]
        q,audit2=run(changed);self.assertEqual(audit,audit2)
        for method in p:
            np.testing.assert_array_equal(p[method],q[method])
            self.assertTrue(np.isfinite(p[method]).all());self.assertTrue(((p[method]>=0)&(p[method]<=1)).all())

    def test_constant_training_matrix_is_stable(self):
        y=np.ones((4,12));fit=fit_basis(y)
        p=latent_predict(fit,np.ones((2,3)),np.arange(3),np.arange(3,12),8,.1)
        np.testing.assert_allclose(p,1)

    def test_complete_frozen_audit(self):
        for path,h in json.loads((ROOT/'data/libero_plus_freeze.json').read_text()).items():
            self.assertEqual(sha((ROOT/path).read_bytes()),h)
        audit=json.loads((ROOT/'results/libero_plus_audit.json').read_text())
        self.assertEqual(len(audit),249)
        for a in audit:
            self.assertFalse(set(a['train_ids'])&set(a['test_ids']))
            self.assertFalse(set(a['train_groups'])&set(a['test_groups']))
            self.assertFalse(set(a['anchor_tasks'])&set(a['target_tasks']))
            self.assertEqual(len(a['target_tasks']),16)
            for inner in a['inner_folds']:
                self.assertFalse(set(inner['train_groups'])&set(inner['validation_groups']))
                self.assertEqual(set(inner['train_groups'])|set(inner['validation_groups']),set(a['train_groups']))

    def test_output_files_have_all_design_rows(self):
        audit=json.loads((ROOT/'results/libero_plus_audit.json').read_text())
        data=pd.read_csv(ROOT/'results/libero_plus_task_metrics.csv.gz')
        expected=sum(6*16*len(a['test_groups']) for a in audit)
        self.assertEqual(len(data),expected)
        ranking=pd.read_csv(ROOT/'results/libero_plus_ranking_metrics.csv.gz')
        self.assertEqual(len(ranking),len(audit)*6*16*7)
        predictions=pd.read_csv(ROOT/'results/libero_plus_predictions.csv.gz')
        expected=sum(6*16*7*len(a['test_ids']) for a in audit if a['fraction']==1 and a['budget']==8 and a['draw']==0)
        self.assertEqual(len(predictions),expected)

if __name__=='__main__':unittest.main()
