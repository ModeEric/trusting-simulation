import unittest,json
import numpy as np
import pandas as pd
from src.common import RAW,PROCESSED,RESULTS,verify_raw,validate_scores
from src.js_literals import Parser

class DataTests(unittest.TestCase):
    def test_frozen_inputs_match_checksums(self):self.assertGreater(len(verify_raw()),10)
    def test_simpler_exact_join_and_no_invented_trials(self):
        s=pd.read_csv(PROCESSED/'simpler_task_scores.csv');p=pd.read_csv(PROCESSED/'simpler_paired.csv')
        self.assertEqual(len(s),84);self.assertEqual(len(p),42);self.assertEqual(p.task_id.nunique(),9)
        self.assertTrue(s.n_trials.isna().all())
        validate_scores(s,['source','domain','embodiment','task_id','policy_id','metric'])
    def test_duplicate_and_scale_guards(self):
        with self.assertRaises(ValueError):validate_scores(pd.DataFrame({'id':[1,1],'score':[.1,.2]}),['id'])
        with self.assertRaises(ValueError):validate_scores(pd.DataFrame({'id':[1],'score':[10]}),['id'])
    def test_robocurve_reproduces_published_cells(self):
        t=pd.read_csv(RAW/'robocurve_trials.csv');c=pd.read_csv(RAW/'robocurve_cells.csv')
        groups=t.groupby(['task','model']).score.agg(['mean','count']).reset_index()
        check=groups.merge(c,on=['task','model'],validate='one_to_one')
        np.testing.assert_allclose(check['mean'],check.mean_score,rtol=0,atol=1e-12)
        self.assertEqual(len(t),360);self.assertTrue((check['count']==20).all())
        self.assertEqual(int(np.isclose(t.score,1).sum()),8)
    def test_episode_links_and_unsupported_external_matches(self):
        e=pd.read_csv(PROCESSED/'robocurve_episodes.csv');self.assertEqual(e.episode_id.nunique(),360)
        self.assertTrue(e.metadata_url.notna().all());self.assertTrue(e.log_url.notna().all())
        cov=pd.read_csv(RESULTS/'robodojo_coverage_matrix.csv')
        self.assertEqual(int(cov.real_task_score_available.sum()),180)
        self.assertFalse((cov.match_quality=='exact').any())
        self.assertFalse(json.loads((RESULTS/'robodojo_feasibility.json').read_text())['go'])
    def test_js_parser_fails_on_code(self):
        with self.assertRaises(ValueError):Parser('fetch("https://example.com")').value()
        self.assertEqual(Parser('{model:"test",average:a(136/300*100,.2)}').value()['average']['score'],136/300*100)

    def test_robocurve_metadata_preserves_csv_task_ids(self):
        meta=pd.read_csv(PROCESSED/'robocurve_task_metadata.csv')
        episodes=pd.read_csv(PROCESSED/'robocurve_episodes.csv')
        self.assertEqual(set(meta.task_id),set(episodes.task_id))
        self.assertEqual(meta.set_index('task_id').loc['Classify Objects','display_name'],'Sort Objects')
