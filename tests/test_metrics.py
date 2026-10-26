import ast,unittest
from typing import Sequence
import numpy as np
from src.common import RAW
from src.ingest_simpler import read_constants
from src.metrics import mmrv,author_pearson,task_pairs

class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wanted={'mean_maximum_rank_violation','pearson_correlation'}
        nodes=[n for n in ast.parse((RAW/'simpler_metrics.py').read_text()).body
               if isinstance(n,ast.FunctionDef) and n.name in wanted]
        env={'np':np,'Sequence':Sequence}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned reference functions>','exec'),env)
        cls.reference=env
    def test_every_published_task_matches_reference(self):
        d=read_constants()
        for task,scores in d['SIMPLER_PERF'].items():
            s=list(scores.values());y=[d['REAL_PERF'][task][p] for p in scores]
            with self.subTest(task=task):
                self.assertAlmostEqual(mmrv(s,y),self.reference['mean_maximum_rank_violation'](s,y),places=14)
                self.assertAlmostEqual(author_pearson(s,y),self.reference['pearson_correlation'](s,y),places=14)
    def test_mmrv_asymmetric_ties_and_argument_order(self):
        self.assertAlmostEqual(mmrv([.5,.5],[0,1]),.5)
        self.assertAlmostEqual(mmrv([0,1],[.5,.5]),0)
        self.assertAlmostEqual(mmrv([1,0],[0,1]),1)
    def test_constant_pearson_matches_author_convention(self):
        self.assertEqual(author_pearson([0,0],[1,1]),1)
        self.assertEqual(author_pearson([0,0],[0,1]),0)
    def test_metrics_randomized_ties(self):
        rng=np.random.default_rng(42)
        for _ in range(100):
            x=rng.integers(0,5,6)/4;y=rng.integers(0,5,6)/4
            self.assertAlmostEqual(mmrv(x,y),self.reference['mean_maximum_rank_violation'](x,y))
    def test_exact_point_one_margin_is_included(self):
        import pandas as pd
        d=pd.DataFrame([dict(task_id='t',embodiment='e',policy_id='a',real=.3,sim=.2,prediction=.2),
                        dict(task_id='t',embodiment='e',policy_id='b',real=.2,sim=.1,prediction=.1)])
        self.assertTrue(task_pairs(d)[0]['eligible'])
