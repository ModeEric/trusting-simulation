"""Meaningful guards against row-span errors and falsely paired checkpoints."""
import unittest
import pandas as pd
from src.common import PROCESSED, RESULTS, RAW
from src.ingest_robodojo_paper import TASK_ORDER, TABLE_IDS, score_pair
from src.paper_tables import read_tables


class RoboDojoPaperTests(unittest.TestCase):
    def test_tasks_use_published_embodiment_order(self):
        d=pd.read_csv(PROCESSED/'robodojo_paper_real_task_scores.csv')
        self.assertEqual(len(d),360)
        self.assertEqual(d[d.metric=='success_rate'].groupby('method_id').size().tolist(),[18]*10)
        for robot,tasks in [('arx_x5',TASK_ORDER['ARX X5']),('piper',TASK_ORDER['Piper']),('piper_x',TASK_ORDER['Piper X'])]:
            x=d[(d.method_id=='Pi-05')&(d.embodiment==robot)&(d.metric=='progress')].sort_values('task_column')
            self.assertEqual(x.task_id.tolist(),tasks)
        lookup=d[d.metric=='progress'].set_index(['method_id','task_id']).score
        self.assertAlmostEqual(lookup.loc['Pi-05','pack_and_pour_fruit'],.47)
        self.assertAlmostEqual(lookup.loc['Pi-05','stack_bowls'],.72)
        self.assertAlmostEqual(lookup.loc['InternVLA-A1','store_in_safe'],.48)

    def test_domains_never_share_checkpoint_identifiers(self):
        real=pd.read_csv(PROCESSED/'robodojo_paper_real_task_scores.csv')
        sim=pd.read_csv(PROCESSED/'robodojo_paper_sim_categories.csv')
        self.assertFalse(set(real.policy_id)&set(sim.policy_id))
        self.assertTrue((real.match_quality=='unmatched').all())
        self.assertTrue((real.n_trials==10).all())
        self.assertEqual(len(set(real.method_id)&set(sim.method_id)),10)
        self.assertEqual(real.source_revision.unique().tolist(),sim.source_revision.unique().tolist())

    def test_physical_aggregates_and_source_revisions(self):
        c=pd.read_csv(RESULTS/'robodojo_paper_aggregate_checks.csv')
        self.assertEqual(len(c),80)
        self.assertTrue(c.passes_rounding_tolerance.all())
        self.assertLessEqual(c.absolute_difference.max(),.001)
        a=read_tables(RAW/'continuation/robodojo_paper_v1.html')
        b=read_tables(RAW/'continuation/robodojo_paper_v3.html')
        for table in [TABLE_IDS[0],TABLE_IDS[2],TABLE_IDS[3]]:
            self.assertEqual(a[table],b[table])
        changes=pd.read_csv(RESULTS/'robodojo_paper_version_changes.csv')
        self.assertEqual(len(changes),2)
        self.assertEqual(changes.table_id.unique().tolist(),['S5.T2.6.1'])
        real=pd.read_csv(PROCESSED/'robodojo_paper_real_task_scores.csv')
        cell=real[(real.method_id=='Spirit v1.5')&(real.task_id=='cap_pen')&(real.metric=='progress')]
        self.assertAlmostEqual(cell.score.iloc[0],.04)

    def test_empty_ranking_comparisons_are_not_zero_error(self):
        d=pd.read_csv(RESULTS/'robodojo_paper_proxy_task_metrics.csv')
        empty=d[d.n_material==0]
        self.assertGreater(len(empty),0)
        self.assertTrue(empty.ranking_error.isna().all())
        self.assertTrue((d.n_wrong<=d.n_material).all())
        self.assertTrue((d.n_pairs==45).all())

    def test_repeat_sd_is_not_raw_trial_data(self):
        d=pd.read_csv(PROCESSED/'robodojo_paper_repeated_run_sd.csv')
        self.assertEqual(len(d),114)
        self.assertFalse(d.raw_round_scores_available.any())
        self.assertTrue((d.n_runs==3).all())
        v=d[(d.method_id=='Pi-05')&(d.task_id=='store_in_safe')&(d.metric=='success_rate')].across_run_sd.iloc[0]
        self.assertAlmostEqual(v,.231)
        cost=pd.read_csv(PROCESSED/'robodojo_paper_evaluation_cost.csv')
        self.assertEqual(len(cost),18);self.assertAlmostEqual(cost.total_minutes.sum(),202)
        self.assertNotIn('connect_charger',set(cost.task_id))

    def test_malformed_numeric_cells_fail(self):
        for got,expected in zip(score_pair('24.6 / 20.0'),(.246,.2)):
            self.assertAlmostEqual(got,expected)
        with self.assertRaises(ValueError):score_pair('24.6 / missing')
        with self.assertRaises(ValueError):score_pair('101 / 0')
