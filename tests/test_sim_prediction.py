import json
import unittest
import numpy as np
import pandas as pd
from src.common import ROOT, RESULTS, PROCESSED, sha256
from src.sim_prediction import folds, fit_training, predict, tune_latent


class SimulationPredictionTests(unittest.TestCase):
    def test_frozen_plan_and_protocol_filters(self):
        for name, digest in json.loads((ROOT/'data/sim_prediction_plan_freeze.json').read_text()).items():
            self.assertEqual(sha256(ROOT/name), digest)
        meta=pd.read_csv(PROCESSED/'sim_prediction_policy_metadata.csv')
        for flag in ['primary_included','broad_included']:
            selected=meta[meta[flag]]
            self.assertFalse(selected.policy_type.str.startswith('agent').any())
            self.assertFalse(selected.display_name.str.contains('Single Task').any())
        self.assertTrue(meta[meta.primary_included].documented_training.all())
        for cohort in ['primary','broad']:
            y=pd.read_csv(PROCESSED/f'sim_prediction_matrix_{cohort}.csv',index_col=0)
            self.assertEqual(y.shape[1],42)
            self.assertFalse(any(t.endswith('_random') for t in y.columns))
            self.assertTrue(np.isfinite(y.to_numpy()).all())
            self.assertTrue(((y>=0)&(y<=1)).all().all())

    def test_nested_folds_and_anchor_target_separation(self):
        audits=json.loads((RESULTS/'sim_prediction_fold_audit.json').read_text())
        for a in audits:
            train=set(a['train_families']);test=set(a['test_families'])
            self.assertFalse(train & test)
            self.assertFalse(set(a['train_policies']) & set(a['test_policies']))
            self.assertFalse(set(a['anchor_tasks']) & set(a['target_tasks']))
            self.assertEqual(len(a['anchor_tasks']),a['budget'])
            self.assertEqual(len(a['target_tasks']),42-a['budget'])
            for inner in a['inner_folds']:
                it=set(inner['train_families']);iv=set(inner['validation_families'])
                self.assertFalse(it & iv)
                self.assertEqual(it | iv,train)
                self.assertFalse((it | iv) & test)
        membership={}
        for a in audits:
            key=(a['cohort'],a['partition'],a['anchor_draw'],a['budget'])
            membership.setdefault(key,[]).extend(a['test_policies'])
        for ids in membership.values():self.assertEqual(len(ids),len(set(ids)))

    def test_actual_pipeline_is_immune_to_test_target_mutation(self):
        rng=np.random.default_rng(24)
        y=rng.uniform(0,1,(18,12));families=np.repeat(np.arange(9),2)
        test=folds(families,13)[0];train=np.setdiff1d(np.arange(len(y)),test)
        anchors=np.array([1,3,5,7]);targets=np.setdiff1d(np.arange(12),anchors)
        def run(table):
            option,audit=tune_latent(table[train],families[train],anchors,49)
            return predict(fit_training(table[train]),table[test][:,anchors],anchors,option),option,audit
        p,option,audit=run(y)
        changed=y.copy();changed[np.ix_(test,targets)]=1-y[np.ix_(test,targets)]
        q,new_option,new_audit=run(changed)
        self.assertEqual(option,new_option);self.assertEqual(audit,new_audit)
        for method in p:np.testing.assert_array_equal(p[method],q[method])
        # Ensure the invariance is meaningful: permitted inputs do affect outputs.
        changed=y.copy();changed[np.ix_(test,anchors)]=0
        q,_,_=run(changed)
        self.assertGreater(np.max(np.abs(p['AnchorMean']-q['AnchorMean'])),.1)
        changed=y.copy();changed[np.ix_(train,targets)]=0
        q,_,_=run(changed)
        self.assertGreater(np.max(np.abs(p['TaskMean']-q['TaskMean'])),.1)

    def test_neighbor_and_scale_against_independent_scalar_computation(self):
        y=np.random.default_rng(3).uniform(0,1,(8,11));anchors=np.array([8,1,7,3,5])
        x=np.random.default_rng(4).uniform(0,1,(3,5));p=predict(fit_training(y),x,anchors)
        for i in range(len(x)):
            distance=[sum((x[i,j]-y[k,t])**2 for j,t in enumerate(anchors))/len(anchors) for k in range(len(y))]
            nearest=sorted(range(len(y)),key=lambda k:distance[k])[:3]
            np.testing.assert_allclose(p['Neighbor3'][i],y[nearest].mean(axis=0))
            mean=y.mean(axis=0)
            scale=(sum(x[i,j]*mean[t] for j,t in enumerate(anchors))+.01)/(sum(mean[t]**2 for t in anchors)+.01)
            np.testing.assert_allclose(p['Scale'][i],np.clip(scale*mean,0,1))
        for prediction in p.values():
            self.assertEqual(prediction.shape,(3,11))
            self.assertTrue(((prediction>=0)&(prediction<=1)).all())

    def test_saved_ranking_counts_use_only_jointly_held_out_policies(self):
        audits=json.loads((RESULTS/'sim_prediction_fold_audit.json').read_text())
        rows=pd.read_csv(RESULTS/'sim_prediction_ranking_metrics.csv')
        for cohort in ['primary','broad']:
            matrix=pd.read_csv(PROCESSED/f'sim_prediction_matrix_{cohort}.csv',index_col=0)
            meta=pd.read_csv(PROCESSED/'sim_prediction_policy_metadata.csv').set_index('policy_id')
            # Independently count all pairs for a complete design, not cross-fold predictions.
            for a in [v for v in audits if v['cohort']==cohort and v['partition']==0 and v['anchor_draw']==0 and v['budget']==10]:
                ids=a['test_policies']
                selected=rows[(rows.cohort==cohort)&(rows.partition==0)&(rows.anchor_draw==0)&(rows.budget==10)&(rows.fold==a['fold'])&(rows.method=='TaskMean')].set_index('task_id')
                for task in a['target_tasks']:
                    eligible=sum(meta.loc[x,'family']!=meta.loc[z,'family'] and abs(matrix.loc[x,task]-matrix.loc[z,task])>=.10-1e-12 for j,x in enumerate(ids) for z in ids[j+1:])
                    self.assertEqual(selected.loc[task,'n_eligible'],eligible)
                    self.assertEqual(selected.loc[task,'loss_sum'],eligible*.5)

    def test_reconstructed_pair_diagnostics_match_primary_metrics(self):
        reference=pd.read_csv(RESULTS/'sim_prediction_summary.csv').set_index(['cohort','budget','method'])
        diagnostics=pd.read_csv(RESULTS/'sim_prediction_gap_diagnostics.csv')
        for row in diagnostics[diagnostics.gap_group=='material'].itertuples():
            self.assertAlmostEqual(row.ranking_error,reference.loc[(row.cohort,row.budget,row.method),'ranking_error'],places=12)
            self.assertEqual(row.n_tasks,reference.loc[(row.cohort,row.budget,row.method),'n_informative_tasks'])

