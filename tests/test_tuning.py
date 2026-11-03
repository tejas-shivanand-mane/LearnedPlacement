import contextlib,copy,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from learnedplacement.runner import ROOT,read_json
from learnedplacement.expanded import make_scenarios
from learnedplacement.expanded_planners import digest
from learnedplacement.learning_runner import run as old_run
from learnedplacement.guided_runner import run as evaluate,method_specs,load_source
from learnedplacement.tuning_runner import (run,prepare,train_job,summary,freeze_selection,
    comparison_source,evaluation_config,export,validate_config)
import test_learning


def config():
    c=test_learning.config();c.update(experiment='tuning_v7',seed=c['seed']+2,instances_per_regime=2,instance_splits=['train','validation'])
    c['tuning']=dict(checkpoints=[2,4],validation_planning_traces=2,validation_traces=2,top_k=2,save_every_episodes=1,
                     selection_metric='objective',training_pool_versions=2,variants=[dict(name='baseline',hidden=4,learning_rate=.001,entropy=.01),dict(name='wide',hidden=6,learning_rate=.001,entropy=.01)])
    return c


class TuningTests(unittest.TestCase):
    def test_default_balanced_cases_and_no_test_inputs(self):
        c=read_json(ROOT/'configs/pilot_v7.json');validate_config(c)
        with tempfile.TemporaryDirectory() as tmp:
            m=prepare(c,Path(tmp));train=read_json(Path(tmp)/'train_scenarios.json');val=read_json(Path(tmp)/'validation_scenarios.json')
            self.assertEqual(len(train),384);self.assertEqual(len(val),24)
            self.assertEqual({v['split'] for v in train},{'train'});self.assertEqual({v['split'] for v in val},{'validation'})
            self.assertEqual({len(v['workload']['nodes']) for v in train},{6,12,24,32,64,96})
            self.assertFalse(set(m['training_cases'])&set(m['validation_cases']))

    def test_checkpoint_validation_resume_does_not_update_model(self):
        import learnedplacement.tuning_runner as module
        from learnedplacement.simulator import BudgetExpired
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=config();a=root/'a';b=root/'b';prepare(c,a);prepare(c,b)
            original=module.simulate;calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==7:raise BudgetExpired()
                return original(*args,**kwargs)
            with patch.object(module,'simulate',side_effect=interrupt):
                with self.assertRaises(BudgetExpired):train_job(a,'baseline',11)
            self.assertEqual(read_json(a/'jobs/baseline-seed-11/state.json')['episode'],2)
            cp=read_json(a/'jobs/baseline-seed-11/checkpoints/2.json')
            train_job(a,'baseline',11);train_job(b,'baseline',11)
            self.assertEqual(cp,read_json(a/'jobs/baseline-seed-11/checkpoints/2.json'))
            aa=read_json(a/'jobs/baseline-seed-11/state.json');bb=read_json(b/'jobs/baseline-seed-11/state.json')
            for key in ('episode','actor','regressor','baselines','history','simulations','return_sum'):self.assertEqual(aa[key],bb[key])
            self.assertEqual(read_json(a/'jobs/baseline-seed-11/validation/4/summary.json'),read_json(b/'jobs/baseline-seed-11/validation/4/summary.json'))

    def test_selected_shared_setting_and_frozen_comparison(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);old=root/'v5';old_run(test_learning.config(),old,max_minutes=None)
            out=root/'tune';c=config();self.assertEqual(run(c,out,workers=1)['state'],'finished')
            selected=read_json(out/'selection.json');groups=selected['groups'];winner=min(groups,key=lambda g:(g['objective'],g['completion_time'],g['episode'],g['variant']))
            self.assertEqual(selected['selection'],winner);self.assertEqual(set(selected['model_hashes']),{'11','29'})
            frozen=read_json(out/'selected_policies.json')['actors']
            self.assertEqual(run(c,out,workers=1)['state'],'finished');self.assertEqual(read_json(out/'selected_policies.json')['actors'],frozen)
            source=comparison_source(out,old,root/'source');cfg=evaluation_config(c,source)
            self.assertEqual(set(cfg['learning']['seeds']),{11,29,1011,1029})
            result=evaluate(cfg,root/'eval',source,workers=1);self.assertEqual(result['state'],'finished')
            em=read_json(root/'eval/manifest.json');cid=em['cases'][0]
            plans=read_json(root/'eval/partitions'/cid/'cases'/cid/'plans.json')
            self.assertEqual(plans['heuristic_top2']['simulations'],4)
            self.assertEqual(plans['heuristic_top2']['candidate_pool_sha256'],plans['rl_1011_top2']['candidate_pool_sha256'])
            with zipfile.ZipFile(export(out)) as z:bundle=json.loads(z.read('analysis.json'))
            self.assertNotIn('"actor":',json.dumps(bundle));self.assertEqual(bundle['tuning']['state'],'finished')
            p=source/'comparison_policies.json';data=read_json(p);data['actors']['11']['params']['a'][0]+=1;p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'bundle changed'):load_source(source,cfg)

    def test_serial_parallel_same_models_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);c=config();a=root/'a';b=root/'b'
            run(c,a,workers=1);run(c,b,workers=2)
            self.assertEqual(read_json(a/'selection.json'),read_json(b/'selection.json'))
            self.assertEqual(read_json(a/'selected_policies.json')['actors'],read_json(b/'selected_policies.json')['actors'])

    def test_reject_test_split_and_mutated_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);c=config();prepare(c,out)
            p=out/'train_scenarios.json';raw=read_json(p);raw[0]['split']='test';p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'TRAIN'):train_job(out,'baseline',11)
            with self.assertRaisesRegex(ValueError,'Modified'):prepare(c,out)
            c['instance_splits']=['train','test']
            with self.assertRaisesRegex(ValueError,'ONLY'):validate_config(c)

    def test_heuristic_ranker_runs_no_event_simulation(self):
        from learnedplacement.ranking import heuristic_score
        from learnedplacement.learning import action_pool
        c=config();s=make_scenarios(c)[0];pool=action_pool(s,c)
        with patch('learnedplacement.expanded.ExpandedSimulation.run',side_effect=AssertionError('No event loop allowed')):
            scores=[heuristic_score(s,p,c['learning']['penalties']) for p in pool]
        self.assertEqual(len(scores),len(pool))

    def test_cached_feature_update_matches_original_math(self):
        from learnedplacement.learning import new_training_state,train_episode,action_pool,matrix,features
        from learnedplacement.tuning_runner import update_from_features
        c=config();s=make_scenarios(c)[0];pool=action_pool(s,c);x=matrix(s,pool)
        state=new_training_state(len(features(s,pool[0])),c,11)
        self.assertEqual(train_episode(s,pool,state,c,2.5,1),update_from_features(s,state,c,2.5,1,x))
