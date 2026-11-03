from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from learnedplacement.expanded import make_scenarios,generate_trace,violations,Decision
from learnedplacement.expanded_planners import digest
from learnedplacement.learning import Network,features,matrix,action_pool,objective,new_training_state,train_episode
from learnedplacement.learning_runner import run,prepare,Experiment,methods,validate_config
from learnedplacement.runner import ROOT,read_json,export_run
from learnedplacement.simulator import BudgetExpired
import test_expanded


def config():
    c=read_json(ROOT/'configs/pilot_v5.json')
    c.update(active_nodes=[4],regimes=['homogeneous'],instances_per_regime=3,instance_splits=['train','validation','test'],
             planning_traces=2,evaluation_traces=2,search_candidates=2,tuned_candidates=2,save_every_traces=1)
    c['expanded_profiles']=c['expanded_profiles'][:1]
    c['workload'].update(steps=2,checkpoint_every=1,backup_every=1,base_step_s=.1)
    c['choices'].update(checkpoint_every=[1,2],backup_every=[1,2])
    c['learning'].update(seeds=[11,29],episodes=3,rollouts_per_episode=2,action_count=3,hidden=4,history_every=1)
    c['methods']=methods(c)
    return c


class LearningMathTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        net=Network(3,4,1);x=np.array([[.1,.2,.4],[.4,.3,.2]])
        g=np.array([.7,-.2]); grads=net.gradients(x,g)
        for key in net.params:
            for idx in np.ndindex(net.params[key].shape):
                original=net.params[key][idx];eps=1e-6
                net.params[key][idx]=original+eps;a=float(net.forward(x)[0]@g)
                net.params[key][idx]=original-eps;b=float(net.forward(x)[0]@g)
                net.params[key][idx]=original
                self.assertAlmostEqual(grads[key][idx],(a-b)/(2*eps),places=6)

    def test_reinforce_increases_rewarded_action_probability(self):
        net=Network(2,4,1);x=np.array([[0.,1.],[1.,0.]])
        before=net.probabilities(x)[1]
        for _ in range(30):net.policy_update(x,1,1.,.01,0.)
        self.assertGreater(net.probabilities(x)[1],before+.1)

    def test_entropy_gradient_matches_objective(self):
        net=Network(2,4,8);x=np.array([[0.,1.],[1.,0.],[1.,1.]])
        p=net.probabilities(x);action=1;adv=.7;entropy=.2
        g=p.copy();g[action]-=1;g*=adv
        logp=np.log(p);g+=entropy*p*(logp-float(p@logp))
        analytic=net.gradients(x,g)
        def loss():
            p=net.probabilities(x);return -adv*np.log(p[action])+entropy*sum(p*np.log(p))
        for idx in np.ndindex(net.params['v'].shape):
            a=net.params['v'][idx];eps=1e-6
            net.params['v'][idx]=a+eps;left=loss()
            net.params['v'][idx]=a-eps;right=loss()
            net.params['v'][idx]=a
            self.assertAlmostEqual(analytic['v'][idx],(left-right)/(2*eps),places=6)

    def test_supervised_control_learns_reward_direction(self):
        net=Network(2,4,2);x=np.array([[0.,1.]])
        before=net.forward(x)[0][0]
        for _ in range(30):net.regression_update(x,2.,.01)
        self.assertGreater(net.forward(x)[0][0],before)

    def test_serialized_optimizer_resume_is_identical(self):
        net=Network(2,4,1);x=np.array([[0.,1.],[1.,0.]])
        net.policy_update(x,1,.5,.01,.01)
        restored=Network.from_dict(json.loads(json.dumps(net.to_dict())))
        net.policy_update(x,0,-.7,.01,.01);restored.policy_update(x,0,-.7,.01,.01)
        self.assertEqual(net.to_dict(),restored.to_dict())

    def test_shared_objective_penalizes_censoring_and_constraints(self):
        s,p=test_expanded.fixture();weights=dict(cost=20.,deadline=20.,censor=100.)
        a=dict(restricted_mean_s=10.,mean_observed_cost_usd=.5,deadline_success=1.,completed_fraction=1.)
        self.assertGreater(objective(s.workload,dict(a,completed_fraction=.5),weights),objective(s.workload,a,weights))
        self.assertGreater(objective(s.workload,dict(a,mean_observed_cost_usd=2.),weights),objective(s.workload,a,weights))
        self.assertGreater(objective(s.workload,dict(a,deadline_success=.5),weights),objective(s.workload,a,weights))

    def test_features_do_not_use_identity_split_or_future_events(self):
        s,p=test_expanded.fixture();a=features(s,p)
        s.split='test';s.workload.case_id='other';s.workload.trace_key='other'
        np.testing.assert_array_equal(a,features(s,p))
        self.assertTrue(np.all(np.isfinite(a)))


class LearningPipelineTests(unittest.TestCase):
    def test_pools_are_feasible_and_reproducible(self):
        c=config();s=make_scenarios(c)[0]
        a=action_pool(s,c);self.assertEqual(a,action_pool(s,c))
        self.assertTrue(all(not violations(s,p) for p in a))

    def test_longer_horizon_preserves_trace_prefix(self):
        s,p=test_expanded.fixture();s.workload.domain_failure_per_hour=1000.
        old=generate_trace(s,42);long=replace(s,workload=replace(s.workload,max_simulated_s=240.))
        new=generate_trace(long,42)
        self.assertEqual([e for e in old if e.time_s<60.],[e for e in new if e.time_s<60.])

    def test_training_rejects_validation_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);c=config();prepare(c,out)
            p=out/'learning/train_scenarios.json';data=read_json(p);data[0]['split']='test';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'non-train'):Experiment().train(c,out,None)

    def test_interrupted_episode_is_replayed_exactly(self):
        import learnedplacement.learning_runner as module
        with tempfile.TemporaryDirectory() as aa,tempfile.TemporaryDirectory() as bb:
            c=config();a=Path(aa);b=Path(bb);prepare(c,a);prepare(c,b)
            original=module.simulate;calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==4:raise BudgetExpired()
                return original(*args,**kwargs)
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(module,'simulate',side_effect=interrupt):
                    with self.assertRaises(BudgetExpired):Experiment().train(c,a,None)
                self.assertEqual(read_json(a/'learning/seed-11.json')['episode'],1)
                Experiment().train(c,a,None);Experiment().train(c,b,None)
            for seed in c['learning']['seeds']:
                x=read_json(a/f'learning/seed-{seed}.json');y=read_json(b/f'learning/seed-{seed}.json')
                for key in ('actor','regressor','episode','baselines','return_sum','simulations'):self.assertEqual(x[key],y[key])

    def test_tiny_end_to_end_frozen_models_matched_pool_and_export(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp);c=config();r=run(c,out,max_minutes=1)
            self.assertEqual(r['state'],'finished')
            manifest=read_json(out/'manifest.json')
            self.assertEqual(len(manifest['cases']),2)
            self.assertEqual(len(manifest['training_cases']),1)
            models=[read_json(out/f'learning/seed-{seed}.json')['actor'] for seed in c['learning']['seeds']]
            for cid in manifest['cases']:
                plans=read_json(out/'cases'/cid/'plans.json')
                common=['random_search','random_unaware','rl_seed_11','rl_seed_29','supervised_control']
                self.assertEqual(len({plans[m]['candidate_pool_sha256'] for m in common}),1)
                for m in common[2:]:self.assertEqual(plans[m]['simulations'],0)
                self.assertEqual(plans['local_search']['parent'],'heuristic_portfolio')
            run(c,out,max_minutes=1)
            self.assertEqual(models,[read_json(out/f'learning/seed-{seed}.json')['actor'] for seed in c['learning']['seeds']])
            import zipfile
            with zipfile.ZipFile(export_run(out)) as z:bundle=json.loads(z.read('analysis.json'))
            self.assertEqual(len(bundle['learning']['seeds']),2)
            self.assertNotIn('actor',bundle['learning']['seeds'][0])
            c['simulation_horizon_factor']=64.
            with self.assertRaisesRegex(ValueError,'NEW output'):run(c,out,max_minutes=1)

    def test_modified_frozen_policy_cannot_mix_evaluation_results(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            c=config();out=Path(tmp);prepare(c,out);Experiment().train(c,out,None)
            path=out/'learning/seed-11.json';state=read_json(path)
            state['actor']['params']['a'][0]+=1.
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'Frozen model changed'):Experiment().train(c,out,None)

    def test_partial_pool_search_resumes_same_actions(self):
        import learnedplacement.learning_runner as module
        with tempfile.TemporaryDirectory() as tmp:
            c=config();out=Path(tmp);prepare(c,out)
            s=next(s for s in make_scenarios(c) if s.split=='test')
            e=Experiment();e.output=out;original=module.simulate;calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==5:raise BudgetExpired()
                return original(*args,**kwargs)
            with patch.object(module,'simulate',side_effect=interrupt):
                with self.assertRaises(BudgetExpired):e.plan(s,'random_search',[[],[]],c,{},None)
            progress=read_json(out/'cases'/s.workload.case_id/'pool_search_progress.json')
            self.assertEqual(len(progress['rows']),1)
            result=e.plan(s,'random_search',[[],[]],c,{},None)
            self.assertEqual(result['candidates'],len(action_pool(s,c)))
            self.assertEqual(result['candidate_pool_sha256'],digest([p.to_dict() for p in action_pool(s,c)]))

    def test_inference_does_not_call_simulator(self):
        c=config();s=next(s for s in make_scenarios(c) if s.split=='test');e=Experiment()
        pool=action_pool(s,c);state=new_training_state(len(features(s,pool[0])),c,11)
        e.states={11:state};e.models={11:Network.from_dict(state['actor'])}
        with patch('learnedplacement.learning_runner.simulate',side_effect=AssertionError('Inference simulated')):
            result=e.plan(s,'rl_seed_11',None,c,{},None)
        self.assertFalse(violations(s,Decision.from_dict(result['plan'])))

    def test_default_config_matrix_and_horizon(self):
        c=read_json(ROOT/'configs/pilot_v5.json');validate_config(c)
        scenarios=make_scenarios(c)
        self.assertEqual(len(scenarios),288)
        self.assertEqual(sum(s.split=='train' for s in scenarios),192)
        self.assertEqual(c['simulation_horizon_factor'],32.)
        self.assertEqual(sum(s.split=='test' for s in scenarios),48)
