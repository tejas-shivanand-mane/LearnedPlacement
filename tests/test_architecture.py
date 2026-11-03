import contextlib, copy, io, json, tempfile, unittest, zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Install the architecture extra for V8 tests')
from learnedplacement import architecture_runner as ar
from learnedplacement.architecture import Encoder, inputs, score_actor, graph, pack
from learnedplacement.expanded import make_scenarios, ExpandedSimulation, Decision
from learnedplacement.learning import action_pool, features
from learnedplacement.runner import ROOT, read_json
from learnedplacement import guided_runner as guided
from learnedplacement.simulator import BudgetExpired
import test_tuning

torch.set_num_threads(1)


def config():
    c=test_tuning.config();c['experiment']='architecture_v8';c['seed']+=10
    c['learning'].update(history_every=2)
    c['tuning'].update(save_every_episodes=2,variants=[dict(name=a,hidden=8,learning_rate=.001,entropy=.01) for a in ('mlp','set','gnn')])
    c['architecture']=dict(batch_episodes=2,layers=2,candidate_chunk_size=2,cache_entries=2,device='cpu',diagnostic_profiles=['reference'],diagnostic_planning_traces=2,diagnostic_evaluation_traces=2)
    return c


class ArchitectureTests(unittest.TestCase):
    def test_default_inputs_and_latency_profiles(self):
        c=read_json(ROOT/'configs/pilot_v8.json');ar.validate_config(c)
        with tempfile.TemporaryDirectory() as tmp:
            m=ar.prepare(c,tmp)
            self.assertEqual(len(m['training_cases']),576);self.assertEqual(len(m['validation_cases']),24)
            self.assertEqual(len(read_json(Path(tmp)/'diagnostic_scenarios.json')),48)
            self.assertFalse(set(m['training_cases'])&set(m['validation_cases']))
        ss=make_scenarios(c);reference=ss[0];latency=next(s for s in ss if s.workload.trace_key==reference.workload.trace_key and s.workload.profile=='high_latency')
        self.assertEqual(reference.workload.nodes,latency.workload.nodes)
        self.assertEqual(reference.workload.cost_budget_usd,latency.workload.cost_budget_usd)
        self.assertEqual(reference.workload.cross_domain_factor,latency.workload.cross_domain_factor)
        p=action_pool(reference,c)[0]
        before=ExpandedSimulation(reference,p,[],None,False).step_seconds(list(range(len(p.groups))))
        after=ExpandedSimulation(latency,p,[],None,False).step_seconds(list(range(len(p.groups))))
        self.assertAlmostEqual(after-before,2*(len(p.groups[0])+len(p.groups)-2)*(.01-.001))
        self.assertFalse(np.array_equal(features(reference,p),features(latency,p)))

    def test_permutation_invariance_and_candidate_equivariance(self):
        c=config();s=make_scenarios(c)[0];pool=action_pool(s,c)
        perm=list(reversed(range(len(s.workload.nodes))));mapping={old:new for new,old in enumerate(perm)}
        ss=replace(s,workload=replace(s.workload,nodes=[s.workload.nodes[i] for i in perm]))
        pp=[replace(p,groups=tuple(tuple(mapping[i] for i in g) for g in p.groups),backups=tuple((j,mapping[i]) for j,i in p.backups),spares=tuple(mapping[i] for i in p.spares)) for p in pool]
        for name in ('mlp','set','gnn'):
            net=Encoder(name,len(features(s,pool[0])),8,2,11)
            a=score_actor(net,s,pool);b=score_actor(net,ss,pp)
            np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(a[::-1],score_actor(net,s,pool[::-1]),rtol=1e-5,atol=1e-6)
            rebuilt=Encoder.from_dict(net.to_dict())
            np.testing.assert_array_equal(a,score_actor(rebuilt,s,pool))

    def test_relation_edges_affect_gnn_and_receive_gradients(self):
        c=config();s=make_scenarios(c)[0];pool=action_pool(s,c);xs=inputs(s,pool,'gnn');net=Encoder('gnn',len(features(s,pool[0])),8,2,11)
        batch=pack(xs,'cpu');a=net(batch)
        no_edges=dict(batch,edges=[e[:,:0] for e in batch['edges']]);b=net(no_edges)
        self.assertFalse(torch.allclose(a,b))
        a.square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.messages.parameters()))
        set_net=Encoder('set',len(features(s,pool[0])),8,2,11)
        torch.testing.assert_close(set_net(batch),set_net(no_edges),rtol=0,atol=0)

    def test_96_machine_pool_forward_backward(self):
        c=read_json(ROOT/'configs/pilot_v8.json')
        c.update(active_nodes=[64],regimes=['correlated'],instances_per_regime=2,instance_splits=['train','validation'])
        c['expanded_profiles']=c['expanded_profiles'][:1]
        scenario=make_scenarios(c)[0];pool=action_pool(scenario,c)
        self.assertEqual(len(scenario.workload.nodes),96);self.assertEqual(len(pool),64)
        for name in ('mlp','set','gnn'):
            model=Encoder(name,len(features(scenario,pool[0])),64,2,11)
            x=inputs(scenario,pool,name);scores=model.scores(x)
            self.assertEqual(tuple(scores.shape),(64,));self.assertTrue(torch.isfinite(scores).all())
            (-torch.log_softmax(scores,0)[0]).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_latency_enters_transfers_without_changing_bandwidth(self):
        c=config();s=make_scenarios(c)[0];p=action_pool(s,c)[0]
        low=ExpandedSimulation(s,p,[],None,False)
        high=ExpandedSimulation(replace(s,network_latency_s=s.network_latency_s+.02),p,[],None,False)
        # One remote-storage-to-machine request; latency increases both endpoint loads.
        a,_=low.transfer([(0,0)],{0:[-1]});b,_=high.transfer([(0,0)],{0:[-1]})
        self.assertAlmostEqual(b-a,.02)

    def test_update_increases_rewarded_probability(self):
        c=config();s=make_scenarios(c)[0];pool=action_pool(s,c)
        for name in ('mlp','set','gnn'):
            net=Encoder(name,len(features(s,pool[0])),8,2,11);xs=inputs(s,pool,name);opt=torch.optim.Adam(net.parameters(),lr=.01)
            before=torch.softmax(net.scores(xs),0)[0].item()
            for _ in range(5):
                opt.zero_grad();loss=-torch.log_softmax(net.scores(xs),0)[0];loss.backward();opt.step()
            self.assertGreater(torch.softmax(net.scores(xs),0)[0].item(),before)

    def test_resume_replays_incomplete_batch_and_validation(self):
        c=config()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            a=Path(tmp)/'a';b=Path(tmp)/'b';ar.prepare(c,a);ar.prepare(c,b)
            original=ar._rollout;calls=0
            def stop(task):
                nonlocal calls
                calls+=1
                if calls==2:raise BudgetExpired()
                return original(task)
            with patch.object(ar,'_rollout',side_effect=stop):
                with self.assertRaises(BudgetExpired):ar.train_job(a,'gnn',11)
            self.assertEqual(read_json(a/'jobs/gnn-seed-11/state.json')['episode'],0)
            original_validation=ar.checkpoint_validation;calls=0
            def stop_validation(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==1:raise BudgetExpired()
                return original_validation(*args,**kwargs)
            with patch.object(ar,'checkpoint_validation',side_effect=stop_validation):
                with self.assertRaises(BudgetExpired):ar.train_job(a,'gnn',11)
            self.assertEqual(read_json(a/'jobs/gnn-seed-11/state.json')['episode'],2)
            ar.train_job(a,'gnn',11);ar.train_job(b,'gnn',11)
            for ep in (2,4):self.assertEqual(read_json(a/f'jobs/gnn-seed-11/checkpoints/{ep}.json'),read_json(b/f'jobs/gnn-seed-11/checkpoints/{ep}.json'))
            self.assertEqual(read_json(a/'jobs/gnn-seed-11/state.json')['history'],read_json(b/'jobs/gnn-seed-11/state.json')['history'])

    def test_end_to_end_separate_modes_diagnostic_export(self):
        c=config()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);out=root/'train'
            self.assertEqual(ar.diagnostic(c,out,workers=1)['state'],'finished')
            report=read_json(out/'diagnostic_summary.json');self.assertEqual(report['total_cases'],1)
            self.assertEqual(ar.run(c,out,workers=1)['state'],'finished')
            selected=ar.freeze_selection(out);self.assertEqual(len(selected['selections']),6)
            frozen=(out/'selected_policies.json').read_bytes()
            ar.run(c,out,workers=1)
            self.assertEqual(frozen,(out/'selected_policies.json').read_bytes())
            report=ar.summary(out)
            for name in ('mlp','set','gnn'):
                for mode in ('direct','top_k'):
                    options=[r for r in selected['groups'] if r['architecture']==name and r['mode']==mode]
                    self.assertEqual(selected['selections'][name+'_'+mode]['episode'],min(options,key=lambda r:(r['objective'],r['completion_time'],r['episode']))['episode'])
            source=ar.comparison_source(out,root/'source');cfg=ar.evaluation_config(c,source)
            self.assertEqual(len(cfg['methods']),16)
            guided.validate_config(cfg)
            self.assertEqual(guided.run(cfg,root/'eval',source,workers=1)['state'],'finished')
            m=read_json(root/'eval/manifest.json');cid=m['cases'][0];plans=read_json(root/'eval/partitions'/cid/'cases'/cid/'plans.json')
            for name in ['mlp_s11','set_s11','gnn_s11']:self.assertEqual(plans[name]['simulations'],0)
            self.assertEqual(plans['gnn_s11_top2']['simulations'],4)
            self.assertEqual(plans['gnn_s11_top2']['candidate_pool_sha256'],plans['pool_search']['candidate_pool_sha256'])
            with zipfile.ZipFile(ar.export(out)) as z:exported=json.loads(z.read('analysis.json'))
            self.assertNotIn('weights',json.dumps(exported));self.assertEqual(exported['diagnostic_summary']['state'],'finished')
            with zipfile.ZipFile(guided.export(root/'eval')) as z:ev=json.loads(z.read('analysis.json'))
            self.assertEqual(ev['status']['state'],'finished')
            mutated=copy.deepcopy(cfg);mutated['architecture_models']['gnn_s11']['seed']=11011
            with self.assertRaisesRegex(ValueError,'modes changed'):guided.load_source(source,mutated)

    def test_parallel_rollouts_and_legacy_comparison(self):
        from learnedplacement.learning_runner import run as old_run
        from learnedplacement import tuning_runner as tuning
        c=config();c['tuning']['checkpoints']=[2]
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);a=root/'a';b=root/'b'
            ar.run(c,a,workers=1);ar.run(c,b,workers=2)
            self.assertEqual(read_json(a/'selection.json'),read_json(b/'selection.json'))
            self.assertEqual(read_json(a/'selected_policies.json')['actors'],read_json(b/'selected_policies.json')['actors'])
            import test_learning
            old=root/'old';old_run(test_learning.config(),old,max_minutes=None)
            v7=root/'v7';tc=test_tuning.config();tuning.run(tc,v7,workers=1)
            legacy=tuning.comparison_source(v7,old,root/'legacy')
            source=ar.comparison_source(a,root/'source',legacy)
            cfg=ar.evaluation_config(c,source);bundle=guided.load_source(source,cfg)
            self.assertEqual(len(bundle['actors']),16)
            self.assertIn('old_small_seed_11',cfg['methods'])

    def test_input_tampering_and_test_split_rejected(self):
        c=config()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ar.prepare(c,root)
            path=root/'train_scenarios.json';raw=read_json(path);raw[0]['split']='test';path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'TRAIN'):ar.train_job(root,'mlp',11)
            with self.assertRaisesRegex(ValueError,'Modified'):ar.prepare(c,root)
        c['instance_splits']=['train','test']
        with self.assertRaises(ValueError):ar.validate_config(c)
