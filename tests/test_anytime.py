import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from learnedplacement import anytime_runner as ar
from learnedplacement.architecture import Encoder
from learnedplacement.expanded import make_scenarios, generate_trace, Decision, violations
from learnedplacement.expanded_planners import base_plan, digest
from learnedplacement.learning import features
from learnedplacement.runner import read_json, atomic_json, source_snapshot, ROOT
import hashlib
import test_learning

torch.set_num_threads(1)


def source(root):
    c = test_learning.config()
    s = make_scenarios(c)[0]
    actors = {}; specs = {}
    for i, arch in enumerate(('mlp','set','gnn')):
        for seed in (11,29):
            key = str(i*100+seed)
            actors[key] = Encoder(arch,len(features(s,base_plan(s,c))),4,1,seed).to_dict()
            specs[f'{arch}_s{seed}'] = dict(kind='direct',seed=int(key))
    bundle = dict(actors=actors,model_specs=specs,source_config=c,source_learning={'selection':{'fixture':True}},excluded_hardware_seeds=[c['seed']])
    atomic_json(root/'selected_policies.json',bundle)
    atomic_json(root/'selected_policy_hash.json',{'sha256':digest(bundle)})
    atomic_json(root/'manifest.json',dict(experiment='architecture_v8',sources={k:hashlib.sha256(v.encode()).hexdigest() for k,v in source_snapshot().items()}))
    return dict(seed=c['seed']+50,case_mode='balanced',search_seeds=[101,202],seconds=[.00001,.00002],candidates=[1,2])


class AnytimeTests(unittest.TestCase):
    def test_checkpoint_no_lookahead_and_unscored_fallback(self):
        trajectory = dict(events=[dict(elapsed_s=1.5,candidates=1,plan={'p':1},objective=9),
                                  dict(elapsed_s=5.1,candidates=2,plan={'p':2},objective=1)],
                          fallback={'p':0},fallback_ready_s=.1)
        rows = ar.checkpoints(trajectory,dict(seconds=[.01,1,2,5,6],candidates=[1,2]))
        self.assertIsNone(rows[0]['plan'])
        self.assertEqual([r['plan'] for r in rows[1:5]],[{'p':0},{'p':1},{'p':1},{'p':2}])
        self.assertEqual(rows[-2]['plan'],{'p':1})
        self.assertEqual(rows[-1]['plan'],{'p':2})

    def test_default_cases_balanced_profiles_and_full(self):
        c = read_json(ROOT/'configs/pilot_v8.json')
        c.update(instances_per_regime=1,instance_splits=['test'],anytime={'case_mode':'balanced'})
        cases = ar.scenarios(c)
        from collections import Counter
        self.assertEqual(len(cases),24)
        self.assertEqual(set(Counter(s.workload.profile for s in cases).values()),{4})
        self.assertEqual({len(s.workload.nodes) for s in cases},{6,12,24,32,64,96})
        c['anytime']['case_mode']='full'
        self.assertEqual(len(ar.scenarios(c)),144)

    def test_model_selection_checks_and_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=source(root)
            c,bundle,specs=ar.configuration(root,settings)
            self.assertEqual(len(specs),6)
            self.assertEqual(len(ar.method_specs(specs)),9)
            self.assertEqual(c['instance_splits'],['test'])
            settings['seed']=bundle['source_config']['seed']+1
            with self.assertRaisesRegex(ValueError,'fresh'):ar.configuration(root,settings)
            path=root/'selected_policies.json';b=read_json(path);b['actors'].pop(next(iter(b['actors'])))
            atomic_json(path,b)
            with self.assertRaisesRegex(ValueError,'changed'):ar.load_source(root)

    def test_search_private_cache_monotonicity_and_all_model_seeds(self):
        from learnedplacement.architecture import load_actor
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=source(root);settings.update(seconds=[.00001],candidates=[2,5])
            c,bundle,specs=ar.configuration(root,settings);s=ar.scenarios(c)[0]
            traces=[generate_trace(s,1),generate_trace(s,2)]
            original=ar.simulate
            for name,spec in ar.method_specs(specs).items():
                model=load_actor(bundle['actors'][str(spec['seed'])]) if 'seed' in spec else None
                with patch.object(ar,'simulate',wraps=original) as spy:
                    trajectory=ar.search(s,c,spec,101,traces,model)
                    self.assertEqual(spy.call_count,trajectory['candidates']*len(traces))
                objectives=[e['objective'] for e in trajectory['events']]
                self.assertEqual(objectives,sorted(objectives,reverse=True))
                for event in trajectory['events']:
                    self.assertFalse(violations(s,Decision.from_dict(event['plan'])))
                self.assertGreaterEqual(trajectory['candidates'],5)

    def test_interrupted_search_restarts_and_worker_stop_is_cooperative(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);src=root/'source';settings=source(src);out=root/'out'
            ar.prepare(src,out,settings,1)
            raw=read_json(out/'scenarios.json')[0];cid=raw['workload']['case_id']
            with patch.object(ar,'simulate',side_effect=InterruptedError('fixture interruption')):
                with self.assertRaises(InterruptedError):ar._case((str(out),raw))
            self.assertFalse(list((out/'cases'/cid/'search').glob('*.json')))
            (out/'.stop').touch()
            with self.assertRaises(InterruptedError):ar._case((str(out),raw))
            self.assertEqual(ar.run(src,out,settings,1)['state'],'finished')

    def test_serial_parallel_resume_export_and_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);src=root/'source';settings=source(src)
            before=(src/'selected_policies.json').read_bytes()
            with patch('learnedplacement.architecture_runner.run',side_effect=AssertionError('No training')):
                for workers in (1,2):
                    out=root/f'out{workers}'
                    self.assertEqual(ar.run(src,out,settings,workers)['state'],'finished')
                    with patch.object(ar,'search',side_effect=AssertionError('Completed search rerun')):
                        ar.run(src,out,settings,workers)
            self.assertEqual(before,(src/'selected_policies.json').read_bytes())
            out=root/'out1';m=read_json(out/'manifest.json');cid=m['cases'][0]
            a=read_json(out/'cases'/cid/'result.json');b=read_json(root/'out2'/'cases'/cid/'result.json')
            aa={(r['method'],r['search_seed'],r['budget']):r for r in a['selections'] if r['axis']=='candidates'}
            bb={(r['method'],r['search_seed'],r['budget']):r for r in b['selections'] if r['axis']=='candidates'}
            for key in aa:
                self.assertEqual(aa[key]['plan'],bb[key]['plan'])
                self.assertEqual(aa[key]['summary'],bb[key]['summary'])
            # Evaluation can resume from cached complete plans without searching.
            (out/'cases'/cid/'result.json').unlink()
            with patch.object(ar,'search',side_effect=AssertionError('Completed search rerun')),patch.object(ar,'simulate',side_effect=AssertionError('Cached evaluation rerun')):
                ar.run(src,out,settings,1)
            with zipfile.ZipFile(ar.export(out)) as archive:
                payload=json.loads(archive.read('analysis.json'))
            self.assertEqual(payload['status']['state'],'finished')
            self.assertNotIn('observations',json.dumps(payload))
            self.assertNotIn('actors',payload)
            with patch('matplotlib.pyplot.show'):
                table=ar.plot_results(out,axis='candidates')
            self.assertEqual(len(table),12)
            with self.assertRaisesRegex(ValueError,'NEW V9'):ar.run(src,out,settings,2)


if __name__ == '__main__':
    unittest.main()
