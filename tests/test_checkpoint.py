from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_simulator
from learnedplacement.checkpoint import checkpoint_search, plan_checkpoint_method
from learnedplacement.model import Case, Plan, make_cases, violations
from learnedplacement.planners import add_backups, neighbor
from learnedplacement.runner import ROOT, export_run, paired_report, read_json, run, validate_config
from learnedplacement.simulator import BudgetExpired, Event, simulate


class CheckpointTests(unittest.TestCase):
    def fixture(self):
        c,p=test_simulator.fixture()
        c.checkpoint_every=2
        cfg=dict(seed=1,checkpoint_intervals=[1,2,6],checkpoint_layouts=4,checkpoint_seconds=10.)
        return c,p,cfg

    def test_interval_tradeoff_healthy_vs_failure(self):
        c,p,_=self.fixture()
        frequent=replace(p,checkpoint_every=2)
        none=replace(p,checkpoint_every=c.steps)
        self.assertAlmostEqual(simulate(c,frequent,[])['completion_s'],6.04)
        self.assertEqual(simulate(c,none,[])['completion_s'],6.)
        trace=sorted([Event(4.5,0,n) for n in range(4)]+[Event(5.5,1,n) for n in range(4)])
        self.assertLess(simulate(c,frequent,trace)['completion_s'],simulate(c,none,trace)['completion_s'])
        self.assertEqual(c.checkpoint_every,2)

    def test_old_and_new_plan_serialization_and_interval_validation(self):
        c,p,_=self.fixture()
        self.assertNotIn('checkpoint_every',p.to_dict())
        self.assertEqual(Plan.from_dict(p.to_dict()),p)
        q=replace(p,checkpoint_every=3)
        self.assertEqual(Plan.from_dict(q.to_dict()),q)
        for invalid in [0,-1,True,1.5]:
            self.assertTrue(violations(c,replace(p,checkpoint_every=invalid)))

    def test_backup_edits_preserve_interval(self):
        import random
        c,p,_=self.fixture();p=replace(p,checkpoint_every=3)
        self.assertEqual(add_backups(c,p).checkpoint_every,3)
        for i in range(20):
            self.assertEqual(neighbor(c,p,random.Random(i)).checkpoint_every,3)

    def test_matched_selectors_use_same_complete_layouts(self):
        c,_,cfg=self.fixture();r=checkpoint_search(c,[[],[]],cfg)
        self.assertEqual(r['completed_layouts'],4)
        self.assertEqual(r['candidates'],12)
        u=r['controls']['checkpoint_unaware'];f=r['controls']['checkpoint_fixed_search']
        self.assertEqual(r['candidate_pool_sha256'],u['candidate_pool_sha256'])
        self.assertEqual(r['layout_pool_sha256'],f['layout_pool_sha256'])
        self.assertEqual(f['candidates'],4)
        self.assertEqual(r['plan']['checkpoint_every'],c.steps)
        self.assertEqual(f['plan']['checkpoint_every'],c.checkpoint_every)
        audit=r['candidate_audit']
        self.assertEqual(r['plan'],min(audit,key=lambda x:x['aware_score'])['plan'])
        fixed=[x for x in audit if x['plan']['checkpoint_every']==c.checkpoint_every]
        self.assertEqual(f['plan'],min(fixed,key=lambda x:x['aware_score'])['plan'])

    def test_layout_generation_and_unaware_choice_ignore_risk(self):
        c,_,cfg=self.fixture()
        a=checkpoint_search(c,[[]],cfg)
        risky=replace(c,nodes=[replace(n,failure_per_hour=100.) for n in c.nodes],domain_failure_per_hour=50.)
        b=checkpoint_search(risky,[[Event(.5,0,0),Event(2.,1,0)]],cfg)
        self.assertEqual(a['layout_pool_sha256'],b['layout_pool_sha256'])
        self.assertEqual(a['controls']['checkpoint_unaware']['plan'],b['controls']['checkpoint_unaware']['plan'])

    def test_partial_layout_block_excluded_from_all_selectors(self):
        c,_,cfg=self.fixture();calls=0
        def limited(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==11:raise BudgetExpired()
            return simulate(*args,**kwargs)
        with patch('learnedplacement.checkpoint.simulate',side_effect=limited):
            r=checkpoint_search(c,[[],[]],cfg)
        self.assertEqual(r['completed_layouts'],1)
        self.assertEqual(r['candidates'],3)
        self.assertEqual(len(r['candidate_audit']),3)
        self.assertEqual(r['controls']['checkpoint_fixed_search']['candidates'],1)
        self.assertEqual(r['controls']['checkpoint_unaware']['candidates'],3)

    def test_fallback_when_no_layout_finishes(self):
        c,_,cfg=self.fixture()
        with patch('learnedplacement.checkpoint.simulate',side_effect=BudgetExpired):
            r=checkpoint_search(c,[[]],cfg)
        self.assertEqual(r['completed_layouts'],0)
        self.assertIsNone(r['planning_point_feasible'])
        self.assertFalse(violations(c,Plan.from_dict(r['plan'])))

    def test_ablations_change_only_intended_field(self):
        c,p,cfg=self.fixture();p=replace(add_backups(c,p),checkpoint_every=6)
        parent={'checkpoint_joint':dict(plan=p.to_dict(),planning_s=1.)}
        a=plan_checkpoint_method(c,'checkpoint_no_backup',[],cfg,parent)
        b=plan_checkpoint_method(c,'checkpoint_fixed_ablation',[],cfg,parent)
        self.assertEqual(Plan.from_dict(a['plan']),replace(p,backups=()))
        self.assertEqual(Plan.from_dict(b['plan']),replace(p,checkpoint_every=2))


class CheckpointRunnerTests(unittest.TestCase):
    def config(self):
        c=test_simulator.RunnerTests().config()
        c.update(methods=read_json(ROOT/'configs/pilot_v3.json')['methods'],
                 checkpoint_intervals=[1,2],checkpoint_layouts=2,checkpoint_seconds=2.)
        c['workload']['checkpoint_every']=1
        return c

    def test_tiny_v3_run_export_and_matched_reports(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'run';cfg=self.config()
            self.assertEqual(run(config=cfg,output=out,max_minutes=1)['state'],'finished')
            directory=out/'cases/n4-homogeneous-i0'
            for method in cfg['methods']:
                self.assertIn(read_json(directory/f'{method}.summary.json')['checkpoint_every'],[1,2])
            for baseline in ['checkpoint_unaware','checkpoint_fixed_search']:
                r=paired_report(out,'checkpoint_joint',baseline,repeats=20)[0]
                self.assertTrue(r['matched_candidate_pool'] or r['matched_layout_pool'])
            with zipfile.ZipFile(export_run(out)) as z:
                b=json.loads(z.read('analysis.json'))
                self.assertNotIn('controls',b['cases'][0]['methods']['checkpoint_joint']['selection'])
            plans=(directory/'plans.json').read_bytes()
            run(config=cfg,output=out,max_minutes=1)
            self.assertEqual(plans,(directory/'plans.json').read_bytes())

    def test_v3_inputs_and_reference_budgets_independent_of_search_grid(self):
        c=read_json(ROOT/'configs/pilot_v3.json');validate_config(c)
        a=make_cases(c);self.assertEqual(len(a),48)
        self.assertEqual({x.replicas for x in a},{1,2,4})
        c['checkpoint_intervals']=[80]
        b=make_cases(c)
        self.assertEqual([x.to_dict() for x in a],[x.to_dict() for x in b])
        for x,y in zip(a[::2],a[1::2]):
            self.assertEqual(x.nodes,y.nodes)
            self.assertEqual(x.replacement_s,y.replacement_s)
            self.assertNotEqual(x.storage_gib_s,y.storage_gib_s)

    def test_missing_reference_interval_rejected(self):
        c=self.config();c['checkpoint_intervals']=[2]
        with self.assertRaisesRegex(ValueError,'reference checkpoint'):
            validate_config(c)
