from dataclasses import replace
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import random

from learnedplacement.expanded import Scenario, Decision, ExpandedSimulation, simulate, generate_trace, violations, make_scenarios
from learnedplacement.expanded_planners import plan, resources, spread_shards, observation, valid_candidates, neighbors
from learnedplacement.expanded_runner import run, validate_config, build_summary
from learnedplacement.runner import ROOT, read_json, export_run
from learnedplacement.simulator import Event, BudgetExpired
from learnedplacement.model import Case, Node


def fixture():
    nodes = [Node(1.,1.,32.,100.,.1,i%2) for i in range(8)]
    c = Case('tiny',nodes,replicas=2,shards=2,state_gib=2.,working_gib=1.,steps=8,
             base_step_s=1.,collective_factor=0.,cross_domain_factor=1.,storage_gib_s=100.,
             backup_budget_gib=8.,backup_every=2,checkpoint_every=2,detection_s=0.,reinit_s=0.,
             replacement_s=2.,deadline_s=30.,cost_budget_usd=1.,max_simulated_s=60.)
    return Scenario(c,network_latency_s=0.,local_disk_gib_s=100.),Decision(((0,1),(2,3)),checkpoint_every=8,backup_every=2)


def tiny_config():
    cfg = read_json(ROOT/'configs/pilot_v4.json')
    cfg.update(active_nodes=[4],regimes=['homogeneous'],instances_per_regime=1,instance_splits=['train'],
               planning_traces=2,evaluation_traces=2,search_candidates=3,tuned_candidates=3,save_every_traces=1,planner_seconds=5.)
    cfg['expanded_profiles'] = cfg['expanded_profiles'][:1]
    cfg['workload'].update(steps=3,checkpoint_every=2,backup_every=2,base_step_s=.1)
    cfg['choices'].update(checkpoint_every=[2,3],backup_every=[1,2])
    return cfg


class ExpandedSimulationTests(unittest.TestCase):
    def test_fixed_work_across_parallelism_and_resources(self):
        s,p = fixture()
        a = simulate(s,p,[])
        b = simulate(s,replace(p,groups=((0,),(1,),(2,),(3,))),[])
        c = simulate(s,replace(p,groups=((0,1,2,3),)),[])
        d = simulate(s,replace(p,groups=((0,1),)),[])
        self.assertEqual(a['completion_s'],8.)
        self.assertEqual(a['completion_s'],b['completion_s'])
        self.assertEqual(a['completion_s'],c['completion_s'])
        self.assertEqual(d['completion_s'],16.)
        self.assertEqual(a['machine_cost_usd'],d['machine_cost_usd'])

    def test_topology_changes_communication(self):
        s,p = fixture(); s.workload.collective_factor = .5; s.workload.cross_domain_factor = .1
        p = replace(p,groups=((0,1),))
        packed = replace(p,groups=((0,2),))
        self.assertNotEqual(simulate(s,p,[])['completion_s'],simulate(s,packed,[])['completion_s'])

    def test_async_save_overlaps_compute_and_has_committed_version(self):
        s,p = fixture(); s.workload.storage_gib_s = .5; s.workload.steps = 16
        sync = replace(p,checkpoint_every=2)
        asynchronous = replace(sync,checkpoint_mode='async',background_fraction=.5)
        a = simulate(s,sync,[],audit=True); b = simulate(s,asynchronous,[],audit=True)
        self.assertLess(b['completion_s'],a['completion_s'])
        self.assertGreater(b['background_elapsed_s'],0.)
        self.assertAlmostEqual(sum(b['phase_seconds'].values()),b['observed_s'])
        self.assertTrue(any(x['event']=='checkpoint_complete' for x in b['audit']))
        self.assertGreater(b['write_cost_usd'],0.)

    def test_failed_pending_snapshot_is_not_committed(self):
        s,p = fixture(); s.workload.storage_gib_s = .5
        p = replace(p,checkpoint_every=2,checkpoint_mode='async',background_fraction=.5)
        trace = sorted([Event(3.,0,n) for n in range(4)]+[Event(5.,1,n) for n in range(4)])
        r = simulate(s,p,trace,audit=True)
        self.assertGreater(r['checkpoint_aborts'],0)
        self.assertEqual(next(x['version'] for x in r['audit'] if x['event']=='rollback'),0)
        self.assertTrue(r['completed'])

    def test_local_checkpoint_lost_with_hosts_remote_survives(self):
        s,p = fixture(); p = replace(p,checkpoint_every=2)
        trace = sorted([Event(3.5,0,n) for n in range(4)]+[Event(5.5,1,n) for n in range(4)])
        a = simulate(s,p,trace)
        b = simulate(s,replace(p,checkpoint_tier='local'),trace)
        self.assertLess(a['lost_steps'],b['lost_steps'])
        self.assertEqual(b['write_cost_usd'],0.)

    def test_spares_have_cost_and_reduce_waiting(self):
        s,p = fixture(); p = replace(p,groups=((0,1),))
        q = replace(p,spares=(4,5))
        healthy = simulate(s,q,[])
        self.assertGreater(healthy['machine_cost_usd'],simulate(s,p,[])['machine_cost_usd'])
        trace = [Event(2.5,0,0),Event(20.,1,0)]
        a = simulate(s,p,trace); b = simulate(s,q,trace)
        self.assertLess(b['completion_s'],a['completion_s'])
        self.assertEqual(b['spare_remaps'],1)
        self.assertAlmostEqual(b['machine_cost_usd'],4*b['observed_s']/3600)

    def test_multiple_copies_per_shard_and_recovery(self):
        s,p = fixture()
        p = replace(p,backups=((0,4),(0,6),(1,5),(1,7)))
        self.assertEqual(violations(s,p),[])
        trace = sorted([Event(3.5,0,n) for n in (0,1,2,3,4,5)]+[Event(5.,1,n) for n in (0,1,2,3,4,5)])
        r = simulate(s,p,trace)
        self.assertEqual(r['lost_steps'],1)
        self.assertTrue(r['completed'])

    def test_defer_policy_does_not_repair_while_group_survives(self):
        s,p = fixture()
        trace = [Event(1.5,0,0),Event(2.,1,0)]
        eager = simulate(s,p,trace)
        deferred = simulate(s,replace(p,recovery='defer'),trace)
        self.assertGreater(eager['recovered_groups'],0)
        self.assertEqual(deferred['recovered_groups'],0)
        self.assertGreater(deferred['completion_s'],eager['completion_s'])

    def test_periodic_repair_changes_rejoin(self):
        s,p = fixture(); trace = [Event(1.5,0,0),Event(2.,1,0)]
        a = simulate(s,replace(p,repair_every=1),trace,audit=True)
        b = simulate(s,replace(p,repair_every=4),trace,audit=True)
        self.assertLess(next(x['time_s'] for x in a['audit'] if x['event']=='restore'),next(x['time_s'] for x in b['audit'] if x['event']=='restore'))

    def test_retention_keeps_common_local_version(self):
        s,p = fixture(); sim = ExpandedSimulation(s,replace(p,checkpoint_tier='local',checkpoint_retention=2),[])
        sim.local = {(0,4):[2,4],(1,5):[2,6]}; sim.version=6
        sim.primary = {n:None for n in p.active}
        self.assertEqual(sim.newest_recoverable(),2)
        sim.local[(0,4)] = [4]
        self.assertEqual(sim.newest_recoverable(),0)

    def test_constraints_account_for_async_buffer_and_disk(self):
        s,p = fixture(); s.workload.nodes[0] = replace(s.workload.nodes[0],memory_gib=2.)
        self.assertFalse(violations(s,p))
        self.assertTrue(violations(s,replace(p,checkpoint_mode='async')))
        s.local_disk_gib=.5
        self.assertTrue(violations(s,replace(p,checkpoint_tier='local')))
        self.assertTrue(violations(s,replace(p,spares=(0,))))
        self.assertTrue(violations(s,replace(p,backups=((0,4),(0,4)))))

    def test_trace_reproducibility_and_domain_outages(self):
        s,p = fixture(); s.workload.domain_failure_per_hour=1000
        s.outage_distribution='lognormal'
        a = generate_trace(s,42); self.assertEqual(a,generate_trace(s,42))
        self.assertNotEqual(a,generate_trace(s,43))
        self.assertEqual(len({e.time_s for e in a if e.kind==0 and e.node==0}-{e.time_s for e in a if e.kind==0 and e.node==2}),0)

    def test_censoring_and_phase_accounting(self):
        s,p=fixture(); s.workload.max_simulated_s=3.
        r=simulate(s,p,[])
        self.assertFalse(r['completed']); self.assertIsNone(r['completion_s'])
        self.assertAlmostEqual(sum(r['phase_seconds'].values()),r['observed_s'])

    def test_outages_overlap_and_failures_during_recovery(self):
        s,p=fixture(); s.workload.reinit_s=2.
        trace=[Event(.5,0,0),Event(.7,0,0),Event(1.,1,0),Event(2.,1,0),Event(3.,0,2),Event(4.,1,2)]
        r=simulate(s,p,trace,audit=True)
        self.assertTrue(r['completed'])
        self.assertEqual([x['time_s'] for x in r['audit'] if x['event']=='replacement' and x['node']==0],[2.])
        self.assertAlmostEqual(sum(r['phase_seconds'].values()),r['observed_s'])

    def test_tiny_policy_matrix_survives_common_domain_loss(self):
        s,p=fixture()
        s.workload.collective_factor=.01
        trace=sorted([Event(1.5,0,n) for n in (0,2,4,6)]+[Event(3.5,1,n) for n in (0,2,4,6)])
        for tier in ('remote','local'):
            for mode in ('sync','async'):
                for recovery in ('eager','defer'):
                    for spare in ((),(6,7)):
                        q=replace(p,checkpoint_every=2,checkpoint_tier=tier,checkpoint_mode=mode,
                                  recovery=recovery,spares=spare,backups=((0,4),(1,5)))
                        result=simulate(s,q,trace)
                        self.assertTrue(result['completed'])
                        self.assertAlmostEqual(sum(result['phase_seconds'].values()),result['observed_s'])
                        self.assertAlmostEqual(result['cost_usd'],sum(result[k] for k in ('machine_cost_usd','storage_cost_usd','write_cost_usd')))

    def test_failure_tied_with_async_completion_cancels_snapshot(self):
        s,p=fixture()
        q=replace(p,checkpoint_every=2,checkpoint_mode='async',background_fraction=.5)
        sim=ExpandedSimulation(s,q,[Event(1.,0,0)])
        sim.pending=dict(version=2,finish=1.,participants={0},destinations=[])
        sim.advance(1.,'waiting')
        self.assertEqual(sim.remote,[0])
        self.assertEqual(sim.checkpoint_aborts,1)

    def test_case_and_decision_serialization_round_trip(self):
        s,p=fixture()
        p=replace(p,spares=(4,),checkpoint_mode='async',checkpoint_tier='local')
        self.assertEqual(Decision.from_dict(json.loads(json.dumps(p.to_dict()))),p)
        self.assertEqual(Scenario.from_dict(json.loads(json.dumps(s.to_dict()))),s)


class ExpandedPlannerTests(unittest.TestCase):
    def test_tuned_baseline_covers_entire_checkpoint_grid_first(self):
        cfg=tiny_config(); cfg['choices']['checkpoint_every']=[1,2,3]
        cfg['tuned_candidates']=3
        s=make_scenarios(cfg)[0]
        r=plan(s,'domain_tuned',[[],[]],cfg)
        self.assertEqual({x['plan']['checkpoint_every'] for x in r['candidate_audit']},{1,2,3})

    def test_domain_permutation_preserves_groups(self):
        s,p=fixture(); groups=spread_shards(s,p.groups)
        self.assertEqual([set(g) for g in p.groups],[set(g) for g in groups])
        self.assertTrue(all(s.workload.nodes[groups[0][j]].domain != s.workload.nodes[groups[1][j]].domain for j in range(2)))

    def test_adapter_and_candidates_are_feasible(self):
        cfg=tiny_config(); s=make_scenarios(cfg)[0]
        obs=observation(s)
        self.assertNotIn('split',obs); self.assertNotIn('case_id',obs['workload'])
        candidates=valid_candidates(s,cfg,42,16)
        self.assertEqual(candidates,valid_candidates(s,cfg,42,16))
        self.assertTrue(all(not violations(s,p) for p in candidates))
        self.assertGreater(len({len(p.groups[0]) for p in candidates}),1)
        self.assertGreater(len({len(p.active) for p in candidates}),1)

    def test_candidate_partial_scores_are_excluded(self):
        cfg=tiny_config(); s=make_scenarios(cfg)[0]; calls=0
        original=simulate
        def interrupted(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==5: raise BudgetExpired()
            return original(*args,**kwargs)
        with patch('learnedplacement.expanded_planners.simulate',side_effect=interrupted):
            r=plan(s,'random_search',[[],[]],cfg)
        self.assertEqual(r['candidates'],1)
        self.assertEqual(r['controls']['random_unaware']['candidates'],1)
        self.assertEqual(r['candidate_pool_sha256'],r['controls']['random_unaware']['candidate_pool_sha256'])

    def test_complete_random_pool_and_selectors(self):
        cfg=tiny_config(); s=make_scenarios(cfg)[0]
        r=plan(s,'random_search',[[],[]],cfg)
        u=r['controls']['random_unaware']
        self.assertEqual(r['plan'],min(r['candidate_audit'],key=lambda x:x['score'])['plan'])
        self.assertEqual(u['plan'],min(r['candidate_audit'],key=lambda x:x['healthy_score'])['plan'])

    def test_split_and_profile_pairing(self):
        cfg=read_json(ROOT/'configs/pilot_v4.json'); validate_config(cfg)
        scenarios=make_scenarios(cfg)
        self.assertEqual(len(scenarios),144)
        for key in {s.workload.trace_key for s in scenarios}:
            family=[s for s in scenarios if s.workload.trace_key==key]
            self.assertEqual(len({s.split for s in family}),1)
            self.assertEqual(len({s.workload.deadline_s for s in family}),1)
            self.assertTrue(all(x.workload.nodes==family[0].workload.nodes for x in family))


class ExpandedRunnerTests(unittest.TestCase):
    def test_tiny_resume_export_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'; cfg=tiny_config()
            self.assertEqual(run(cfg,out,max_minutes=1e-9)['state'],'paused_time_limit')
            self.assertEqual(run(cfg,out,max_minutes=1)['state'],'finished')
            m=read_json(out/'manifest.json'); d=out/'cases'/m['cases'][0]
            before=(d/'plans.json').read_bytes()
            run(cfg,out,max_minutes=1)
            self.assertEqual(before,(d/'plans.json').read_bytes())
            rows=build_summary(out); self.assertEqual(len(rows),6)
            import zipfile
            with zipfile.ZipFile(export_run(out)) as z:
                self.assertEqual(z.namelist(),['analysis.json'])
                bundle=json.loads(z.read('analysis.json'))
            self.assertEqual(len(bundle['cases'][0]['paired_comparisons']),15)
            self.assertNotIn('candidate_audit',bundle['cases'][0]['methods']['random_search']['selection'])
            cfg['seed']+=1
            with self.assertRaisesRegex(ValueError,'NEW output'): run(cfg,out,max_minutes=1)

    def test_lock_and_optional_time_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'.run.lock').write_text('1')
            with self.assertRaisesRegex(RuntimeError,'locked'): run(tiny_config(),out)
            self.assertTrue((out/'.run.lock').exists())
            with self.assertRaisesRegex(RuntimeError,'locked'): run(tiny_config(),out,max_minutes=31)
            with self.assertRaisesRegex(RuntimeError,'locked'): run(tiny_config(),out,max_minutes=None)
            with self.assertRaises(ValueError): run(tiny_config(),out,max_minutes=0)

    def test_config_invalid_choice_rejected(self):
        cfg=tiny_config(); cfg['choices']['checkpoint_mode']=['magic']
        with self.assertRaises(ValueError): validate_config(cfg)
