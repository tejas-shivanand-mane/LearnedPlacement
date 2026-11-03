"""Small deterministic checks, not an experiment run."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import test_simulator
from test_simulator import fixture
from learnedplacement.model import Plan, make_cases, step_seconds, violations
from learnedplacement.planners import initial, matched_search
from learnedplacement.runner import ROOT, paired_report, read_json, run, validate_config
from learnedplacement.simulator import BudgetExpired, Event, simulate


class MatchedSearchTests(unittest.TestCase):
    def config(self):
        return dict(seed=1, matched_candidates=12, matched_seconds=10)

    def test_speed_grouping_removes_mixed_speed_bottlenecks(self):
        c, _ = fixture()
        c.nodes = [replace(n, speed=3. if i < 2 else 1.) for i, n in enumerate(c.nodes)]
        grouped, spread = initial(c, 'speed_grouped'), initial(c, 'speed_first')
        self.assertEqual(grouped.active, spread.active)
        self.assertLess(step_seconds(c, grouped, range(2)), step_seconds(c, spread, range(2)))
        self.assertEqual(violations(c, grouped), [])

    def test_equal_candidates_scores_and_no_failure_winners(self):
        c, _ = fixture()
        result = matched_search(c, [[], []], self.config())
        other = result['unaware_result']
        self.assertEqual(result['candidates'], 12)
        self.assertEqual(result['candidate_pool_sha256'], other['candidate_pool_sha256'])
        self.assertEqual(result['plan'], other['plan'])
        self.assertEqual(result['candidates'], other['candidates'])
        for item in result['candidate_audit']:
            self.assertEqual(item['aware_score'], item['unaware_score'])
            self.assertEqual(violations(c, Plan.from_dict(item['plan'])), [])

    def test_proposals_and_unaware_choice_do_not_use_failure_rates(self):
        c, _ = fixture()
        a = matched_search(c, [[], []], self.config())
        risky = replace(c, nodes=[replace(n, failure_per_hour=100.) for n in c.nodes], domain_failure_per_hour=50.)
        b = matched_search(risky, [[Event(.5, 0, 0), Event(2., 1, 0)]], self.config())
        self.assertEqual(a['candidate_pool_sha256'], b['candidate_pool_sha256'])
        self.assertEqual(a['unaware_result']['plan'], b['unaware_result']['plan'])

    def test_interrupted_candidate_is_excluded_from_both_objectives(self):
        c, _ = fixture()
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise BudgetExpired()
            return simulate(*args, **kwargs)
        with patch('learnedplacement.planners.simulate', side_effect=interrupted):
            result = matched_search(c, [[], []], self.config())
        self.assertEqual(result['status'], 'wall_clock_limit')
        self.assertEqual(result['candidates'], 1)
        self.assertEqual(result['unaware_result']['candidates'], 1)
        self.assertEqual(len(result['candidate_audit']), 1)
        self.assertEqual(result['plan'], result['candidate_audit'][0]['plan'])

    def test_backups_can_help_when_all_primary_state_is_lost(self):
        c, p = fixture()
        trace = sorted([Event(3.5, 0, n) for n in range(4)] + [Event(5.5, 1, n) for n in range(4)])
        protected = Plan(p.groups, ((0, 4), (1, 5)))
        self.assertLess(simulate(c, protected, trace)['completion_s'], simulate(c, p, trace)['completion_s'])


class V2RunnerTests(unittest.TestCase):
    def config(self):
        c = test_simulator.RunnerTests().config()
        c.update(methods=['speed_grouped', 'matched_failure_aware', 'matched_failure_unaware', 'matched_no_backup'],
                 matched_candidates=4, matched_seconds=2., profiles=[{'name':'reference'}, {'name':'tight', 'deadline_factor':1.15}])
        return c

    def test_profiles_preserve_hardware_and_failure_seed_key(self):
        cases = make_cases(self.config())
        self.assertEqual(len(cases), 2)
        a, b = cases
        self.assertEqual(a.nodes, b.nodes)
        self.assertEqual(a.trace_key, b.trace_key)
        self.assertNotEqual(a.case_id, b.case_id)
        self.assertLess(b.deadline_s, a.deadline_s)

    def test_dependency_and_duplicate_profile_validation(self):
        c = self.config()
        c['methods'].reverse()
        with self.assertRaisesRegex(ValueError, 'dependencies'):
            validate_config(c)
        c = self.config()
        c['profiles'] *= 2
        with self.assertRaisesRegex(ValueError, 'Profile names'):
            validate_config(c)

    def test_v2_resume_diagnostics_ablation_and_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            c = self.config()
            out = Path(temp)
            self.assertEqual(run(config=c, output=out, max_minutes=1)['state'], 'finished')
            manifest = read_json(out/'manifest.json')
            for cid in manifest['cases']:
                d = out/'cases'/cid
                plans = read_json(d/'plans.json')
                self.assertEqual(plans['matched_no_backup']['plan']['groups'], plans['matched_failure_aware']['plan']['groups'])
                self.assertEqual(plans['matched_no_backup']['plan']['backups'], [])
                for method in manifest['methods']:
                    s = read_json(d/f'{method}.summary.json')
                    self.assertAlmostEqual(s['failure_penalty_s'], s['mean_completion_s']-s['failure_free_completion_s'])
                    self.assertTrue((d/f'{method}.failure_free.json').exists())
            report = paired_report(out, 'matched_failure_aware', 'matched_failure_unaware', repeats=20)
            self.assertEqual(len(report), 2)
            for r in report:
                self.assertTrue(r['matched_candidate_pool'])
                self.assertEqual(r['matched_candidates'], 4)
                self.assertAlmostEqual(r['time_difference_s'], r['failure_free_time_difference_s']+r['failure_penalty_difference_s'])
            saved = (out/'cases'/manifest['cases'][0]/'plans.json').read_bytes()
            run(config=c, output=out, max_minutes=1)
            self.assertEqual(saved, (out/'cases'/manifest['cases'][0]/'plans.json').read_bytes())

    def test_default_v2_config_has_48_valid_cases(self):
        c = read_json(ROOT/'configs'/'pilot_v2.json')
        validate_config(c)
        self.assertEqual(len(make_cases(c)), 48)
