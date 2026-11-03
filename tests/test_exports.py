import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from learnedplacement.runner import atomic_json, export_run


class ExportTests(unittest.TestCase):
    def fixture(self, parent):
        out = Path(parent)/'run'
        d = out/'cases'/'tiny'
        atomic_json(out/'manifest.json', dict(cases=['tiny'], methods=['a', 'b'], config={'evaluation_traces':3}))
        atomic_json(out/'status.json', {'state':'finished'})
        atomic_json(d/'case.json', {'case_id':'tiny'})
        atomic_json(d/'plans.json', {m:dict(plan={'groups':[[0]],'backups':[]}, planning_s=.1,
                    candidate_pool_sha256='same', candidate_audit=[{'large':'unneeded'}], unaware_result={'duplicate':True}) for m in ['a','b']})
        for m, times in [('a',[9.,8.,7.]),('b',[10.,10.,10.])]:
            atomic_json(d/f'{m}.observations.json', [dict(trace_index=i,trace_seed=i+100,completed=True,
                        observed_s=t,cost_usd=t/100,deadline_hit=True,unneeded_detail='raw') for i,t in enumerate(times)])
            atomic_json(d/f'{m}.summary.json', {'mean_completion_s':sum(times)/3})
            atomic_json(d/f'{m}.failure_free.json', {'completion_s':6.})
        return out

    def read_export(self, out):
        path = export_run(out)
        with zipfile.ZipFile(path) as z:
            self.assertEqual(z.namelist(), ['analysis.json'])
            return json.loads(z.read('analysis.json'))

    def test_default_compact_preserves_local_raw_and_paired_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            out = self.fixture(temp)
            before = {str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            b = self.read_export(out)
            after = {str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            case = b['cases'][0]
            selection = case['methods']['a']['selection']
            self.assertNotIn('candidate_audit', selection)
            self.assertNotIn('unaware_result', selection)
            self.assertIn('candidate_pool_sha256', selection)
            pair = case['paired_comparisons'][0]
            self.assertEqual(pair['time_difference']['mean'], -2.)
            self.assertEqual(pair['time_difference']['sample_variance'], 1.)
            self.assertEqual(pair['failure_penalty_difference_s'], -2.)
            self.assertTrue(pair['full_evaluation'])
            with zipfile.ZipFile(export_run(out, full=True)) as z:
                self.assertIn('cases/tiny/a.observations.json',z.namelist())
                self.assertIn('candidate_audit',z.read('cases/tiny/plans.json').decode())

    def test_partial_censored_export_keeps_restricted_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.fixture(temp)
            p=out/'cases/tiny/a.observations.json'
            rows=json.loads(p.read_text())[:2]
            rows[0]['completed']=False
            atomic_json(p,rows)
            pair=self.read_export(out)['cases'][0]['paired_comparisons'][0]
            self.assertEqual(pair['paired_traces'],2)
            self.assertFalse(pair['full_evaluation'])
            self.assertFalse(pair['all_completed'])
            self.assertEqual(pair['time_quantity'],'restricted_observed_s')
            self.assertNotIn('failure_penalty_difference_s',pair)
            self.assertEqual(pair['time_difference']['mean'],-1.5)

    def test_mismatched_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.fixture(temp)
            p=out/'cases/tiny/a.observations.json'
            rows=json.loads(p.read_text()); rows[1]['trace_seed']+=1
            atomic_json(p,rows)
            with self.assertRaisesRegex(ValueError,'Paired trace mismatch'):
                export_run(out)

    def test_old_run_without_failure_free_data_exports_without_simulations(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.fixture(temp)
            for p in out.rglob('*.failure_free.json'): p.unlink()
            pair=self.read_export(out)['cases'][0]['paired_comparisons'][0]
            self.assertNotIn('failure_penalty_difference_s',pair)
