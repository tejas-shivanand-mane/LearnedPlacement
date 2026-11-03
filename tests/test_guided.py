import contextlib,copy,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from learnedplacement.runner import read_json,ROOT
from learnedplacement.learning_runner import run as train_run,methods
from learnedplacement.expanded import make_scenarios
from learnedplacement.guided_runner import run,prepare,Experiment,method_specs,export,validate_config,find_model_run
from learnedplacement.simulator import BudgetExpired
import test_learning


def setup(root):
    source=root/'v5';c=test_learning.config();c['learning']['seeds']=[11];c['methods']=methods(c)
    with contextlib.redirect_stdout(io.StringIO()):train_run(c,source,max_minutes=None)
    g=copy.deepcopy(c);g.update(experiment='guided_v6',seed=c['seed']+1,instances_per_regime=2,instance_splits=['validation','test'],split='validation')
    g['guided']=dict(top_k=[1,2],candidate_budgets=[3],seconds_budgets=[.02]);g['methods']=list(method_specs(g))
    return source,g


class GuidedTests(unittest.TestCase):
    def test_end_to_end_frozen_budgets_export_resume(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);source,c=setup(root);out=root/'v6';self.assertEqual(find_model_run(root),source)
            before=(source/'learning/seed-11.json').read_bytes()
            self.assertEqual(run(c,out,source)['state'],'finished')
            m=read_json(out/'manifest.json');cid=m['cases'][0];plans=read_json(out/'partitions'/cid/'cases'/cid/'plans.json')
            self.assertEqual(plans['rl_11']['simulations'],0)
            for name,k in [('random_top1',1),('rl_11_top1',1),('random_top2',2),('rl_11_top2',2)]:
                self.assertEqual(plans[name]['candidates'],k);self.assertEqual(plans[name]['simulations'],k*c['planning_traces'])
                self.assertEqual(plans[name]['candidate_pool_sha256'],plans['pool_search']['candidate_pool_sha256'])
            for name in ['portfolio_local_c3','rl_11_local_c3']:self.assertLessEqual(plans[name]['candidates'],3)
            self.assertEqual(plans['random_top1']['ranked_actions'],plans['random_top2']['ranked_actions'][:1])
            self.assertEqual(before,(source/'learning/seed-11.json').read_bytes());self.assertEqual(run(c,out,None)['state'],'finished')
            # Recover observations even if a forced stop lost the shared cache.
            import shutil
            directory=out/'partitions'/cid/'cases'/cid;method='rl_11'
            original=read_json(directory/f'{method}.observations.json')
            (directory/f'{method}.summary.json').unlink()
            (directory/f'{method}.observations.json').write_text(json.dumps(original[:1]))
            shutil.rmtree(directory/'evaluation_cache')
            self.assertEqual(run(c,out,None)['state'],'finished')
            self.assertEqual(read_json(directory/f'{method}.observations.json'),original)
            with zipfile.ZipFile(export(out)) as z:bundle=json.loads(z.read('analysis.json'))
            self.assertEqual(bundle['status']['state'],'finished');self.assertEqual(len(bundle['cases']),1)
            self.assertNotIn('actors',bundle['learning'])
            c['guided']['candidate_budgets']=[4];c['methods']=list(method_specs(c))
            with self.assertRaisesRegex(ValueError,'NEW output'):run(c,out,None)

    def test_interrupted_search_preserves_completed_candidates(self):
        import learnedplacement.guided_runner as module
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);source,c=setup(root);records=[]
            for idx in range(2):
                out=root/str(idx);m=prepare(c,out,source);cid=m['cases'][0];s=next(s for s in make_scenarios(c) if s.split=='validation')
                e=Experiment(out);e.prepare_worker(c,out/'partitions'/cid)
                if idx==0:
                    original=module.simulate;calls=0
                    def interrupt(*args,**kwargs):
                        nonlocal calls
                        calls+=1
                        if calls==3:raise BudgetExpired()
                        return original(*args,**kwargs)
                    original_check=module.check_budget
                    def check(deadline):
                        if calls>=3:raise BudgetExpired()
                        return original_check(deadline)
                    with patch.object(module,'simulate',side_effect=interrupt),patch.object(module,'check_budget',side_effect=check):
                        with self.assertRaises(BudgetExpired):e.plan(s,'pool_search',[[],[]],c,{},None)
                    self.assertEqual(len(read_json(e.output/'search/pool_search.json')['rows']),1)
                records.append(e.plan(s,'pool_search',[[],[]],c,{},None))
            for key in ['plan','candidates','simulations','candidate_pool_sha256','search_summary']:self.assertEqual(records[0][key],records[1][key])
            self.assertGreater(records[0]['simulation_attempts'],records[1]['simulation_attempts'])

    def test_tampered_models_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);source,c=setup(root);out=root/'v6';prepare(c,out,source)
            path=out/'frozen_policies.json';b=read_json(path);b['actors']['11']['params']['a'][0]+=1;path.write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'Frozen policies changed'):Experiment(out)
            with self.assertRaisesRegex(ValueError,'NEW output'):run(c,out,None)

    def test_large_sizes_and_active_options(self):
        c=read_json(ROOT/'configs/pilot_v6.json');c.update(active_nodes=[24,48,64],available_nodes={'24':32,'48':64,'64':96})
        c['choices']['replicas']=[1,2,4,6,8,12,16];c['methods']=list(method_specs(c));validate_config(c);scenarios=make_scenarios(c)
        self.assertEqual({len(s.workload.nodes) for s in scenarios},{32,64,96})
        self.assertEqual({s.workload.shards*s.workload.replicas for s in scenarios},{24,48,64})

    def test_parallel_matches_serial(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);source,c=setup(root);c['regimes']=['homogeneous','fast_risky'];a=root/'serial';b=root/'parallel'
            self.assertEqual(run(c,a,source,workers=1)['state'],'finished');self.assertEqual(run(c,b,source,workers=2)['state'],'finished')
            for cid in read_json(a/'manifest.json')['cases']:
                aa=read_json(a/'partitions'/cid/'cases'/cid/'plans.json');bb=read_json(b/'partitions'/cid/'cases'/cid/'plans.json')
                for method,spec in method_specs(c).items():
                    if spec.get('seconds') is not None:continue
                    self.assertEqual(aa[method]['plan'],bb[method]['plan']);self.assertEqual(aa[method]['search_summary'],bb[method]['search_summary'])
                    self.assertEqual(read_json(a/'partitions'/cid/'cases'/cid/f'{method}.observations.json'),read_json(b/'partitions'/cid/'cases'/cid/f'{method}.observations.json'))
