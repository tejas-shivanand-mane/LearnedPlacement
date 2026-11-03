"""Frozen v5 models, guided search and CPU process parallelism; no training.

Each worker owns a separate run directory. Candidate summaries and evaluation
traces are checkpointed. No search method receives held-out failure traces.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import zipfile
import numpy as np

from .expanded import Scenario, Decision, make_scenarios, simulate, violations
from .expanded_planners import base_plan, neighbors, random_plan, digest
from .expanded_runner import run as evaluation_run, validate_config as validate_v4
from .exports import analysis_bundle
from .learning import Network, FEATURE_VERSION, action_pool, features, matrix, objective
from .model import seed_for
from .metrics import summarize
from .runner import ROOT, atomic_json, read_json, source_snapshot
from .simulator import BudgetExpired, check_budget


def method_specs(config):
    g=config['guided']; specs={'pool_search':dict(kind='pool')}
    for k in g['top_k']: specs[f'random_top{k}']=dict(kind='random',k=k)
    for seed in config['learning']['seeds']:
        specs[f'rl_{seed}']=dict(kind='direct',seed=seed)
        for k in g['top_k']: specs[f'rl_{seed}_top{k}']=dict(kind='ranked',seed=seed,k=k)
    # The original portfolio pipeline has at most 1+24+64+64 evaluations.
    specs['local_reference']=dict(kind='local',seed=None,cap=1+config['tuned_candidates']+2*config['search_candidates'],seconds=None)
    for label,values in [('c',g['candidate_budgets']),('s',g['seconds_budgets'])]:
        for value in values:
            budget=dict(cap=value if label=='c' else None,seconds=value if label=='s' else None)
            specs[f'portfolio_local_{label}{value}']=dict(kind='local',seed=None,**budget)
            for seed in config['learning']['seeds']:
                specs[f'rl_{seed}_local_{label}{value}']=dict(kind='local',seed=seed,**budget)
    return specs


def validate_config(c):
    validate_v4(dict(c,experiment='expanded_v4',methods=['local_search']))
    if c.get('experiment')!='guided_v6':raise ValueError('Expected guided_v6')
    if c['split'] not in ('validation','test'):raise ValueError('Select validation or test, separately')
    if c['instance_splits']!=['validation','test']:raise ValueError('Expected separate fresh validation and test instances')
    g=c['guided']
    for key in ('top_k','candidate_budgets'):
        if not g[key] or any(type(n) is not int or n<=0 for n in g[key]) or len(set(g[key]))!=len(g[key]):raise ValueError('Invalid '+key)
    if any(n>c['learning']['action_count'] for n in g['top_k']):raise ValueError('top_k exceeds pool size')
    if not g['seconds_budgets'] or any(not math.isfinite(n) or n<=0 for n in g['seconds_budgets']):raise ValueError('Invalid seconds budgets')
    if not c['learning']['seeds'] or len(set(c['learning']['seeds']))!=len(c['learning']['seeds']):raise ValueError('Invalid seeds')
    if c['methods']!=list(method_specs(c)):raise ValueError('Regenerate methods with method_specs after editing budgets/seeds')
    if any(type(v) is not int or v<int(k) for k,v in c.get('available_nodes',{}).items()):raise ValueError('Invalid available_nodes')


def find_model_run(results):
    """Resolve a unique completed model set, never select by evaluation quality."""
    found={}
    for path in sorted(Path(results).glob('*/learning/frozen_models.json')):
        run=path.parent.parent
        try:
            m=read_json(run/'manifest.json')
            if m.get('experiment')=='learning_v5':
                frozen=read_json(path)
                if all(read_json(run/'learning'/f'seed-{seed}.json')['episode']==m['config']['learning']['episodes'] for seed in m['config']['learning']['seeds']):
                    found.setdefault(digest(frozen),run)
        except (OSError,KeyError,ValueError):continue
    if len(found)!=1:
        raise ValueError('Set SOURCE_RUN explicitly to your completed v5 results folder. Found distinct model sets: '+str(list(map(str,found.values()))))
    return next(iter(found.values()))


def load_source(source,c):
    source=Path(source).resolve();m=read_json(source/'manifest.json')
    if m.get('experiment')!='learning_v5':raise ValueError('SOURCE_RUN must be a v5 run folder, not an analysis ZIP')
    for name in ('learnedplacement/learning.py','learnedplacement/expanded.py','learnedplacement/metrics.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=m['sources'].get(name):
            raise ValueError('Source model uses incompatible simulator/features: '+name)
    frozen=read_json(source/'learning/frozen_models.json')
    summary=read_json(source/'learning_summary.json')
    if summary['feature_version']!=FEATURE_VERSION:raise ValueError('Incompatible feature version')
    if c['seed']==m['config']['seed']:raise ValueError('Use a fresh evaluation seed')
    if c['learning']['penalties']!=m['config']['learning']['penalties']:raise ValueError('Keep the trained objective unchanged')
    actors={}
    for seed in c['learning']['seeds']:
        state=read_json(source/'learning'/f'seed-{seed}.json')
        if state['episode']!=m['config']['learning']['episodes'] or digest(state['actor'])!=frozen[str(seed)]['actor']:
            raise ValueError('Incomplete or modified frozen policy')
        Network.from_dict(state['actor'])
        actors[str(seed)]=state['actor']
    return dict(feature_version=FEATURE_VERSION,actors=actors,source_fingerprint=m['fingerprint'],
                source_config=m['config'],source_training_cases=m['training_cases'],source_learning=summary)


def prepare(c,output,source_run=None):
    validate_config(c);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    snapshot=output/'frozen_policies.json'
    if snapshot.exists():
        bundle=read_json(snapshot)
        if source_run is not None and digest(load_source(source_run,c))!=digest(bundle):raise ValueError('Source models changed')
    else:
        if source_run is None:raise ValueError('SOURCE_RUN required on first invocation')
        bundle=load_source(source_run,c)
    sources=source_snapshot();hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
    fingerprint=digest(dict(config=c,sources=hashes,models=digest(bundle),numpy=np.__version__))
    path=output/'manifest.json'
    if path.exists():
        m=read_json(path)
        if m['fingerprint']!=fingerprint:raise ValueError('Configuration/source/models changed: choose a NEW output directory')
        return m
    scenarios=[s for s in make_scenarios(c) if s.split==c['split']]
    # Fresh seeds prevent reuse of previous traces/heterogeneous hardware draws.
    # Homogeneous instances can still be identical: this is not IID/OOD proof.
    for s in scenarios:
        p=base_plan(s,c)
        for actor in bundle['actors'].values():
            if Network.from_dict(actor).params['w'].shape[0]!=len(features(s,p)):raise ValueError('Feature dimension mismatch')
        atomic_json(output/'scenario_inputs'/f'{s.workload.case_id}.json',s.to_dict())
    for name,content in sources.items():
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(content)
    m=dict(schema=6,experiment='guided_v6',fingerprint=fingerprint,config=c,sources=hashes,numpy=np.__version__,
           cases=[s.workload.case_id for s in scenarios],methods=c['methods'],created_unix_s=time.time(),
           scenario_inputs_sha256=digest([s.to_dict() for s in scenarios]),frozen_policies_sha256=digest(bundle),
           model_hashes={seed:digest(actor) for seed,actor in bundle['actors'].items()},source_fingerprint=bundle['source_fingerprint'],
           source_config=bundle['source_config'],method_specs=method_specs(c),
           split_contract='Frozen existing v5 policies. Fresh hardware seed and planning/evaluation trace namespace. Validation and test run separately; all profiles share instance split. Larger sizes are extrapolation.',
           caveat='Synthetic simulator, not real training. Fixed 3 failure domains at all sizes. Budgeted search uses planning traces only. No guarantees of optimality or deadline/cost feasibility.')
    atomic_json(snapshot,bundle);atomic_json(output/'learning_summary.json',bundle['source_learning']);atomic_json(path,m)
    return m


class SearchStopped(Exception):pass


class Experiment:
    def __init__(self,root):
        self.root=Path(root);self.output=None
        bundle=read_json(self.root/'frozen_policies.json')
        m=read_json(self.root/'manifest.json')
        if digest(bundle)!=m['frozen_policies_sha256']:raise ValueError('Frozen policies changed')
        self.actors=bundle['actors'];self.models={int(k):Network.from_dict(v) for k,v in self.actors.items()}

    def prepare_worker(self,c,output):
        self.output=output;m=read_json(self.root/'manifest.json');cid=output.name
        scenario=read_json(self.root/'scenario_inputs'/f'{cid}.json')
        atomic_json(output/'cases'/cid/'case.json',scenario)
        worker=dict(m,cases=[cid]);path=output/'manifest.json'
        if path.exists() and read_json(path)['fingerprint']!=m['fingerprint']:raise ValueError('Worker configuration changed')
        atomic_json(path,worker);return worker

    def plan(self,s,method,traces,c,prior,stop_at):
        spec=method_specs(c)[method];path=self.output/'search'/f'{method}.json'
        state=read_json(path) if path.exists() else dict(rows=[],elapsed_s=0.,simulation_attempts=0)
        start=time.monotonic();previous=state['elapsed_s']
        cap=spec.get('cap');seconds=spec.get('seconds')
        deadline=None if seconds is None else start+max(0.,seconds-previous)
        def elapsed():return previous+time.monotonic()-start
        def persist():
            state['elapsed_s']=elapsed();atomic_json(path,state)
        def limits():
            check_budget(stop_at)
            if deadline is not None and time.monotonic()>=deadline:raise SearchStopped('wall_clock_budget')
        cache={digest(r['plan']):r for r in state['rows']}
        def score(r):return (objective(s.workload,r['summary'],c['learning']['penalties']),r['summary']['restricted_mean_s'])
        best=min(state['rows'],key=score) if state['rows'] else None
        fallback=base_plan(s,c,domain=False)
        pool_hash=None;pool_size=None;ranked_indices=None;model_hash=None;initialization=None;pool=None;indices=None
        reason='complete';rng=random.Random(seed_for(c['seed'],s.workload.case_id,'local-search',spec.get('seed')))
        def evaluate(p):
            nonlocal best
            key=digest(p.to_dict())
            if key in cache:return cache[key]
            limits()
            if cap is not None and len(state['rows'])>=cap:raise SearchStopped('candidate_budget')
            rows=[]
            limit=stop_at if deadline is None else deadline if stop_at is None else min(deadline,stop_at)
            try:
                for trace in traces:
                    state['simulation_attempts']+=1
                    rows.append(simulate(s,p,trace,limit))
            except BudgetExpired:
                check_budget(stop_at)  # global interruption resumes; per-method timeout finalizes
                raise SearchStopped('wall_clock_budget')
            r=dict(plan=p.to_dict(),summary=summarize(s.workload,rows))
            cache[key]=r;state['rows'].append(r)
            if best is None or score(r)<score(best):best=r
            persist();return r
        def explore(initial,quota,shuffle):
            current=evaluate(initial);p=Decision.from_dict(current['plan']);seen={p};used=1
            while used<quota:
                limits();items=[]
                for q in neighbors(s,p,c):
                    limits();items.append(q)
                if shuffle:rng.shuffle(items)
                improved=False
                for q in items:
                    limits()
                    if q in seen:continue
                    seen.add(q);r=evaluate(q);used+=1
                    if score(r)<score(current):current=r;p=q;improved=True;break
                    if used>=quota:return current
                if improved:continue
                if not shuffle:return current
                for _ in range(max(100,quota*50)):
                    limits();q=random_plan(s,c,rng)
                    if q is not None and q not in seen:break
                else:return current
                seen.add(q);r=evaluate(q);used+=1
                if score(r)<score(current):current=r;p=q
            return current
        try:
            if spec['kind']!='local' or spec.get('seed') is not None:
                limits();pool=action_pool(s,c);limits()
                pool_hash=digest([p.to_dict() for p in pool]);pool_size=len(pool)
                indices=list(range(len(pool)))
                if spec.get('seed') is not None:
                    seed=spec['seed'];scores,_=self.models[seed].forward(matrix(s,pool))
                    indices=sorted(indices,key=lambda i:(-float(scores[i]),i));model_hash=digest(self.actors[str(seed)])
                elif spec['kind']=='random':
                    # Same permutation for k=4 and k=8 gives nested random controls.
                    random.Random(seed_for(c['seed'],s.workload.case_id,'random-subset')).shuffle(indices)
                limits();fallback=pool[indices[0]]
                if spec['kind']=='direct':ranked_indices=indices
                elif spec['kind'] in ('pool','random','ranked'):
                    ranked_indices=indices[:spec.get('k',len(indices))]
                    for i in ranked_indices:evaluate(pool[i])
                else:
                    initialization='frozen_rl';explore(fallback,cap or 100000,True)
            else:
                initialization='heuristic_portfolio'
                speed=evaluate(fallback);base=base_plan(s,c);tuned=evaluate(base)
                intervals=[v for v in c['choices']['checkpoint_every'] if v!=base.checkpoint_every]
                for i in range(c['tuned_candidates']-1):
                    limits()
                    q=replace(base,checkpoint_every=intervals[i]) if i<len(intervals) else random_plan(s,c,rng,base)
                    if q is not None:
                        r=evaluate(q)
                        if score(r)<score(tuned):tuned=r
                coord=explore(Decision.from_dict(tuned['plan']),c['search_candidates'],False)
                initial=min([speed,tuned,coord],key=score)
                # Count budget applies to entire pipeline, including initialization.
                explore(Decision.from_dict(initial['plan']),c['search_candidates'] if method=='local_reference' else cap or 100000,True)
        except SearchStopped as e:reason=str(e)
        except (BudgetExpired,KeyboardInterrupt):
            persist();raise
        persist()
        selected=best['plan'] if best else fallback.to_dict()
        if violations(s,Decision.from_dict(selected)):raise AssertionError('Invalid selected plan')
        return dict(plan=selected,planning_s=state['elapsed_s'],candidates=len(state['rows']),
                    simulations=len(state['rows'])*len(traces),simulation_attempts=state['simulation_attempts'],
                    search_summary=best['summary'] if best else None,planning_point_feasible=best['summary']['point_estimate_feasible'] if best else None,
                    status=reason,planning_time_kind='measured_total_including_initialization_and_resume_overhead',
                    candidate_pool_sha256=pool_hash,pool_size=pool_size,model_sha256=model_hash,
                    ranked_actions=ranked_indices,initial_pool_action=indices[0] if indices else None,
                    selected_pool_action=next((i for i,p in enumerate(pool) if p.to_dict()==selected),None) if pool is not None else None,
                    initialization=initialization,budget_candidates=cap,budget_seconds=seconds,
                    budget_overrun_s=max(0.,state['elapsed_s']-seconds) if seconds is not None else 0.,
                    unscored_fallback=best is None and spec['kind']!='direct',candidate_audit=state['rows'])


def _worker(root,cid,remaining_minutes):
    root=Path(root);c=read_json(root/'manifest.json')['config'];experiment=Experiment(root)
    import contextlib
    directory=root/'partitions'/cid;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'worker.log').open('a') as log,contextlib.redirect_stdout(log):
        return evaluation_run(config=c,output=directory,max_minutes=remaining_minutes,
                              prepare_fn=experiment.prepare_worker,plan_fn=experiment.plan,trace_namespace='v6',reuse_evaluations=True)


def status(output):
    output=Path(output);path=output/'manifest.json'
    if not path.exists():return dict(state='not_started')
    m=read_json(path);done=sum((output/'partitions'/cid/'cases'/cid/f'{method}.summary.json').exists() for cid in m['cases'] for method in m['methods'])
    total=len(m['cases'])*len(m['methods'])
    return dict(state='finished' if done==total else 'incomplete',completed_comparisons=done,total_comparisons=total,fraction_complete=done/total)


def build_summary(output):
    output=Path(output);m=read_json(output/'manifest.json');results=[]
    for cid in m['cases']:
        s=Scenario.from_dict(read_json(output/'scenario_inputs'/f'{cid}.json'))
        for method in m['methods']:
            p=output/'partitions'/cid/'cases'/cid/f'{method}.summary.json'
            if p.exists():
                selection=read_json(p.parent/'plans.json')[method]
                results.append(dict(case_id=cid,split=s.split,profile=s.workload.profile,regime=s.workload.regime,
                                    available_nodes=len(s.workload.nodes),reference_nodes=s.workload.shards*s.workload.replicas,
                                    method=method,simulations=selection['simulations'],unscored_fallback=selection['unscored_fallback'],**read_json(p)))
    atomic_json(output/'summary.json',dict(results=results,status=status(output)))
    if results:
        import csv,io
        flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]
        buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=sorted(set().union(*(r.keys() for r in flat))));writer.writeheader();writer.writerows(flat)
        (output/'summary.csv').write_text(buf.getvalue())
    return results


def run(config=None,output='results/pilot_v6_small_validation',source_run=None,workers=1,max_minutes=None):
    """Unlimited by default. Independent case processes, one lock per root run."""
    if type(workers) is not int or workers<1:raise ValueError('workers must be a positive integer')
    if max_minutes is not None and (not math.isfinite(max_minutes) or max_minutes<=0):raise ValueError('Invalid optional time limit')
    c=config if config is not None else read_json(ROOT/'configs/pilot_v6.json')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);lock=output/'.run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Run locked; confirm old workers stopped before removing .run.lock.') from None
    os.write(fd,str(os.getpid()).encode());os.close(fd);prepared=False
    stop=None if max_minutes is None else time.monotonic()+max_minutes*60
    try:
        m=prepare(c,output,source_run);prepared=True
        execution_path=output/'execution.json'
        if execution_path.exists() and read_json(execution_path)['workers']!=workers:
            raise ValueError('Worker count changed; choose a NEW output directory for comparable timings')
        if not execution_path.exists():
            import platform
            atomic_json(execution_path,dict(workers=workers,cpu_count=os.cpu_count(),platform=platform.platform(),python=platform.python_version(),blas_threads_in_workers=1,backend='CPU processes',time_limit_minutes=max_minutes))
        raw=[read_json(output/'scenario_inputs'/f'{cid}.json') for cid in m['cases']]
        if digest(raw)!=m['scenario_inputs_sha256']:raise ValueError('Scenario inputs changed')
        pending=[cid for cid in m['cases'] if any(not (output/'partitions'/cid/'cases'/cid/f'{method}.summary.json').exists() for method in m['methods'])]
        # Avoid nested BLAS threads competing with simulator processes.
        old={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        try:
            for key in old:os.environ[key]='1'
            if workers==1:
                for cid in pending:
                    check_budget(stop);remaining=None if stop is None else max(.000001,(stop-time.monotonic())/60)
                    print('Running case:',cid,flush=True)
                    _worker(str(output),cid,remaining);build_summary(output)
            elif pending:
                import multiprocessing as mp
                with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
                    # Submit at most workers cases; queued tasks do not receive stale deadlines.
                    todo=iter(pending);active={}
                    def submit():
                        check_budget(stop)
                        cid=next(todo,None)
                        if cid is not None:
                            remaining=None if stop is None else max(.000001,(stop-time.monotonic())/60)
                            print('Running case:',cid,flush=True)
                            active[pool.submit(_worker,str(output),cid,remaining)]=cid
                    for _ in range(min(workers,len(pending))):submit()
                    while active:
                        future=next(as_completed(active));cid=active.pop(future);future.result()
                        print('Completed case:',cid,flush=True);build_summary(output);submit()
        finally:
            for key,value in old.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    except BudgetExpired:print('Paused at optional time limit; rerun unchanged to resume.',flush=True)
    finally:
        try:
            if prepared:build_summary(output)
        finally:lock.unlink(missing_ok=True)
    return status(output)


def export(output):
    output=Path(output).resolve();m=read_json(output/'manifest.json');cases=[]
    for cid in m['cases']:
        part=output/'partitions'/cid
        if (part/'manifest.json').exists():cases.extend(analysis_bundle(part)['cases'])
        else:cases.append(dict(case=read_json(output/'scenario_inputs'/f'{cid}.json'),methods={},paired_comparisons=[]))
    for case in cases:
        pools={r['selection']['candidate_pool_sha256'] for r in case['methods'].values() if r.get('selection',{}).get('candidate_pool_sha256') is not None}
        if len(pools)>1:raise ValueError('Mismatched common action pools')
        for method,r in case['methods'].items():
            spec=m['method_specs'][method]
            if spec.get('seed') is not None and r.get('selection',{}).get('model_sha256') is not None:
                if r['selection']['model_sha256']!=m['model_hashes'][str(spec['seed'])]:raise ValueError('Mismatched frozen model')
    bundle=dict(format='learnedplacement-analysis',schema=1,manifest=m,learning=read_json(output/'learning_summary.json'),
                status=status(output),cases=cases,execution=read_json(output/'execution.json') if (output/'execution.json').exists() else None,interpretation=dict(
                    difference_direction='selected minus baseline; negative time/cost favors selected',
                    intervals='Paired normal 95% intervals, not multiplicity corrected; hardware profiles are dependent.',
                    timing='Total measured planning, including initialization, features, proposals and resume overhead. Cold model load/shared trace generation excluded. Parallel CPU contention affects latency.',
                    budgets='Candidate budgets count unique fully evaluated plans across initialization and local search. Each uses planning_traces rollouts. Attempted simulations include interrupted work. Wall budgets can overshoot during indivisible operations.',
                    scope='Synthetic workload; fixed three failure domains. Larger sizes extrapolate frozen small-instance policies. No guarantee of optimality or constraint satisfaction.'))
    target=output.with_suffix('.analysis.zip');temp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:z.writestr('analysis.json',json.dumps(bundle,separators=(',',':')))
    os.replace(temp,target);return target


def plot_results(output,profile='reference',available_nodes=None):
    """Paired complete-case dots and size curves, with no error bars/captions."""
    import pandas as pd
    import matplotlib.pyplot as plt
    output=Path(output);df=pd.DataFrame(read_json(output/'summary.json')['results'])
    if df.empty:return df
    methods=read_json(output/'manifest.json')['methods'];complete=df.groupby('case_id').method.nunique()
    df=df[df.case_id.isin(complete[complete==len(methods)].index)]
    if profile is not None:df=df[df.profile==profile]
    if available_nodes is not None:df=df[df.available_nodes==available_nodes]
    if df.empty:print('No complete matched cases for this filter.');return df
    avg=df.groupby(['available_nodes','method']).agg(execution_s=('restricted_mean_s','mean'),planning_s=('planning_s','mean'),cost=('mean_observed_cost_usd','mean'),screen_rate=('statistical_screen_passed','mean'),simulations=('simulations','mean'))
    if available_nodes is not None:
        table=avg.droplevel(0).reindex(methods);fig,axes=plt.subplots(1,2,figsize=(12,max(5,.3*len(methods))),layout='constrained')
        for ax,col,title in zip(axes,['execution_s','planning_s'],['Observed execution time (s)','Planning time (s)']):ax.scatter(table[col],table.index);ax.set_title(title);ax.grid(axis='x',alpha=.2)
    else:
        fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
        for method in methods:
            sub=avg.xs(method,level='method')
            for ax,col in zip(axes,['execution_s','planning_s']):ax.plot(sub.index,sub[col],marker='o',label=method)
        for ax,title in zip(axes,['Observed execution time (s)','Planning time (s)']):ax.set_xlabel('Available machines');ax.set_title(title);ax.grid(alpha=.2)
        axes[1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1))
    plt.show();return avg
