"""Run/resume the v4 CPU-only experiment; nothing executes on import."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

from .expanded import Scenario, Decision, make_scenarios, simulate, generate_trace, violations
from .expanded_planners import METHODS, POLICY_FIELDS, plan, spaces
from .metrics import summarize
from .model import seed_for
from .runner import ROOT, atomic_json, read_json, source_snapshot, status, export_run
from .simulator import BudgetExpired, check_budget


def validate_config(cfg):
    if cfg.get('experiment') != 'expanded_v4': raise ValueError('Expected expanded_v4 config')
    for key in ('seed','instances_per_regime','planning_traces','evaluation_traces','save_every_traces','search_candidates','tuned_candidates'):
        if type(cfg.get(key)) is not int or cfg[key] <= 0: raise ValueError(f'{key} must be a positive integer')
    if cfg['evaluation_traces'] < 2: raise ValueError('At least two evaluation traces required')
    if not math.isfinite(cfg['planner_seconds']) or cfg['planner_seconds'] <= 0: raise ValueError('Invalid planner_seconds')
    if len(cfg['instance_splits']) != cfg['instances_per_regime']: raise ValueError('One split per instance required')
    if any(x not in ('train','validation','test') for x in cfg['instance_splits']): raise ValueError('Unknown split')
    methods = cfg['methods']
    if not methods or len(set(methods)) != len(methods) or set(methods)-set(METHODS): raise ValueError('Invalid methods')
    if 'random_unaware' in methods and ('random_search' not in methods or methods.index('random_search') > methods.index('random_unaware')):
        raise ValueError('random_search must precede random_unaware')
    choice = cfg['choices']
    required = set(POLICY_FIELDS) | {'shards','replicas','backup_copies','spare_count'}
    if set(choice) != required: raise ValueError(f'Choices must be exactly {sorted(required)}')
    for key, values in choice.items():
        if not isinstance(values,list) or not values or len(set(values)) != len(values): raise ValueError(f'Invalid grid {key}')
    for key in ('shards','replicas','checkpoint_every','backup_every','checkpoint_retention','repair_every'):
        if any(type(v) is not int or v <= 0 for v in choice[key]): raise ValueError(f'Invalid {key}')
    for key in ('spare_count','backup_copies'):
        if any(type(v) is not int or v < 0 for v in choice[key]): raise ValueError(f'Invalid {key}')
    enums = dict(checkpoint_tier={'remote','local'},checkpoint_mode={'sync','async'},recovery={'eager','defer'},
                 recovery_order={'fastest','smallest'},source_policy={'balanced','first'})
    for key, allowed in enums.items():
        if set(choice[key])-allowed: raise ValueError(f'Invalid {key}')
    if any(not isinstance(v,(int,float)) or not 0 < v < 1 for v in choice['background_fraction']): raise ValueError('Invalid background_fraction')
    for key in ('checkpoint_every','backup_every','shards'):
        if cfg['workload'][key] not in choice[key]: raise ValueError(f'Reference {key} must be in grid')
    if not cfg['active_nodes'] or any(type(n) is not int or n <= 0 or n % cfg['workload']['shards'] for n in cfg['active_nodes']): raise ValueError('Invalid reference node counts')
    if any(n//cfg['workload']['shards'] not in choice['replicas'] for n in cfg['active_nodes']): raise ValueError('Reference replicas must be in grid')
    names = [p['name'] for p in cfg['expanded_profiles']]
    if not names or len(set(names)) != len(names) or any(not n.replace('_','').isalnum() for n in names): raise ValueError('Invalid profiles')
    for p in cfg['expanded_profiles']:
        if set(p)-{'name','description','storage_multiplier','network_multiplier','replacement_multiplier','environment'}:
            raise ValueError('Unknown profile key')
    baseline = Decision(((0,),),checkpoint_every=cfg['workload']['checkpoint_every'],backup_every=cfg['workload']['backup_every'])
    for key in POLICY_FIELDS:
        if getattr(baseline,key) not in choice[key]:
            raise ValueError(f'Reference policy {key} must be in grid')
    for key in ('backup_copies','spare_count'):
        if 0 not in choice[key]: raise ValueError(f'{key} must include zero for reference baseline')
    if len(set(cfg['regimes'])) != len(cfg['regimes']): raise ValueError('Duplicate regimes')


def prepare(cfg, output):
    validate_config(cfg)
    sources = source_snapshot()
    hashes = {name:hashlib.sha256(content.encode()).hexdigest() for name,content in sources.items()}
    fingerprint = hashlib.sha256(json.dumps(dict(config=cfg,sources=hashes),sort_keys=True).encode()).hexdigest()
    path = output/'manifest.json'
    if path.exists():
        m = read_json(path)
        if m['fingerprint'] != fingerprint: raise ValueError('Configuration or source changed. Choose a NEW output directory.')
        return m
    cases = make_scenarios(cfg)
    if not cases: raise ValueError('No scenarios')
    for s in cases:
        if not spaces(s,cfg): raise ValueError('No feasible resource counts')
        atomic_json(output/'cases'/s.workload.case_id/'case.json',s.to_dict())
    for name,content in sources.items():
        dest = output/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(content)
    manifest = dict(schema=4,experiment='expanded_v4',fingerprint=fingerprint,config=cfg,sources=hashes,
                    cases=[s.workload.case_id for s in cases],methods=cfg['methods'],python=platform.python_version(),
                    platform=platform.platform(),created_unix_s=time.time(),
                    caveat='Synthetic fixed-work surrogate. No real training, RL, exact oracle, or published-system reproduction.',
                    split_contract='Split by hardware instance; profiles share split. Future learning fits TRAIN only; reserve fresh final test seeds after exploratory analysis.')
    atomic_json(path,manifest)
    return manifest


def build_summary(output):
    output = Path(output)
    manifest = read_json(output/'manifest.json')
    results = []
    for cid in manifest['cases']:
        d = output/'cases'/cid
        s = Scenario.from_dict(read_json(d/'case.json'))
        for method in manifest['methods']:
            path = d/f'{method}.summary.json'
            if path.exists():
                results.append(dict(case_id=cid,regime=s.workload.regime,profile=s.workload.profile,
                                    split=s.split,reference_nodes=s.workload.shards*s.workload.replicas,
                                    method=method,**read_json(path)))
    atomic_json(output/'summary.json',dict(results=results,status=status(output)))
    if results:
        flat = [{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]
        stream = io.StringIO(); writer = csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader(); writer.writerows(flat)
        tmp = output/'summary.csv.tmp'; tmp.write_text(stream.getvalue()); os.replace(tmp,output/'summary.csv')
    return results


def run(config=None, output='results/pilot_v4', max_minutes=30, config_path=None,
        prepare_fn=None, plan_fn=None, before_evaluate=None, trace_namespace='v4'):
    if not isinstance(max_minutes,(int,float)) or not 0 < max_minutes <= 30:
        raise ValueError('max_minutes must be in (0,30]')
    prepare_fn = prepare_fn or prepare
    plan_fn = plan_fn or plan
    cfg = config if config is not None else read_json(config_path or ROOT/'configs/pilot_v4.json')
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    lock = output/'.run.lock'
    try: fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError('Run is locked. Confirm old runner stopped before removing .run.lock.') from None
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    stop_at = time.monotonic()+60*max_minutes
    current = {}
    prepared = False
    try:
        m = prepare_fn(cfg,output); prepared = True
        atomic_json(output/'status.json',dict(state='running'))
        if before_evaluate is not None:
            current = dict(stage='training')
            before_evaluate(cfg,output,stop_at)
        for cid in m['cases']:
            check_budget(stop_at)
            d = output/'cases'/cid
            s = Scenario.from_dict(read_json(d/'case.json')); c = s.workload
            selected_path = d/'plans.json'
            selected = read_json(selected_path) if selected_path.exists() else {}
            planning_traces, eval_cache = None,{}
            for method in m['methods']:
                current = dict(case_id=cid,method=method)
                if (d/f'{method}.summary.json').exists(): continue
                check_budget(stop_at)
                if method not in selected:
                    if planning_traces is None and method != 'random_unaware':
                        planning_traces = [generate_trace(s,seed_for(cfg['seed'],c.trace_key,trace_namespace+'-planning',i),stop_at) for i in range(cfg['planning_traces'])]
                    print(f'{cid}: planning {method}',flush=True)
                    record = plan_fn(s,method,planning_traces,cfg,selected,stop_at)
                    check_budget(stop_at)
                    selected[method] = record; atomic_json(selected_path,selected)
                p = Decision.from_dict(selected[method]['plan'])
                if violations(s,p): raise AssertionError('Planner returned infeasible plan')
                path = d/f'{method}.failure_free.json'
                if not path.exists(): atomic_json(path,simulate(s,p,[],stop_at))
                healthy = read_json(path)
                path = d/f'{method}.observations.json'
                rows = read_json(path) if path.exists() else []
                print(f'{cid}: evaluating {method} ({len(rows)}/{cfg["evaluation_traces"]})',flush=True)
                try:
                    for i in range(len(rows),cfg['evaluation_traces']):
                        check_budget(stop_at)
                        seed = seed_for(cfg['seed'],c.trace_key,trace_namespace+'-evaluation',i)
                        if i not in eval_cache: eval_cache[i] = generate_trace(s,seed,stop_at)
                        row = simulate(s,p,eval_cache[i],stop_at)
                        rows.append(dict(trace_index=i,trace_seed=seed,**row))
                        if len(rows) % cfg['save_every_traces'] == 0:
                            atomic_json(path,rows); atomic_json(output/'status.json',dict(state='running',**current,saved_traces=len(rows)))
                finally: atomic_json(path,rows)
                summary = summarize(c,rows)
                summary.update(planning_s=selected[method]['planning_s'],candidates=selected[method]['candidates'],
                               planner_status=selected[method]['status'],structural_feasible=True,
                               shards=len(p.groups[0]),replicas=len(p.groups),active_nodes=len(p.active),allocated_nodes=len(p.allocated),
                               spare_count=len(p.spares),backup_copies=len(p.backups),backup_gib=len(p.backups)*c.state_gib/len(p.groups[0]),
                               **{key:getattr(p,key) for key in POLICY_FIELDS},
                               failure_free_completion_s=healthy['completion_s'],
                               failure_penalty_s=(summary['mean_completion_s']-healthy['completion_s'] if summary['mean_completion_s'] is not None and healthy['completed'] else None),
                               estimated_execution_plus_planning_s=summary['restricted_mean_s']+selected[method]['planning_s'])
                for key in ('spare_remaps','checkpoint_writes','checkpoint_aborts','checkpoint_attempted_gib','background_elapsed_s',
                            'machine_cost_usd','storage_cost_usd','write_cost_usd'):
                    summary['mean_'+key] = statistics.mean(r[key] for r in rows)
                atomic_json(d/f'{method}.summary.json',summary)
                print(f'  time={summary["mean_completion_s"]}, cost={summary["mean_cost_usd"]}, screen={summary["statistical_screen_passed"]}',flush=True)
            build_summary(output)
        atomic_json(output/'status.json',dict(state='finished'))
    except BudgetExpired:
        atomic_json(output/'status.json',dict(state='paused_time_limit',**current))
        print('Paused at time limit. Run the same cell to resume.',flush=True)
    except KeyboardInterrupt:
        atomic_json(output/'status.json',dict(state='paused_interrupt',**current))
        print('Interrupted; saved complete observations. Run again to resume.',flush=True)
    except Exception as e:
        if prepared: atomic_json(output/'status.json',dict(state='error',error=repr(e),**current))
        raise
    finally:
        try:
            if prepared: build_summary(output)
        finally: lock.unlink(missing_ok=True)
    return status(output)


def plot_results(output, profile=None, reference_nodes=None, split=None):
    """Read-only dot plots; no vertical bars or bottom captions."""
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(read_json(Path(output)/'summary.json')['results'])
    if df.empty: print('No completed results yet.'); return df
    if split is not None: df = df[df.split == split]
    if profile is not None: df = df[df.profile == profile]
    if reference_nodes is not None: df = df[df.reference_nodes == reference_nodes]
    if df.empty: return df
    # Avoid misleading aggregates from different completed case subsets.
    methods = read_json(Path(output)/'manifest.json')['methods']
    complete = df.groupby('case_id').method.nunique()
    df = df[df.case_id.isin(complete[complete == len(methods)].index)]
    if df.empty: print('No case has all methods evaluated yet.'); return df
    avg = df.groupby('method').agg(restricted_mean_s=('restricted_mean_s','mean'),
                                  mean_observed_cost_usd=('mean_observed_cost_usd','mean'),
                                  planning_s=('planning_s','mean'),screen_rate=('statistical_screen_passed','mean'),
                                  censored_traces=('censored_traces','sum')).reindex(methods)
    fig, axes = plt.subplots(1,3,figsize=(13,max(4,.4*len(methods))),layout='constrained')
    for ax,key,title in zip(axes,('restricted_mean_s','mean_observed_cost_usd','planning_s'),
                           ('Observed execution time (s)','Observed cost ($)','Planning time (s)')):
        ax.scatter(avg[key],avg.index)
        ax.set_title(title); ax.grid(axis='x',alpha=.2)
    plt.show()
    return avg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/pilot_v4.json'))
    parser.add_argument('--output',default='results/pilot_v4')
    parser.add_argument('--max-minutes',type=float,default=30)
    parser.add_argument('--status',action='store_true')
    args = parser.parse_args()
    print(json.dumps(status(args.output) if args.status else run(config_path=args.config,output=args.output,max_minutes=args.max_minutes),indent=2))


if __name__ == '__main__': main()
