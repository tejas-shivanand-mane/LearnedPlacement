"""V9 frozen-model initialization study. No training and no work on import.

One trajectory supplies both wall-time and candidate-count checkpoints. Planning
caches are private to trajectories; held-out evaluation caches are case-local.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import time
import zipfile

from .expanded import Scenario, Decision, make_scenarios, generate_trace, simulate, violations
from .expanded_planners import base_plan, neighbors, random_plan, digest
from .learning import action_pool, objective
from .metrics import summarize
from .model import seed_for
from .runner import ROOT, read_json, atomic_json, source_snapshot


class SearchComplete(Exception):
    pass


def load_source(source):
    """Read existing frozen V8 weights; never rerun selection or training."""
    source = Path(source)
    manifest = read_json(source / 'manifest.json')
    if manifest['experiment'] == 'architecture_v8':
        bundle = read_json(source / 'selected_policies.json')
        expected = read_json(source / 'selected_policy_hash.json')['sha256']
    elif manifest['experiment'] == 'comparison_v8':
        bundle = read_json(source / 'comparison_policies.json')
        expected = manifest['bundle_sha256']
    else:
        raise ValueError('SOURCE must be a completed V8 tuning/models folder, not an analysis ZIP')
    if digest(bundle) != expected:
        raise ValueError('Frozen model bundle changed')
    for name in ('learning.py', 'expanded.py', 'metrics.py', 'architecture.py'):
        key = 'learnedplacement/' + name
        if hashlib.sha256((ROOT / key).read_bytes()).hexdigest() != manifest['sources'].get(key):
            raise ValueError('Incompatible simulator/model source: ' + key)
    specs = {name: spec for name, spec in bundle['model_specs'].items()
             if re.fullmatch(r'(mlp|set|gnn)_s\d+', name) and spec['kind'] == 'direct'}
    if {name.split('_')[0] for name in specs} != {'mlp', 'set', 'gnn'}:
        raise ValueError('Need frozen direct checkpoints for MLP, Set and GNN')
    seeds = [{int(n.split('_s')[1]) for n in specs if n.startswith(a + '_')}
             for a in ('mlp', 'set', 'gnn')]
    if not all(s == seeds[0] for s in seeds):
        raise ValueError('Architectures must retain the same model seeds')
    return bundle, specs


def configuration(source, settings=None):
    settings = copy.deepcopy(settings or read_json(ROOT / 'configs/pilot_v9.json'))
    bundle, specs = load_source(source)
    c = copy.deepcopy(bundle['source_config'])
    c.pop('tuning', None); c.pop('architecture', None)
    c.update(experiment='anytime_v9', seed=settings['seed'], instances_per_regime=1,
             instance_splits=['test'], anytime=settings)
    if c['seed'] in set(bundle.get('excluded_hardware_seeds', [])) | {
            bundle['source_config']['seed'], bundle['source_config']['seed'] + 1}:
        raise ValueError('Choose a fresh hardware seed, different from training and V8 evaluation')
    for key in ('seconds', 'candidates'):
        values = settings[key]
        if not values or values != sorted(set(values)) or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Budgets must be sorted, unique, positive: ' + key)
    if any(type(v) is not int for v in settings['candidates']):
        raise ValueError('Candidate budgets must be integers')
    if not settings['search_seeds'] or len(set(settings['search_seeds'])) != len(settings['search_seeds']):
        raise ValueError('Provide distinct search seeds')
    if settings['case_mode'] not in ('balanced', 'full'):
        raise ValueError('case_mode must be balanced or full')
    return c, bundle, specs


def scenarios(c):
    values = make_scenarios(c)
    if c['anytime']['case_mode'] == 'full':
        return values
    profiles = [p['name'] for p in c['expanded_profiles']]
    return [s for s in values if s.workload.profile == profiles[
        (c['active_nodes'].index(s.workload.shards * s.workload.replicas)
         + c['regimes'].index(s.workload.regime)) % len(profiles)]]


def method_specs(specs):
    result = {name: dict(kind=name, family=name) for name in ('heuristic', 'random', 'portfolio')}
    result.update({name: dict(spec, kind='rl', family=name.split('_')[0]) for name, spec in specs.items()})
    return result


def search(s, c, spec, search_seed, traces, model=None, clock=time.perf_counter, cancel=None):
    """Adaptive first-improvement search; eval data is not an argument.

    Finish enough of ONE trajectory for both maximum budgets. A candidate may
    cross a wall-time boundary; it is unavailable at that earlier checkpoint.
    """
    from .architecture import score_actor
    cfg = c['anytime']; start = clock(); cache = {}; events = []; best = None
    rng = random.Random(seed_for(c['seed'], s.workload.case_id, 'v9-neighbors', search_seed))
    init_rng = random.Random(seed_for(c['seed'], s.workload.case_id, 'v9-initialization', search_seed))
    fallback = base_plan(s, c, domain=False)
    ready = clock() - start
    pool_hash = None

    def score(row):
        return (row['objective'], row['summary']['restricted_mean_s'])

    def evaluate(p):
        nonlocal best
        if cancel is not None:
            cancel()
        if clock() - start >= max(cfg['seconds']) and len(cache) >= max(cfg['candidates']):
            raise SearchComplete
        key = digest(p.to_dict())
        if key in cache:
            return cache[key]
        if violations(s, p):
            raise ValueError('Structurally invalid candidate')
        observations = []
        for trace in traces:
            if cancel is not None:
                cancel()
            observations.append(simulate(s, p, trace))
        summary = summarize(s.workload, observations)
        row = dict(plan=p.to_dict(), summary=summary,
                   objective=objective(s.workload, summary, c['learning']['penalties']))
        cache[key] = row
        if best is None or score(row) < score(best):
            best = row
        events.append(dict(elapsed_s=clock()-start, candidates=len(cache),
                           plan=best['plan'], objective=best['objective']))
        return row

    def explore(initial, quota=None, shuffle=True):
        current = evaluate(initial); used = 1; seen = {digest(current['plan'])}
        while quota is None or used < quota:
            if cancel is not None:
                cancel()
            if clock()-start >= max(cfg['seconds']) and len(cache) >= max(cfg['candidates']):
                raise SearchComplete
            p = Decision.from_dict(current['plan']); items = list(neighbors(s, p, c))
            if shuffle:
                rng.shuffle(items)
            improved = False
            for q in items:
                key = digest(q.to_dict())
                if key in seen:
                    continue
                seen.add(key); trial = evaluate(q); used += 1
                if score(trial) < score(current):
                    current = trial; improved = True; break
                if quota is not None and used >= quota:
                    return current
            if improved:
                continue
            if not shuffle:
                return current
            for _ in range(1000):
                q = random_plan(s, c, rng)
                if q is not None and digest(q.to_dict()) not in seen:
                    break
            else:
                return current
            seen.add(digest(q.to_dict())); trial = evaluate(q); used += 1
            if score(trial) < score(current):
                current = trial
        return current

    exhausted = False
    try:
        if spec['kind'] in ('rl', 'random'):
            pool = action_pool(s, c)
            pool_hash = digest([p.to_dict() for p in pool])
            if spec['kind'] == 'rl':
                scores = score_actor(model, s, pool)
                initial = pool[max(range(len(pool)), key=lambda i: (float(scores[i]), -i))]
            else:
                initial = init_rng.choice(pool)
            explore(initial)
        elif spec['kind'] == 'heuristic':
            explore(base_plan(s, c))
        else:
            speed = evaluate(fallback); base = base_plan(s, c); tuned = evaluate(base)
            intervals = [v for v in c['choices']['checkpoint_every'] if v != base.checkpoint_every]
            for i in range(c['tuned_candidates'] - 1):
                q = (replace(base, checkpoint_every=intervals[i]) if i < len(intervals)
                     else random_plan(s, c, init_rng, base))
                if q is not None:
                    trial = evaluate(q)
                    if score(trial) < score(tuned):
                        tuned = trial
            coord = explore(Decision.from_dict(tuned['plan']), c['search_candidates'], False)
            initial = min([speed, tuned, coord], key=score)
            # Same portfolio initialization as V8; final search continues to the
            # requested budgets instead of inheriting V8's 153-candidate ceiling.
            explore(Decision.from_dict(initial['plan']))
        exhausted = True
    except SearchComplete:
        pass
    return dict(events=events, fallback=fallback.to_dict(), fallback_ready_s=ready,
                elapsed_s=clock()-start, exhausted=exhausted, pool_sha256=pool_hash,
                candidates=len(cache))


def checkpoints(trajectory, settings):
    """Only decisions available at the requested budget; never look ahead."""
    rows = []
    for axis in ('seconds', 'candidates'):
        field = 'elapsed_s' if axis == 'seconds' else 'candidates'
        for budget in settings[axis]:
            eligible = [e for e in trajectory['events'] if e[field] <= budget]
            event = eligible[-1] if eligible else None
            ready = axis == 'candidates' or trajectory['fallback_ready_s'] <= budget
            rows.append(dict(axis=axis, budget=budget,
                plan=event['plan'] if event else trajectory['fallback'] if ready else None,
                unscored_fallback=event is None, candidates=event['candidates'] if event else 0,
                decision_ready_s=event['elapsed_s'] if event else trajectory['fallback_ready_s'],
                planning_objective=event['objective'] if event else None))
    return rows


def prepare(source, output, settings, workers):
    import numpy as np
    import torch
    c, bundle, specs = configuration(source, settings)
    sources = source_snapshot(); cases = [s.to_dict() for s in scenarios(c)]
    methods = method_specs(specs)
    actors = {str(spec['seed']): bundle['actors'][str(spec['seed'])] for spec in specs.values()}
    identity = dict(config=c, sources={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
                    models={k:digest(v) for k,v in actors.items()}, workers=workers,
                    numpy=np.__version__, torch=str(torch.__version__), cases_sha256=digest(cases))
    out = Path(output); path = out/'manifest.json'
    if path.exists():
        manifest = read_json(path)
        if manifest['fingerprint'] != digest(identity):
            raise ValueError('Config/source/models/workers/dependencies changed; use a NEW V9 output')
        if digest(read_json(out/'scenarios.json')) != identity['cases_sha256']:
            raise ValueError('Saved scenarios changed')
        if digest(read_json(out/'actors.json')) != digest(actors):
            raise ValueError('Saved models changed')
        return manifest
    manifest = dict(identity, fingerprint=digest(identity), experiment='anytime_v9', methods=methods,
                    cases=[s['workload']['case_id'] for s in cases], source_bundle_sha256=digest(bundle),
                    selected_checkpoints=bundle['source_learning']['selection'],
                    source=str(Path(source).resolve()), created_unix_s=time.time())
    atomic_json(out/'actors.json', actors); atomic_json(out/'scenarios.json', cases)
    for name, content in sources.items():
        dest = out/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(content)
    atomic_json(path, manifest)
    return manifest


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _case(task):
    from .architecture import load_actor
    root, raw = task; root = Path(root); m = read_json(root/'manifest.json'); c = m['config']
    def cancel():
        if (root/'.stop').exists():
            raise InterruptedError('V9 interrupted; completed work saved')
    cancel()
    s = Scenario.from_dict(raw); cid = s.workload.case_id; directory = root/'cases'/cid
    output = directory/'result.json'
    if output.exists():
        return cid
    actors = read_json(root/'actors.json')
    models = {k:load_actor(v) for k,v in actors.items()}
    planning = [generate_trace(s, seed_for(c['seed'], s.workload.trace_key, 'v9-planning', i))
                for i in range(c['planning_traces'])]
    jobs = [(name, seed) for name in m['methods'] for seed in c['anytime']['search_seeds']]
    random.Random(seed_for(c['seed'], cid, 'v9-job-order')).shuffle(jobs)
    selections = []; logs = []
    for name, seed in jobs:
        cancel()
        spec = m['methods'][name]; path = directory/'search'/f'{name}-r{seed}.json'
        if path.exists():
            trajectory = read_json(path)
        else:
            trajectory = search(s, c, spec, seed, planning, models.get(str(spec.get('seed'))), cancel=cancel)
            atomic_json(path, trajectory)
        for selection in checkpoints(trajectory, c['anytime']):
            selections.append(dict(method=name, family=spec['family'], search_seed=seed, **selection))
        logs.append(dict(method=name, search_seed=seed, **{k:v for k,v in trajectory.items()
                    if k not in ('events', 'fallback')}, progress=[{k:v for k,v in e.items() if k != 'plan'} for e in trajectory['events']]))
        print(f'{cid}: {name}, search seed {seed}, {trajectory["candidates"]} candidates', flush=True)
    # Held-out traces generated only AFTER every planning trajectory is complete.
    evaluation = [generate_trace(s, seed_for(c['seed'], s.workload.trace_key, 'v9-evaluation', i))
                  for i in range(c['evaluation_traces'])]
    summaries = {}
    for row in selections:
        cancel()
        if row['plan'] is None:
            row['summary'] = None; continue
        key = digest(row['plan']); path = directory/'evaluation'/f'{key}.json'
        if key not in summaries:
            state = read_json(path) if path.exists() else dict(plan=row['plan'], observations=[])
            if state['plan'] != row['plan']:
                raise ValueError('Evaluation plan mismatch')
            if 'summary' not in state:
                p = Decision.from_dict(row['plan'])
                if violations(s, p):
                    raise ValueError('Invalid selected plan')
                try:
                    for trace in evaluation[len(state['observations']):]:
                        cancel()
                        state['observations'].append(simulate(s, p, trace))
                        if len(state['observations']) % c['save_every_traces'] == 0:
                            atomic_json(path, state)
                finally:
                    atomic_json(path, state)
                state['summary'] = summarize(s.workload, state['observations'])
                state['summary']['objective'] = objective(s.workload, state['summary'], c['learning']['penalties'])
                atomic_json(path, state)
            summaries[key] = state['summary']
        row['summary'] = summaries[key]
    atomic_json(output, dict(case=raw, selections=selections, searches=logs,
                            unique_evaluated_plans=len(summaries), finished=True))
    return cid


def status(output):
    root = Path(output); m = read_json(root/'manifest.json')
    finished = sum((root/'cases'/cid/'result.json').exists() for cid in m['cases'])
    return dict(state='finished' if finished == len(m['cases']) else 'incomplete',
                completed_cases=finished, total_cases=len(m['cases']),
                methods=len(m['methods']), search_seeds=m['config']['anytime']['search_seeds'])


def run(source, output='results/pilot_v9_anytime_w4', settings=None, workers=4):
    """No overall timeout. Resume complete searches/evaluation traces.

    An interrupted search trajectory restarts so timing is not distorted by
    restoring a warm cache. Completed trajectories are never re-timed.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    from .architecture_runner import locked
    out = Path(output).resolve()
    with locked(out):
        prepare(source, out, settings, workers)
        (out/'.stop').unlink(missing_ok=True)
        raw = read_json(out/'scenarios.json')
        tasks = [(str(out), s) for s in raw if not (out/'cases'/s['workload']['case_id']/'result.json').exists()]
        old = {k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        for k in old:
            os.environ[k] = '1'
        try:
            print(f'V9: {len(tasks)} remaining cases, {workers} CPU workers, no training or overall timeout.', flush=True)
            if workers == 1:
                _init_worker()
                for task in tasks:
                    _case(task); print(status(out), flush=True)
            else:
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=_init_worker)
                try:
                    futures = [pool.submit(_case, task) for task in tasks]
                    for future in as_completed(futures):
                        future.result(); print(status(out), flush=True)
                except BaseException:
                    (out/'.stop').touch()
                    raise
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
        finally:
            for k,v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            atomic_json(out/'status.json', status(out))
    return status(out)


def export(output):
    out = Path(output); m = read_json(out/'manifest.json')
    cases = [read_json(out/'cases'/cid/'result.json') for cid in m['cases'] if (out/'cases'/cid/'result.json').exists()]
    payload = dict(format='learnedplacement-anytime-analysis', manifest=m, status=status(out), cases=cases,
                   interpretation='Synthetic workloads; independent held-out traces; frozen models. CPU timing includes pool/features/inference and search, excludes cold model load, trace generation and result writes. No across-method planning cache. Full trajectories can extend beyond the largest wall budget to reach count checkpoints; later decisions never enter earlier checkpoints.')
    target = out.with_suffix('.analysis.zip'); temp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('analysis.json', json.dumps(payload, separators=(',',':'), allow_nan=False))
    os.replace(temp, target)
    return target


def plot_results(output, axis='seconds', profile=None, size=None):
    """All seeds averaged within case before cases; only complete matched cases."""
    import pandas as pd
    import matplotlib.pyplot as plt
    if axis not in ('seconds', 'candidates'):
        raise ValueError('axis must be seconds or candidates')
    out = Path(output); m = read_json(out/'manifest.json'); rows = []
    expected = len(m['methods']) * len(m['config']['anytime']['search_seeds'])
    for cid in m['cases']:
        path = out/'cases'/cid/'result.json'
        if not path.exists():
            continue
        result = read_json(path); w = result['case']['workload']
        if profile is not None and w['profile'] != profile:
            continue
        if size is not None and len(w['nodes']) != size:
            continue
        for budget in m['config']['anytime'][axis]:
            selected = [r for r in result['selections'] if r['axis']==axis and r['budget']==budget and r['summary'] is not None]
            if len(selected) != expected:
                continue
            for r in selected:
                q = r['summary']
                rows.append(dict(case=cid, family=r['family'], method=r['method'], budget=budget,
                    objective=q['objective'], completion_s=q['restricted_mean_s'], cost=q['mean_observed_cost_usd'],
                    deadline_success=q['deadline_success'], completed=q['completed_fraction'],
                    screen_passed=float(q['statistical_screen_passed']), unscored_fallback=float(r['unscored_fallback'])))
    if not rows:
        print('No complete matched cases yet.'); return pd.DataFrame()
    frame = pd.DataFrame(rows)
    avg = frame.groupby(['case','family','budget']).mean(numeric_only=True).groupby(['family','budget']).mean()
    labels = dict(heuristic='Heuristic + search', random='Random + search', portfolio='Heuristic portfolio + search',
                  mlp='RL MLP + search', set='RL Set + search', gnn='RL GNN + search')
    fig, axes = plt.subplots(1, 2, figsize=(12,4), layout='constrained')
    for family, values in avg.groupby(level='family'):
        for ax, metric in zip(axes, ['objective','completion_s']):
            ax.plot(values.index.get_level_values('budget'), values[metric], marker='o', label=labels[family])
    for ax, title in zip(axes, ['Evaluation objective ↓', 'Completion time under failures (s) ↓']):
        ax.set_title(title); ax.set_xlabel('Planning budget (seconds)' if axis=='seconds' else 'Unique candidate evaluations')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    coverage = frame.groupby('budget')['case'].nunique()
    print('Matched cases per budget:', coverage.to_dict())
    plt.show()
    return avg
