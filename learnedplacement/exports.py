"""Compact, read-only analysis exports; never run or resume an experiment."""
import itertools
import math
import statistics


def difference_stats(values):
    """Approximate normal interval for a paired mean, not a bootstrap interval."""
    mean = statistics.mean(values)
    variance = statistics.variance(values) if len(values) > 1 else None
    se = math.sqrt(variance/len(values)) if variance is not None else None
    return dict(mean=mean, sample_variance=variance, standard_error=se,
                approximate_mean_ci95=[mean-1.959963984540054*se, mean+1.959963984540054*se]
                if se is not None else None)


def analysis_bundle(output):
    # Import lazily to avoid a cycle with runner.export_run.
    from .runner import read_json, status
    manifest = read_json(output/'manifest.json')
    cases = []
    for cid in manifest['cases']:
        directory = output/'cases'/cid
        plans_path = directory/'plans.json'
        plans = read_json(plans_path) if plans_path.exists() else {}
        methods, observations = {}, {}
        for method in manifest['methods']:
            record = {}
            if method in plans:
                # Keep the chosen plan, timing, score and candidate-pool hash.
                # The full pool and duplicate unaware record remain local.
                record['selection'] = {k: v for k, v in plans[method].items()
                                       if k not in ('candidate_audit', 'unaware_result', 'controls')}
            for suffix, key in [('summary', 'summary'), ('failure_free', 'failure_free')]:
                path = directory/f'{method}.{suffix}.json'
                if path.exists():
                    record[key] = read_json(path)
            path = directory/f'{method}.observations.json'
            if path.exists():
                rows = read_json(path)
                observations[method] = rows
                record['saved_observations'] = len(rows)
                record['evaluation_finished'] = len(rows) == manifest['config']['evaluation_traces']
            if record:
                methods[method] = record
        pairs = []
        for selected, baseline in itertools.combinations(observations, 2):
            a, b = observations[selected], observations[baseline]
            count = min(len(a), len(b))
            if not count:
                continue
            # Interrupted exports compare the shared saved prefix only. Never
            # pair unrelated seeds or silently discard censored observations.
            a, b = a[:count], b[:count]
            if any((x['trace_index'], x['trace_seed']) != (y['trace_index'], y['trace_seed'])
                   for x, y in zip(a, b)):
                raise ValueError(f'Paired trace mismatch: {cid}, {selected}, {baseline}')
            completed = all(r['completed'] for r in a+b)
            time_stats = difference_stats([x['observed_s']-y['observed_s'] for x, y in zip(a, b)])
            pair = dict(selected=selected, baseline=baseline, paired_traces=count,
                        full_evaluation=count == manifest['config']['evaluation_traces'],
                        all_completed=completed,
                        time_quantity='completion_s' if completed else 'restricted_observed_s',
                        time_difference=time_stats,
                        cost_quantity='total_cost_usd' if completed else 'observed_cost_usd',
                        cost_difference=difference_stats([x['cost_usd']-y['cost_usd'] for x, y in zip(a, b)]),
                        deadline_success_difference=difference_stats([int(x['deadline_hit'])-int(y['deadline_hit']) for x, y in zip(a, b)]))
            healthy = [methods[m].get('failure_free', {}).get('completion_s') for m in (selected, baseline)]
            # Old v1 runs have no saved failure-free observations. No simulations
            # are needed to export them; this diagnostic is simply unavailable.
            if completed and all(t is not None for t in healthy):
                delta = healthy[0]-healthy[1]
                interval = time_stats['approximate_mean_ci95']
                pair['failure_free_time_difference_s'] = delta
                pair['failure_penalty_difference_s'] = time_stats['mean']-delta
                pair['failure_penalty_approximate_mean_ci95'] = [v-delta for v in interval] if interval else None
            pairs.append(pair)
        cases.append(dict(case=read_json(directory/'case.json'), methods=methods, paired_comparisons=pairs))
    return dict(format='learnedplacement-analysis', schema=1, manifest=manifest,
                status=status(output),
                interpretation={
                    'difference_direction': 'selected minus baseline; negative time/cost favors selected',
                    'intervals': 'Approximate two-sided 95% normal intervals for paired means; not bootstrap or simultaneous guarantees.',
                    'partial_runs': 'Pairs use the shared saved prefix and are labeled full_evaluation=false until complete.',
                    'censoring': 'Censored pairs retain all observations and compare restricted time and observed cost, not expected total completion/cost.',
                    'scope': 'Enough for routine plan/constraint/mean-effect analysis. Raw distribution reanalysis and exact replay require local raw files.',
                    'resume': 'This compact bundle is not a resumable run directory.'},
                cases=cases)
