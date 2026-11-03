"""Mixed-size RL training with resumable validation and seed-group selection.

No work executes on import. No test instances are generated during tuning.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import statistics as stats
import zipfile
import numpy as np

from .expanded import Scenario,make_scenarios,generate_trace,simulate
from .expanded_planners import digest
from .expanded_runner import validate_config as validate_simulation
from .learning import Network,FEATURE_VERSION,action_pool,features,matrix,objective,new_training_state
from .model import seed_for
from .metrics import summarize
from .runner import ROOT,read_json,atomic_json,source_snapshot
from .simulator import BudgetExpired,check_budget


def validate_config(c):
    validate_simulation(dict(c,experiment='expanded_v4',methods=['local_search']))
    if c.get('experiment')!='tuning_v7':raise ValueError('Expected tuning_v7')
    if set(c['instance_splits'])!={'train','validation'}:raise ValueError('Tuning requires train and validation ONLY')
    t=c['tuning'];points=t['checkpoints']
    if not points or any(type(n) is not int or n<=0 for n in points) or points!=sorted(set(points)):raise ValueError('Invalid checkpoints')
    for name in ('validation_planning_traces','validation_traces','save_every_episodes','top_k','training_pool_versions'):
        if type(t[name]) is not int or t[name]<=0:raise ValueError('Invalid '+name)
    if t['validation_traces']<2 or t['top_k']>c['learning']['action_count']:raise ValueError('Invalid validation size/top-k')
    if t['selection_metric'] not in ('objective','completion_time'):raise ValueError('Invalid selection metric')
    names=[v['name'] for v in t['variants']]
    if not names or len(set(names))!=len(names) or any(not n.replace('_','').isalnum() for n in names):raise ValueError('Invalid variant names')
    for v in t['variants']:
        if set(v)!={'name','hidden','learning_rate','entropy'}:raise ValueError('Invalid variant fields')
        if type(v['hidden']) is not int or v['hidden']<=0:raise ValueError('Invalid hidden size')
        if not math.isfinite(v['learning_rate']) or v['learning_rate']<=0 or not math.isfinite(v['entropy']) or v['entropy']<0:raise ValueError('Invalid optimizer setting')
    seeds=c['learning']['seeds']
    if not seeds or len(set(seeds))!=len(seeds) or any(type(n) is not int or not 0<=n<1000 for n in seeds):raise ValueError('Seeds must be distinct integers below 1000')
    for key in ('rollouts_per_episode','action_count','history_every'):
        if type(c['learning'][key]) is not int or c['learning'][key]<=0:raise ValueError('Invalid '+key)
    if set(c['learning']['penalties'])!={'cost','deadline','censor'} or any(not math.isfinite(v) or v<=0 for v in c['learning']['penalties'].values()):raise ValueError('Invalid penalties')


def variant_config(c,name):
    cfg=copy.deepcopy(c);v=next(v for v in c['tuning']['variants'] if v['name']==name)
    cfg['learning'].update({k:v[k] for k in ('hidden','learning_rate','entropy')})
    return cfg


def prepare(c,out):
    validate_config(c);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sources=source_snapshot();hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
    fingerprint=digest(dict(config=c,sources=hashes,numpy=np.__version__))
    path=out/'manifest.json'
    if path.exists():
        m=read_json(path)
        if m['fingerprint']!=fingerprint:raise ValueError('Config/source/NumPy changed; choose a NEW tuning output')
        for split in ('train','validation'):
            if digest(read_json(out/f'{split}_scenarios.json'))!=m[split+'_sha256']:raise ValueError('Modified '+split+' inputs')
        return m
    scenarios=make_scenarios(c);train=[s.to_dict() for s in scenarios if s.split=='train']
    # One deterministic profile per size/regime/instance: 24 balanced tuning
    # cases by default. Keep all four profiles for subsequent fresh evaluation.
    validation=[]
    for s in scenarios:
        if s.split!='validation':continue
        n=s.workload.shards*s.workload.replicas
        index=(c['active_nodes'].index(n)+c['regimes'].index(s.workload.regime))%len(c['expanded_profiles'])
        if s.workload.profile==c['expanded_profiles'][index]['name']:validation.append(s.to_dict())
    if not train or not validation:raise ValueError('Empty training/validation set')
    jobs=[f"{v['name']}-seed-{seed}" for v in c['tuning']['variants'] for seed in c['learning']['seeds']]
    m=dict(experiment='tuning_v7',schema=7,config=c,sources=hashes,fingerprint=fingerprint,numpy=np.__version__,
           train_sha256=digest(train),validation_sha256=digest(validation),training_cases=[s['workload']['case_id'] for s in train],
           validation_cases=[s['workload']['case_id'] for s in validation],jobs=jobs,
           split_contract='Only TRAIN updates parameters. VALIDATION chooses one shared variant/checkpoint across every seed. No test cases are generated or read. Final comparison uses a separate hardware seed.',
           created_unix_s=time.time())
    for name,value in [('train_scenarios',train),('validation_scenarios',validation),('manifest',m)]:atomic_json(out/f'{name}.json',value)
    for name,content in sources.items():
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    return m


def checkpoint_validation(root,job,c,episode,actor,stop_at=None):
    root=Path(root);directory=root/'jobs'/job/'validation'/str(episode)
    model_hash=digest(actor);net=Network.from_dict(actor);t=c['tuning'];records=[]
    raw=read_json(root/'validation_scenarios.json')
    if digest(raw)!=read_json(root/'manifest.json')['validation_sha256']:raise ValueError('Validation inputs changed')
    for value in raw:
        check_budget(stop_at);s=Scenario.from_dict(value)
        if s.split!='validation':raise ValueError('Checkpoint validation cannot read train/test data')
        path=directory/(s.workload.case_id+'.json')
        record=read_json(path) if path.exists() else dict(model_sha256=model_hash,candidates=[],observations={})
        if record['model_sha256']!=model_hash:raise ValueError('Validation model changed')
        if record.get('finished'):
            records.append(record['summary']);continue
        pool=action_pool(s,c);scores,_=net.forward(matrix(s,pool));order=sorted(range(len(pool)),key=lambda i:(-float(scores[i]),i))
        signature=digest([p.to_dict() for p in pool])
        if record.get('pool_sha256',signature)!=signature:raise ValueError('Validation pool changed')
        record['pool_sha256']=signature
        traces=[generate_trace(s,seed_for(c['seed'],s.workload.trace_key,'v7-tune-planning',i),stop_at) for i in range(t['validation_planning_traces'])]
        for index in order[:t['top_k']][len(record['candidates']):]:
            rows=[simulate(s,pool[index],trace,stop_at) for trace in traces]
            summary=summarize(s.workload,rows)
            record['candidates'].append(dict(index=index,loss=objective(s.workload,summary,c['learning']['penalties'])))
            atomic_json(path,record)
        winner=min(record['candidates'],key=lambda r:(r['loss'],order.index(r['index'])))['index']
        # Paired held-out validation traces; reused only for identical actions.
        for index in dict.fromkeys([order[0],winner]):
            key=str(index);rows=record['observations'].setdefault(key,[])
            try:
                for i in range(len(rows),t['validation_traces']):
                    trace=generate_trace(s,seed_for(c['seed'],s.workload.trace_key,'v7-tune-evaluation',i),stop_at)
                    rows.append(simulate(s,pool[index],trace,stop_at))
                    if len(rows)%8==0:atomic_json(path,record)
            finally:atomic_json(path,record)
        summaries={}
        for label,index in [('direct',order[0]),('top_k',winner)]:
            q=summarize(s.workload,record['observations'][str(index)])
            summaries[label]=dict(objective=objective(s.workload,q,c['learning']['penalties']),
                                  completion_time=q['restricted_mean_s'],cost=q['mean_observed_cost_usd'],
                                  screen_passed=q['statistical_screen_passed'],point_feasible=q['point_estimate_feasible'],
                                  censored=q['censored_traces'],deadline_success=q['deadline_success'])
        record['summary']=dict(case_id=s.workload.case_id,available_nodes=len(s.workload.nodes),**summaries)
        record['finished']=True;atomic_json(path,record);records.append(record['summary'])
    result=dict(episode=episode,model_sha256=model_hash,cases=records)
    for label in ('direct','top_k'):
        result[label]={key:stats.mean(r[label][key] for r in records) for key in ('objective','completion_time','cost','screen_passed','point_feasible','censored','deadline_success')}
    atomic_json(directory/'summary.json',result);return result



def update_from_features(s,state,c,loss,action,x):
    """Same REINFORCE/Huber update as v5, using cached immutable features."""
    lc=c['learning'];actor=Network.from_dict(state['actor']);reg=Network.from_dict(state['regressor'])
    reward=-float(loss);key=s.workload.case_id;baseline=state['baselines'].get(key,-1.)
    actor.policy_update(x,action,reward-baseline,lc['learning_rate'],lc['entropy'])
    reg.regression_update(x[action:action+1],reward,lc['learning_rate'])
    result=dict(state,actor=actor.to_dict(),regressor=reg.to_dict(),episode=state['episode']+1,
                return_sum=state['return_sum']+reward,baselines=dict(state['baselines']))
    result['baselines'][key]=.9*baseline+.1*reward
    return result

def train_job(root,variant,seed,stop_at=None):
    root=Path(root);m=read_json(root/'manifest.json');c=variant_config(m['config'],variant)
    lc=c['learning'];t=c['tuning'];job=f'{variant}-seed-{seed}';directory=root/'jobs'/job;directory.mkdir(parents=True,exist_ok=True)
    raw=read_json(root/'train_scenarios.json')
    if digest(raw)!=m['train_sha256'] or any(v['split']!='train' for v in raw):raise ValueError('Invalid TRAIN inputs')
    scenarios=[Scenario.from_dict(v) for v in raw]
    state_path=directory/'state.json'
    state=read_json(state_path) if state_path.exists() else new_training_state(len(features(scenarios[0],action_pool(scenarios[0],c)[0])),c,seed)
    before=state['elapsed_s'];started=time.monotonic();pool_cache={}
    def save():
        state['elapsed_s']=before+time.monotonic()-started;atomic_json(state_path,state)
    def validate_saved(episode):
        cp_path=directory/'checkpoints'/f'{episode}.json'
        if not cp_path.exists():
            if state['episode']!=episode:raise ValueError('Missing historical checkpoint')
            atomic_json(cp_path,dict(episode=episode,actor=state['actor'],sha256=digest(state['actor'])))
        cp=read_json(cp_path)
        if digest(cp['actor'])!=cp['sha256']:raise ValueError('Checkpoint modified')
        report_path=directory/'validation'/str(episode)/'summary.json'
        if report_path.exists():
            if read_json(report_path)['model_sha256']!=cp['sha256']:raise ValueError('Validation checkpoint hash mismatch')
        else:
            checkpoint_validation(root,job,c,episode,cp['actor'],stop_at)
    try:
        for episode in t['checkpoints']:
            if episode<=state['episode']:validate_saved(episode)
        while state['episode']<max(t['checkpoints']):
            check_budget(stop_at);episode=state['episode']
            # Identical context and rollout seeds across settings, independent of
            # their chosen actions. Each seed remains a separate training run.
            rng=random.Random(seed_for(c['seed'],seed,episode,'v7-training-case'))
            s=rng.choice(scenarios);version=rng.randrange(t['training_pool_versions'])
            cache_key=(s.workload.case_id,version)
            if cache_key not in pool_cache:
                pool=action_pool(s,c,key=f'v7-training-bank-{version}')
                pool_cache[cache_key]=(pool,matrix(s,pool))
            pool,x=pool_cache[cache_key]
            actor=Network.from_dict(state['actor']);probs=actor.probabilities(x)
            action=rng.choices(range(len(pool)),weights=probs.tolist(),k=1)[0]
            rows=[]
            for i in range(lc['rollouts_per_episode']):
                trace=generate_trace(s,seed_for(c['seed'],seed,episode,'v7-training-rollout',i),stop_at)
                rows.append(simulate(s,pool[action],trace,stop_at))
            loss=objective(s.workload,summarize(s.workload,rows),lc['penalties'])
            state=update_from_features(s,state,c,loss,action,x);state['simulations']+=len(rows)
            if state['episode']%lc['history_every']==0 or state['episode'] in t['checkpoints']:
                previous=state['history'][-1] if state['history'] else dict(episode=0,return_sum=0.)
                state['history'].append(dict(episode=state['episode'],return_sum=state['return_sum'],
                                            mean_reward=state['return_sum']/state['episode'],
                                            recent_mean_reward=(state['return_sum']-previous['return_sum'])/(state['episode']-previous['episode'])))
            if state['episode']%t['save_every_episodes']==0 or state['episode'] in t['checkpoints']:save()
            if state['episode'] in t['checkpoints']:validate_saved(state['episode'])
    finally:save()
    return dict(job=job,episodes=state['episode'])


def _worker(root,variant,seed,stop_at):
    directory=Path(root)/'jobs'/f'{variant}-seed-{seed}';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'worker.log').open('a') as log,contextlib.redirect_stdout(log):
        try:return train_job(root,variant,seed,stop_at)
        except BudgetExpired:return dict(job=directory.name,state='paused_time_limit')


def summary(root):
    root=Path(root);m=read_json(root/'manifest.json');c=m['config'];rows=[];jobs=[]
    for variant in c['tuning']['variants']:
        for seed in c['learning']['seeds']:
            job=f"{variant['name']}-seed-{seed}";d=root/'jobs'/job;path=d/'state.json'
            state=read_json(path) if path.exists() else {}
            jobs.append(dict(job=job,seed=seed,variant=variant['name'],episodes=state.get('episode',0),
                             simulations=state.get('simulations',0),elapsed_s=state.get('elapsed_s',0),history=state.get('history',[])))
            for episode in c['tuning']['checkpoints']:
                path=d/'validation'/str(episode)/'summary.json'
                if path.exists():
                    r=read_json(path);rows.append(dict(variant=variant['name'],seed=seed,episode=episode,model_sha256=r['model_sha256'],direct=r['direct'],top_k=r['top_k'],cases=r['cases']))
    expected=len(c['tuning']['variants'])*len(c['learning']['seeds'])*len(c['tuning']['checkpoints'])
    result=dict(state='finished' if len(rows)==expected else 'incomplete',validation_checkpoints=len(rows),expected_validation_checkpoints=expected,jobs=jobs,validation=rows)
    atomic_json(root/'tuning_summary.json',result);return result


def freeze_selection(root):
    root=Path(root);m=read_json(root/'manifest.json');c=m['config'];report=summary(root)
    if report['state']!='finished':raise ValueError('Finish all configured seeds/settings/checkpoints before selecting')
    groups=[];metric=c['tuning']['selection_metric']
    for v in c['tuning']['variants']:
        for episode in c['tuning']['checkpoints']:
            rows=[r for r in report['validation'] if r['variant']==v['name'] and r['episode']==episode]
            groups.append(dict(variant=v['name'],episode=episode,
                               objective=stats.mean(r['top_k']['objective'] for r in rows),
                               completion_time=stats.mean(r['top_k']['completion_time'] for r in rows),
                               screen_rate=stats.mean(r['top_k']['screen_passed'] for r in rows),
                               cost=stats.mean(r['top_k']['cost'] for r in rows)))
    winner=min(groups,key=lambda g:(g[metric],g['objective'] if metric=='completion_time' else g['completion_time'],g['episode'],g['variant']))
    actors={};hashes={}
    for seed in c['learning']['seeds']:
        p=root/'jobs'/f"{winner['variant']}-seed-{seed}"/'checkpoints'/f"{winner['episode']}.json"
        cp=read_json(p)
        if digest(cp['actor'])!=cp['sha256']:raise ValueError('Checkpoint modified')
        val=next(r for r in report['validation'] if r['variant']==winner['variant'] and r['seed']==seed and r['episode']==winner['episode'])
        if val['model_sha256']!=cp['sha256']:raise ValueError('Selected model not validated')
        actors[str(seed)]=cp['actor'];hashes[str(seed)]=cp['sha256']
    selected=dict(selection=winner,selection_metric=metric,groups=groups,model_hashes=hashes,
                  rule='One variant and episode selected by mean top-k validation metric across ALL seeds; no seed selection.')
    path=root/'selection.json'
    if path.exists() and read_json(path)!=selected:raise ValueError('Frozen selection changed; use a NEW output')
    bundle=dict(feature_version=FEATURE_VERSION,actors=actors,source_fingerprint=m['fingerprint'],source_config=c,
                source_training_cases=m['training_cases'],selection=selected,source_learning=report)
    bpath=root/'selected_policies.json'
    if bpath.exists() and digest(read_json(bpath))!=digest(bundle):
        # Timing/history provenance can be updated on resume; actor identity cannot.
        if read_json(bpath)['actors']!=actors:raise ValueError('Frozen selected actors changed')
    atomic_json(bpath,bundle);atomic_json(path,selected)
    atomic_json(root/'selected_policy_hash.json',dict(sha256=digest(bundle)))
    return selected


def run(config=None,output='results/pilot_v7_tuning',workers=4,max_minutes=None):
    if type(workers) is not int or workers<1:raise ValueError('Invalid worker count')
    if max_minutes is not None and (not math.isfinite(max_minutes) or max_minutes<=0):raise ValueError('Invalid optional time limit')
    c=config if config is not None else read_json(ROOT/'configs/pilot_v7.json');out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'.run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Tuning locked; confirm old processes stopped before removing lock') from None
    os.write(fd,str(os.getpid()).encode());os.close(fd);prepared=False
    stop=None if max_minutes is None else time.monotonic()+60*max_minutes
    old={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    try:
        m=prepare(c,out);prepared=True
        execution=out/'execution.json'
        if execution.exists() and read_json(execution)['workers']!=workers:raise ValueError('Worker count changed; use new tuning output')
        if not execution.exists():atomic_json(execution,dict(workers=workers,cpu_count=os.cpu_count(),backend='CPU processes',max_minutes=max_minutes))
        jobs=[]
        for v in c['tuning']['variants']:
            for seed in c['learning']['seeds']:
                d=out/'jobs'/f"{v['name']}-seed-{seed}"
                if not (d/'validation'/str(max(c['tuning']['checkpoints']))/'summary.json').exists():jobs.append((v['name'],seed))
        for key in old:os.environ[key]='1'
        if workers==1:
            for variant,seed in jobs:
                check_budget(stop);print('Training:',variant,'seed',seed,flush=True);_worker(str(out),variant,seed,stop);summary(out)
        elif jobs:
            print(f'Training {len(jobs)} setting/seed jobs using {workers} CPU workers. Progress is saved under jobs/.',flush=True)
            import multiprocessing as mp
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
                active={pool.submit(_worker,str(out),v,seed,stop):(v,seed) for v,seed in jobs}
                for future in as_completed(active):
                    print('Saved:',future.result(),flush=True);summary(out)
        if summary(out)['state']=='finished':freeze_selection(out)
    except BudgetExpired:print('Paused at optional time limit; rerun unchanged.',flush=True)
    finally:
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        try:
            if prepared:summary(out)
        finally:lock.unlink(missing_ok=True)
    result=summary(out)
    return {k:v for k,v in result.items() if k not in ('jobs','validation')}


def comparison_source(tuning_output,old_source,output):
    """Freeze both old and newly selected families for the same comparison run."""
    from .guided_runner import load_source
    root=Path(tuning_output);m=read_json(root/'manifest.json');selected=read_json(root/'selected_policies.json')
    if digest(selected)!=read_json(root/'selected_policy_hash.json')['sha256']:raise ValueError('Selected policy bundle changed')
    c=m['config'];cfg=copy.deepcopy(c);cfg['seed']=c['seed']+1
    old=load_source(old_source,cfg)
    actors=dict(old['actors']);labels={key:f'old_small_seed_{key}' for key in actors}
    for seed,actor in selected['actors'].items():
        key=str(1000+int(seed));actors[key]=actor;labels[key]=f'mixed_selected_seed_{seed}'
    bundle=dict(feature_version=FEATURE_VERSION,actors=actors,source_fingerprint=digest([old['source_fingerprint'],m['fingerprint']]),
                source_config=c,source_training_cases=m['training_cases'],source_learning=dict(old=old['source_learning'],tuning=selected['source_learning'],selection=selected['selection']),
                model_labels=labels,excluded_hardware_seeds=[old['source_config']['seed'],c['seed']])
    output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/'comparison_policies.json'
    if path.exists() and read_json(path)!=bundle:raise ValueError('Comparison source changed; use a new comparison source folder')
    atomic_json(path,bundle)
    atomic_json(output/'manifest.json',dict(experiment='comparison_v7',fingerprint=digest(bundle),sources=m['sources'],bundle_sha256=digest(bundle)))
    return output


def evaluation_config(tuning_config,source,split='validation'):
    from .guided_runner import method_specs
    c=copy.deepcopy(tuning_config);c.pop('tuning',None)
    c.update(experiment='guided_v6',seed=tuning_config['seed']+1,instances_per_regime=2,instance_splits=['validation','test'],split=split)
    c['learning']['seeds']=[int(k) for k in read_json(Path(source)/'comparison_policies.json')['actors']]
    c['guided']=dict(top_k=[tuning_config['tuning']['top_k']],candidate_budgets=[32],seconds_budgets=[5],ranking_only=True,heuristic_ranker=True)
    c['methods']=list(method_specs(c));return c


def export(output):
    output=Path(output);report=summary(output);selection=output/'selection.json'
    bundle=dict(format='learnedplacement-tuning-analysis',manifest=read_json(output/'manifest.json'),execution=read_json(output/'execution.json'),
                tuning=report,selection=read_json(selection) if selection.exists() else None)
    target=output.with_suffix('.analysis.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('analysis.json',json.dumps(bundle,separators=(',',':')))
    return target


def plot_progress(output):
    """Read-only recent reward and held-out checkpoint curves; no error bars."""
    import pandas as pd
    import matplotlib.pyplot as plt
    report=read_json(Path(output)/'tuning_summary.json')
    history=[dict(job=j['job'],variant=j['variant'],seed=j['seed'],**h) for j in report['jobs'] for h in j['history']]
    if history:
        df=pd.DataFrame(history);fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        for name,g in df.groupby('job'):ax.plot(g.episode,g.recent_mean_reward,label=name,alpha=.75)
        ax.set_title('Recent training reward');ax.set_xlabel('Completed episodes');ax.legend(fontsize=7);ax.grid(alpha=.2);plt.show()
    rows=[dict(variant=r['variant'],seed=r['seed'],episode=r['episode'],**r['top_k']) for r in report['validation']]
    if not rows:return pd.DataFrame()
    df=pd.DataFrame(rows)
    # Do not compare a partially finished seed group with a complete one.
    expected=len(read_json(Path(output)/'manifest.json')['config']['learning']['seeds'])
    counts=df.groupby(['variant','episode']).seed.transform('nunique');df=df[counts==expected]
    if df.empty:return df
    avg=df.groupby(['variant','episode']).mean(numeric_only=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for variant,g in avg.groupby(level='variant'):
        for ax,key in zip(axes,['completion_time','objective']):ax.plot(g.index.get_level_values('episode'),g[key],marker='o',label=variant)
    for ax,title in zip(axes,['Validation completion time (s)','Validation penalized objective']):ax.set_title(title);ax.set_xlabel('Training episodes');ax.legend();ax.grid(alpha=.2)
    plt.show();return avg
