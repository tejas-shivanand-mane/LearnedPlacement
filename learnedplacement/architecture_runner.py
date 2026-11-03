"""V8: matched encoder experiments; CPU simulation workers, one GPU owner.

No experiments run on import. Train/validation only during model development;
final evaluation uses the existing paired, resumable guided runner.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics as stats
import time
import zipfile
import numpy as np
import torch
from .architecture import Encoder, inputs, load_actor, score_actor, VERSION
from .expanded import Scenario, make_scenarios, generate_trace, simulate
from .expanded_planners import digest
from .learning import action_pool, features, objective
from .metrics import summarize
from .model import seed_for
from .runner import ROOT, read_json, atomic_json, source_snapshot
from .simulator import check_budget, BudgetExpired
from . import tuning_runner as tuning


def validate_config(c):
    proxy=copy.deepcopy(c);proxy['experiment']='tuning_v7';tuning.validate_config(proxy)
    if c.get('experiment')!='architecture_v8':raise ValueError('Expected architecture_v8')
    if {v['name'] for v in c['tuning']['variants']}!={'mlp','set','gnn'}:raise ValueError('Compare mlp, set and gnn')
    settings={(v['hidden'],v['learning_rate'],v['entropy']) for v in c['tuning']['variants']}
    if len(settings)!=1:raise ValueError('Use matched hidden width and optimizer settings')
    a=c['architecture']
    for k in ('batch_episodes','layers','candidate_chunk_size','cache_entries','diagnostic_planning_traces','diagnostic_evaluation_traces'):
        if type(a[k]) is not int or a[k]<=0:raise ValueError('Invalid '+k)
    if any(n%a['batch_episodes'] for n in c['tuning']['checkpoints']):raise ValueError('Checkpoints must align with batch_episodes')
    if c['tuning']['save_every_episodes']%a['batch_episodes']:raise ValueError('Save interval must align with batch')
    if a['device'] not in ('auto','cpu','cuda'):raise ValueError('Choose auto, cpu or cuda')
    if not set(a['diagnostic_profiles'])<=set(p['name'] for p in c['expanded_profiles']):raise ValueError('Unknown diagnostic profile')
    if min(a['diagnostic_planning_traces'],a['diagnostic_evaluation_traces'])<2:raise ValueError('Diagnostic requires at least two traces')


def prepare(c, output):
    validate_config(c);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    sources=source_snapshot();hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
    fingerprint=digest(dict(config=c,sources=hashes,numpy=np.__version__,torch=str(torch.__version__)))
    path=out/'manifest.json'
    if path.exists():
        m=read_json(path)
        if m['fingerprint']!=fingerprint:raise ValueError('Config/source/dependencies changed; use a NEW V8 output')
        for split in ('train','validation','diagnostic'):
            if digest(read_json(out/f'{split}_scenarios.json'))!=m[split+'_sha256']:raise ValueError('Modified '+split+' scenarios')
        return m
    scenarios=make_scenarios(c);train=[];validation=[];diagnostic=[]
    for s in scenarios:
        if s.split=='train':train.append(s.to_dict());continue
        if s.split!='validation':raise ValueError('No test data during development')
        n=s.workload.shards*s.workload.replicas
        index=(c['active_nodes'].index(n)+c['regimes'].index(s.workload.regime))%len(c['expanded_profiles'])
        if s.workload.profile==c['expanded_profiles'][index]['name']:validation.append(s.to_dict())
        if s.workload.profile in c['architecture']['diagnostic_profiles']:diagnostic.append(s.to_dict())
    if not train or not validation:raise ValueError('Empty training/validation cases')
    m=dict(experiment='architecture_v8',schema=8,config=c,sources=hashes,fingerprint=fingerprint,
           numpy=np.__version__,torch=str(torch.__version__),created_unix_s=time.time(),
           training_cases=[s['workload']['case_id'] for s in train],validation_cases=[s['workload']['case_id'] for s in validation],
           split_contract='Only TRAIN updates parameters. Per architecture and mode, select one checkpoint across all seeds. Diagnostics use validation only; no final test data is read.')
    for name,rows in [('train',train),('validation',validation),('diagnostic',diagnostic)]:
        atomic_json(out/f'{name}_scenarios.json',rows);m[name+'_sha256']=digest(rows)
    for name,content in sources.items():
        path=out/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    atomic_json(out/'manifest.json',m);return m


@contextmanager
def locked(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);path=out/'.run.lock'
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('V8 locked; confirm prior workers stopped before removing .run.lock') from None
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:yield
    finally:path.unlink(missing_ok=True)


def resolve_device(name):
    if name=='auto':return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable; install a compatible PyTorch build or choose cpu')
    return name


def _init_worker():
    torch.set_num_threads(1)


def _rollout(task):
    raw,plan,c,seed,episode,stop=task
    from .expanded import Decision
    s=Scenario.from_dict(raw);p=Decision.from_dict(plan)
    rows=[simulate(s,p,generate_trace(s,seed_for(c['seed'],seed,episode,'v8-training-rollout',i),stop),stop)
          for i in range(c['learning']['rollouts_per_episode'])]
    return objective(s.workload,summarize(s.workload,rows),c['learning']['penalties'])


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    return copy.deepcopy(value)


def save_state(path,state):
    temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)
    atomic_json(path.with_suffix('.json'),{k:v for k,v in state.items() if k not in ('weights','optimizer','baselines')})


def checkpoint_validation(root,job,c,episode,actor,stop_at=None):
    # Reuse the tested resumable validator; it dispatches to the V8 encoder.
    return tuning.checkpoint_validation(root,job,c,episode,actor,stop_at)


def train_job(root, architecture, seed, device='cpu', executor=None, stop_at=None):
    root=Path(root);m=read_json(root/'manifest.json');c=tuning.variant_config(m['config'],architecture)
    lc,t,a=c['learning'],c['tuning'],c['architecture'];job=f'{architecture}-seed-{seed}'
    directory=root/'jobs'/job;directory.mkdir(parents=True,exist_ok=True);path=directory/'state.pt'
    raw=read_json(root/'train_scenarios.json')
    if digest(raw)!=m['train_sha256'] or any(s['split']!='train' for s in raw):raise ValueError('Invalid TRAIN inputs')
    scenarios=[Scenario.from_dict(s) for s in raw]
    dimension=len(features(scenarios[0],action_pool(scenarios[0],c)[0]))
    model=Encoder(architecture,dimension,lc['hidden'],a['layers'],seed).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lc['learning_rate'])
    state=dict(episode=0,seed=seed,architecture=architecture,baselines={},return_sum=0.,simulations=0,elapsed_s=0.,history=[],
               parameter_count=sum(p.numel() for p in model.parameters()),weights=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()))
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(state['weights']);optimizer.load_state_dict(state['optimizer'])
    before=state['elapsed_s'];start=time.monotonic();cache=OrderedDict()
    def save():
        state['elapsed_s']=before+time.monotonic()-start;save_state(path,state)
    def validate_saved(episode):
        cp_path=directory/'checkpoints'/f'{episode}.json'
        if not cp_path.exists():
            if state['episode']!=episode:raise ValueError('Missing historical checkpoint')
            actor=model.to_dict();atomic_json(cp_path,dict(episode=episode,actor=actor,sha256=digest(actor)))
        cp=read_json(cp_path)
        if digest(cp['actor'])!=cp['sha256']:raise ValueError('Modified checkpoint')
        report=directory/'validation'/str(episode)/'summary.json'
        if report.exists():
            if read_json(report)['model_sha256']!=cp['sha256']:raise ValueError('Validation model mismatch')
        else:
            print(f'Validating {job} at {episode} episodes',flush=True)
            checkpoint_validation(root,job,c,episode,cp['actor'],stop_at)
    try:
        for ep in t['checkpoints']:
            if ep<=state['episode']:validate_saved(ep)
        while state['episode']<max(t['checkpoints']):
            check_budget(stop_at);contexts=[];tasks=[]
            # Freeze policy across this on-policy mini-batch. All architectures
            # see identical context/trace seeds; their sampled actions can differ.
            for episode in range(state['episode'],state['episode']+a['batch_episodes']):
                rng=random.Random(seed_for(c['seed'],seed,episode,'v8-training-case'))
                index=rng.randrange(len(scenarios));s=scenarios[index];version=rng.randrange(t['training_pool_versions'])
                key=(index,version)
                if key not in cache:
                    pool=action_pool(s,c,key=f'v8-training-bank-{version}')
                    cache[key]=(pool,inputs(s,pool,architecture))
                cache.move_to_end(key);pool,x=cache[key]
                if len(cache)>a['cache_entries']:cache.popitem(last=False)
                with torch.no_grad():probs=torch.softmax(model.scores(x,a['candidate_chunk_size']),0).cpu().tolist()
                action=rng.choices(range(len(pool)),weights=probs,k=1)[0]
                contexts.append((s,x,action));tasks.append((raw[index],pool[action].to_dict(),c,seed,episode,stop_at))
            losses=list(executor.map(_rollout,tasks)) if executor is not None else [_rollout(task) for task in tasks]
            optimizer.zero_grad(set_to_none=True);baselines=dict(state['baselines']);reward_sum=0.
            for (s,x,action),loss in zip(contexts,losses):
                check_budget(stop_at);reward=-float(loss);key=s.workload.case_id;baseline=baselines.get(key,-1.)
                logits=model.scores(x,a['candidate_chunk_size']);logp=torch.log_softmax(logits,0);probs=logp.exp()
                entropy=-(probs*logp).sum()
                policy_loss=(-(reward-baseline)*logp[action]-lc['entropy']*entropy)/len(contexts)
                if not torch.isfinite(policy_loss):raise FloatingPointError('Nonfinite policy loss')
                policy_loss.backward();baselines[key]=.9*baseline+.1*reward;reward_sum+=reward
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            # State is committed only after every rollout and the complete update.
            # Interruptions before this assignment replay the same batch on resume.
            state=dict(state,weights=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                       episode=state['episode']+len(contexts),baselines=baselines,return_sum=state['return_sum']+reward_sum,
                       simulations=state['simulations']+len(contexts)*lc['rollouts_per_episode'])
            if state['episode']%lc['history_every']==0 or state['episode'] in t['checkpoints']:
                previous=state['history'][-1] if state['history'] else dict(episode=0,return_sum=0.)
                state['history']=state['history']+[dict(episode=state['episode'],return_sum=state['return_sum'],mean_reward=state['return_sum']/state['episode'],
                    recent_mean_reward=(state['return_sum']-previous['return_sum'])/(state['episode']-previous['episode']))]
                print(f"{job}: {state['episode']}/{max(t['checkpoints'])}; recent reward {state['history'][-1]['recent_mean_reward']:.3f}",flush=True)
            if state['episode']%t['save_every_episodes']==0 or state['episode'] in t['checkpoints']:save()
            if state['episode'] in t['checkpoints']:validate_saved(state['episode'])
    finally:save()
    return dict(job=job,episodes=state['episode'])


def summary(root):
    root=Path(root);c=read_json(root/'manifest.json')['config'];jobs=[];rows=[]
    for v in c['tuning']['variants']:
        for seed in c['learning']['seeds']:
            job=f"{v['name']}-seed-{seed}";d=root/'jobs'/job;path=d/'state.json'
            state=read_json(path) if path.exists() else {}
            jobs.append(dict(job=job,variant=v['name'],seed=seed,**{k:state.get(k,[] if k=='history' else 0) for k in ('episode','simulations','elapsed_s','parameter_count','history')}))
            for ep in c['tuning']['checkpoints']:
                path=d/'validation'/str(ep)/'summary.json'
                if path.exists():
                    record=read_json(path);cp=read_json(d/'checkpoints'/f'{ep}.json')
                    if digest(cp['actor'])!=cp['sha256'] or cp['sha256']!=record['model_sha256']:raise ValueError('Modified validation checkpoint')
                    rows.append(dict(variant=v['name'],seed=seed,**record))
    expected=len(c['tuning']['variants'])*len(c['learning']['seeds'])*len(c['tuning']['checkpoints'])
    result=dict(state='finished' if len(rows)==expected else 'incomplete',validation_checkpoints=len(rows),expected_validation_checkpoints=expected,jobs=jobs,validation=rows)
    atomic_json(root/'tuning_summary.json',result);return result


def freeze_selection(root):
    root=Path(root);m=read_json(root/'manifest.json');c=m['config'];report=summary(root)
    if report['state']!='finished':raise ValueError('Finish every architecture/seed/checkpoint before selection')
    metric=c['tuning']['selection_metric'];selections={};actors={};labels={};specs={};groups=[]
    for vi,v in enumerate(c['tuning']['variants']):
        for mi,mode in enumerate(('direct','top_k')):
            options=[]
            for ep in c['tuning']['checkpoints']:
                rs=[r for r in report['validation'] if r['variant']==v['name'] and r['episode']==ep]
                options.append(dict(architecture=v['name'],mode=mode,episode=ep,**{k:stats.mean(r[mode][k] for r in rs) for k in ('objective','completion_time','cost','screen_passed')}))
            best=min(options,key=lambda r:(r[metric],r['objective'],r['completion_time'],r['episode']));groups+=options
            selections[v['name']+'_'+mode]=best
            for seed in c['learning']['seeds']:
                cp=read_json(root/'jobs'/f"{v['name']}-seed-{seed}"/'checkpoints'/f"{best['episode']}.json")
                expected_hash=next(r['model_sha256'] for r in report['validation'] if r['variant']==v['name'] and r['seed']==seed and r['episode']==best['episode'])
                if digest(cp['actor'])!=cp['sha256'] or cp['sha256']!=expected_hash:raise ValueError('Unvalidated checkpoint')
                key=str(10000+vi*2000+mi*1000+seed);actors[key]=cp['actor']
                name=f"{v['name']}_s{seed}"+('' if mode=='direct' else f"_top{c['tuning']['top_k']}")
                labels[key]=name;specs[name]=dict(kind='direct' if mode=='direct' else 'ranked',seed=int(key))
                if mode=='top_k':specs[name]['k']=c['tuning']['top_k']
    selection=dict(selections=selections,groups=groups,selection_metric=metric,rule='One checkpoint per architecture AND deployment mode, averaged over all seeds; no seed selection.',model_hashes={k:digest(v) for k,v in actors.items()})
    bundle=dict(feature_version=1,actors=actors,model_labels=labels,model_specs=specs,source_fingerprint=m['fingerprint'],source_config=c,
                source_training_cases=m['training_cases'],source_learning=dict(selection=selection,tuning=report),excluded_hardware_seeds=[c['seed']])
    path=root/'selected_policies.json'
    if path.exists() and read_json(path)['actors']!=actors:raise ValueError('Frozen actors changed')
    atomic_json(path,bundle);atomic_json(root/'selection.json',selection);atomic_json(root/'selected_policy_hash.json',dict(sha256=digest(bundle)))
    return selection


def run(config=None,output='results/pilot_v8_tuning',workers=4,max_minutes=None):
    c=config if config is not None else read_json(ROOT/'configs/pilot_v8.json');out=Path(output).resolve()
    if type(workers) is not int or workers<1:raise ValueError('Invalid workers')
    if max_minutes is not None and (not math.isfinite(max_minutes) or max_minutes<=0):raise ValueError('Invalid optional timeout')
    stop=None if max_minutes is None else time.monotonic()+max_minutes*60
    with locked(out):
        prepare(c,out);device=resolve_device(c['architecture']['device']);torch.set_num_threads(1)
        execution=dict(workers=workers,device=device,torch=str(torch.__version__),cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name() if device=='cuda' else None,backend='Single model owner; CPU rollout processes',max_minutes=max_minutes)
        ep=out/'execution.json'
        if ep.exists() and any(read_json(ep)[k]!=execution[k] for k in ('workers','device','torch','cuda','gpu')):raise ValueError('Execution backend changed; use a new output')
        atomic_json(ep,execution)
        old={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        for k in old:os.environ[k]='1'
        pool=None
        try:
            if workers>1:pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_init_worker)
            print(f'V8 training on {device}; {workers} CPU rollout workers; no mandatory timeout.',flush=True)
            for v in c['tuning']['variants']:
                for seed in c['learning']['seeds']:
                    check_budget(stop)
                    d=out/'jobs'/f"{v['name']}-seed-{seed}"
                    complete=all((d/'validation'/str(ep)/'summary.json').exists() for ep in c['tuning']['checkpoints'])
                    if not complete:train_job(out,v['name'],seed,device,pool,stop)
                    summary(out)
            freeze_selection(out)
        except BudgetExpired:print('Paused at optional time limit; rerun unchanged.',flush=True)
        finally:
            if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
            for k,val in old.items():
                if val is None:os.environ.pop(k,None)
                else:os.environ[k]=val
            summary(out)
    return {k:v for k,v in summary(out).items() if k not in ('jobs','validation')}


def comparison_source(tuning_output,output,legacy_source=None):
    root=Path(tuning_output);m=read_json(root/'manifest.json');bundle=read_json(root/'selected_policies.json')
    if digest(bundle)!=read_json(root/'selected_policy_hash.json')['sha256']:raise ValueError('Selected policies changed')
    bundle=copy.deepcopy(bundle)
    if legacy_source is not None:
        from .guided_runner import load_source
        lm=read_json(Path(legacy_source)/'manifest.json')
        if lm['experiment']!='comparison_v7':raise ValueError('Legacy source must be the frozen v7 models folder')
        old=read_json(Path(legacy_source)/'comparison_policies.json')
        cfg=copy.deepcopy(m['config']);cfg['seed']+=1;cfg['learning']['seeds']=list(map(int,old['actors']))
        old=load_source(legacy_source,cfg)
        for key,actor in old['actors'].items():
            if key in bundle['actors']:raise ValueError('Overlapping model IDs')
            bundle['actors'][key]=actor;label=old['model_labels'][key];bundle['model_labels'][key]=label
            bundle['model_specs'][label]=dict(kind='direct',seed=int(key))
            bundle['model_specs'][label+f"_top{m['config']['tuning']['top_k']}"]=dict(kind='ranked',seed=int(key),k=m['config']['tuning']['top_k'])
        bundle['excluded_hardware_seeds']+=old['excluded_hardware_seeds'];bundle['source_learning']['legacy']=old['source_learning']
    output=Path(output);output.mkdir(parents=True,exist_ok=True);p=output/'comparison_policies.json'
    if p.exists() and digest(read_json(p))!=digest(bundle):raise ValueError('Comparison source changed; use new folder')
    atomic_json(p,bundle)
    atomic_json(output/'manifest.json',dict(experiment='comparison_v8',fingerprint=digest(bundle),sources=m['sources'],bundle_sha256=digest(bundle)))
    return output


def evaluation_config(tuning_config,source,split='validation'):
    from .guided_runner import method_specs
    c=copy.deepcopy(tuning_config);c.pop('tuning');c.pop('architecture')
    c.update(experiment='guided_v6',seed=tuning_config['seed']+1,instances_per_regime=2,instance_splits=['validation','test'],split=split)
    bundle=read_json(Path(source)/'comparison_policies.json')
    c['learning']['seeds']=list(map(int,bundle['actors']))
    c['architecture_models']=bundle['model_specs']
    c['guided']=dict(top_k=[tuning_config['tuning']['top_k']],candidate_budgets=[32],seconds_budgets=[5],ranking_only=True,heuristic_ranker=True)
    c['methods']=list(method_specs(c));return c


def export(output):
    output=Path(output);bundle=dict(format='learnedplacement-architecture-analysis',manifest=read_json(output/'manifest.json'),tuning=summary(output))
    for key in ('execution','selection','diagnostic_summary'):
        path=output/(key+'.json');bundle[key]=read_json(path) if path.exists() else None
    target=output.with_suffix('.analysis.zip');tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('analysis.json',json.dumps(bundle,separators=(',',':')))
    os.replace(tmp,target);return target


def plot_progress(output):
    import pandas as pd
    import matplotlib.pyplot as plt
    report=summary(output);rows=[]
    for r in report['validation']:
        for mode in ('direct','top_k'):rows.append(dict(architecture=r['variant'],seed=r['seed'],episode=r['episode'],mode=mode,**r[mode]))
    if not rows:return pd.DataFrame()
    df=pd.DataFrame(rows);seeds=read_json(Path(output)/'manifest.json')['config']['learning']['seeds']
    df=df[df.groupby(['architecture','episode','mode']).seed.transform('nunique')==len(seeds)]
    if df.empty:return df
    avg=df.groupby(['architecture','mode','episode']).mean(numeric_only=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for (arch,mode),g in avg.groupby(level=['architecture','mode']):
        for ax,col in zip(axes,['completion_time','objective']):ax.plot(g.index.get_level_values('episode'),g[col],marker='o',label=f'{arch} {mode}')
    for ax,title in zip(axes,['Validation completion time (s)','Validation penalized objective']):ax.set_title(title);ax.set_xlabel('Episodes');ax.grid(alpha=.2);ax.legend(fontsize=7)
    plt.show();return avg


def _diagnostic_case(task):
    root,cid,stop=task;root=Path(root);m=read_json(root/'manifest.json');c=m['config'];a=c['architecture']
    raw=read_json(root/'diagnostic_scenarios.json')
    if digest(raw)!=m['diagnostic_sha256']:raise ValueError('Diagnostic inputs changed')
    value=next(v for v in raw if v['workload']['case_id']==cid)
    s=Scenario.from_dict(value)
    if s.split!='validation':raise ValueError('Diagnostic must use validation only')
    path=root/'diagnostic'/f'{cid}.json';pool=action_pool(s,c);signature=digest([p.to_dict() for p in pool])
    state=read_json(path) if path.exists() else dict(pool_sha256=signature,candidates=[],observations={})
    if state['pool_sha256']!=signature:raise ValueError('Diagnostic pool changed')
    if state.get('finished'):return state['summary']
    traces=[generate_trace(s,seed_for(c['seed'],s.workload.trace_key,'v8-diagnostic-planning',i),stop) for i in range(a['diagnostic_planning_traces'])]
    for i in range(len(state['candidates']),len(pool)):
        check_budget(stop);healthy=simulate(s,pool[i],[],stop)
        q=summarize(s.workload,[simulate(s,pool[i],t,stop) for t in traces])
        state['candidates'].append(dict(index=i,summary=q,healthy_time=healthy['observed_s'],objective=objective(s.workload,q,c['learning']['penalties'])))
        atomic_json(path,state)
    rows=state['candidates'];eligible=[r for r in rows if r['summary']['point_estimate_feasible']]
    candidates={'objective':min(rows,key=lambda r:(r['objective'],r['index']))['index'],
                'fastest':min(rows,key=lambda r:(r['summary']['restricted_mean_s'],r['index']))['index']}
    if eligible:candidates['planning_feasible']=min(eligible,key=lambda r:(r['summary']['restricted_mean_s'],r['index']))['index']
    confirmed={}
    for name,index in candidates.items():
        observed=state['observations'].setdefault(str(index),[])
        try:
            for i in range(len(observed),a['diagnostic_evaluation_traces']):
                check_budget(stop);trace=generate_trace(s,seed_for(c['seed'],s.workload.trace_key,'v8-diagnostic-confirmation',i),stop)
                observed.append(simulate(s,pool[index],trace,stop))
                if len(observed)%8==0:atomic_json(path,state)
        finally:atomic_json(path,state)
        confirmed[name]=dict(index=index,plan=pool[index].to_dict(),summary=summarize(s.workload,observed))
    state['summary']=dict(case_id=cid,available_nodes=len(s.workload.nodes),profile=s.workload.profile,
        deadline_s=s.workload.deadline_s,cost_budget_usd=s.workload.cost_budget_usd,
        network_latency_s=s.network_latency_s,cross_domain_factor=s.workload.cross_domain_factor,
        pool_size=len(pool),planning_point_feasible_candidates=len(eligible),
        minimum_healthy_time_s=min(r['healthy_time'] for r in rows),confirmed=confirmed,
        pool_sha256=signature,planning_candidates=rows)
    state['finished']=True;atomic_json(path,state);return state['summary']


def diagnostic(config=None,output='results/pilot_v8_tuning',workers=4,max_minutes=None):
    """Search the pool on planning traces, independently confirm <=3 choices.

    Absence of a confirmed feasible candidate is not an infeasibility proof.
    No model learns from these validation results.
    """
    c=config if config is not None else read_json(ROOT/'configs/pilot_v8.json');out=Path(output).resolve()
    if type(workers) is not int or workers<1:raise ValueError('Invalid workers')
    if max_minutes is not None and (not math.isfinite(max_minutes) or max_minutes<=0):raise ValueError('Invalid optional timeout')
    stop=None if max_minutes is None else time.monotonic()+max_minutes*60
    with locked(out):
        m=prepare(c,out);raw=read_json(out/'diagnostic_scenarios.json');tasks=[(str(out),v['workload']['case_id'],stop) for v in raw]
        results=[]
        try:
            if workers==1:results=[_diagnostic_case(t) for t in tasks]
            else:
                with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_init_worker) as pool:
                    for result in pool.map(_diagnostic_case,tasks):
                        results.append(result);print('Diagnostic:',result['case_id'],flush=True)
        except BudgetExpired:print('Diagnostic paused; rerun unchanged.',flush=True)
        finally:
            results=[]
            for _,cid,_ in tasks:
                p=out/'diagnostic'/f'{cid}.json'
                if p.exists() and read_json(p).get('finished'):results.append(read_json(p)['summary'])
            report=dict(state='finished' if len(results)==len(tasks) else 'incomplete',completed_cases=len(results),total_cases=len(tasks),cases=results,
                        interpretation='Validation-only diagnostic. Planning feasibility is a point estimate; independent confirmation covers at most three nominated plans, not the whole feasible space. No infeasibility or optimality proof. Healthy times are references, not rigorous lower bounds.')
            atomic_json(out/'diagnostic_summary.json',report)
    return {k:v for k,v in report.items() if k!='cases'}
