"""Resumable v5: actual bandit-policy training, then frozen-policy evaluation."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import random
import math
import numpy as np

from .expanded import Scenario, Decision, make_scenarios, generate_trace, simulate, violations
from .expanded_planners import plan as heuristic_plan, digest
from .expanded_runner import run as evaluation_run, validate_config as validate_v4, plot_results
from .learning import Network, features, matrix, action_pool, objective, new_training_state, train_episode, FEATURE_VERSION
from .metrics import summarize
from .model import seed_for
from .runner import ROOT, atomic_json, read_json, source_snapshot, status, export_run
from .simulator import check_budget

BASE_METHODS=['speed_fixed','domain_tuned','coordinate_search','heuristic_portfolio','local_search','random_search','random_unaware']


def methods(config):
    return BASE_METHODS+[f'rl_seed_{seed}' for seed in config['learning']['seeds']]+['supervised_control']


def validate_config(config):
    c=dict(config,experiment='expanded_v4',methods=['speed_fixed','domain_tuned','coordinate_search','random_search','local_search','random_unaware'])
    validate_v4(c)
    if config.get('experiment')!='learning_v5':raise ValueError('Expected learning_v5')
    lc=config['learning']
    for key in ('episodes','rollouts_per_episode','action_count','hidden','history_every'):
        if type(lc[key]) is not int or lc[key]<=0:raise ValueError(f'Invalid {key}')
    if lc['action_count']<2:raise ValueError('At least two actions required')
    if not lc['seeds'] or len(set(lc['seeds']))!=len(lc['seeds']) or any(type(n) is not int or n<0 for n in lc['seeds']):raise ValueError('Invalid learning seeds')
    for key in ('learning_rate','entropy'):
        if not math.isfinite(lc[key]) or lc[key]<0:raise ValueError(f'Invalid {key}')
    if lc['learning_rate']==0:raise ValueError('learning_rate must be positive')
    if set(lc['penalties'])!={'cost','deadline','censor'} or any(not math.isfinite(x) or x<=0 for x in lc['penalties'].values()):raise ValueError('Invalid penalties')
    if not {'train','validation','test'}<=set(config['instance_splits']):raise ValueError('Separate train, validation and test instances required')
    if config['methods']!=methods(config):raise ValueError('Missing or misordered learning methods')


def prepare(config,output):
    validate_config(config)
    sources=source_snapshot()
    hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
    fingerprint=digest(dict(config=config,sources=hashes,numpy=np.__version__))
    path=output/'manifest.json'
    if path.exists():
        m=read_json(path)
        if m['fingerprint']!=fingerprint:raise ValueError('Configuration/source/NumPy changed. Choose a NEW output directory.')
        return m
    scenarios=make_scenarios(config)
    train=[s for s in scenarios if s.split=='train'];evaluation=[s for s in scenarios if s.split!='train']
    for s in evaluation:atomic_json(output/'cases'/s.workload.case_id/'case.json',s.to_dict())
    atomic_json(output/'learning'/'train_scenarios.json',[s.to_dict() for s in train])
    for name,content in sources.items():
        p=output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    m=dict(schema=5,experiment='learning_v5',config=config,sources=hashes,fingerprint=fingerprint,
           numpy=np.__version__,cases=[s.workload.case_id for s in evaluation],methods=methods(config),
           training_cases=[s.workload.case_id for s in train],training_inputs_sha256=digest([s.to_dict() for s in train]),created_unix_s=time.time(),
           split_contract='Training updates and regression labels use TRAIN only. Final fixed-episode policies are frozen before validation/test. No validation/test model selection.',
           caveat='One-step contextual-bandit RL selecting feasible full plans. Synthetic training-workload simulation; no tensors/Ray/cloud/Z3. Policies are not fault-time controllers.')
    atomic_json(path,m);return m


def training_summary(states, config):
    return dict(algorithm='REINFORCE contextual bandit with entropy regularization',feature_version=FEATURE_VERSION,
                episodes_per_seed=config['learning']['episodes'],penalties=config['learning']['penalties'],
                seeds=[dict(seed=s['seed'],episodes=s['episode'],simulations=s['simulations'],elapsed_s=s['elapsed_s'],
                            mean_training_reward=s['return_sum']/max(1,s['episode']),policy_sha256=digest(s['actor']),
                            regressor_sha256=digest(s['regressor']),history=s['history']) for s in states],
                supervised_control='Same sampled state-action rewards as first RL seed; Huber regression, not optimal labels.',
                model_selection='Final fixed-episode checkpoints; no validation/test tuning.')


class Experiment:
    def __init__(self):self.states={};self.output=None;self.models={}

    def train(self,config,output,stop_at):
        self.output=output;lc=config['learning']
        raw=read_json(output/'learning/train_scenarios.json')
        scenarios=[Scenario.from_dict(x) for x in raw]
        if not scenarios or any(s.split!='train' for s in scenarios):raise ValueError('Training attempted on non-train case')
        manifest=read_json(output/'manifest.json')
        if digest(raw)!=manifest['training_inputs_sha256']:raise ValueError('Training inputs changed; choose a new output directory')
        frozen_path=output/'learning/frozen_models.json'
        frozen=read_json(frozen_path) if frozen_path.exists() else None
        for seed in lc['seeds']:
            check_budget(stop_at)
            path=output/'learning'/f'seed-{seed}.json'
            if path.exists():state=read_json(path)
            else:
                p=action_pool(scenarios[0],config)[0]
                state=new_training_state(len(features(scenarios[0],p)),config,seed)
            if frozen is not None:
                expected=frozen[str(seed)]
                if state['episode']!=lc['episodes'] or digest(state['actor'])!=expected['actor'] or digest(state['regressor'])!=expected['regressor']:
                    raise ValueError('Frozen model changed; choose a NEW output directory')
            print(f'RL seed {seed}: {state["episode"]}/{lc["episodes"]} episodes saved',flush=True)
            started=time.monotonic();previous_elapsed=state['elapsed_s']
            try:
                while state['episode']<lc['episodes']:
                    check_budget(stop_at)
                    episode=state['episode']
                    rng=random.Random(seed_for(config['seed'],seed,episode,'train-case'))
                    s=rng.choice(scenarios)
                    # Fresh candidate sets across episodes discourage index memorization.
                    pool=action_pool(s,config,key=f'train-{seed}-{episode}')
                    x=matrix(s,pool);actor=Network.from_dict(state['actor'])
                    probs=actor.probabilities(x)
                    action=rng.choices(range(len(pool)),weights=probs.tolist(),k=1)[0]
                    rows=[]
                    for i in range(lc['rollouts_per_episode']):
                        trace_seed=seed_for(config['seed'],seed,episode,'rl-training-rollout',i)
                        trace=generate_trace(s,trace_seed,stop_at)
                        rows.append(simulate(s,pool[action],trace,stop_at))
                    loss=objective(s.workload,summarize(s.workload,rows),lc['penalties'])
                    # Atomic episode: no parameter update until ALL rollouts complete.
                    state=train_episode(s,pool,state,config,loss,action)
                    state['simulations']+=len(rows)
                    state['elapsed_s']=previous_elapsed+time.monotonic()-started
                    if state['episode']%lc['history_every']==0 or state['episode']==lc['episodes']:
                        state['history'].append(dict(episode=state['episode'],mean_reward=state['return_sum']/state['episode'],last_reward=-loss))
                        print(f'  seed {seed}: episode {state["episode"]}, average reward {state["return_sum"]/state["episode"]:.4f}',flush=True)
                    atomic_json(path,state)
                    atomic_json(output/'status.json',dict(state='running',stage='training',learning_seed=seed,episodes=state['episode'],target_episodes=lc['episodes']))
            finally:
                state['elapsed_s']=previous_elapsed+time.monotonic()-started
                atomic_json(path,state)
                self.states[seed]=state
                atomic_json(output/'learning_summary.json',training_summary(list(self.states.values()),config))
            self.models[seed]=Network.from_dict(state['actor'])
        model_hashes={str(seed):dict(actor=digest(state['actor']),regressor=digest(state['regressor'])) for seed,state in self.states.items()}
        if frozen is not None and frozen!=model_hashes:raise ValueError('Frozen models changed')
        atomic_json(frozen_path,model_hashes)
        first=lc['seeds'][0]
        self.regressor=Network.from_dict(self.states[first]['regressor'])
        atomic_json(output/'learning_summary.json',training_summary(list(self.states.values()),config))

    def plan(self,s,method,traces,config,prior,stop_at):
        penalties=config['learning']['penalties']
        score=lambda c,summary:(objective(c,summary,penalties),summary['restricted_mean_s'])
        started=time.monotonic()
        if method in ('speed_fixed','domain_tuned','coordinate_search','local_search'):
            parent='heuristic_portfolio' if method=='local_search' else 'domain_tuned' if method=='coordinate_search' else None
            initial=Decision.from_dict(prior[parent]['plan']) if parent else None
            r=heuristic_plan(s,method,traces,config,prior,stop_at,initial_plan=initial,score_fn=score)
            r['own_planning_s']=r['planning_s']
            if parent:
                r['parent']=parent;r['parent_planning_s']=prior[parent]['planning_s']
                r['planning_s']+=r['parent_planning_s']
                r['planning_time_kind']='measured_search_plus_required_parent'
            return r
        if method=='heuristic_portfolio':
            choices=['speed_fixed','domain_tuned','coordinate_search']
            if any(prior[m]['search_summary'] is None for m in choices):
                candidates=[m for m in choices if prior[m]['search_summary'] is not None]
            else:candidates=choices
            winner=min(candidates,key=lambda m:score(s.workload,prior[m]['search_summary'])) if candidates else 'domain_tuned'
            r=prior[winner]
            return dict(plan=r['plan'],planning_s=time.monotonic()-started+prior['speed_fixed']['planning_s']+prior['coordinate_search']['planning_s'],
                        candidates=sum(prior[m]['candidates'] for m in choices),simulations=sum(prior[m]['simulations'] for m in choices),
                        search_summary=r['search_summary'],planning_point_feasible=r.get('planning_point_feasible'),
                        status='planning_only_portfolio_selection',parent=winner,planning_time_kind='sum_of_required_heuristics')
        if method=='random_unaware':return prior['random_search']['controls']['random_unaware']
        pool=action_pool(s,config)
        pool_hash=digest([p.to_dict() for p in pool])
        if method=='random_search':
            # Entire common action pool must be scored. Save completed candidates
            # so a global timeout can resume without changing the RL action space.
            path=self.output/'cases'/s.workload.case_id/'pool_search_progress.json'
            progress=read_json(path) if path.exists() else dict(pool_sha256=pool_hash,rows=[],elapsed_s=0.,healthy_s=0.)
            if progress['pool_sha256']!=pool_hash:raise ValueError('Action pool changed')
            old_time=progress['elapsed_s']
            try:
                for p in pool[len(progress['rows']):]:
                    check_budget(stop_at)
                    tick=time.monotonic()
                    healthy=simulate(s,p,[],stop_at)
                    progress['healthy_s']+=time.monotonic()-tick
                    rows=[simulate(s,p,t,stop_at) for t in traces]
                    summary=summarize(s.workload,rows);hs=summarize(s.workload,[healthy])
                    progress['rows'].append(dict(plan=p.to_dict(),summary=summary,healthy_summary=hs,
                                                 score=list(score(s.workload,summary)),healthy_score=list(score(s.workload,hs))))
                    progress['elapsed_s']=old_time+time.monotonic()-started
                    atomic_json(path,progress)
            finally:
                progress['elapsed_s']=old_time+time.monotonic()-started;atomic_json(path,progress)
            best=min(progress['rows'],key=lambda x:tuple(x['score']));healthy=min(progress['rows'],key=lambda x:tuple(x['healthy_score']))
            common=dict(candidates=len(pool),candidate_pool_sha256=pool_hash,status='complete_common_pool',planning_point_feasible=None)
            u=dict(common,plan=healthy['plan'],planning_s=progress['healthy_s'],simulations=len(pool),
                   search_summary=healthy['healthy_summary'],planning_time_kind='attributed_healthy_scoring_only')
            return dict(common,plan=best['plan'],planning_s=progress['elapsed_s'],simulations=len(pool)*(len(traces)+1),
                        search_summary=best['summary'],planning_time_kind='measured_complete_pool_search',
                        candidate_audit=progress['rows'],controls={'random_unaware':u})
        x=matrix(s,pool)
        if method.startswith('rl_seed_'):
            seed=int(method.removeprefix('rl_seed_'));model=self.models[seed]
            scores,_=model.forward(x);model_hash=digest(self.states[seed]['actor'])
        elif method=='supervised_control':
            scores,_=self.regressor.forward(x)
            model_hash=digest(self.states[config['learning']['seeds'][0]]['regressor'])
        else:raise ValueError('Unknown learning method')
        index=int(np.argmax(scores));p=pool[index]
        if violations(s,p):raise AssertionError('Policy action failed structural validation')
        return dict(plan=p.to_dict(),planning_s=time.monotonic()-started,candidates=len(pool),simulations=0,
                    candidate_pool_sha256=pool_hash,model_sha256=model_hash,search_summary=None,
                    planning_point_feasible=None,status='frozen_model_inference',
                    planning_time_kind='measured_candidate_generation_features_and_inference',
                    selected_action=index)


def run(config=None,output='results/pilot_v5',max_minutes=30,config_path=None):
    cfg=config if config is not None else read_json(config_path or ROOT/'configs/pilot_v5.json')
    experiment=Experiment()
    return evaluation_run(config=cfg,output=output,max_minutes=max_minutes,prepare_fn=prepare,
                          before_evaluate=experiment.train,plan_fn=experiment.plan,trace_namespace='v5')


def export(output):
    # runner.export_run automatically includes the small learning summary in v5.
    return export_run(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/pilot_v5.json'))
    parser.add_argument('--output',default='results/pilot_v5')
    parser.add_argument('--max-minutes',type=float,default=30)
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    print(json.dumps(status(args.output) if args.status else run(config_path=args.config,output=args.output,max_minutes=args.max_minutes),indent=2))


if __name__=='__main__':main()
