"""NumPy contextual-bandit RL and a same-experience supervised control.

One complete plan is one action; reward comes from simulator rollouts. This is
REINFORCE, not imitation of optimal labels and not dynamic failure-time control.
No tensor training workload is executed. NumPy is only required for v5.
"""
from dataclasses import replace
import math
import random
import numpy as np

from .expanded import Decision, violations
from .expanded_planners import base_plan, layout, valid_candidates, digest
from .model import seed_for

FEATURE_VERSION = 1


def objective(c, summary, weights):
    """A finite, shared penalized objective, lower is better (no guarantees)."""
    return (summary['restricted_mean_s']/c.deadline_s
            + weights['cost']*max(0.,summary['mean_observed_cost_usd']/c.cost_budget_usd-1.)
            + weights['deadline']*max(0.,c.required_success-summary['deadline_success'])
            + weights['censor']*(1.-summary['completed_fraction']))


def action_pool(s, config, key='deployment'):
    """Identical deterministic feasible candidates for learned/pool-search policies.

    Cheap speed/domain/price/risk seeds and cadence variants precede random
    proposals. No simulator results or test traces guide pool construction.
    """
    pool=[]
    for kind in ('speed','domain','price','risk'):
        p=layout(s,s.workload.shards,s.workload.replicas,kind)
        if p is not None:
            for interval in config['choices']['checkpoint_every']:
                q=replace(p,checkpoint_every=interval)
                if not violations(s,q) and q not in pool: pool.append(q)
    seed=seed_for(config['seed'],s.workload.trace_key,s.workload.profile,key,'action-pool')
    for p in valid_candidates(s,config,seed,config['learning']['action_count']):
        if p not in pool: pool.append(p)
    if not pool: pool=[base_plan(s,config)]
    return pool[:config['learning']['action_count']]


def features(s,p):
    """Deterministic dimensionless inputs; no IDs/splits/outcomes or fitted stats.

    Log transforms are fixed a priori. Domain collision and topology features
    expose sharded redundancy rather than relying on candidate list positions.
    """
    c=s.workload; shards=len(p.groups[0]); replicas=len(p.groups)
    nodes=[c.nodes[n] for n in sorted(p.active)]
    def stats(values):
        return [min(values),sum(values)/len(values),max(values)]
    price=sum(c.nodes[n].price_per_hour for n in p.allocated)
    speed=sum(min(c.nodes[n].speed for n in g) for g in p.groups)
    compute=c.base_step_s*c.shards*c.replicas/(shards*speed)
    fail=sum(n.failure_per_hour for n in nodes)*s.hazard_multiplier
    collisions=0; domain_counts=[]
    for j in range(shards):
        holders=[g[j] for g in p.groups]+[n for k,n in p.backups if k==j]
        domains=[c.nodes[n].domain for n in holders]
        collisions+=sum(domains[a]==domains[b] for a in range(len(domains)) for b in range(a))
        domain_counts.append(len(set(domains)))
    raw=[len(c.nodes)/24, len(p.active)/24,len(p.allocated)/24,shards/8,replicas/4,
         c.shards/8,c.replicas/4,c.state_gib/16,c.working_gib/4,c.steps/240,
         c.base_step_s/3,c.deadline_s/1000,c.cost_budget_usd/3,
         c.max_simulated_s/c.deadline_s,c.storage_gib_s/2,s.local_disk_gib_s,
         c.cross_domain_factor,c.collective_factor,s.network_latency_s*1000,
         c.replacement_s/30,c.reinit_s/4,c.detection_s/2,
         c.domain_failure_per_hour*s.hazard_multiplier,fail,
         len(p.backups)/shards,len(p.spares)/2,
         p.checkpoint_every/c.steps,p.backup_every/c.steps,p.checkpoint_retention/2,
         p.background_fraction,p.repair_every/c.steps,
         float(p.checkpoint_tier=='local'),float(p.checkpoint_mode=='async'),
         float(p.recovery=='defer'),float(p.recovery_order=='smallest'),float(p.source_policy=='first'),
         sum(len({c.nodes[n].domain for n in g}) for g in p.groups)/(replicas*shards),
         collisions/max(1,shards*replicas*replicas),min(domain_counts)/4,
         price,compute,compute*c.steps/c.deadline_s,
         price*compute*c.steps/3600/c.cost_budget_usd,
         fail*compute*c.steps/3600,
         c.state_gib/c.storage_gib_s/max(1.,p.checkpoint_every*compute),
         c.state_gib/shards/min(n.memory_gib for n in nodes),
         float(s.outage_distribution=='lognormal'),s.outage_cv]
    raw+=stats([n.speed for n in nodes])+stats([n.bandwidth_gib_s/4 for n in nodes])
    raw+=stats([n.memory_gib/24 for n in nodes])+stats([n.failure_per_hour*s.hazard_multiplier for n in nodes])
    # Preserve nonnegative feature ordering and bound extreme scales smoothly.
    return np.log1p(np.asarray(raw,dtype=np.float64))


def matrix(s,pool):
    return np.stack([features(s,p) for p in pool])


class Network:
    """Small shared candidate scorer: tanh MLP, explicit gradients and Adam."""
    def __init__(self, dimension, hidden=32, seed=0):
        rng=np.random.default_rng(seed)
        self.params={'w':rng.normal(0,1/math.sqrt(dimension),(dimension,hidden)),
                     'b':np.zeros(hidden),'v':rng.normal(0,.05,(hidden,)),'a':np.zeros(1)}
        self.m={k:np.zeros_like(v) for k,v in self.params.items()}
        self.v={k:np.zeros_like(v) for k,v in self.params.items()}
        self.steps=0

    def forward(self,x):
        h=np.tanh(x@self.params['w']+self.params['b'])
        return h@self.params['v']+self.params['a'][0],h

    def gradients(self,x,output_gradient):
        _,h=self.forward(x)
        g=np.asarray(output_gradient)
        dh=g[:,None]*self.params['v'][None,:]*(1-h*h)
        return {'w':x.T@dh,'b':dh.sum(axis=0),'v':h.T@g,'a':np.array([g.sum()])}

    def update(self,grads,lr):
        norm=math.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        scale=min(1.,5./max(norm,1e-12))
        self.steps+=1
        for k,g in grads.items():
            g=g*scale
            self.m[k]=.9*self.m[k]+.1*g
            self.v[k]=.999*self.v[k]+.001*g*g
            mh=self.m[k]/(1-.9**self.steps);vh=self.v[k]/(1-.999**self.steps)
            self.params[k]-=lr*mh/(np.sqrt(vh)+1e-8)

    def probabilities(self,x):
        logits,_=self.forward(x)
        e=np.exp(logits-np.max(logits))
        return e/e.sum()

    def policy_update(self,x,action,advantage,lr,entropy):
        probs=self.probabilities(x)
        # Gradient of -advantage*log pi(action) - entropy_weight*H(pi).
        g=probs.copy();g[action]-=1.;g*=advantage
        logp=np.log(np.maximum(probs,1e-300))
        g+=entropy*probs*(logp-float(probs@logp))
        self.update(self.gradients(x,g),lr)

    def regression_update(self,x,target,lr):
        prediction,_=self.forward(x)
        # Huber derivative bounds the effect of rare extreme failure returns.
        g=np.clip(prediction-target,-1.,1.)/len(x)
        self.update(self.gradients(x,g),lr)

    def to_dict(self):
        return dict(params={k:v.tolist() for k,v in self.params.items()},
                    m={k:v.tolist() for k,v in self.m.items()},v={k:v.tolist() for k,v in self.v.items()},steps=self.steps)

    @classmethod
    def from_dict(cls,d):
        obj=cls.__new__(cls)
        for key in ('params','m','v'):setattr(obj,key,{k:np.asarray(v,dtype=np.float64) for k,v in d[key].items()})
        obj.steps=d['steps'];return obj


def new_training_state(dimension,config,seed):
    lc=config['learning']
    return dict(schema=1,feature_version=FEATURE_VERSION,episode=0,seed=seed,
                actor=Network(dimension,lc['hidden'],seed).to_dict(),
                regressor=Network(dimension,lc['hidden'],seed+100000).to_dict(),
                baselines={},return_sum=0.,simulations=0,elapsed_s=0.,history=[])


def train_episode(s, pool, state, config, returns, action):
    """Commit one completed on-policy interaction; incomplete rollouts do not update."""
    lc=config['learning'];x=matrix(s,pool)
    actor=Network.from_dict(state['actor']);reg=Network.from_dict(state['regressor'])
    reward=-float(returns)
    key=s.workload.case_id
    # Action-independent baseline from PREVIOUS visits, not the current reward.
    baseline=state['baselines'].get(key,-1.)
    actor.policy_update(x,action,reward-baseline,lc['learning_rate'],lc['entropy'])
    # Exactly the same selected-action feedback, with no oracle labels.
    reg.regression_update(x[action:action+1],reward,lc['learning_rate'])
    result=dict(state,actor=actor.to_dict(),regressor=reg.to_dict(),episode=state['episode']+1,
                return_sum=state['return_sum']+reward,baselines=dict(state['baselines']))
    result['baselines'][key]=.9*baseline+.1*reward
    return result
