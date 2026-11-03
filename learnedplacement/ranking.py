"""Cheap analytical ranking control; no event rollouts or learned weights."""
import math
from .expanded import ExpandedSimulation


def heuristic_score(s,p,weights):
    c=s.workload
    # Evaluate only the algebraic healthy step expression, not the event loop.
    step=ExpandedSimulation(s,p,[]).step_seconds(range(len(p.groups)))
    shards=len(p.groups[0]); shard=c.state_gib/shards
    checkpoint=(c.state_gib/c.storage_gib_s if p.checkpoint_tier=='remote'
                else shard/s.local_disk_gib_s)
    writes=(c.steps-1)//p.checkpoint_every
    saving=checkpoint*writes
    if p.checkpoint_mode=='async':saving*=p.background_fraction
    backup=len(p.backups)*shard/min(c.nodes[n].bandwidth_gib_s for n in p.active)
    saving+=backup*((c.steps-1)//p.backup_every)
    healthy=c.steps*step+saving
    # First-order recovery estimate: redundancy reduces exposure; domain sharing
    # reduces effective redundancy. This is a ranking approximation, not a bound.
    effective=[]
    for j in range(shards):
        hosts=[g[j] for g in p.groups]+[n for k,n in p.backups if k==j]
        effective.append(len({c.nodes[n].domain for n in hosts}))
    redundancy=max(1.,sum(effective)/shards)
    rate=s.hazard_multiplier*(sum(c.nodes[n].failure_per_hour for n in p.active)/redundancy
                             +c.domain_failure_per_hour/max(1.,redundancy))
    lost=.5*p.checkpoint_every*step
    recovery=c.detection_s+c.reinit_s+checkpoint+c.replacement_s/(1+len(p.spares))
    expected=healthy*(1+rate*(lost+recovery)/3600)
    price=sum(c.nodes[n].price_per_hour for n in p.allocated)
    cost=expected*price/3600
    if p.checkpoint_tier=='remote':
        cost+=c.state_gib*p.checkpoint_retention*s.remote_storage_usd_gib_hour*expected/3600
        cost+=writes*c.state_gib*s.remote_write_usd_gib
    return (expected/c.deadline_s+weights['cost']*max(0.,cost/c.cost_budget_usd-1)
            +weights['deadline']*max(0.,expected/c.deadline_s-1),expected)
