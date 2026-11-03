"""Pilot v4: configurable resources, sharding, protection and recovery.

An analytical fixed-work surrogate, NOT a tensor execution engine. No RL yet.
All runtime decisions below use observed state, never future trace entries.
"""
from dataclasses import asdict, dataclass, replace
import math
import random

from .model import Case, Plan, seed_for, make_cases as old_cases
from .simulator import Simulation, Event, check_budget, common_version


@dataclass
class Scenario:
    workload: Case
    split: str = 'train'
    network_latency_s: float = .001
    local_disk_gib_s: float = 1.0
    local_disk_gib: float = 64.0
    remote_write_usd_gib: float = .00001
    remote_storage_usd_gib_hour: float = .00003
    outage_distribution: str = 'fixed'
    outage_cv: float = .5
    hazard_multiplier: float = 1.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        d['workload'] = Case.from_dict(d['workload'])
        return cls(**d)


@dataclass(frozen=True)
class Decision:
    groups: tuple[tuple[int, ...], ...]
    backups: tuple[tuple[int, int], ...] = ()
    spares: tuple[int, ...] = ()
    checkpoint_every: int = 80
    backup_every: int = 20
    checkpoint_tier: str = 'remote'  # remote durable, or per-rank local disks
    checkpoint_mode: str = 'sync'
    checkpoint_retention: int = 1
    background_fraction: float = .25
    recovery: str = 'eager'         # eager repair, or defer while any group lives
    repair_every: int = 1          # observed committed-step boundary
    recovery_order: str = 'fastest'
    source_policy: str = 'balanced'

    @property
    def active(self):
        return {n for g in self.groups for n in g}

    @property
    def allocated(self):
        return self.active | {n for _, n in self.backups} | set(self.spares)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        for key in ('groups', 'backups'):
            d[key] = tuple(tuple(x) for x in d.get(key, ()))
        d['spares'] = tuple(d.get('spares', ()))
        return cls(**d)


def effective_case(s, p):
    c = s.workload
    shards, replicas = len(p.groups[0]), len(p.groups)
    # Preserve total unit-speed worker-seconds per global step across layouts.
    return replace(c, shards=shards, replicas=replicas,
                   base_step_s=c.base_step_s*c.shards*c.replicas/(shards*replicas),
                   backup_every=p.backup_every, checkpoint_every=p.checkpoint_every)


def validate_scenario(s):
    from .model import validate_case
    validate_case(s.workload)
    if s.split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown scenario split')
    if s.outage_distribution not in ('fixed', 'lognormal'):
        raise ValueError('Unknown outage distribution')
    for k in ('network_latency_s', 'remote_write_usd_gib', 'remote_storage_usd_gib_hour',
              'outage_cv', 'hazard_multiplier'):
        if not math.isfinite(getattr(s, k)) or getattr(s, k) < 0:
            raise ValueError(f'Invalid {k}')
    if any(not math.isfinite(v) or v <= 0 for v in (s.local_disk_gib_s,s.local_disk_gib)):
        raise ValueError('Local disk bandwidth/capacity must be positive')


def violations(s, p):
    c = s.workload
    errors = []
    if not p.groups or not p.groups[0] or any(len(g) != len(p.groups[0]) for g in p.groups):
        return ['nonempty equal-sized groups required']
    ids = [n for g in p.groups for n in g]
    other = list(p.spares) + [n for _, n in p.backups]
    if any(type(n) is not int or not 0 <= n < len(c.nodes) for n in ids+other):
        return ['invalid node']
    if len(set(ids)) != len(ids) or len(set(p.spares)) != len(p.spares):
        errors.append('duplicate active or spare assignment')
    if p.active & set(p.spares):
        errors.append('spare is already active')
    if len(set(p.backups)) != len(p.backups):
        errors.append('duplicate backup copy')
    for k in ('checkpoint_every', 'backup_every', 'checkpoint_retention', 'repair_every'):
        if type(getattr(p, k)) is not int or getattr(p, k) <= 0:
            errors.append(f'{k} must be a positive integer')
    enums = dict(checkpoint_tier=('remote', 'local'), checkpoint_mode=('sync', 'async'),
                 recovery=('eager', 'defer'), recovery_order=('fastest', 'smallest'),
                 source_policy=('balanced', 'first'))
    for k, values in enums.items():
        if getattr(p, k) not in values:
            errors.append(f'unknown {k}')
    if not 0 < p.background_fraction < 1:
        errors.append('background fraction must be between zero and one')
    shard_gib = c.state_gib/len(p.groups[0])
    usage = [0.]*len(c.nodes)
    for n in p.active | set(p.spares):
        usage[n] += shard_gib + c.working_gib
        if p.checkpoint_mode == 'async':
            usage[n] += shard_gib  # frozen outgoing snapshot; one job in flight
    for shard, n in p.backups:
        if type(shard) is not int or not 0 <= shard < len(p.groups[0]):
            return errors + ['invalid backup shard']
        if any(g[shard] == n for g in p.groups) or n in p.spares:
            errors.append('backup overlaps its primary or a reserved spare')
        usage[n] += 2*shard_gib  # committed copy plus staging
    if len(p.backups)*shard_gib > c.backup_budget_gib + 1e-9:
        errors.append('backup budget exceeded')
    if any(v > n.memory_gib + 1e-9 for v, n in zip(usage, c.nodes)):
        errors.append('node memory exceeded including staging')
    if p.checkpoint_tier == 'local' and shard_gib*p.checkpoint_retention > s.local_disk_gib:
        errors.append('local checkpoint capacity exceeded')
    return errors


def generate_trace(s, seed, stop_at=None):
    """Common exogenous node/domain starts and repairs for EVERY plan.

    Lognormal applies to outage duration, not interarrival times. Failure starts
    remain Poisson. Correlated outage members share one duration.
    """
    c = s.workload
    events = []
    def process(rate, members, key):
        rate *= s.hazard_multiplier
        if rate == 0:
            return
        starts = random.Random(seed_for(seed, key, 'start'))
        repairs = random.Random(seed_for(seed, key, 'repair'))
        t = starts.expovariate(rate/3600)
        while t < c.max_simulated_s:
            check_budget(stop_at)
            duration = c.replacement_s
            if s.outage_distribution == 'lognormal' and duration:
                sigma = math.sqrt(math.log(1+s.outage_cv**2))
                duration = repairs.lognormvariate(math.log(duration)-sigma*sigma/2, sigma)
            for n in members:
                events.extend((Event(t, 0, n), Event(t+duration, 1, n)))
            t += starts.expovariate(rate/3600)
    for n, node in enumerate(c.nodes):
        process(node.failure_per_hour, [n], ('node', n))
    for domain in sorted({n.domain for n in c.nodes}):
        process(c.domain_failure_per_hour, [i for i,n in enumerate(c.nodes) if n.domain == domain], ('domain', domain))
    return sorted(events)


class ExpandedSimulation(Simulation):
    def __init__(self, s, p, events, stop_at=None, audit=False):
        errors = violations(s, p)
        if errors:
            raise ValueError('; '.join(errors))
        c = effective_case(s, p)
        super().__init__(c, Plan(p.groups, checkpoint_every=p.checkpoint_every), events, stop_at, audit)
        self.s, self.p = s, p
        self.reserved = set(p.allocated)
        self.backup = {(shard,n): None for shard,n in p.backups}
        self.local = {}  # (shard, node) -> retained complete versions
        self.remote = [0]  # initial version 0 is always durable and free
        self.pending = None
        self.last_checkpoint = -1
        self.checkpoint_writes = self.checkpoint_aborts = self.remaps = 0
        self.write_gib = self.overlap_s = 0.
        self.initial_rate = sum(c.nodes[n].price_per_hour for n in self.reserved)

    def handle(self, event):
        failed = super().handle(event)
        if failed is not None:
            # Node loss also destroys local disks. This is an explicit assumption.
            for key in list(self.local):
                if key[1] == failed:
                    del self.local[key]
            if self.pending and failed in self.pending['participants']:
                self.checkpoint_aborts += 1
                self.pending = None
        return failed

    def finish_checkpoint(self, job):
        version = job['version']
        if self.p.checkpoint_tier == 'remote':
            self.remote = [0] + sorted(set(self.remote+[version])-{0})[-self.p.checkpoint_retention:]
            self.persistent = max(self.remote)
        else:
            for shard, n in job['destinations']:
                key = (shard, n)
                self.local[key] = sorted(set(self.local.get(key, [])+[version]))[-self.p.checkpoint_retention:]
        self.checkpoint_writes += 1
        self.record('checkpoint_complete', snapshot_version=version, tier=self.p.checkpoint_tier)

    def advance(self, duration, phase, critical=()):
        check_budget(self.stop_at)
        start, target = self.t, self.t+duration
        end = min(target, self.c.max_simulated_s)
        critical = set(critical)
        while self.t <= end:
            check_budget(self.stop_at)
            event_time = self.events[self.e].time_s if self.e < len(self.events) else math.inf
            finish_time = self.pending['finish'] if self.pending else math.inf
            next_time = min(event_time, finish_time, end)
            if next_time < self.t-1e-9:
                raise ValueError('Events must be chronological')
            if self.pending:
                self.overlap_s += next_time-self.t
            self.t = next_time
            aborted = False
            # Loss wins a tie with checkpoint completion.
            while self.e < len(self.events) and self.events[self.e].time_s == self.t:
                failed = self.handle(self.events[self.e]); self.e += 1
                aborted = aborted or failed in critical
            if self.pending and self.pending['finish'] <= self.t:
                job, self.pending = self.pending, None
                self.finish_checkpoint(job)
            if aborted or self.t >= end:
                self.seconds[phase] += self.t-start
                return not aborted and target <= self.c.max_simulated_s

    def sources(self, version):
        result = [[] for _ in range(self.c.shards)]
        for n,v in self.primary.items():
            if not self.down[n] and v == version:
                result[self.shard[n]].append(n)
        for (shard,n),v in self.backup.items():
            if not self.down[n] and v == version:
                result[shard].append(n)
        for (shard,n),versions in self.local.items():
            if not self.down[n] and version in versions:
                result[shard].append(-2-n)  # local disk endpoint token
        if version in self.remote:
            for group in result:
                group.append(-1)
        return result

    def newest_recoverable(self):
        versions = [set(self.remote) for _ in range(self.c.shards)]
        for n,v in self.primary.items():
            if not self.down[n] and v is not None:
                versions[self.shard[n]].add(v)
        for (shard,n),v in self.backup.items():
            if not self.down[n] and v is not None:
                versions[shard].add(v)
        for (shard,n),vv in self.local.items():
            if not self.down[n]:
                versions[shard].update(vv)
        return common_version(versions, self.version)

    def transfer(self, requests, sources):
        c = self.c
        load, participants = {}, set()
        def host(token):
            return token if token >= 0 else (-2-token if token <= -2 else None)
        def bw(token):
            return c.storage_gib_s if token == -1 else (self.s.local_disk_gib_s if token <= -2 else c.nodes[token].bandwidth_gib_s)
        for shard,dst in requests:
            options = []
            for src in sources[shard]:
                a,b = host(src),host(dst)
                if src == dst:
                    options.append((0.,src,0.,0.)); continue
                factor = c.cross_domain_factor if a is not None and b is not None and c.nodes[a].domain != c.nodes[b].domain else 1.
                x = c.shard_gib/(bw(src)*factor)
                y = c.shard_gib/(bw(dst)*factor)
                if a != b:
                    x += self.s.network_latency_s; y += self.s.network_latency_s
                options.append((max(load.get(src,0)+x,load.get(dst,0)+y),src,x,y))
            if not options:
                raise ValueError('No consistent source')
            _,src,x,y = min(options) if self.p.source_policy == 'balanced' else options[0]
            load[src] = load.get(src,0)+x; load[dst] = load.get(dst,0)+y
            participants.update(n for n in (host(src),host(dst)) if n is not None)
        duration = max(load.values(), default=0.)
        # Conservative bandwidth reservation while an asynchronous save is pending.
        if self.pending:
            duration /= 1-self.p.background_fraction
        return duration, participants

    def step_seconds(self, live):
        c,p = self.c,self.p
        compute = c.base_step_s*c.replicas/sum(min(c.nodes[n].speed for n in p.groups[r]) for r in live)
        def collective(members, volume):
            if len(members) <= 1:
                return 0.
            bw = min(c.nodes[n].bandwidth_gib_s for n in members)
            if len({c.nodes[n].domain for n in members}) > 1:
                bw *= c.cross_domain_factor
            return volume/bw + self.s.network_latency_s*2*(len(members)-1)
        # Explicit two-level topology: shard collectives inside replica groups,
        # then gradient synchronization between holders of the same shard.
        intra = max(collective(p.groups[r],c.collective_factor*c.state_gib*(c.shards-1)/c.shards) for r in live)
        inter = max(collective([p.groups[r][shard] for r in live],c.collective_factor*c.shard_gib) for shard in range(c.shards))
        communication = intra+inter
        if self.pending:
            communication /= 1-p.background_fraction
        return compute+communication

    def remap_to_spares(self):
        available = sorted((n for n in self.p.spares if not self.down[n]), key=lambda n:(-self.c.nodes[n].speed,n))
        groups = [list(g) for g in self.p.groups]
        used = []
        for g in groups:
            for shard,old in enumerate(g):
                if self.down[old] and available:
                    n = available.pop(0); used.append(n); g[shard] = n
                    self.primary.pop(old); self.shard.pop(old)
                    self.primary[n] = None; self.shard[n] = shard
                    self.remaps += 1
                    self.record('spare_remap', old=old, new=n)
        if used:
            self.p = replace(self.p, groups=tuple(tuple(g) for g in groups),
                             spares=tuple(n for n in self.p.spares if n not in used))

    def start_checkpoint(self, sources):
        # Only primaries own the dedicated outgoing checkpoint buffer. Backup
        # buffers may be overwritten by refresh and cannot double as this buffer.
        sources = [[n for n in options if n in self.primary and self.shard[n] == shard]
                   for shard,options in enumerate(sources)]
        if not all(sources): return False
        if self.p.checkpoint_tier == 'remote':
            destinations = [(shard,-1) for shard in range(self.c.shards)]
        else:
            # Choose an up-to-date primary for each shard; no disk-to-disk copy.
            destinations = []
            for shard,options in enumerate(sources):
                nodes = [n for n in options if n >= 0]
                if not nodes:
                    return False
                destinations.append((shard,-2-min(nodes)))
        duration,participants = self.transfer(destinations,sources)
        local_dest = [(shard,-2-token) for shard,token in destinations] if self.p.checkpoint_tier == 'local' else []
        job = dict(version=self.version, participants=participants, destinations=local_dest)
        if self.p.checkpoint_tier == 'remote':
            # Bill full attempted payload, even when a transfer later aborts.
            self.write_gib += self.c.state_gib
        if self.p.checkpoint_mode == 'async':
            job['finish'] = self.t+duration/self.p.background_fraction
            self.pending = job
            self.record('checkpoint_start', snapshot_version=self.version)
        elif self.advance(duration,'checkpoint',participants):
            self.finish_checkpoint(job)
        else:
            self.checkpoint_aborts += 1
        self.last_checkpoint = self.version
        return True

    def run(self):
        c = self.c
        self.record('start')
        while self.version < c.steps and self.t < c.max_simulated_s:
            check_budget(self.stop_at)
            if self.t < self.detect_until:
                self.advance(self.detect_until-self.t,'detection'); continue
            sources = self.sources(self.version)
            if not all(sources):
                old = self.version
                self.version = self.newest_recoverable()
                self.rollbacks += 1; self.lost_steps += old-self.version
                self.record('rollback',from_version=old)
                for n,v in self.primary.items():
                    if v is not None and v > self.version: self.primary[n] = None
                for key,v in self.backup.items():
                    if v is not None and v > self.version: self.backup[key] = None
                for key,vv in self.local.items():
                    self.local[key] = [v for v in vv if v <= self.version]
                self.remote = [v for v in self.remote if v <= self.version]
                # Old execution histories cannot complete an outgoing snapshot.
                self.pending = None
                self.backup_done = self.last_checkpoint = -1
                sources = self.sources(self.version)
            live = [r for r,g in enumerate(self.p.groups) if all(not self.down[n] and self.primary[n] == self.version for n in g)]
            # Failure-triggered admission plus periodic step-boundary repair.
            repair_now = not live or (self.p.recovery == 'eager' and self.version % self.p.repair_every == 0)
            if repair_now:
                self.remap_to_spares()
                repair = [r for r,g in enumerate(self.p.groups) if r not in live and all(not self.down[n] for n in g)]
                if repair:
                    def order(r):
                        g = self.p.groups[r]
                        return (-min(c.nodes[n].speed for n in g),r) if self.p.recovery_order == 'fastest' else (sum(self.primary[n] != self.version for n in g),r)
                    r = min(repair,key=order)
                    requests = [(shard,n) for shard,n in enumerate(self.p.groups[r]) if self.primary[n] != self.version]
                    duration,participants = self.transfer(requests,sources)
                    participants.update(self.p.groups[r])
                    if self.advance(c.reinit_s+duration,'recovery',participants):
                        for n in self.p.groups[r]: self.primary[n] = self.version
                        self.recovered_groups += 1; self.record('restore',groups=[r])
                    continue
            if not live:
                next_event = self.events[self.e].time_s if self.e < len(self.events) else c.max_simulated_s
                self.advance(max(0.,next_event-self.t),'waiting'); continue
            if self.version > 0 and self.version % self.p.checkpoint_every == 0 and self.last_checkpoint != self.version and not self.pending:
                if self.start_checkpoint(sources): continue
            if self.version % self.p.backup_every == 0 and self.backup_done != self.version:
                requests = [(shard,n) for shard,n in self.p.backups if not self.down[n]]
                duration,participants = self.transfer(requests,sources)
                if self.advance(duration,'backup',participants):
                    for key in requests: self.backup[key] = self.version
                    self.backup_done = self.version
                continue
            if self.advance(self.step_seconds(live),'training',[n for r in live for n in self.p.groups[r]]):
                self.version += 1
                for r in live:
                    for n in self.p.groups[r]: self.primary[n] = self.version
                self.record('commit',live_groups=len(live))
            else:
                self.aborted_steps += 1
        completed = self.version == c.steps
        remote_storage = c.state_gib*self.p.checkpoint_retention if self.p.checkpoint_tier == 'remote' else 0.
        machine_cost = self.initial_rate*self.t/3600
        storage_cost = remote_storage*self.s.remote_storage_usd_gib_hour*self.t/3600
        write_cost = self.write_gib*self.s.remote_write_usd_gib
        result = dict(completed=completed, completion_s=self.t if completed else None,
                      observed_s=self.t,cost_usd=machine_cost+storage_cost+write_cost,
                      machine_cost_usd=machine_cost,storage_cost_usd=storage_cost,write_cost_usd=write_cost,
                      cost_censored=not completed,deadline_hit=completed and self.t <= c.deadline_s,
                      committed_steps=self.version,failures=self.failures,rollbacks=self.rollbacks,
                      lost_steps=self.lost_steps,aborted_steps=self.aborted_steps,recovered_groups=self.recovered_groups,
                      phase_seconds=self.seconds,spare_remaps=self.remaps,checkpoint_writes=self.checkpoint_writes,
                      checkpoint_aborts=self.checkpoint_aborts,checkpoint_attempted_gib=self.write_gib,
                      background_elapsed_s=self.overlap_s)
        if self.log is not None: result['audit'] = self.log
        return result


def simulate(s, p, events, stop_at=None, audit=False):
    return ExpandedSimulation(s,p,events,stop_at,audit).run()


def make_scenarios(config):
    """Case-level split: all profiles of a hardware instance share the split."""
    base = dict(config)
    base['profiles'] = [{'name':'reference'}]
    scenarios = []
    for c in old_cases(base):
        i = int(c.case_id.split('-i')[1].split('-')[0])
        split = config['instance_splits'][i]
        for profile in config['expanded_profiles']:
            workload = replace(c, case_id=c.case_id.removesuffix('-reference')+'-'+profile['name'],profile=profile['name'])
            # Explicit heterogeneous RAM and NIC capabilities. Identical across
            # environment profiles; not selected or modified by planners.
            workload.nodes = [replace(n, memory_gib=(16.,24.,32.)[j%3],
                                      bandwidth_gib_s=(2.,4.,8.)[j%3]) for j,n in enumerate(c.nodes)]
            workload.storage_gib_s *= profile.get('storage_multiplier',1.)
            workload.cross_domain_factor *= profile.get('network_multiplier',1.)
            workload.replacement_s *= profile.get('replacement_multiplier',1.)
            # Fixed budget anchors from original workload, independent of choices
            # and profile. More adverse profiles are genuinely harder.
            s = Scenario(workload,split,**profile.get('environment',{}))
            validate_scenario(s); scenarios.append(s)
    return scenarios
