"""Step-atomic training with event-driven failures and versioned recovery.

No tensors, worker processes, network calls, optimizers or GPU libraries.
"""
from dataclasses import dataclass
import math
import random
import time

from .model import seed_for, step_seconds, violations


class BudgetExpired(Exception):
    """A wall-clock budget expired; a partial observation must not be reported."""


def check_budget(stop_at):
    if stop_at is not None and time.monotonic() >= stop_at:
        raise BudgetExpired()


@dataclass(frozen=True, order=True)
class Event:
    time_s: float
    # down=0 sorts before up=1 at the same timestamp.
    kind: int
    node: int


def generate_trace(c, seed, stop_at=None):
    """Exogenous Poisson outages, reproducible and independent of the plan.

    Overlapping individual/domain outages are reference-counted. Replacements
    have the same attributes and fresh empty memory. Outage duration is fixed.
    """
    events = []

    def process(rate, members, key):
        if rate == 0:
            return
        rng = random.Random(seed_for(seed, key))
        t = rng.expovariate(rate / 3600)
        while t < c.max_simulated_s:
            check_budget(stop_at)
            for n in members:
                events.append(Event(t, 0, n))
                events.append(Event(t + c.replacement_s, 1, n))
            t += rng.expovariate(rate / 3600)
    for i, node in enumerate(c.nodes):
        process(node.failure_per_hour, [i], ("node", i))
    for domain in sorted({n.domain for n in c.nodes}):
        process(c.domain_failure_per_hour,
                [i for i, n in enumerate(c.nodes) if n.domain == domain],
                ("domain", domain))
    return sorted(events)


def common_version(versions, ceiling):
    """Newest version present for every shard. Never stitch different versions."""
    common = set.intersection(*(set(v) for v in versions))
    return max((v for v in common if v <= ceiling), default=None)


class Simulation:
    def __init__(self, c, p, events, stop_at=None, audit=False):
        errors = violations(c, p)
        if errors:
            raise ValueError("; ".join(errors))
        self.c, self.p = c, p
        self.checkpoint_every = c.checkpoint_every if p.checkpoint_every is None else p.checkpoint_every
        self.events = events
        self.stop_at = stop_at
        self.t = 0.
        self.e = 0
        self.version = 0
        self.persistent = 0  # Durable initial state; creation excluded for all plans.
        self.down = [0] * len(c.nodes)
        self.primary = {n: 0 for n in p.active}
        self.shard = {n: s for g in p.groups for s, n in enumerate(g)}
        self.backup = {(s, n): None for s, n in p.backups}
        self.backup_done = -1
        self.detect_until = 0.
        self.seconds = {k: 0. for k in ("training", "recovery", "backup", "checkpoint", "waiting", "detection")}
        self.failures = self.rollbacks = self.lost_steps = self.aborted_steps = 0
        self.recovered_groups = 0
        self.log = [] if audit else None

    def record(self, event, **kwargs):
        if self.log is not None:
            self.log.append(dict(time_s=self.t, event=event, version=self.version, **kwargs))

    def handle(self, event):
        n = event.node
        if event.kind == 0:
            self.down[n] += 1
            if self.down[n] == 1:
                if n in self.primary:
                    self.primary[n] = None
                for key in self.backup:
                    if key[1] == n:
                        self.backup[key] = None
                if n in self.p.allocated:
                    self.failures += 1
                    self.detect_until = max(self.detect_until, self.t + self.c.detection_s)
                    self.record("failure", node=n)
                return n
        else:
            self.down[n] -= 1
            if self.down[n] < 0:
                raise ValueError("Trace contains unmatched up event")
            if self.down[n] == 0 and n in self.p.allocated:
                self.record("replacement", node=n)
        return None

    def advance(self, duration, phase, critical=()):
        """Perform an atomic operation. A failed participant aborts its commit.

        Time/traffic already spent is charged even when the operation aborts.
        Unrelated failures do not invalidate the operation.
        """
        check_budget(self.stop_at)
        start = self.t
        target = self.t + duration
        end = min(target, self.c.max_simulated_s)
        critical = set(critical)
        while self.e < len(self.events) and self.events[self.e].time_s <= end:
            event = self.events[self.e]
            if event.time_s < self.t - 1e-9:
                raise ValueError("Events must be chronological")
            self.t = event.time_s
            aborted = False
            # A simultaneous domain outage is one state transition. Do not let
            # the planner act between failures at the same timestamp.
            while self.e < len(self.events) and self.events[self.e].time_s == self.t:
                event = self.events[self.e]
                self.e += 1
                failed = self.handle(event)
                aborted = aborted or failed in critical
            if aborted:
                self.seconds[phase] += self.t - start
                return False
        self.t = end
        self.seconds[phase] += self.t - start
        return target <= self.c.max_simulated_s

    def sources(self, version):
        result = [[] for _ in range(self.c.shards)]
        for n, v in self.primary.items():
            if not self.down[n] and v == version:
                result[self.shard[n]].append(n)
        for (s, n), v in self.backup.items():
            if not self.down[n] and v == version:
                result[s].append(n)
        if self.persistent == version:
            for sources in result:
                sources.append(-1)  # Persistent storage.
        return result

    def newest_recoverable(self):
        versions = [{self.persistent} for _ in range(self.c.shards)]
        for n, v in self.primary.items():
            if not self.down[n] and v is not None:
                versions[self.shard[n]].add(v)
        for (s, n), v in self.backup.items():
            if not self.down[n] and v is not None:
                versions[s].add(v)
        return common_version(versions, self.version)

    def transfer(self, requests, sources):
        """Greedy source selection and shared-endpoint transfer time.

        Each request is (shard, destination). Endpoint traffic is summed, then
        endpoints overlap. Cross-domain transfers consume extra endpoint time.
        Storage is one shared endpoint. No fabric-level contention/compute overlap.
        """
        load = {}
        participants = set()
        for shard, dst in requests:
            options = []
            for src in sources[shard]:
                if src == dst:
                    options.append((0., src, 0., 0.))
                    continue
                src_bw = self.c.storage_gib_s if src == -1 else self.c.nodes[src].bandwidth_gib_s
                dst_bw = self.c.storage_gib_s if dst == -1 else self.c.nodes[dst].bandwidth_gib_s
                factor = 1.
                if src >= 0 and dst >= 0 and self.c.nodes[src].domain != self.c.nodes[dst].domain:
                    factor = self.c.cross_domain_factor
                a = self.c.shard_gib / (src_bw * factor)
                b = self.c.shard_gib / (dst_bw * factor)
                options.append((max(load.get(src, 0.) + a, load.get(dst, 0.) + b), src, a, b))
            if not options:
                raise ValueError("No compatible source for required shard")
            _, src, a, b = min(options)
            load[src] = load.get(src, 0.) + a
            load[dst] = load.get(dst, 0.) + b
            participants.update(n for n in (src, dst) if n >= 0)
        return max(load.values(), default=0.), participants

    def run(self):
        c, p = self.c, self.p
        self.record("start")
        while self.version < c.steps and self.t < c.max_simulated_s:
            check_budget(self.stop_at)
            if self.t < self.detect_until:
                self.advance(self.detect_until - self.t, "detection")
                continue
            sources = self.sources(self.version)
            if not all(sources):
                old = self.version
                self.version = self.newest_recoverable()
                if self.version is None:
                    raise AssertionError("Initial durable state should always be available")
                self.rollbacks += 1
                self.lost_steps += old - self.version
                self.record("rollback", from_version=old)
                # Future snapshots cannot be used after rolling back execution.
                for n, v in self.primary.items():
                    if v is not None and v > self.version:
                        self.primary[n] = None
                for key, v in self.backup.items():
                    if v is not None and v > self.version:
                        self.backup[key] = None
                self.backup_done = -1
                sources = self.sources(self.version)
            live = [r for r, g in enumerate(p.groups)
                    if all(not self.down[n] and self.primary[n] == self.version for n in g)]
            repair = [r for r, g in enumerate(p.groups)
                      if r not in live and all(not self.down[n] for n in g)]
            if repair:
                requests = [(s, n) for r in repair for s, n in enumerate(p.groups[r])
                            if self.primary[n] != self.version]
                duration, participants = self.transfer(requests, sources)
                participants.update(n for r in repair for n in p.groups[r])
                if self.advance(c.reinit_s + duration, "recovery", participants):
                    for r in repair:
                        for n in p.groups[r]:
                            self.primary[n] = self.version
                    self.recovered_groups += len(repair)
                    self.record("restore", groups=repair)
                continue
            if not live:
                if self.e == len(self.events):
                    self.advance(c.max_simulated_s - self.t, "waiting")
                    break
                self.advance(self.events[self.e].time_s - self.t, "waiting")
                continue
            # Fixed synchronous snapshot policy; all copies commit atomically.
            if self.version > 0 and self.version % self.checkpoint_every == 0 and self.persistent != self.version:
                duration, participants = self.transfer([(s, -1) for s in range(c.shards)], sources)
                if self.advance(duration, "checkpoint", participants):
                    self.persistent = self.version
                    self.record("checkpoint")
                continue
            if self.version % c.backup_every == 0 and self.backup_done != self.version:
                requests = [(s, n) for s, n in p.backups if not self.down[n]]
                duration, participants = self.transfer(requests, sources)
                if self.advance(duration, "backup", participants):
                    for key in requests:
                        self.backup[key] = self.version
                    self.backup_done = self.version
                    self.record("backup", copies=len(requests))
                continue
            duration = step_seconds(c, p, live)
            if self.advance(duration, "training", [n for r in live for n in p.groups[r]]):
                self.version += 1
                for r in live:
                    for n in p.groups[r]:
                        self.primary[n] = self.version
                self.record("commit", live_groups=len(live))
            else:
                self.aborted_steps += 1
        completed = self.version == c.steps
        rate = sum(c.nodes[n].price_per_hour for n in p.allocated)
        # Reserved-capacity accounting, including downtime and dedicated backups.
        result = dict(completed=completed, completion_s=self.t if completed else None,
                      observed_s=self.t, cost_usd=rate*self.t/3600,
                      cost_censored=not completed,
                      deadline_hit=completed and self.t <= c.deadline_s,
                      committed_steps=self.version, failures=self.failures,
                      rollbacks=self.rollbacks, lost_steps=self.lost_steps,
                      aborted_steps=self.aborted_steps, recovered_groups=self.recovered_groups,
                      phase_seconds=self.seconds)
        if self.log is not None:
            result["audit"] = self.log
        return result


def simulate(c, p, events, stop_at=None, audit=False):
    return Simulation(c, p, events, stop_at, audit).run()
