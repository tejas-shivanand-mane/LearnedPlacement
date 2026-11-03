"""Input models, deterministic synthetic instances, and structural validation."""
from dataclasses import asdict, dataclass, field
import hashlib
import math
import random


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Node:
    speed: float
    price_per_hour: float
    memory_gib: float
    bandwidth_gib_s: float
    failure_per_hour: float
    domain: int


@dataclass
class Case:
    case_id: str
    nodes: list[Node]
    replicas: int
    shards: int = 4
    state_gib: float = 16.0
    working_gib: float = 4.0
    steps: int = 240
    base_step_s: float = 3.0
    collective_factor: float = 0.12
    cross_domain_factor: float = 0.55
    storage_gib_s: float = 2.0
    backup_every: int = 20
    checkpoint_every: int = 80
    backup_budget_gib: float = 8.0
    detection_s: float = 2.0
    reinit_s: float = 4.0
    replacement_s: float = 30.0
    domain_failure_per_hour: float = 0.0
    deadline_s: float = 2000.0
    cost_budget_usd: float = 20.0
    required_success: float = 0.95
    max_simulated_s: float = 8000.0
    regime: str = "custom"

    @property
    def shard_gib(self):
        return self.state_gib / self.shards

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        obj = dict(obj)
        obj["nodes"] = [Node(**n) for n in obj["nodes"]]
        return cls(**obj)


@dataclass(frozen=True)
class Plan:
    # groups[r][s] is the machine holding shard s in replica group r.
    groups: tuple[tuple[int, ...], ...]
    # At most one additional snapshot per shard: (shard index, host machine).
    backups: tuple[tuple[int, int], ...] = ()

    def to_dict(self):
        return {"groups": [list(g) for g in self.groups],
                "backups": [list(b) for b in self.backups]}

    @classmethod
    def from_dict(cls, obj):
        return cls(tuple(tuple(g) for g in obj["groups"]),
                   tuple(tuple(b) for b in obj["backups"]))

    @property
    def active(self):
        return set(n for g in self.groups for n in g)

    @property
    def allocated(self):
        return self.active | {n for _, n in self.backups}


def validate_case(c):
    for name in ("replicas", "shards", "steps", "backup_every", "checkpoint_every"):
        if type(getattr(c, name)) is not int or getattr(c, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if len(c.nodes) < c.replicas * c.shards:
        raise ValueError("Not enough candidate machines")
    for name in ("state_gib", "base_step_s", "storage_gib_s", "cross_domain_factor",
                 "deadline_s", "cost_budget_usd", "max_simulated_s"):
        if not math.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
            raise ValueError(f"Invalid {name}")
    for name in ("working_gib", "backup_budget_gib", "collective_factor", "detection_s",
                 "reinit_s", "replacement_s", "domain_failure_per_hour"):
        if not math.isfinite(getattr(c, name)) or getattr(c, name) < 0:
            raise ValueError(f"Invalid {name}")
    if not 0 < c.required_success < 1 or c.max_simulated_s < c.deadline_s:
        raise ValueError("Invalid success target or simulation horizon")
    for n in c.nodes:
        if any(not math.isfinite(v) or v <= 0 for v in
               (n.speed, n.price_per_hour, n.memory_gib, n.bandwidth_gib_s)):
            raise ValueError("Node speed, price, memory and bandwidth must be positive")
        if not math.isfinite(n.failure_per_hour) or n.failure_per_hour < 0:
            raise ValueError("Invalid node failure rate")


def violations(c, p):
    errors = []
    if len(p.groups) != c.replicas or any(len(g) != c.shards for g in p.groups):
        errors.append("wrong group dimensions")
    ids = [n for g in p.groups for n in g]
    if any(type(n) is not int or not 0 <= n < len(c.nodes) for n in ids):
        return errors + ["invalid training node"]
    if len(ids) != len(set(ids)):
        errors.append("training node assigned more than once")
    if len({s for s, _ in p.backups}) != len(p.backups):
        errors.append("more than one backup per shard")
    if len(p.backups) * c.shard_gib > c.backup_budget_gib + 1e-9:
        errors.append("backup storage exceeded")
    usage = [0.0] * len(c.nodes)
    for n in ids:
        usage[n] += c.shard_gib + c.working_gib
    for s, n in p.backups:
        if type(s) is not int or type(n) is not int or not 0 <= s < c.shards or not 0 <= n < len(c.nodes):
            return errors + ["invalid backup"]
        if any(len(g) == c.shards and g[s] == n for g in p.groups):
            errors.append("backup duplicates local primary shard")
        # Double buffering keeps the previous snapshot valid until refresh commits.
        usage[n] += 2 * c.shard_gib
    if any(v > n.memory_gib + 1e-9 for v, n in zip(usage, c.nodes)):
        errors.append("node memory exceeded (includes backup staging buffer)")
    return errors


def step_seconds(c, p, groups):
    """Analytical surrogate, not a measured HSDP performance model.

    Fixed samples per global optimizer step; work is balanced between live groups.
    Within-group throughput is limited by the slowest member. Collective traffic
    is a configurable fraction of state size and serialized with computation.
    """
    speeds = [min(c.nodes[n].speed for n in p.groups[r]) for r in groups]
    compute = c.base_step_s * c.replicas / sum(speeds)
    members = [n for r in groups for n in p.groups[r]]
    bandwidth = min(c.nodes[n].bandwidth_gib_s for n in members)
    if len({c.nodes[n].domain for n in members}) > 1:
        bandwidth *= c.cross_domain_factor
    return compute + c.collective_factor * c.shard_gib / bandwidth


def make_case(config, active_nodes, regime, instance):
    rng = random.Random(seed_for(config["seed"], active_nodes, regime, instance))
    count = active_nodes + active_nodes // 2
    nodes = []
    for i in range(count):
        risky = i % 2 == 1
        speed, price, hazard = 1., 1., 0.12
        if regime == "cheap_risky":
            price, hazard = (0.35, 2.0) if risky else (1., 0.12)
        elif regime == "fast_risky":
            speed, price, hazard = (1.7, 0.8, 2.0) if risky else (1., 1., 0.12)
        elif regime == "correlated":
            speed, price, hazard = (1.3, 0.55, 0.2) if risky else (1., 1., 0.12)
        elif regime != "homogeneous":
            raise ValueError(f"Unknown regime: {regime}")
        jitter = 1. if regime == "homogeneous" else rng.uniform(0.95, 1.05)
        nodes.append(Node(speed*jitter, price, 24., 4., hazard, i % 3))
    workload = config["workload"]
    c = Case(f"n{active_nodes}-{regime}-i{instance}", nodes,
             replicas=active_nodes // workload["shards"], regime=regime,
             **workload)
    c.domain_failure_per_hour = 1.0 if regime == "correlated" else 0.0
    # Fixed, planner-independent reference: unit-speed job on the required count
    # of $1/hour machines, with the same persistent checkpoint overhead.
    ref_step = c.base_step_s + c.collective_factor*c.shard_gib/(4*c.cross_domain_factor)
    ref_time = c.steps * ref_step + ((c.steps-1)//c.checkpoint_every)*c.state_gib/c.storage_gib_s
    c.deadline_s = config["deadline_factor"] * ref_time
    c.cost_budget_usd = config["cost_budget_factor"] * ref_time * active_nodes / 3600
    c.max_simulated_s = config["simulation_horizon_factor"] * ref_time
    validate_case(c)
    return c
