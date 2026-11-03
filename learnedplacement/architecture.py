"""V8 candidate encoders. No simulator outcomes or node IDs enter the inputs.

PyTorch is optional for older pilots. Graphs use machines, shards, replica
 groups and failure domains, with directed typed placement/membership edges.
The set model uses the same node attributes and one-hop neighborhood summaries,
but not the edges. Every encoder also receives the original global features.
"""
import numpy as np
import torch
from torch import nn
from .learning import features, matrix, Network

VERSION = 1
BASE_DIM = 18
RELATIONS = 8
NODE_DIM = BASE_DIM + RELATIONS * (BASE_DIM + 1)


def graph(s, p):
    c = s.workload
    n, shards, groups = len(c.nodes), len(p.groups[0]), len(p.groups)
    domains = sorted({v.domain for v in c.nodes})
    domain_index = {d: n + shards + groups + i for i, d in enumerate(domains)}
    x = np.zeros((n + shards + groups + len(domains), BASE_DIM), np.float32)
    types = np.zeros(len(x), np.int64)
    edges = [[] for _ in range(RELATIONS)]
    def link(a, b, rel):
        edges[rel].append((a, b)); edges[rel + 1].append((b, a))
    for i, node in enumerate(c.nodes):
        x[i, 0] = 1
        x[i, 4:13] = [node.speed, node.price_per_hour, node.memory_gib/24,
                      node.bandwidth_gib_s/4, node.failure_per_hour*s.hazard_multiplier,
                      float(i in p.active), sum(host == i for _, host in p.backups),
                      float(i in p.spares), float(i in p.allocated)]
        link(i, domain_index[node.domain], 0)
    for r, members in enumerate(p.groups):
        index = n + shards + r; types[index] = 2; x[index, 2] = 1
        x[index, 4:9] = [min(c.nodes[i].speed for i in members),
                        sum(c.nodes[i].price_per_hour for i in members),
                        min(c.nodes[i].memory_gib for i in members)/24,
                        min(c.nodes[i].bandwidth_gib_s for i in members)/4,
                        sum(c.nodes[i].failure_per_hour for i in members)*s.hazard_multiplier]
        x[index, 13:16] = [len(members)/8, len({c.nodes[i].domain for i in members})/3, c.state_gib/16]
        for j, i in enumerate(members):
            link(i, index, 2); link(i, n+j, 4)
    for j, i in p.backups: link(i, n+j, 6)
    for j in range(shards):
        index = n+j; types[index] = 1; x[index, 1] = 1
        holders = [g[j] for g in p.groups] + [i for k, i in p.backups if k == j]
        x[index, 13:18] = [len(holders)/4, len({c.nodes[i].domain for i in holders})/3,
                          c.state_gib/shards/16, c.working_gib/shards/4, groups/4]
    for d, index in domain_index.items():
        types[index] = 3; x[index, 3] = 1
        x[index, 8] = c.domain_failure_per_hour*s.hazard_multiplier
        x[index, 13] = sum(v.domain == d for v in c.nodes)/24
    x = np.log1p(x)
    # Explicit local placement attributes for the set baseline. No arbitrary IDs.
    local = []
    edge_arrays = []
    for pairs in edges:
        e = np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T
        edge_arrays.append(e)
        total = np.zeros_like(x); count = np.zeros((len(x), 1), np.float32)
        if e.shape[1]:
            np.add.at(total, e[1], x[e[0]]); np.add.at(count, e[1], 1)
        local.extend([np.log1p(count), total/np.maximum(count, 1)])
    return dict(nodes=np.concatenate([x] + local, axis=1), types=types,
                edges=edge_arrays, global_features=features(s, p).astype(np.float32))


def inputs(s, pool, architecture):
    if architecture == 'mlp': return matrix(s, pool).astype(np.float32)
    return [graph(s, p) for p in pool]


def pack(items, device):
    offsets = np.cumsum([0] + [len(g['nodes']) for g in items[:-1]])
    return dict(
        x=torch.as_tensor(np.concatenate([g['nodes'] for g in items]), device=device),
        types=torch.as_tensor(np.concatenate([g['types'] for g in items]), device=device),
        batch=torch.as_tensor(np.concatenate([np.full(len(g['nodes']), i, np.int64) for i,g in enumerate(items)]), device=device),
        edges=[torch.as_tensor(np.concatenate([g['edges'][r]+off for g,off in zip(items, offsets)], axis=1), device=device) for r in range(RELATIONS)],
        global_features=torch.as_tensor(np.stack([g['global_features'] for g in items]), device=device))


class Encoder(nn.Module):
    def __init__(self, architecture, dimension=60, hidden=64, layers=2, seed=0):
        super().__init__()
        if architecture not in ('mlp', 'set', 'gnn'): raise ValueError('Unknown architecture')
        self.spec = dict(architecture=architecture, dimension=dimension, hidden=hidden, layers=layers, seed=seed)
        # Preserve global RNG state; no dropout or stochastic operations in forward.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if architecture == 'mlp':
                self.scorer = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            else:
                self.embed = nn.Linear(NODE_DIM, hidden)
                self.self_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
                if architecture == 'gnn':
                    self.messages = nn.ModuleList([nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(RELATIONS)]) for _ in range(layers)])
                self.scorer = nn.Sequential(nn.Linear(4*hidden+dimension, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, data):
        if self.spec['architecture'] == 'mlp': return self.scorer(data).squeeze(-1)
        h = torch.tanh(self.embed(data['x']))
        for layer, transform in enumerate(self.self_layers):
            z = transform(h)
            if self.spec['architecture'] == 'gnn':
                for rel, edge in enumerate(data['edges']):
                    src, dst = edge
                    if src.numel() == 0: continue
                    # Transform nodes once, then gather messages; sparse O(EH).
                    message = self.messages[layer][rel](h)[src]
                    agg = torch.zeros_like(h).index_add(0, dst, message)
                    count = torch.zeros((len(h), 1), device=h.device).index_add(0, dst, torch.ones((len(dst), 1), device=h.device))
                    z = z + agg/count.clamp_min(1)
            h = torch.tanh(z)
        batch_count = len(data['global_features'])
        index = data['batch']*4 + data['types']
        pooled = h.new_zeros((batch_count*4, h.shape[1])).index_add(0, index, h)
        counts = h.new_zeros((batch_count*4, 1)).index_add(0, index, h.new_ones((len(h), 1)))
        pooled = (pooled/counts.clamp_min(1)).reshape(batch_count, -1)
        return self.scorer(torch.cat([pooled, data['global_features']], dim=1)).squeeze(-1)

    def scores(self, data, chunk_size=8):
        device = next(self.parameters()).device
        if self.spec['architecture'] == 'mlp':
            return self(torch.as_tensor(data, device=device))
        return torch.cat([self(pack(data[i:i+chunk_size], device)) for i in range(0, len(data), chunk_size)])

    def to_dict(self):
        return dict(architecture_version=VERSION, **self.spec,
                    weights={k:v.detach().cpu().tolist() for k,v in self.state_dict().items()})

    @classmethod
    def from_dict(cls, value, device='cpu'):
        if value['architecture_version'] != VERSION: raise ValueError('Encoder version mismatch')
        obj = cls(**{k:value[k] for k in ('architecture','dimension','hidden','layers','seed')})
        obj.load_state_dict({k:torch.tensor(v, dtype=torch.float32) for k,v in value['weights'].items()}, strict=True)
        return obj.to(device)


def load_actor(actor, device='cpu'):
    if 'architecture_version' in actor: return Encoder.from_dict(actor, device)
    return Network.from_dict(actor)


def score_actor(net, s, pool):
    if isinstance(net, Network): return net.forward(matrix(s, pool))[0]
    with torch.no_grad():
        return net.scores(inputs(s, pool, net.spec['architecture'])).cpu().numpy()
