"""Offline CUDA Graph bank with captured route stats and async CPU feedback.

No capture, eager model call, tensor.item(), GPU->CPU synchronous copy or CUDA
synchronize occurs during replay. Routing feedback may lag; the cost table's
shape prior is used on missing/expired feedback. No native service imports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.mixed_selection import ACTIONS, FEATURE_COUNT, RouteFeatures


@triton.jit
def _route_stats(
    P,
    D,
    OUT,
    NP: tl.constexpr,
    ND: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
):
    offsets = tl.arange(0, B)
    pmask, dmask = offsets < NP * K, offsets < ND * K
    p = tl.load(P + offsets, pmask, 0).to(tl.int64)
    d = tl.load(D + offsets, dmask, 0).to(tl.int64)
    pvalid, dvalid = (p >= 0) & (p < E), (d >= 0) & (d < E)
    hp = tl.histogram(p.to(tl.int32), E, mask=pmask & pvalid)
    hd = tl.histogram(d.to(tl.int32), E, mask=dmask & dvalid)
    h = hp + hd
    mass = tl.sum(h, 0).to(tl.float32)
    collisions = tl.sum(h * h, 0).to(tl.float32) - mass
    # Exclude within-token top-k pairs (IDs are distinct). This corrects the
    # sample-size bias of sum(hist)^2/sum(hist^2) across different P sizes.
    effective = tl.minimum(E, mass * (mass - K) / tl.maximum(collisions, 1.0))
    tl.store(OUT + 0, mass)
    tl.store(
        OUT + 1,
        tl.sum((pmask & ~pvalid).to(tl.int32), 0)
        + tl.sum((dmask & ~dvalid).to(tl.int32), 0),
    )
    tl.store(OUT + 2, tl.sum((h > 0).to(tl.int32), 0))
    tl.store(OUT + 3, effective)
    tl.store(OUT + 4, tl.max(h, 0) / tl.maximum(mass, 1.0))
    tl.store(OUT + 5, tl.sum((hp > 0).to(tl.int32), 0))
    tl.store(OUT + 6, tl.sum((hd > 0).to(tl.int32), 0))
    tl.store(OUT + 7, tl.sum(((hp > 0) & (hd > 0)).to(tl.int32), 0))
    for i in tl.static_range(3):
        block = 8 << i
        tl.store(OUT + 8 + i, tl.sum((hp + block - 1) // block, 0))
        tl.store(OUT + 11 + i, tl.sum((hd + block - 1) // block, 0))
        tl.store(OUT + 14 + i, tl.sum((h + block - 1) // block, 0))


def route_features(p_ids, d_ids, *, experts=256, out=None):
    if (
        not p_ids.is_cuda
        or p_ids.device != d_ids.device
        or p_ids.ndim != 2
        or d_ids.ndim != 2
        or p_ids.shape[1] != d_ids.shape[1]
        or p_ids.dtype not in (torch.int32, torch.int64)
        or d_ids.dtype != p_ids.dtype
        or not p_ids.is_contiguous()
        or not d_ids.is_contiguous()
        or experts < 8
        or experts > 256
        or experts & (experts - 1)
    ):
        raise ValueError("Unsupported route tensor layout")
    if out is None:
        out = torch.empty(FEATURE_COUNT, device=p_ids.device, dtype=torch.float32)
    if (
        out.device != p_ids.device
        or out.dtype != torch.float32
        or out.shape != (FEATURE_COUNT,)
        or not out.is_contiguous()
    ):
        raise ValueError("Invalid route feature storage")
    _route_stats[(1,)](
        p_ids,
        d_ids,
        out,
        len(p_ids),
        len(d_ids),
        p_ids.shape[1],
        experts,
        triton.next_power_of_2(max(p_ids.numel(), d_ids.numel(), 1)),
        num_warps=8,
    )
    return out


class RouteMailbox:
    """Two pinned slots, only read after their recorded CUDA event completes."""

    def __init__(self):
        self.slots = [
            dict(
                host=torch.empty(FEATURE_COUNT, dtype=torch.float32, pin_memory=True),
                event=torch.cuda.Event(),
                busy=False,
                step=0,
                context=0,
            )
            for _ in range(2)
        ]
        self.submitted = self.dropped = self.completed = self.invalid = 0

    def submit(self, stats, *, step, context):
        for slot in self.slots:
            if not slot["busy"]:
                slot["host"].copy_(stats, non_blocking=True)
                slot["event"].record()
                slot.update(busy=True, step=step, context=context)
                self.submitted += 1
                return True
        self.dropped += 1
        return False

    def poll(self, *, step, context, max_age):
        latest = None
        for slot in self.slots:
            if slot["busy"] and slot["event"].query():
                slot["busy"] = False
                self.completed += 1
                if (
                    slot["context"] != context
                    or not 0 <= step - slot["step"] <= max_age
                ):
                    continue
                try:
                    features = RouteFeatures(tuple(slot["host"].tolist()))
                except ValueError:
                    self.invalid += 1
                    continue
                if latest is None or slot["step"] >= latest[0]:
                    latest = (slot["step"], features)
        return latest


@dataclass
class GraphEntry:
    graph: torch.cuda.CUDAGraph
    outputs: object
    stats: torch.Tensor
    keepalive: object
    validated: bool = False


class OfflineAdaptiveGraphs:
    """Select only prevalidated, captured graphs; fixed kernel config per graph."""

    def __init__(
        self, entries, policy, *, operator, p, capacity, sample_every=8, max_age=16
    ):
        if set(entries) != set(ACTIONS) or not all(
            e.validated for e in entries.values()
        ):
            raise ValueError("All six prevalidated Separate/Mixed graphs are required")
        if (
            any(
                not e.stats.is_cuda
                or e.stats.dtype != torch.float32
                or e.stats.shape != (FEATURE_COUNT,)
                for e in entries.values()
            )
            or len({e.stats.device for e in entries.values()}) != 1
        ):
            raise ValueError("Graph statistics must share CUDA feature storage layout")
        if not 0 < p <= capacity or sample_every < 1 or max_age < 0:
            raise ValueError("Invalid graph-bank geometry or feedback interval")
        self.entries, self.policy = dict(entries), policy
        self.operator, self.p, self.capacity = operator, p, capacity
        self.sample_every, self.max_age = sample_every, max_age
        self.mailbox = RouteMailbox()
        self.context = None
        self.feedback = None
        self.current = None
        self.replays = self.transitions = 0

    def replay(self, *, step=0, context=0):
        if type(step) is not int or type(context) is not int or step < 0:
            raise ValueError(
                "Step/context must be CPU integers; step must be nonnegative"
            )
        context_changed = context != self.context
        if context_changed:
            self.context, self.feedback, self.current = context, None, None
        fresh = self.mailbox.poll(step=step, context=context, max_age=self.max_age)
        if fresh is not None:
            self.feedback = fresh
        route = None
        if self.feedback is not None and 0 <= step - self.feedback[0] <= self.max_age:
            route = self.feedback[1]
        decision = self.policy.choose(
            self.operator, self.p, self.capacity, route, current=self.current
        )
        if decision.action != self.current:
            self.transitions += 1
        self.current = decision.action
        entry = self.entries[decision.action]
        entry.graph.replay()
        if context_changed or self.replays % self.sample_every == 0:
            self.mailbox.submit(entry.stats, step=step, context=context)
        self.replays += 1
        return decision
