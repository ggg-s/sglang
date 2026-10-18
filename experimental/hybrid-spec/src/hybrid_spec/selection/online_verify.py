"""Target-feedback calibration and prefix-closed, batch-wide verification budgets.

No draft logits are required. Both sources learn conditional correctness from
the same target feedback. The tables live on the worker device; timing samples
are collected asynchronously outside model CUDA graphs.
"""

from __future__ import annotations

from collections import deque

import torch


class OnlineAcceptance:
    # Neural; five match-length bins each for local and shared retrieval.
    NUM_BUCKETS = 11

    def __init__(self, config, device):
        self.config = config
        self.device = torch.device(device)
        shape = (self.NUM_BUCKETS, config.num_drafts)
        self.check_ct = torch.zeros(shape, dtype=torch.int64, device=device)
        self.correct_ct = torch.zeros_like(self.check_ct)
        self.depths = torch.arange(config.num_drafts, device=device)
        self.match_edges = torch.tensor([4, 8, 16, 32], device=device)
        self.gpu_ops = None
        if self.device.type == "cuda":
            from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton

            self.gpu_ops = GPUSuffixAutomaton.load_module()

    def buckets(self, view):
        metadata = view.query_metadata
        if metadata is None:
            raise RuntimeError("dynamic verify requires retrieval query metadata")
        if self.gpu_ops is not None:
            return self.gpu_ops.calibration_buckets(view.use_retrieval, metadata)
        match_bin = torch.bucketize(metadata[:, 0].contiguous(), self.match_edges)
        source = metadata[:, 1].clamp(0, 1)
        return torch.where(view.use_retrieval, 1 + source * 5 + match_bin, 0).long()

    def survival(self, buckets):
        config = self.config
        strength = config.dynamic_verify_prior_strength
        conditional = (
            self.correct_ct[buckets].float()
            + strength * config.dynamic_verify_prior_rate
        ) / (self.check_ct[buckets].float() + strength)
        return conditional.cumprod(dim=1)

    def update(self, buckets, num_drafts, num_accept_tokens):
        """First rejection is labelled; cropped/unreachable descendants are not."""
        if self.gpu_ops is not None:
            self.gpu_ops.calibration_feedback(
                self.check_ct,
                self.correct_ct,
                buckets,
                num_drafts.to(torch.int32),
                num_accept_tokens.to(torch.int32),
            )
            return
        num_correct = num_accept_tokens.long() - 1
        healthy = (num_correct >= 0) & (num_correct <= num_drafts)
        depth = self.depths[None, :]
        check = (depth < num_drafts[:, None]) & (depth <= num_correct[:, None])
        correct = depth < num_correct[:, None]
        self.check_ct.index_add_(0, buckets, (check & healthy[:, None]).long())
        self.correct_ct.index_add_(
            0, buckets, (correct & check & healthy[:, None]).long()
        )


class DynamicVerify:
    def __init__(self, config, device, tp_sync=None):
        self.config = config
        self.model = OnlineAcceptance(config, device)
        self.tp_sync = tp_sync
        self.costs = {}
        self.draft_costs = {}
        self.rounds = {}
        self.trials = {}
        self.timings = deque()
        self.pending = None
        self.start = None
        self.verify_start = None
        self.completed = None
        self.context_key = None
        self.verify_ct = 0
        self.explore_ct = 0
        self.cost_sample_ct = 0
        self.grids = {}

    def minimum(self, values):
        return self.tp_sync.minimum(values) if self.tp_sync is not None else values

    def observe_cost(self, key, graph_tokens, draft_ms, verify_ms):
        """EMA per batch/context/physical graph shape, never per logical length."""
        if draft_ms < 0 or verify_ms <= 0:
            return
        costs = self.costs.setdefault(key, {})
        old = costs.get(graph_tokens)
        sample = (float(draft_ms), float(verify_ms))
        costs[graph_tokens] = (
            sample
            if old is None
            else tuple(0.9 * a + 0.1 * b for a, b in zip(old, sample))
        )
        self.cost_sample_ct += 1
        old_draft = self.draft_costs.get(key, draft_ms)
        self.draft_costs[key] = 0.9 * old_draft + 0.1 * draft_ms

    def poll_timings(self):
        while self.timings and self.timings[0][-1].query():
            key, size, start, verify_start, end = self.timings.popleft()
            self.observe_cost(
                key,
                size,
                start.elapsed_time(verify_start),
                verify_start.elapsed_time(end),
            )

    def begin(self, batch):
        self.poll_timings()
        # A failed/aborted iteration cannot label a later request's candidates.
        self.pending = None
        self.start = self.verify_start = None
        # Host request lengths are a coarse context key, not scheduler metadata.
        context = max(
            (len(req.origin_input_ids) + len(req.output_ids) for req in batch.reqs),
            default=1,
        )
        self.context_key = (len(batch.reqs), max(512, 1 << (context - 1).bit_length()))
        if self.model.device.type == "cuda":
            self.start = torch.cuda.Event(enable_timing=True)
            self.start.record(torch.cuda.current_stream(self.model.device))
            if self.completed is not None:
                # Close the previous decode cycle only after its feedback and
                # native draft-extend have been enqueued. There is no wait.
                if len(self.timings) < 8:
                    self.timings.append((*self.completed, self.start))
                self.completed = None

    def discard_timing(self):
        """Prefill/idle must not be charged to an unrelated decode shape."""
        self.completed = None
        self.start = self.verify_start = None
        self.pending = None

    def select(self, view, caps, graph_tokens, key):
        """Scores and allocation stay on GPU; caller reads only the final sum."""
        config, model = self.config, self.model
        bs, width = len(caps), config.num_drafts
        buckets = self.minimum(model.buckets(view))
        full = len(graph_tokens) - 1
        rounds = self.rounds.get(key, 0) + 1
        self.rounds[key] = rounds
        explore = -1
        if rounds % config.dynamic_verify_explore_interval == 0:
            trial = self.trials.get(key, 0)
            self.trials[key] = trial + 1
            explore = full if trial % 2 == 0 else (trial // 2) % len(graph_tokens)
            self.explore_ct += 1
        costs = self.costs.get(key, {})
        draft_ms = self.draft_costs.get(key, 0.0)
        price_values = [
            draft_ms + costs[n][1] if n in costs else torch.inf for n in graph_tokens
        ]
        if model.gpu_ops is not None and bs * width <= 256 and len(graph_tokens) <= 256:
            grid_key = tuple(graph_tokens)
            if grid_key not in self.grids:
                self.grids[grid_key] = torch.tensor(
                    graph_tokens, device=caps.device, dtype=torch.long
                )
            lengths = model.gpu_ops.calibration_select(
                model.check_ct,
                model.correct_ct,
                buckets,
                caps.to(torch.int32),
                self.grids[grid_key],
                torch.tensor(price_values, device=caps.device, dtype=torch.float32),
                config.dynamic_verify_min_samples,
                config.dynamic_verify_min_drafts,
                config.dynamic_verify_prior_strength,
                config.dynamic_verify_prior_rate,
                bool(costs),
                explore,
            )
            return self.finish_selection(buckets, lengths)
        survival = model.survival(buckets)
        cold = model.check_ct[buckets, 0] < config.dynamic_verify_min_samples
        minimum = torch.where(cold, caps, config.dynamic_verify_min_drafts)
        valid = model.depths[None, :] < caps[:, None]
        mandatory = model.depths[None, :] < minimum[:, None]
        scores = torch.where(valid, survival, 0)
        priority = torch.where(valid, scores, -torch.inf)
        priority = torch.where(mandatory, torch.inf, priority)
        # Stable row-major ties keep each parent before its child, including
        # underflow to zero and mandatory positions with equal infinite scores.
        order = torch.argsort(priority.flatten(), descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(0, order, torch.arange(bs * width, device=caps.device))
        prefix = scores.flatten()[order].cumsum(0)
        capacities = torch.tensor(graph_tokens, device=caps.device, dtype=torch.long)
        budgets = torch.minimum(capacities - bs, caps.sum())
        utility = bs + prefix[(budgets - 1).clamp(min=0)]
        feasible = budgets >= minimum.sum()

        # All alternatives share this round's drafting cost. Source mix changes
        # that cost, so use its recent EMA as the fixed term.
        priced = torch.tensor(
            price_values,
            dtype=torch.float32,
            device=caps.device,
        )
        # Source caps can make several capacities represent the same logical
        # prefix. Price the graph that packing will actually select, including
        # when a larger captured graph happens to have a cheaper measurement.
        physical = torch.searchsorted(capacities, budgets + bs)
        priced = priced[physical]
        # Until a shape has been measured, preserve the fixed source budgets.
        if not costs:
            priced[full] = 1
        objective = torch.where(
            feasible & priced.isfinite(), utility / priced, -torch.inf
        )
        chosen = objective.argmax()
        if explore >= 0:
            # Alternate full-label exploration with timing different graph
            # capacities. Cold buckets always retain their full source budget.
            chosen = torch.where(
                feasible[explore], chosen.new_tensor(explore), chosen.new_tensor(full)
            )
        # A newly encountered source/context may make every measured shape
        # infeasible. Full capacity always accommodates all fixed source caps.
        chosen = torch.where(
            objective.isfinite().any() | (explore >= 0),
            chosen,
            chosen.new_tensor(full),
        )
        lengths = (
            ((rank.view(bs, width) < budgets[chosen]) & valid).sum(1).to(torch.int32)
        )
        return self.finish_selection(buckets, lengths)

    def finish_selection(self, buckets, lengths):
        lengths = self.minimum(lengths)
        # Both are fresh outputs, never aliases of captured query buffers.
        self.pending = (buckets, lengths)
        self.verify_ct += 1
        return lengths + 1

    def mark_verify(self, graph_tokens):
        self.graph_tokens = graph_tokens
        if self.start is not None:
            self.verify_start = torch.cuda.Event(enable_timing=True)
            self.verify_start.record(torch.cuda.current_stream(self.model.device))

    def feedback(self, num_accept_tokens):
        pending, self.pending = self.pending, None
        if pending is None:
            return
        if self.verify_start is not None:
            self.completed = (
                self.context_key,
                self.graph_tokens,
                self.start,
                self.verify_start,
            )
        self.start = self.verify_start = None
        buckets, lengths = pending
        self.model.update(buckets, lengths, self.minimum(num_accept_tokens))

    def snapshot(self):
        self.poll_timings()
        return {
            "verify_ct": self.verify_ct,
            "explore_ct": self.explore_ct,
            "cost_sample_ct": self.cost_sample_ct,
            "check_ct": self.model.check_ct.cpu().tolist(),
            "correct_ct": self.model.correct_ct.cpu().tolist(),
            "costs_ms": {
                f"{key[0]}:{key[1]}": {str(n): list(v) for n, v in costs.items()}
                for key, costs in self.costs.items()
            },
        }
