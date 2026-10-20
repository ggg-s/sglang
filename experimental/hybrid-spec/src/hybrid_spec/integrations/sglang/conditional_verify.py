"""Device-selected compact target tiers with a fixed dense accept ABI.

There is one model leaf per existing token tier, independent of live batch
size. Each leaf packs its own input, prepares attention metadata, executes G
target rows, and scatters to a shared request-major output. Only diagnostics
read the cumulative counters back to the host.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager

import torch


@dataclasses.dataclass
class DeviceVerifyPlan:
    verify_lens: torch.Tensor
    logical_tokens: torch.Tensor
    graph_tokens: torch.Tensor
    branch: torch.Tensor
    admission_tokens: int

    @classmethod
    def create(cls, verify_lens, capacities, admission_tokens, floor=0):
        logical = verify_lens.sum()
        # The grid covers the full source window; consensus lengths cannot
        # exceed that window. Lower-bound selection also preserves TP alignment.
        index = torch.searchsorted(capacities, logical.clamp(min=floor))
        return cls(
            verify_lens,
            logical,
            capacities.gather(0, index.reshape(1)).reshape(()),
            (index + 1).to(torch.int32),
            admission_tokens,
        )


class DeviceVerifyCounters:
    def __init__(self, grid, device):
        self.grid = tuple(grid)
        self.values = torch.zeros(4, dtype=torch.int64, device=device)
        self.histogram = torch.zeros(len(grid), dtype=torch.int64, device=device)
        self.one = torch.ones(1, dtype=torch.int64, device=device)

    def add(self, plan, dense_tokens):
        self.values.add_(
            torch.stack(
                (
                    plan.logical_tokens.new_ones(()),
                    plan.logical_tokens,
                    plan.graph_tokens,
                    plan.logical_tokens.new_full((), dense_tokens),
                )
            )
        )
        self.histogram.scatter_add_(0, (plan.branch.long() - 1).reshape(1), self.one)

    def snapshot(self):
        values = self.values.cpu().tolist()
        histogram = self.histogram.cpu().tolist()
        return dict(
            zip(
                ("verify_ct", "logical_token_ct", "target_token_ct", "dense_token_ct"),
                values,
            ),
            graph_token_histogram={
                str(n): count for n, count in zip(self.grid, histogram) if count
            },
            conditional_verify_graph=True,
        )


def packed_indices(verify_lens, capacity, width):
    """Fixed-shape gather map; padded lanes name the sink, never a live row."""
    ends = verify_lens.cumsum(0)
    lanes = torch.arange(capacity, device=verify_lens.device)
    rows = torch.searchsorted(ends, lanes, right=True)
    valid = rows < verify_lens.numel()
    safe_rows = rows.clamp(max=verify_lens.numel() - 1)
    starts = ends - verify_lens
    columns = lanes - starts[safe_rows]
    indices = safe_rows * width + columns
    return indices.clamp(max=verify_lens.numel() * width - 1), safe_rows, columns, valid


def padded_query_lens(verify_lens, capacity, width):
    """Match native tier padding without reading the live request count.

    Live requests form a positive prefix. Distribute padding among trailing
    sink requests; treating those zeros as live would put all padding in the
    last row and lose query coverage on sparse token grids.
    """
    rows = torch.arange(verify_lens.numel(), device=verify_lens.device)
    live = (verify_lens > 0).sum()
    padding = verify_lens.numel() - live
    leftover = capacity - verify_lens.sum()
    divisor = padding.clamp(min=1)
    base = leftover // divisor
    remainder = leftover % divisor
    sink_lens = base + ((rows - live) < remainder)
    padded = torch.where(rows >= live, sink_lens, verify_lens)
    # Native behavior when every captured slot is a live request.
    padded = padded + ((padding == 0) & (rows == rows.numel() - 1)) * leftover
    return padded.clamp(max=width).to(torch.int32)


class ConditionalVerifyBackend:
    conditional_verify = True

    def __init__(self, delegate, runner):
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )
        from .conditional_graph import _switch_adapter, require_conditional_graph

        require_conditional_graph("switch")
        if type(delegate) is not FullCudaGraphBackend:
            raise ValueError("conditional verify requires the full CUDA graph backend")
        # Metadata prep must be entirely GPU-driven, including the KV index
        # space. Do not silently accept an unported pool/backend combination.
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        attention = runner.attn_backend
        dsv4 = (
            type(attention).__module__
            == "sglang.srt.layers.attention.deepseek_v4_backend"
            and type(attention).__name__ == "DeepseekV4AttnBackend"
        )
        if (
            not (type(attention) is TritonAttnBackend or dsv4)
            or runner.is_encoder_decoder
            or runner.enable_pdmux
            or runner.enable_two_batch_overlap
            or runner.require_mlp_sync
            or runner.model_runner.lora_manager is not None
        ):
            raise ValueError(
                "conditional verify requires Triton or DSV4 metadata, no DP/PP, LoRA, "
                "PDMux or two-batch overlap"
            )
        self.delegate, self.runner = delegate, runner
        self.online_c128 = attention.online_c128_mtp if dsv4 else None
        if self.online_c128 is not None:
            self.online_c128.enable_ragged_verify()
        self.adapter = _switch_adapter()
        self.grid = tuple(runner.capture_num_tokens)
        self.capacities = torch.tensor(
            self.grid, device=runner.device, dtype=torch.long
        )
        self.counters = DeviceVerifyCounters(self.grid, runner.device)
        self.leaves = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.owners = []
        self.outputs = {}
        self.parent = None
        self.stream = None
        self.groups = ()
        slots, width = runner.max_bs, runner.captured_req_width
        self.drafts = torch.zeros(slots * width, device=runner.device, dtype=torch.long)
        self.cache_locs = torch.zeros_like(self.drafts)
        self.lens = torch.zeros(slots, device=runner.device, dtype=torch.int32)
        self.seq_lens = torch.zeros_like(runner.buffers.seq_lens)
        self.req_indices = torch.zeros_like(runner.buffers.req_pool_indices)
        self.branch = torch.ones((), device=runner.device, dtype=torch.int32)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def prepare_capture_inputs(self, forward_batch):
        forward_batch.conditional_verify_lens = (
            forward_batch.spec_info.ragged_verify_layout.verify_lens
        )

    @contextmanager
    def capture_session(self, stream):
        from sglang.srt.distributed.parallel_state import (
            conditional_graph_capture_scope,
        )

        self.stream = stream
        try:
            with conditional_graph_capture_scope() as groups:
                self.groups = groups
                yield
                # Exactly the original tier grid; no B x G model captures.
                graphs = [self.leaves[n] for n in self.grid]
                self.parent = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.parent, stream=stream):
                    self.adapter.capture_switch(
                        self.branch.data_ptr(),
                        [graph.raw_cuda_graph() for graph in graphs],
                        stream.cuda_stream,
                    )
        finally:
            self.stream = None

    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
            ScatterCompactToStrided,
        )
        from msgspec.structs import replace
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        fb = capture_inputs
        if fb is None or shape_key.size in self.leaves:
            raise ValueError(
                "conditional verify requires one capture input per token tier"
            )
        runner, capacity = self.runner, shape_key.size
        slots, width = fb.batch_size, runner.captured_req_width
        original_layout = fb.spec_info.ragged_verify_layout
        self.lens.zero_()
        self.lens[:slots].copy_(original_layout.verify_lens)
        self.seq_lens.fill_(runner.seq_len_fill_value)
        fb.seq_lens_cpu = None
        fb.seq_lens_sum = None

        def composed():
            lens = self.lens[:slots]
            fb.conditional_verify_lens = lens
            layout = RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=lens, graph_num_tokens=capacity
            )
            metadata_layout = replace(
                RaggedVerifyLayout.from_verify_lens_device(
                    verify_lens=padded_query_lens(lens, capacity, width),
                    graph_num_tokens=capacity,
                ),
                cap=width,
            )
            runner._stage_ragged_verify_layout(metadata_layout, capacity)
            indices, rows, columns, valid = packed_indices(lens, capacity, width)
            fb.input_ids.copy_(torch.where(valid, self.drafts[indices], 0))
            fb.positions.copy_(torch.where(valid, self.seq_lens[rows] + columns, 0))
            fb.seq_lens.copy_(self.seq_lens[:slots])
            fb.req_pool_indices.copy_(self.req_indices[:slots])
            # Rebuild the plan during capture, rather than caching a warmup
            # plan's already-translated addresses. Replay then refreshes both
            # full/SWA writes and translated read tables from live GPU inputs.
            translator = runner.model_runner.kv_index_translator
            kv_plan = translator.plan(
                req_pool_indices=fb.req_pool_indices,
                seq_lens=fb.seq_lens,
                seq_lens_cpu=None,
                write_virtual=self.cache_locs,
                read_extent=width,
            )
            kv_plan.bind(fb, translator, cols=torch.where(valid, indices, -1))
            # This prep belongs to the selected branch. Staging all tiers
            # eagerly would overwrite their shared attention buffers.
            runner.attn_backend.init_forward_metadata_out_graph(fb)
            out = forward_fn()
            for name in ("next_token_logits", "hidden_states"):
                packed = getattr(out, name)
                if packed is None:
                    continue
                if name not in self.outputs:
                    self.outputs[name] = packed.new_empty(
                        (runner.max_bs * width, *packed.shape[1:])
                    )
                dense = ScatterCompactToStrided.execute(
                    compact=packed,
                    layout=layout,
                    fill_value=0.0,
                    verify_num_draft_tokens=width,
                )
                self.outputs[name][: slots * width].copy_(dense)
            return out

        # Compile/allocate before capture. Leaves execute serially and copy all
        # externally consumed values into buffers allocated outside this pool,
        # so their model temporaries can reuse one pool like native target tiers.
        for _ in range(2):
            runner.device_module.synchronize()
            composed()
            if post_warmup_hook is not None:
                post_warmup_hook()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
            out = composed()
        self.leaves[capacity] = graph
        self.owners.append((fb, out))
        self.output_template = out

    def prepare(self, verify_input, batch, target_worker):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        plan = verify_input.conditional_verify_plan
        runner = target_worker.model_runner
        batch.input_ids = verify_input.draft_token
        batch.out_cache_loc = verify_input.kv_loc_plan.write_virtual
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        hidden_mode = (
            CaptureHiddenMode.NULL
            if runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        fb = ForwardBatch.init_new(
            batch,
            runner,
            capture_hidden_mode=hidden_mode,
            return_hidden_states_before_norm=False,
            kv_loc_plan=verify_input.kv_loc_plan,
        )
        fb.spec_info = DFlashVerifyInput(
            draft_token=verify_input.draft_token,
            positions=verify_input.positions,
            draft_token_num=verify_input.draft_token_num,
            capture_hidden_mode=hidden_mode,
            ragged_verify_layout=verify_input.compact_verify_layout,
            kv_loc_plan=verify_input.kv_loc_plan,
        )
        if not self.runner.can_run_graph(fb):
            raise ValueError(
                "conditional verify batch does not fit the captured token grid"
            )
        fb.conditional_verify_plan = plan
        fb.conditional_verify_write_virtual = verify_input.kv_loc_plan.write_virtual
        fb.mark_forward_metadata_ready()
        return fb, True

    def replay_dynamic(self, forward_batch):
        plan = forward_batch.conditional_verify_plan
        bs, width = forward_batch.batch_size, self.runner.captured_req_width
        stream = torch.cuda.current_stream()
        for group in self.groups:
            group.check_conditional_graph_replay_stream(stream)
        # No host read of either lengths or tier. Zero stale padded request rows.
        self.lens.zero_()
        self.seq_lens.fill_(self.runner.seq_len_fill_value)
        self.req_indices.zero_()
        self.cache_locs.zero_()
        self.lens[:bs].copy_(plan.verify_lens)
        self.seq_lens[:bs].copy_(forward_batch.seq_lens)
        self.req_indices[:bs].copy_(forward_batch.req_pool_indices)
        self.drafts[: bs * width].copy_(forward_batch.input_ids)
        self.cache_locs[: bs * width].copy_(
            forward_batch.conditional_verify_write_virtual
        )
        self.branch.copy_(plan.branch)
        if self.online_c128 is not None:
            self.online_c128.arm_ragged_verify(
                self.req_indices, self.seq_lens, self.lens
            )
        self.parent.replay()
        self.counters.add(plan, bs * width)
        return dataclasses.replace(
            self.output_template,
            **{name: value[: bs * width] for name, value in self.outputs.items()},
        )

    def cleanup(self):
        if self.parent is not None:
            self.parent.reset()
            self.parent = None
        for graph in self.leaves.values():
            graph.reset()
        self.leaves.clear()
        self.owners.clear()
        self.outputs.clear()
        self.pool = None
        self.groups = ()
        self.delegate.cleanup()
