"""Check Flash-0731's split frontier with CUDA compressors and Mooncake TCP."""

import json
import os
from dataclasses import dataclass
from itertools import pairwise
from types import SimpleNamespace as NS

import torch
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
)
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_remote_attn import dsv4_split_prefill, pd_mesh
from sglang_remote_attn import qwen_split_prefill as split
from sglang_remote_attn.split_decode_budget import DecodeBudget

PAGE, WINDOW = 256, 128
LENGTH, PREFIX_HIT = 9215, 4096


@dataclass
class Peer:
    dst_state_indices: list


class Transport:
    """Stage the small CUDA fixtures through registered host memory."""

    def __enter__(self):
        from mooncake.engine import TransferEngine

        self.source, self.destination = TransferEngine(), TransferEngine()
        for engine in (self.source, self.destination):
            assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
        self.session = f"127.0.0.1:{self.destination.get_rpc_port()}"
        self.transfers = 0
        return self

    def __exit__(self, *_):
        self.source = self.destination = None

    def copy(self, source, target):
        assert source.shape == target.shape and source.dtype == target.dtype
        host_source = torch.empty_like(source, device="cpu", pin_memory=True)
        host_target = torch.empty_like(host_source, pin_memory=True)
        host_source.copy_(source)
        addresses = [host_source.data_ptr(), host_target.data_ptr()]
        size = host_source.nbytes
        engines = (self.source, self.destination)
        for engine, address in zip(engines, addresses):
            assert engine.register_memory(address, size) == 0
        try:
            assert (
                self.source.batch_transfer_sync_write(
                    self.session, addresses[:1], addresses[1:], [size]
                )
                == 0
            )
            target.copy_(host_target)
            self.transfers += 1
        finally:
            for engine, address in zip(engines, addresses):
                assert engine.unregister_memory(address) == 0


def mapping(first_page, *, reverse=False):
    pages = torch.arange(
        first_page, first_page + (LENGTH + PAGE) // PAGE, device="cuda"
    )
    if reverse:
        pages = pages.flip(0)
    slots = (pages[:, None] * PAGE + torch.arange(PAGE, device="cuda")).flatten()
    table = slots.to(torch.int32).repeat(2, 1)
    full_to_swa = torch.arange((first_page + len(pages)) * PAGE, device="cuda")
    return pages, table, full_to_swa


def compressor(ratio, dim, online, rid, table, full_to_swa):
    ring_size = 1 if online else (8 if ratio == 4 else 128)
    state_rows = 3 if online else (full_to_swa.numel() // PAGE + 2) * ring_size
    width = dim * (3 if online else (4 if ratio == 4 else 2))
    state = torch.full((state_rows, width), float("nan"), device="cuda")
    inputs = torch.randn((LENGTH + 1, dim * (4 if ratio == 4 else 2)), device="cuda")
    ape = torch.randn((8 if ratio == 4 else 128, dim), device="cuda")
    req = torch.tensor([rid], device="cuda")
    live_plans = []

    def run(start, end, *, decode=False):
        lengths = torch.tensor([end], dtype=torch.int64)
        if decode:
            args = {
                "seq_lens": lengths.cuda(),
                "req_pool_indices": req,
                "req_to_token": table,
            }
            plan = (
                CompressorDecodePlan.generate_online(**args)
                if online
                else CompressorDecodePlan.generate(
                    ratio,
                    **args,
                    full_to_state=full_to_swa,
                    swa_page_size=PAGE,
                    ring_size=ring_size,
                )
            )
        else:
            args = {
                "req_pool_indices": req,
                "req_to_token": table,
                "seq_lens": lengths,
                "extend_lens": torch.tensor([end - start]),
                "num_q_tokens": end - start,
            }
            plan = (
                CompressorPrefillPlan.generate_online(**args)
                if online
                else CompressorPrefillPlan.generate(
                    ratio,
                    **args,
                    full_to_state=full_to_swa,
                    swa_page_size=PAGE,
                    ring_size=ring_size,
                )
            )
        # Native forwards retain plan metadata through CUDA completion.
        # CPU-generated plans own pinned memory used by asynchronous copies.
        live_plans.append(plan)
        return compress_forward(
            state.view(-1, 1 if online else ratio, width),
            inputs[start:end],
            ape,
            plan,
            head_dim=dim,
            compress_ratio=ratio,
            is_online=online,
        )

    return NS(run=run, state=state, inputs=inputs, ape=ape, ring_size=ring_size)


def measure_forward(controller, phase, tokens, forward):
    worker = NS(model_runner=NS(_pre_layerwise=controller))
    batch = NS(
        forward_mode=NS(
            is_extend=lambda: phase != "decode",
            is_decode=lambda: phase == "decode",
        ),
        _hybrid_split_extend=phase == "suffix",
        extend_num_tokens=tokens,
        batch_size=lambda: 1,
    )
    result = split.time_suffix_forward(lambda *_: NS(output=forward()), worker, batch)
    torch.cuda.synchronize()
    split.record_completed_suffix(controller, result)


def negotiate(adaptive, schedule, p_table, p_swa, d_pages):
    budget = (
        DecodeBudget(
            target=0.5,
            stall_ms=600,
            patience_ms=200,
            pinned_chunk=False,
            max_cap=LENGTH // 2,
            min_cap=2560,
        )
        if adaptive
        else None
    )
    controller = NS(
        rank=0,
        split_prefill_adaptive=adaptive,
        split_prefill_decode_budget=budget,
        split_prefill_page_size=PAGE,
        split_prefill_chunk_size=1024,
        split_prefill_budget=1536,
        split_prefill_auto_cap=0,
        split_prefill_defer_ticks=150,
        split_prefill_patience_ticks=15,
        split_prefill_prefix_rate=None,
        split_prefill_cut_ratio=0.75,
        split_prefill_cut_basis="miss",
        split_prefill_suffix_tokens=0,
        split_prefill_suffix_cap=None if adaptive else 4096,
        dsv4_split_schema="a" * 64,
        scheduler=schedule,
        agree=lambda values: values,
    )
    if adaptive:
        fixture = compressor(4, 512, False, 0, p_table, p_swa)
        fixture.run(0, 1024)
        fixture.run(1024, 1025, decode=True)
        torch.cuda.synchronize()
        measure_forward(controller, "suffix", 1024, lambda: fixture.run(0, 1024))
        measure_forward(
            controller, "decode", 1, lambda: fixture.run(1024, 1025, decode=True)
        )
        assert budget.suffix_rate > 0 and budget.decode_ms > 0

    request = NS(origin_input_ids=range(LENGTH), bootstrap_room=1)
    offer, reason = split.negotiate_decode_offer(controller, request, None)
    assert offer is not None, reason
    reserve_start = LENGTH // 2 // PAGE * PAGE
    earliest = max(reserve_start + WINDOW, (LENGTH + 1) // 2)
    if offer.suffix_cap:
        earliest = max(earliest, LENGTH - offer.suffix_cap)
    min_cut = (earliest + PAGE - 1) // PAGE * PAGE
    receiver = NS(
        kv_mgr=NS(),
        _dsv4_split_window=(
            controller.dsv4_split_schema,
            reserve_start,
            d_pages[reserve_start // PAGE :].tolist(),
        ),
    )
    split.publish_decode_offer(controller, receiver, LENGTH, min_cut, offer)
    frames = pd_mesh.bootstrap(
        lambda _, **kwargs: kwargs["message"],
        receiver,
        bootstrap_info={"is_dummy": False},
        message=[b""] * 10,
        operation="send_metadata",
    )
    peer = pd_mesh.parse_transfer(
        lambda **_: Peer([[int(d_pages[-1])], [1]]), msg=frames
    )
    assert dsv4_split_prefill.compatible_offer(controller, peer, LENGTH, min_cut)
    rates = None
    if adaptive:
        chunk, _, _, rate, step_us = receiver._hybrid_split_schedule
        assert rate == peer._hybrid_split_suffix_rate and step_us > 0
        assert chunk == offer.chunk == 1536
        # The fixture uses the same compressor on both sides.
        rates = rate, rate
    else:
        assert not hasattr(receiver, "_hybrid_split_schedule")
    cut, reason = split.select_cut(
        controller,
        LENGTH,
        PREFIX_HIT,
        (*peer._hybrid_split_limits, getattr(peer, "_hybrid_split_suffix_cap", 0)),
        rates,
    )
    assert reason == "selected"
    assert cut == (6656 if adaptive else 7936)
    return controller, peer, reserve_start, cut, offer


def run(transport, online, adaptive):
    torch.manual_seed(731)
    p_pages, p_table, p_swa = mapping(1)
    d_pages, d_table, d_swa = mapping(32, reverse=True)
    controller, peer, reserve_start, cut, offer = negotiate(
        adaptive, NS(sliding_window_size=WINDOW), p_table, p_swa, d_pages
    )
    bounds = list(range(cut, LENGTH, offer.chunk)) + [LENGTH]
    assert len(bounds) >= 3
    p_swa_rows = torch.arange(p_swa.numel(), device="cuda", dtype=torch.float32).view(
        -1, PAGE
    )
    d_swa_rows = torch.full((d_swa.numel() // PAGE, PAGE), -1.0, device="cuda")
    sender = NS(curr_idx=3, kv_mgr=NS(transfer_infos={1: {"decode": peer}}))
    req = NS(
        bootstrap_room=1,
        extend_end=cut,
        disagg_kv_sender=sender,
        _hybrid_split_at=cut,
        _hybrid_prefill_controller=controller,
    )
    split.seal(req, cut)
    assert sender.num_kv_indices == cut // PAGE

    compressors = []
    for ratio, dim in ((4, 512), (4, 128), (128, 512)):
        is_online = online and ratio == 128
        p = compressor(ratio, dim, is_online, 0, p_table, p_swa)
        d = compressor(ratio, dim, is_online, 1, d_table, d_swa)
        reference = compressor(ratio, dim, is_online, 0, p_table, p_swa)
        for other in (d, reference):
            other.inputs.copy_(p.inputs)
            other.ape.copy_(p.ape)
        prefix = p.run(0, cut)
        d_prefix = torch.empty_like(prefix)
        transport.copy(prefix, d_prefix)
        compressors.append((ratio, p, d, reference, d_prefix))

    source_page = int(p_pages[cut // PAGE - 1])
    dest_page = int(d_pages[cut // PAGE - 1])
    manager = NS(kv_args=NS(state_types=[StateType.SWA, StateType.DSV4_REQUEST_STATE]))

    def send_states(_, target, states, *_args):
        assert states == [[source_page], None]
        assert target.dst_state_indices == [[dest_page], [1]]
        transport.copy(p_swa_rows[source_page], d_swa_rows[dest_page])
        for ratio, p, d, _, _ in compressors:
            if ratio == 4:
                width = p.ring_size
                transport.copy(
                    p.state[source_page * width : (source_page + 1) * width],
                    d.state[dest_page * width : (dest_page + 1) * width],
                )

    dsv4_split_prefill.send_states(
        send_states, manager, peer, [[source_page], [0]], None
    )
    torch.testing.assert_close(
        d_swa_rows[dest_page, -WINDOW:], p_swa_rows[source_page, -WINDOW:]
    )
    assert torch.all(d_swa_rows[d_pages[cut // PAGE]] == -1)

    # The paged DSpark draft uses its own KV buffer at the same SWA page IDs.
    # Exercise that second component and its positional request-row transfer.
    draft_peer = Peer([*peer.dst_state_indices, peer.dst_state_indices[0], [1]])
    draft_peer._dsv4_split_swa_destination = peer._dsv4_split_swa_destination
    draft_manager = NS(
        kv_args=NS(
            state_types=[
                *manager.kv_args.state_types,
                StateType.SWA,
                StateType.REQUEST_ROW,
            ]
        )
    )
    p_draft = p_swa_rows + 1
    d_draft = torch.full_like(d_swa_rows, -1)
    p_valid_from = torch.tensor([7, 0], dtype=torch.int64, device="cuda")
    d_valid_from = torch.full_like(p_valid_from, -1)

    def send_draft_states(manager, target, states, *_args):
        send_states(manager, Peer(target.dst_state_indices[:2]), states[:2])
        assert states[2:] == [[source_page], [0]]
        assert target.dst_state_indices[2:] == [[dest_page], [1]]
        transport.copy(p_draft[states[2][0]], d_draft[target.dst_state_indices[2][0]])
        transport.copy(p_valid_from[0:1], d_valid_from[1:2])

    dsv4_split_prefill.send_states(
        send_draft_states,
        draft_manager,
        draft_peer,
        [[source_page], [0], [source_page], [0]],
        None,
    )
    torch.testing.assert_close(d_draft[dest_page], p_draft[source_page])
    assert torch.all(d_draft[d_pages[cut // PAGE]] == -1)
    assert d_valid_from.tolist() == [-1, 7]

    # Exercise the same CUDA mapping release used by D between suffix chunks.
    # All compressors share this FULL->SWA map.
    released = []
    allocator = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
    allocator.page_size = PAGE
    allocator._swa_req_ring = False
    allocator.full_to_swa_index_mapping = d_swa
    allocator.free_group = None
    allocator.swa_attn_allocator = NS(
        debug_mode=False,
        free_page_ids=lambda pages: released.extend(pages.tolist()),
        available_size=lambda: len(released) * PAGE,
        size=d_swa.numel(),
    )
    cursor = NS(value=reserve_start)
    cursor.get_evicted_seqlen = lambda component: cursor.value
    cursor.set_evicted_seqlen = lambda component, value: setattr(cursor, "value", value)
    cursor.req_pool_idx = 1
    d_req = NS(kv=cursor)
    controller.scheduler.disagg_decode_prealloc_queue = NS(
        req_to_token_pool=NS(req_to_token=d_table),
        token_to_kv_pool_allocator=allocator,
    )
    outputs = [[] for _ in compressors]
    for start, end in pairwise(bounds):
        dsv4_split_prefill.trim_swa(controller, d_req, start)
        assert cursor.get_evicted_seqlen(ComponentType.SWA) == start - PAGE
        assert released == d_pages[reserve_start // PAGE : start // PAGE - 1].tolist()
        retired = d_table[1, reserve_start : start - PAGE].long()
        assert not d_swa[retired].any()
        retained = d_table[1, start - PAGE :].long()
        assert torch.equal(d_swa[retained], retained)
        for pieces, (_, _, d, _, _) in zip(outputs, compressors):
            pieces.append(d.run(start, end))
        torch.cuda.synchronize()

    errors = []
    decode_errors = []
    for pieces, (ratio, _, d, reference, prefix) in zip(outputs, compressors):
        actual = torch.cat((prefix, *pieces))
        expected = reference.run(0, LENGTH)
        label = f"C{ratio} dim={actual.size(-1)}"
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-5,
            rtol=2e-5,
            msg=lambda error, label=label: f"{label}: {error}",
        )
        errors.append(float((actual - expected).abs().max()))
        # The first decode step closes both blocks and consumes the carry
        # produced by D's last, partial prefill chunk.
        first_decode = d.run(LENGTH, LENGTH + 1, decode=True)
        expected_decode = reference.run(LENGTH, LENGTH + 1, decode=True)
        torch.testing.assert_close(first_decode, expected_decode, atol=2e-5, rtol=2e-5)
        decode_errors.append(float((first_decode - expected_decode).abs().max()))
    return {
        "online_c128": online,
        "cut": cut,
        "chunks": [end - start for start, end in pairwise(bounds)],
        "offer_cap": offer.suffix_cap,
        "decode_schedule": offer.schedule,
        "max_compressor_error": max(errors),
        "max_first_decode_error": max(decode_errors),
        "retired_swa_pages": len(released),
        "draft_swa_and_request_row": "pass",
    }


if __name__ == "__main__":
    os.environ["MC_FORCE_TCP"] = "1"
    torch.cuda.set_device(0)
    adaptive = os.environ.get("HANDOFF_SPLIT_ADAPTIVE", "0") == "1"
    with Transport() as transport:
        result = [run(transport, online, adaptive) for online in (False, True)]
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "mode": "adaptive" if adaptive else "static",
                    "transfers": transport.transfers,
                    "cases": result,
                }
            )
        )
