"""Small real-CUDA proof/lease probe; no model, Mooncake or TP simulation claim.

Run with PYTHONPATH=plugins/remote_attn. Model compute/native envelope submission
are small tensor/queue fixtures. Plugin page-index copies, CUDA events, launch
plans, IO ledger, deferred release and ordered TP1 retirement are real.
"""

import argparse
import hashlib
import json
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang_remote_attn import prefill_overlap as po
from sglang_remote_attn.pre_layerwise import Config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-cache", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    raw = (torch.arange(257, device="cuda", dtype=torch.int32) + 8) * 3
    # Warm lazy kernels, NumPy and the pinned allocator before timing a pending
    # event; first cudaHostAlloc/module loading may synchronize the device.
    warm_indices, warm_event, warm_refs = po.page_indices(raw, 8)
    warm_event.synchronize()
    del warm_indices, warm_event, warm_refs
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        indices, event, keepalive = po.page_indices(raw, 8)
    pending_observed = not event.query()
    event.synchronize()
    expected = (raw[::8] // 8).cpu().numpy()
    assert (indices == expected).all()
    assert keepalive[1].is_pinned()
    assert pending_observed, "GPU completed before the probe could check pending state"

    controller = SimpleNamespace(rank=0, tp_size=1, sources=[])
    controller.scheduler = scheduler = SimpleNamespace()
    controller.config = Config("layerwise", args.output.parent, drain_timeout=5)
    controller.agree = lambda flags: flags
    overlap = po.PrefillOverlap(controller)
    cache = None
    if args.prefix_cache:
        from sglang_remote_attn.source_cache import CacheGuard

        class PrefixCacheBoundary:
            def __init__(self):
                self.refs = {}
                self.operations = []

            def lock(self, node):
                refs = self.refs.setdefault(node, {"FULL": 0, "SWA": 0})
                for ct in refs:
                    refs[ct] += 1
                return SimpleNamespace(node=node, closed=False)

            def unlock(self, receipt):
                assert not receipt.closed
                for ct in self.refs[receipt.node]:
                    assert self.refs[receipt.node][ct] > 0
                    self.refs[receipt.node][ct] -= 1
                receipt.closed = True

            def checkpoint(self, req, **kwargs):
                self.operations.append("checkpoint")

            def finish(self, handle, outcome):
                self.operations.append("finish")

            def reset(self):
                assert not any(n for refs in self.refs.values() for n in refs.values())

        cache = PrefixCacheBoundary()
        controller.cache_guard = CacheGuard(cache)
    tasks = queue.Queue()
    registration = SimpleNamespace(dst_kv_ptrs=[11, 21], dst_kv_item_len=8)
    peer = SimpleNamespace(mooncake_session_id="peer:1", is_dummy=False)
    manager = SimpleNamespace(
        _dsv41_p_overlap=overlap,
        transfer_infos={},
        decode_kv_args_table={"peer:1": registration},
    )
    freed, observed, worker_errors = [], [], []
    storage = torch.zeros(64, device="cuda")

    def add(manager, room, indices, span, last, **kwargs):
        tasks.put(SimpleNamespace(room=room, indices=indices, last=last, **kwargs))

    def enqueue(room, wait):
        po.track_transfer(
            add,
            manager,
            room,
            indices,
            slice(0, len(indices)),
            False,
            wait_event=wait,
            indices_event=event,
        )

    def request(room):
        request = SimpleNamespace(
            rid=str(room),
            bootstrap_room=room,
            kv=SimpleNamespace(req_pool_idx=room),
            cache_request_handle=(room, 0),
            start_send_idx=0,
            extend_end=16,
            origin_input_ids=list(range(33)),
            pending_bootstrap=False,
            extend_len=16,
            disagg_kv_sender=SimpleNamespace(
                poll=lambda: "success", kv_mgr=manager, curr_idx=0
            ),
        )
        root = args.output.parent / str(room) / "0"
        request.offer = (root, {"generation": str(room)})
        write_json(root / "offer.json", request.offer[1])
        manager.transfer_infos[room] = {"peer:1": peer}
        request.disagg_kv_sender.abort = lambda: None
        if cache is not None:
            request.kv.cache_protected_len = 8
            request.last_node = room
            request.lock = cache.lock(room)
        return request

    req = request(1)
    batch = SimpleNamespace(
        reqs=[req],
        forward_mode=SimpleNamespace(
            is_extend=lambda: True,
            is_prebuilt=lambda: False,
            is_split_prefill=lambda: False,
            is_mixed=lambda: False,
        ),
        is_prefill_only=False,
        return_hidden_states=False,
        input_embeds=None,
        replace_embeds=None,
    )
    batch.copy = lambda: SimpleNamespace(reqs=list(batch.reqs))
    controller.candidates = lambda batch, **kw: [
        (r, r.disagg_kv_sender, *r.offer, ((peer, registration),), {})
        if not r.pending_bootstrap
        else None
        for r in batch.reqs
    ]
    controller.prepare = lambda batch, **kw: []

    def native_send(scheduler, req, *, last_chunk, end_idx=None):
        req.start_send_idx = req.extend_end if end_idx is None else end_idx
        req.disagg_kv_sender.curr_idx = req.start_send_idx // 8
        wait = getattr(req.disagg_kv_sender, "_early_send_wait_event", None)
        if last_chunk:
            wait = req._dsv41_p_lease.plans[-1].forward_done
        enqueue(req.bootstrap_room, wait)

    scheduler.send_kv_chunk = lambda req, **kw: overlap.send_chunk(
        native_send, scheduler, req, end_idx=kw.pop("end_idx", None), **kw
    )

    def forward(runner, fb):
        torch.cuda._sleep(60_000_000)
        storage.add_(1)

    def launch(scheduler, batch):
        with torch.cuda.stream(stream):
            overlap.forward(forward, None, batch)
        return SimpleNamespace(
            delay_sample_func=lambda: None, copy_done=torch.cuda.Event()
        )

    def sampled(result):
        with torch.cuda.stream(stream):
            result.copy_done.record()
        result.delay_sample_func = None

    def native_release(req, label):
        def release():
            if cache is not None:
                if label == "normal":
                    cache.checkpoint(req, up_to=req.extend_end)
                cache.finish(
                    req.cache_request_handle,
                    SimpleNamespace(name="SUCCESS" if label == "normal" else "ABORT"),
                )
                cache.unlock(req.lock)
                req.lock = None
            freed.append(label)

        return release

    def io_worker():
        try:
            while True:
                task = tasks.get(timeout=5)
                if task is None:
                    break
                if task.wait_event is not None:
                    task.wait_event.synchronize()
                task.indices_event.synchronize()
                if cache is not None:
                    assert all(ref > 0 for ref in cache.refs[task.room].values())
                observed.append(storage.cpu().tolist())
                task.on_complete()
        except BaseException as exc:
            worker_errors.append(repr(exc))

    polls = SimpleNamespace(KVPoll=SimpleNamespace(Success="success", Failed="failed"))
    try:
        with patch.dict(sys.modules, {"sglang.srt.disaggregation.base.conn": polls}):
            middle = overlap.run_batch(launch, scheduler, batch)
            lease = req._dsv41_p_lease
            req.extend_end = 33
            checks_before_final = overlap.admission.checks
            final = overlap.run_batch(launch, scheduler, batch)
            assert overlap.admission.checks == checks_before_final == 1
            assert middle._dsv41_p_plan is not final._dsv41_p_plan
            assert len(lease.plans) == 2
            po.after_result(
                lambda s, b, r: s.send_kv_chunk(req, last_chunk=False, end_idx=16),
                SimpleNamespace(
                    _pre_layerwise=SimpleNamespace(prefill_overlap=overlap),
                    send_kv_chunk=scheduler.send_kv_chunk,
                ),
                None,
                middle,
            )
            assert tasks.qsize() == 1
            po.after_result(
                lambda s, b, r: scheduler.send_kv_chunk(req, last_chunk=True),
                SimpleNamespace(
                    _pre_layerwise=SimpleNamespace(prefill_overlap=overlap)
                ),
                None,
                final,
            )
            overlap.defer_release(req, native_release(req, "normal"))
            if cache is not None:
                cache.checkpoint(req, up_to=req.extend_end)
                assert not cache.operations
            assert not lease.ready_to_finish()  # deferred sample/copy plus queued IO
            sampled(middle)
            sampled(final)
            stream.synchronize()
            assert not lease.ready_to_finish()  # GPU done, queue still owns reads
            worker = threading.Thread(target=io_worker)
            worker.start()
            tasks.put(None)
            worker.join(6)
            assert not worker.is_alive() and not worker_errors
            assert lease.ready_to_finish()
            overlap.retirement.observe()
            command = overlap.retirement.poll_command()
            overlap.retirement.arm(command)
            overlap.retirement.finalize()
            assert freed == ["normal"] and not overlap.leases
            # Cancellation while a forward and a native envelope are outstanding.
            cancel = request(2)
            batch.reqs = [cancel]
            result = overlap.run_batch(launch, scheduler, batch)
            lease = cancel._dsv41_p_lease
            overlap.defer_release(cancel, native_release(cancel, "cancel"))
            before = len(observed)
            worker = threading.Thread(target=io_worker)
            worker.start()
            tasks.put(None)
            overlap.drain(cancel)
            worker.join(6)
            assert not worker.is_alive() and not worker_errors
            assert len(observed) == before + 1
            assert (
                result.delay_sample_func is not None
            )  # never wait an unrecorded copy_done
            assert freed == ["normal", "cancel"]
            storage.fill_(777)
            torch.cuda.synchronize()
            assert storage.eq(777).all().item()
            assert all(all(value < 777 for value in row) for row in observed)
            # A cached negative result waits for a bootstrap edge, then one
            # ordered revalidation. Stable checks below perform no new vote.
            third = request(3)
            third.pending_bootstrap = True
            batch.reqs = [third]
            assert overlap.admission.select(batch)[1] == [0]
            for _ in range(32):
                assert overlap.admission.select(batch)[1] == [0]
            third.pending_bootstrap = False

            def controls():
                from sglang_remote_attn.prefill_admission import CONTROL

                overlap.poll()
                for item in overlap.inject([]) or []:
                    if CONTROL in item:
                        overlap.admission.arm(item[CONTROL])
                overlap.finalize_control()

            controls()
            assert overlap.admission.select(batch)[1] == [1]
            storage.zero_()
            torch.cuda.synchronize()
            revoke_result = overlap.run_batch(launch, scheduler, batch)
            lease = third._dsv41_p_lease
            overlap.defer_release(third, native_release(third, "revoke"))
            replacement = SimpleNamespace(dst_kv_ptrs=[31, 41], dst_kv_item_len=8)
            guard = overlap.admission.guard(manager)
            guard.publish(
                "peer:1",
                replacement,
                lambda: manager.decode_kv_args_table.__setitem__("peer:1", replacement),
            )
            assert manager.decode_kv_args_table["peer:1"] is registration

            def abort():
                assert lease.closed and not lease.ledger.pending
                assert manager.decode_kv_args_table["peer:1"] is registration

            third.disagg_kv_sender.abort = abort
            worker = threading.Thread(target=io_worker)
            worker.start()
            tasks.put(None)
            with patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.utils": SimpleNamespace(
                        prepare_abort=lambda req, message: setattr(
                            req, "finished_reason", message
                        )
                    )
                },
            ):
                controls()  # waits real GPU source readers + queued IO before unpin/abort
            worker.join(6)
            assert not worker.is_alive() and not worker_errors
            assert manager.decode_kv_args_table["peer:1"] is replacement
            assert freed == ["normal", "cancel", "revoke"]
            assert overlap.admission.checks == 3
            assert overlap.admission.revalidations == overlap.admission.revocations == 1
            votes = overlap.retirement.final_votes
            for _ in range(100):
                overlap.poll()
                overlap.inject([])
            assert overlap.retirement.final_votes == votes == 3
            for plan in list(overlap.plans.values()):
                if plan.proof.result is result or plan.proof.result is revoke_result:
                    sampled(plan.proof.result)
                    plan.processed = True
            overlap.poll()
            storage.fill_(999)
            torch.cuda.synchronize()
            assert storage.eq(999).all().item()
            assert not overlap.admission.guard(manager).pins
            if cache is not None:
                assert cache.operations == ["checkpoint", "finish", "finish", "finish"]
                assert not controller.cache_guard.leases
                assert all(
                    value == 0
                    for refs in cache.refs.values()
                    for value in refs.values()
                )
                cache.reset()
        plugin = Path(po.__file__).parent
        result = dict(
            device=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            pending_indices_observed=pending_observed,
            pinned_indices_equal=True,
            prefix_cache=args.prefix_cache,
            cache_reads_drained=(not controller.cache_guard.leases)
            if cache is not None
            else None,
            cache_backend="lock/checkpoint boundary fixture; real hybrid receipt tests on CPU"
            if cache is not None
            else None,
            launch_plans=4,
            admission_checks=overlap.admission.checks,
            revalidations=overlap.admission.revalidations,
            revocations=overlap.admission.revocations,
            stable_chunk_admission_votes=0,
            io_callbacks=len(observed),
            releases=freed,
            final_votes=votes,
            empty_poll_votes=0,
            cancel_with_unsubmitted_sampling=True,
            no_post_release_write=True,
            source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), *sorted(plugin.glob("*.py"))]
            },
            limitations=[
                "No DSV4.1 model load or token accuracy/performance measurement",
                "IO callback fixture; no Mooncake TE or SMG",
                "TP1 CUDA; real TP2 Gloo is covered separately on CPU",
            ],
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))
    finally:
        overlap.admission.close()
        overlap.retirement.close()


if __name__ == "__main__":
    main()
