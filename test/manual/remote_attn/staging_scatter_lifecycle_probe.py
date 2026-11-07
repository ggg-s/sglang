"""Exercise failed deferred-scatter publication with real CUDA copies.

Uses the production DecodeStagingHandler lifecycle loaded from source. The
scatter body is a tiny pinned-host H2D copy; allocator, event-record failures,
and stream-drain failures are explicit fixtures. No model, network, or RDMA
completion semantics are simulated as real hardware validation.
"""

import argparse
import ast
import hashlib
import json
import logging
import sys
import threading
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def load_handler(root):
    path = root / "python/sglang/srt/disaggregation/common/staging_handler.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "DecodeStagingHandler"
    )
    source = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )

    class RecordFailure:
        def record(self, stream):
            raise RuntimeError("injected event publication failure")

    namespace = dict(
        torch=torch,
        threading=threading,
        deque=deque,
        logger=logging.getLogger(__name__),
        create_device_event=lambda device: RecordFailure(),
    )
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), namespace)
    return namespace["DecodeStagingHandler"]


def run_case(handler_type, *, drain_fails):
    torch.cuda.set_device(0)
    cuda_stream = torch.cuda.Stream()
    host = torch.arange(65536, dtype=torch.int32).pin_memory()
    destination = torch.full(host.shape, -1, dtype=host.dtype, device="cuda")
    torch.cuda.synchronize()
    copied = torch.cuda.Event()
    order = []
    freed = []

    class TrackedStream:
        fail = drain_fails

        def synchronize(self):
            order.append("drain")
            if self.fail:
                raise RuntimeError("injected stream drain failure")
            cuda_stream.synchronize()
            assert copied.query()
            order.append("drained")

    stream = TrackedStream()
    receiver = SimpleNamespace(
        chunk_staging_infos=[(5, 0, 0, host.numel() * host.element_size(), 3)],
        chunk_staging_layers={0: ((0, 2), 0, 3)},
    )
    request = SimpleNamespace(
        _chunk_events=[],
        _staging_layer_events={},
        _staging_completed_layers=[],
        _staging_scatter_streams=[],
    )
    handler = handler_type.__new__(handler_type)
    handler.device = torch.device("cuda", 0)
    handler.tp_rank = 0
    handler.staging_allocator = SimpleNamespace(_scatter_stream=None)
    handler._scatter_lock = threading.RLock()
    handler._staging_lifecycle = threading.Condition(handler._scatter_lock)
    handler._room_to_decode_req = {7: request}
    handler._room_to_receiver = {7: receiver}
    handler._room_tokens = {7: object()}
    handler._inflight_submits = {}
    handler._claimed_chunks = set()
    handler._defer_scatter = True
    handler._writer_counts = {}
    handler._pending_scatters = deque()
    handler._closing_rooms = set()
    handler.kv_manager = SimpleNamespace(
        _staging_ctx=SimpleNamespace(
            room_receivers={7: receiver}, room_bootstrap={7: object()}
        )
    )

    def scatter_region(*args, **kwargs):
        assert args[-1] is stream
        assert kwargs["layer_ids"] == (0, 2)
        assert kwargs["staging_bytes"] == host.numel() * host.element_size()
        with torch.cuda.stream(cuda_stream):
            torch.cuda._sleep(1_000_000)
            destination.copy_(host, non_blocking=True)
            # Observation only: the production completion-event record below
            # fails. Teardown must rely on its retained stream, not this event.
            copied.record(cuda_stream)
        order.append("copy_enqueued")
        return True

    def free(alloc_id, decode_req):
        assert alloc_id == 5 and decode_req is request
        assert copied.query(), "freed staging while CUDA still reads it"
        freed.append(alloc_id)
        order.append("free")

    handler._scatter_region = scatter_region
    handler._free_and_send_watermark = free
    module = types.ModuleType("sglang.srt.disaggregation.common.staging_buffer")
    module.StagingAllocator = SimpleNamespace(CHUNK_SCATTERED=-3)
    with patch.dict(sys.modules, {module.__name__: module}):
        try:
            handler.submit_chunk_scatter(7, 0, 0, 3, stream=stream)
        except RuntimeError as error:
            assert str(error) == "injected event publication failure"
        else:
            raise AssertionError("event publication fault did not reach caller")
    assert request._chunk_events == []
    assert request._staging_layer_events == {}
    assert request._staging_completed_layers == []
    assert request._staging_scatter_streams == [stream]
    assert receiver.chunk_staging_infos[0][0] == 5
    assert handler._inflight_submits == {} and handler._claimed_chunks == set()
    if drain_fails:
        try:
            handler.unregister_decode_req(7)
        except RuntimeError as error:
            assert str(error) == "injected stream drain failure"
        else:
            raise AssertionError("drain failure did not stop reclamation")
        assert handler.is_staging_room(7)
        assert 7 not in handler._room_tokens
        assert freed == []
        assert receiver.chunk_staging_infos[0][0] == 5
        assert request._staging_scatter_streams == [stream]
        stream.fail = False
    handler.unregister_decode_req(7)
    assert order[-3:] == ["drain", "drained", "free"]
    assert not handler.is_staging_room(7)
    assert freed == [5]
    assert request._staging_scatter_streams == []
    assert torch.equal(destination.cpu(), host)
    return {
        "event_record_failed": True,
        "stream_drain_failed": drain_fails,
        "retained_until_drain": True,
        "copy_matches": True,
        "real_cuda": True,
        "order": order,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    handler_type = load_handler(args.source_root)
    files = [
        "python/sglang/srt/disaggregation/common/staging_handler.py",
        "test/manual/remote_attn/staging_scatter_lifecycle_probe.py",
    ]
    result = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "real_rdma": False,
        "scope": __doc__,
        "cases": [run_case(handler_type, drain_fails=value) for value in (False, True)],
        "source_sha256": {
            name: hashlib.sha256((args.source_root / name).read_bytes()).hexdigest()
            for name in files
        },
    }
    content = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
