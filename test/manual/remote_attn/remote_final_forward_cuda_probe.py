"""Full final-forward ordering on CUDA; tiny Qwen/DSV ABI fixtures, no model load."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang_remote_attn.dsv41_layerwise import DSV41Controller
from sglang_remote_attn.pre_layerwise import Config, Controller


def qwen(dtype):
    source = torch.zeros((4, 4096), device="cuda", dtype=dtype)
    destination = torch.zeros_like(source)
    torch.cuda.synchronize()
    tasks, finished, calls = [], [], []
    stream = torch.cuda.Stream()

    def submit(*args, **kw):
        assert finished and calls == ["native"]
        tasks.append(kw)

    sender = NS(
        kv_mgr=NS(add_transfer_request=submit),
        bootstrap_room=1,
        trace_ctx=NS(copy_for_thread=lambda: None),
    )
    job = NS(
        req=NS(disagg_kv_sender=sender),
        indices=[0],
        index_slice=slice(0, 1),
        stager=None,
        native_staging=False,
        complete=lambda layers: None,
    )
    controller = Controller.__new__(Controller)
    controller.config = Config("remote", Path("/tmp/unused"))
    controller.plan = [(job, [((0, 1), (0, 1)), ((2, 3), (2, 3))])]
    batch = NS(forward_mode="extend", split_index=19)

    def native(runner, fb, **kw):
        assert fb.forward_mode == "extend" and fb.split_index == 19 and not kw
        calls.append("native")
        for layer in range(4):
            torch.cuda._sleep(20_000_000)
            source[layer].fill_(layer + 1)
            assert not tasks
        finished.append(True)
        return source

    with patch.dict(
        sys.modules,
        {
            "sglang.srt.model_executor.forward_batch_info": NS(
                ForwardMode=NS(SPLIT_PREFILL="split")
            )
        },
    ):
        result = controller.forward(native, None, batch)
    assert result is source and len(tasks) == 2
    assert tasks[0]["wait_event"] is tasks[1]["wait_event"]
    asynchronous = not tasks[0]["wait_event"].query()
    assert asynchronous, "probe must exercise a still-running GPU forward"
    with torch.cuda.stream(stream):
        for task in tasks:
            stream.wait_event(task["wait_event"])
            for layer in task["kv_buffer_indices"]:
                destination[layer].copy_(source[layer])
    stream.synchronize()
    for layer in range(4):
        torch.testing.assert_close(
            destination[layer], torch.full_like(destination[layer], layer + 1)
        )
    return dict(
        dtype=str(dtype),
        native_calls=1,
        transfers=2,
        shared_full_forward_event=True,
        event_pending_at_submission=asynchronous,
        consumer_wait_correct=True,
    )


def dsv():
    source = torch.zeros((4, 4096), device="cuda")
    destination = torch.zeros_like(source)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        source[3].fill_(4)
    model = NS(decoder_replay_tail=graph)
    runner = NS(model=NS(model=model))
    proofs = []
    job = NS(enqueue=lambda layer, event, **kw: proofs.append((layer, event, kw)))
    controller = DSV41Controller.__new__(DSV41Controller)
    controller.dsv41_remote = True
    controller.scheduler = NS(disaggregation_mode=NS(value="prefill"))
    controller.layers, controller.plan = 4, [job]

    def native(runner, batch):
        assert runner.model.model.decoder_replay_tail is graph
        for layer in range(3):
            torch.cuda._sleep(20_000_000)
            source[layer].fill_(layer + 1)
        graph.replay()
        assert not proofs
        return source

    assert controller.forward(native, runner, None) is source
    assert len(proofs) == 1 and proofs[0][0] == 3 and proofs[0][2] == {"flush": True}
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        consumer.wait_event(proofs[0][1])
        destination.copy_(source)
    consumer.synchronize()
    for layer in range(4):
        torch.testing.assert_close(
            destination[layer], torch.full_like(destination[layer], layer + 1)
        )
    assert model.decoder_replay_tail is graph and controller.executing is None
    return dict(
        native_tail_graph_preserved=True, full_forward_proof_covers_graph_writes=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        scope=__doc__,
        qwen=[qwen(dtype) for dtype in (torch.bfloat16, torch.float16)],
        dsv=dsv(),
    )
    root = args.source_root.resolve()
    files = [
        root / "plugins/remote_attn/sglang_remote_attn" / name
        for name in ("pre_layerwise.py", "dsv41_layerwise.py", "prefill_overlap.py")
    ]
    files.append(Path(__file__).resolve())
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
