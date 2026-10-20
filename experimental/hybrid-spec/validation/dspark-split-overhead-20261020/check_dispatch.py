"""Bounded old/new hook microcheck. No model, SA query, or target verification.

Pass the pre-change dspark.py as argv[1]. Latency is dispatch overhead only;
CPU/CUDA profiler counts identify nonzero and runtime synchronization calls.
"""

import ast
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import msgspec
import torch
from hybrid_spec.integrations.sglang import dspark
from hybrid_spec.integrations.sglang.gated_override import GatedOverride


def main():
    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/srt/speculative/dspark_components/dspark_draft.py"
    tree = ast.parse(source.read_text())
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in ("DraftBlockResult", "DraftProposal")
    ]
    namespace = dict(torch=torch, msgspec=msgspec, Optional=Optional)
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    module_name = "sglang.srt.speculative.dspark_components.dspark_draft"
    dto = ModuleType(module_name)
    dto.DraftBlockResult = namespace["DraftBlockResult"]
    dto.DraftProposal = namespace["DraftProposal"]
    sys.modules[module_name] = dto
    spec = importlib.util.spec_from_file_location("baseline_dspark", sys.argv[1])
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    def runner(module, hit):
        view = GatedOverride(
            torch.tensor([[10, 11, 12, 13, 14]], device="cuda"),
            torch.tensor([hit], device="cuda"),
        )
        extension = SimpleNamespace(
            config=SimpleNamespace(draft_graph_mode="split"),
            pending_view=view,
            dspark_draft_counters=dict.fromkeys(
                (
                    "proposal_ct",
                    "request_ct",
                    "neural_request_ct",
                    "skip_request_ct",
                    "all_sa_ct",
                    "mixed_ct",
                ),
                0,
            ),
            dspark_split_buffers=dspark._SplitProposalBuffers(1, 5, "cuda", 0),
        )
        proposer = SimpleNamespace(_hybrid_spec_extension=extension, _mask_token_id=0)
        neural = dto.DraftProposal(
            draft_block_ids=torch.tensor([[42, 0, 0, 0, 0]], device="cuda"),
            draft_block=dto.DraftBlockResult(
                torch.tensor([[1, 2, 3, 4, 5]], device="cuda"),
                None,
                torch.ones(1, dtype=torch.bool, device="cuda"),
                torch.ones(1, device="cuda"),
                True,
            ),
            folded=True,
        )
        kwargs = dict(
            bs=1,
            draft_input=SimpleNamespace(bonus_tokens=torch.tensor([42], device="cuda")),
        )

        def call():
            return module.propose(lambda *a, **kw: neural, proposer, **kwargs)

        expected = [[10, 11, 12, 13, 14]] if hit else [[1, 2, 3, 4, 5]]
        result = call()
        assert result.draft_block.draft_tokens.tolist() == expected
        assert result.draft_block_ids[:, 0].tolist() == [42]
        assert result.folded == (not hit)
        return call

    output = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "scope": "B=1, K=5; cached synthetic neural result; dispatch only",
        "repeats": 5,
        "iterations_per_repeat": 200,
        "cases": {},
    }
    for case, hit in (("all_sa", True), ("all_neural", False)):
        calls = {"before": runner(baseline, hit), "after": runner(dspark, hit)}
        samples = {name: [] for name in calls}
        for call in calls.values():
            for _ in range(20):
                call()
        # Alternate order to reduce systematic first/last-run bias.
        for repeat in range(output["repeats"]):
            order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
            for name in order:
                torch.cuda.synchronize()
                begin = time.perf_counter_ns()
                for _ in range(output["iterations_per_repeat"]):
                    calls[name]()
                torch.cuda.synchronize()
                samples[name].append(
                    (time.perf_counter_ns() - begin)
                    / output["iterations_per_repeat"]
                    / 1000
                )
        results = {}
        for name, call in calls.items():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as trace:
                for _ in range(4):
                    call()
            events = {
                event.key: event.count
                for event in trace.key_averages()
                if event.key
                in (
                    "aten::nonzero",
                    "aten::clone",
                    "aten::full_like",
                    "aten::ones",
                )
                or (
                    event.key.startswith("cuda")
                    and ("Synchronize" in event.key or "Memcpy" in event.key)
                )
            }
            results[name] = {
                "median_us": statistics.median(samples[name]),
                "repeat_us": samples[name],
                "events_per_four_calls": events,
            }
        output["cases"][case] = results
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
