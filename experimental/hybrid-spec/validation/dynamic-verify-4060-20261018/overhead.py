"""TP1 chooser + feedback overhead, synthetic scores/costs, no target model."""

import json
import statistics
import time
from types import SimpleNamespace as NS

import torch
from hybrid_spec.config import HybridConfig
from hybrid_spec.integrations.sglang.compact_verify import token_buckets
from hybrid_spec.selection.online_verify import DynamicVerify

config = HybridConfig(
    8,
    2,
    16,
    128,
    history_mode="device",
    draft_graph_mode="split",
    num_neural_drafts=3,
    verify_mode="compact",
    dynamic_verify=True,
    dynamic_verify_explore_interval=1000000,
)
grid = token_buckets(config, [9, 18, 36, 72, 144], 16)
results = []
for bs in (1, 4, 16):
    chooser = DynamicVerify(config, "cuda")
    chooser.model.check_ct.fill_(10000)
    chooser.model.correct_ct.fill_(9700)
    chooser.model.correct_ct[0].fill_(6000)
    view = NS(
        tokens=torch.zeros((bs, 8), dtype=torch.long, device="cuda"),
        use_retrieval=torch.arange(bs, device="cuda") % 2 == 0,
        query_metadata=torch.tensor([[8, i % 2] for i in range(bs)], device="cuda"),
    )
    caps = torch.where(view.use_retrieval, 8, 3)
    key = (bs, 512)
    choices = [n for n in grid if bs * 2 <= n <= bs * 9]
    for n in choices:
        chooser.observe_cost(key, n, 1.0, 0.1 + 0.02 * n)

    def run():
        lengths = chooser.select(view, caps, choices, key)
        total = int(lengths.sum().item())
        chooser.feedback(torch.minimum(lengths, torch.where(view.use_retrieval, 9, 2)))
        return total

    for _ in range(20):
        run()
    samples = []
    for repeat in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            total = run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 100)
    results.append(
        dict(
            batch_size=bs,
            median_us=statistics.median(samples),
            samples_us=samples,
            last_logical_tokens=total,
        )
    )
print(
    json.dumps(
        dict(scope=__doc__, iterations_per_sample=100, results=results), indent=2
    )
)
