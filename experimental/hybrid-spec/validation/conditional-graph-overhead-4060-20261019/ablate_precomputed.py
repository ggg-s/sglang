"""Fixed-predicate controls: isolate comparison kernels from handle/IF costs."""

import gc
import json
import sys
from pathlib import Path

import torch
from ablate_dispatch import Case

if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])
    from dispatch_probe import make_record

    directory = Path(sys.argv[2])
    directory.mkdir(exist_ok=True)
    gc.collect()
    gc.disable()
    case = Case(32, 16, make_record)
    count = case.record.num_neural
    root_predicate = count > 0
    low = {middle: count <= middle for middle in range(1, 32)}
    high = {middle: count > middle for middle in range(1, 32)}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    for kind in ("single_if_precomputed", "tree_precomputed"):
        graph = torch.cuda.CUDAGraph()

        def branch(predicate, body, graph=graph):
            graph.begin_capture_to_if_node(predicate)
            try:
                body()
            finally:
                graph.end_capture_to_conditional_node()

        def tree(first, last):
            if first == last:
                case.record.tails[first].forward()
            else:
                middle = (first + last) // 2
                branch(low[middle], lambda: tree(first, middle))
                branch(high[middle], lambda: tree(middle + 1, last))

        with torch.cuda.graph(
            graph, pool=torch.cuda.graph_pool_handle(), stream=stream
        ):
            if kind == "single_if_precomputed":
                branch(root_predicate, case.record.tails[16].forward)
            else:
                branch(root_predicate, lambda: tree(1, 32))
        case.graphs[kind] = graph
        case.dispatch[kind] = graph.replay
        case.modes.append(kind)
        case.run(kind)
        assert case.record.output.cpu().tolist() == [
            row * 3 + 7 if row < 16 else -1 for row in range(32)
        ]
    gc.enable()
    result = {
        "batch_size": 32,
        "selected": 16,
        "timings": case.measure(),
        "profiles": {
            mode: case.profile(mode, directory)
            for mode in (
                "direct",
                "single_if",
                "single_if_precomputed",
                "tree",
                "tree_precomputed",
                "production",
            )
        },
    }
    (directory / "precomputed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            mode: round(t["dispatch_gpu_us_median"], 3)
            for mode, t in result["timings"].items()
            if t["dispatch_gpu_us_median"] is not None
        }
    )
    case.close()
