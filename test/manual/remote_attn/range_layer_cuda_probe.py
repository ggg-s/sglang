"""Real Qwen layer-group x token-range transport; no model-weight claim."""

import argparse
import json
from pathlib import Path

import torch


def main():
    from cuda_handoff_probe import LAYERS, TOKENS, transfer_probe
    from sglang_remote_attn.qwen_ranges import batches

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for mode, dtype, group, slots, width in (
        ("fast-async", "float16", 1, 2, 1024),
        ("fast-async", "float16", 3, 2, 257),
        ("fast-async", "bfloat16", 4, 1, 513),
        ("fast-sync", "float16", 2, 3, 511),
        ("cpu-layer", "bfloat16", 3, 0, 511),
    ):
        folder = args.output.parent / f"qwen-range-{len(results)}"
        folder.mkdir(exist_ok=True)
        result = transfer_probe(
            mode, dtype, slots, 4, folder, group_size=group, token_range=width
        )
        count = min(group, max(1, slots or group))
        expected = [
            len(g)
            for g, lo, hi in batches(
                dict(
                    chunks=[[]] * LAYERS,
                    kv_committed_len=TOKENS,
                    d2d_group_size=group,
                    d2d_token_range=width,
                ),
                count,
            )
        ]
        actual = result["source"]["native_batches"]
        assert [b["descriptors"] for b in actual] == expected, actual
        assert len(actual) > LAYERS
        assert result["source"]["registration"]["live"] == 0
        result.update(
            group=group,
            token_range=width,
            native_batches_count=len(actual),
            bounded_host_slots=True,
        )
        results.append(result)
    for cancel, remote_drop in ((False, True), (True, False)):
        folder = args.output.parent / (
            "qwen-range-cancel" if cancel else "qwen-range-drop"
        )
        folder.mkdir(exist_ok=True)
        results.append(
            transfer_probe(
                "fast-async",
                "float16",
                2,
                4,
                folder,
                cancel=cancel,
                remote_drop=remote_drop,
                group_size=3,
                token_range=511,
            )
        )
    folder = args.output.parent / "qwen-range-status-loss"
    folder.mkdir(exist_ok=True)
    results.append(
        transfer_probe(
            "fast-async",
            "float16",
            2,
            4,
            folder,
            group_size=3,
            token_range=511,
            fault="lost-status",
        )
    )
    report = dict(
        status="PASS",
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cases=results,
        scope=__doc__,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", cases=len(results))))


if __name__ == "__main__":
    main()
