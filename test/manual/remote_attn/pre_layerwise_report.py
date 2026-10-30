"""Summarize one same-host P5.19 journal; not an end-to-end benchmark.

Native bench_serving supplies TTFT/ITL/E2E. These monotonic timestamps explain
activation, transfer/compute overlap, early admission and Remote duration.
"""

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def summarize(root):
    rows = []
    offers = list(root.glob("*/*/offer.json"))
    for prefill_path in sorted(root.glob("*/*/*/prefill.json")):
        prefill = read(prefill_path)
        path = prefill_path.parent
        generation = prefill["generation"]

        def matching(name):
            value = read(path / name)
            return value if value and value.get("generation") == generation else None

        admitted = matching("admitted.json")
        done = matching("done.json")
        ready = matching("ready.json")
        released = matching("released.json")
        layers = [matching(p.name) for p in path.glob("layer-*.json")]
        times = [p["at"] for p in layers if p]
        first_layer = min(times) if times else None
        all_layers = max(times) if times else None
        rows.append(
            {
                "room": path.parent.parent.name,
                "rank": path.parent.name,
                "generation": generation,
                "rid": prefill["rid"],
                "prefix_len": prefill["prefix_len"],
                "final_chunk_tokens": prefill["final_chunk_end"]
                - prefill["final_chunk_start"],
                "prefill_chunk_tokens": prefill.get("prefill_chunk_tokens"),
                "layer_group_size": prefill.get("layer_group_size"),
                "host_staging": prefill.get("host_staging", False),
                "recv_host_staging": prefill.get("recv_host_staging", False),
                "native_staging": prefill.get("native_staging", False),
                "completed": done is not None,
                "released": released is not None,
                "p_radix_read_lease": prefill.get("p_radix_read_lease", False),
                "cached_prefix_tokens": prefill.get("cached_prefix_tokens", 0),
                "p_radix_lease_closed": released.get("p_radix_lease_closed")
                if released
                else None,
                "deferred_cache_checkpoints": released.get("deferred_cache_checkpoints")
                if released
                else None,
                "error": matching("error.json"),
                "early_admission": bool(admitted),
                "remote_calls": done.get("remote_calls", 0) if done else None,
                "scheduler_overlap": done.get("scheduler_overlap", False)
                if done
                else False,
                "overlap_forwards": done.get("overlap_forwards") if done else None,
                "gpu_proven_forwards": done.get("gpu_proven_forwards")
                if done
                else None,
                "ready_layers_at_admission": admitted["ready_layers"]
                if admitted
                else None,
                "layers_completed_before_prefill_finalize": sum(
                    t < prefill["at"] for t in times
                ),
                "first_layer_lead_ms": max(0, (prefill["at"] - first_layer) * 1000)
                if first_layer
                else None,
                "metadata_publish_delay_ms": (ready["at"] - prefill["at"]) * 1000
                if ready
                else None,
                "prefill_to_admit_ms": (admitted["at"] - prefill["at"]) * 1000
                if admitted
                else None,
                "prefill_to_first_step_ms": (done["first_step_at"] - prefill["at"])
                * 1000
                if done and done.get("first_step_at") is not None
                else None,
                "remaining_kv_at_finalize_ms": max(
                    0, (all_layers - prefill["at"]) * 1000
                )
                if all_layers
                else None,
                "observed_remote_window_ms": max(
                    0, (done["import_done_at"] - admitted["at"]) * 1000
                )
                if done and admitted and done.get("import_done_at") is not None
                else None,
            }
        )
    # Collapse TP shards by taking the slowest rank for each room's latency.
    metrics = {}
    for key in (
        "first_layer_lead_ms",
        "metadata_publish_delay_ms",
        "prefill_to_admit_ms",
        "prefill_to_first_step_ms",
        "remaining_kv_at_finalize_ms",
        "observed_remote_window_ms",
    ):
        rooms = {}
        for row in rows:
            if row[key] is not None:
                rooms[row["room"]] = max(rooms.get(row["room"], 0), row[key])
        values = sorted(rooms.values())
        metrics[key] = {
            "n": len(values),
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "offered_shards": len(offers),
        "activated_shards": len(rows),
        "early_admitted_shards": sum(r["early_admission"] for r in rows),
        "completed_shards": sum(r["completed"] for r in rows),
        "released_shards": sum(r["released"] for r in rows),
        "p_radix_leased_shards": sum(r["p_radix_read_lease"] for r in rows),
        "p_radix_closed_shards": sum(r["p_radix_lease_closed"] is True for r in rows),
        "p_cached_prefix_shards": sum(r["cached_prefix_tokens"] > 0 for r in rows),
        "metrics": metrics,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--require-remote", action="store_true")
    parser.add_argument("--require-overlap", action="store_true")
    parser.add_argument("--require-p-radix", action="store_true")
    parser.add_argument("--require-d-overlap", action="store_true")
    args = parser.parse_args()
    report = summarize(args.journal)
    print(json.dumps(report, indent=2))
    if not report["activated_shards"]:
        raise SystemExit(
            "No activation: inspect P/D mode, offers and startup constraints"
        )
    if report["released_shards"] != report["activated_shards"] or any(
        r["error"] for r in report["rows"]
    ):
        raise SystemExit(
            "Incomplete or failed run: do not treat as a performance result"
        )
    if args.require_p_radix and (
        report["p_radix_leased_shards"] != report["activated_shards"]
        or report["p_radix_closed_shards"] != report["activated_shards"]
    ):
        raise SystemExit("P radix lease missing or not closed on every activated shard")
    if args.require_d_overlap:
        early = [r for r in report["rows"] if r["early_admission"]]
        if (
            not early
            or any(
                not r["scheduler_overlap"]
                or r["overlap_forwards"] is None
                or r["overlap_forwards"] != r["gpu_proven_forwards"]
                for r in early
            )
            or not any((r["overlap_forwards"] or 0) > 0 for r in early)
        ):
            raise SystemExit(
                "D overlap missing, no decode forward exercised, or GPU completion unproven"
            )
    if args.require_remote and not any(
        (r["remote_calls"] or 0) > 0 for r in report["rows"]
    ):
        raise SystemExit(
            "No Remote calls: early admission did not exercise Remote Attention"
        )
    if args.require_overlap and not any(
        r["layers_completed_before_prefill_finalize"] for r in report["rows"]
    ):
        raise SystemExit(
            "No observed layer completion before Prefill finalized; inspect a GPU trace"
        )


if __name__ == "__main__":
    main()
