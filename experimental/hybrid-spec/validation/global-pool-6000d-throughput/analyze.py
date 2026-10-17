"""Summarize repeated official benchmark runs without extrapolating task accuracy."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    rows = json.loads((root / "results.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    groups = {}
    for row in rows:
        groups.setdefault(
            (row["scenario"], row["concurrency"], row["mode"]), []
        ).append(row)
    summary = []
    for (scenario, concurrency, mode), samples in sorted(groups.items()):
        throughputs = [r["output_throughput"] for r in samples]
        record = {
            "scenario": scenario,
            "concurrency": concurrency,
            "mode": mode,
            "rounds": len(samples),
            "tok_s_median": statistics.median(throughputs),
            "tok_s_min": min(throughputs),
            "tok_s_max": max(throughputs),
            "mean_ttft_ms_median": statistics.median(
                r["mean_ttft_ms"] for r in samples
            ),
            "mean_tpot_ms_median": statistics.median(
                r["mean_tpot_ms"] for r in samples
            ),
            "mean_e2e_ms_median": statistics.median(
                r["mean_e2e_latency_ms"] for r in samples
            ),
            "p99_ttft_ms_median": statistics.median(r["p99_ttft_ms"] for r in samples),
            "p99_tpot_ms_median": statistics.median(r["p99_tpot_ms"] for r in samples),
            "sa_fraction_median": statistics.median(r["sa_fraction"] for r in samples),
            "mean_accept_length_median": statistics.median(
                r["mean_accept_length"] for r in samples
            ),
            "skip_batches_total": sum(
                r["split_delta"].get("skip_ct", 0) for r in samples
            ),
            "skip_request_rows_total": sum(
                r["split_delta"].get("skip_request_ct", 0) for r in samples
            ),
            "mixed_batches_total": sum(
                r["split_delta"].get("mixed_ct", 0) for r in samples
            ),
            "completed": sum(r["completed"] for r in samples),
            "status": "ok" if all(r["status"] == "ok" for r in samples) else "failed",
        }
        base = groups.get((scenario, concurrency, "full-off"), [])
        baseline = {r["round"]: r for r in base}
        paired = [
            100
            * (r["output_throughput"] / baseline[r["round"]]["output_throughput"] - 1)
            for r in samples
            if r["round"] in baseline
        ]
        if paired:
            record["paired_gain_pct_median"] = statistics.median(paired)
            record["paired_gain_pct_min"] = min(paired)
            record["paired_gain_pct_max"] = max(paired)
        graph_baseline = "split-off" if mode.startswith("split-") else "full-off"
        by_round = {
            r["round"]: r
            for r in groups.get((scenario, concurrency, graph_baseline), [])
        }
        pool_effect = [
            100
            * (r["output_throughput"] / by_round[r["round"]]["output_throughput"] - 1)
            for r in samples
            if r["round"] in by_round
        ]
        if pool_effect:
            record["global_pool_gain_pct_median"] = statistics.median(pool_effect)
            record["global_pool_gain_pct_min"] = min(pool_effect)
            record["global_pool_gain_pct_max"] = max(pool_effect)
        warm = [r for r in samples if r["seed"]]
        if warm:
            record["seed_seconds_median"] = statistics.median(
                r["seed"]["wall_s"] for r in warm
            )
            record["seed_inclusive_tok_s_median"] = statistics.median(
                (r["total_output_tokens"] + r["seed"]["completion_tokens"])
                / (r["duration"] + r["seed"]["wall_s"])
                for r in warm
            )
        summary.append(record)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "summary.csv").open("w") as stream:
        columns = list(dict.fromkeys(k for record in summary for k in record))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    parity = []
    for (scenario, concurrency, mode), samples in sorted(groups.items()):
        if mode == "full-off":
            continue
        baseline = {
            r["round"]: r for r in groups.get((scenario, concurrency, "full-off"), [])
        }
        for row in samples:
            if row["round"] not in baseline:
                continue
            ref = baseline[row["round"]]
            candidate = root / Path(row["official_result"]).name
            reference = root / Path(ref["official_result"]).name
            left = json.loads(reference.read_text().splitlines()[0])["generated_texts"]
            right = json.loads(candidate.read_text().splitlines()[0])["generated_texts"]
            mismatch_indices = [
                i for i, (a, b) in enumerate(zip(left, right)) if a != b
            ]
            examples = []
            for index in mismatch_indices[:3]:
                a, b = left[index], right[index]
                prefix = next(
                    (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                    min(len(a), len(b)),
                )
                examples.append(
                    {
                        "request_index": index,
                        "first_differing_char": prefix,
                        "baseline_context": a[max(0, prefix - 60) : prefix + 140],
                        "candidate_context": b[max(0, prefix - 60) : prefix + 140],
                    }
                )
            parity.append(
                {
                    "mode": mode,
                    "scenario": scenario,
                    "concurrency": concurrency,
                    "round": row["round"],
                    "mismatch_indices": mismatch_indices,
                    "examples": examples,
                    "requests": len(left),
                    "scope": "Decoded text comparison; not an accuracy evaluation or exact token-ID assertion",
                }
            )
    (root / "output-parity.json").write_text(json.dumps(parity, indent=2) + "\n")

    lines = [
        "# RTX 6000D 跨请求 SA 吞吐对照",
        "",
        "Qwen3.6-35B-A3B-FP8，单卡 TP=1，K=8，gate=2，min_match_len=4。",
        "全局池保留 16 个完成请求，4 个活跃槽位，H=4096，外部上下文上限 128。",
        "每个测量点 80 请求，固定生成 128 token，greedy、ignore_eos=True；KV 前缀缓存统一关闭。",
        "官方 sglang.benchmark.serving 流式客户端，burst 闭环负载，并发 1/4。无指定 SLA。",
        "三轮测量取中位数，范围为三轮最小/最大值；收益为相同轮次相对 full-off 的比例。",
        "full-off 保留本请求 SA 和 MTP，仅关闭跨请求全局池。",
        "",
        "unique-cold：80 个不同的真实代码编辑输入，空池起步。",
        "repeat-cold：其中 16 个输入各重放 5 次，空池起步，建池成本包含在计时内。",
        "repeat-warm：同样的 80 请求，之前通过 16 个真实生成请求预先建池，预热不计入主表。",
        "重复场景是刻意允许内容复用的对照，不能代表普通生产流量的重复率。",
        "未将数据集参考答案放入输入或历史；原始输入索引、长度和哈希见 provenance.json。",
        "",
        "| 负载 | 并发 | 配置 | 轮数 | output tok/s 中位数 [范围] | 对 full-off 收益 [范围] | TTFT ms | TPOT ms | SA decode 行占比 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary:
        gain = (
            f"{r['paired_gain_pct_median']:+.1f}% [{r['paired_gain_pct_min']:+.1f}, {r['paired_gain_pct_max']:+.1f}]"
            if "paired_gain_pct_median" in r
            else "—"
        )
        lines.append(
            f"| {r['scenario']} | {r['concurrency']} | {r['mode']} | {r['rounds']} | "
            f"{r['tok_s_median']:.2f} [{r['tok_s_min']:.2f}, {r['tok_s_max']:.2f}] | {gain} | "
            f"{r['mean_ttft_ms_median']:.2f} | {r['mean_tpot_ms_median']:.2f} | {100 * r['sa_fraction_median']:.1f}% |"
        )
    lines += [
        "",
        "TTFT/TPOT 列为每轮均值的中位数；p99、接受长度、split 分支计数见 summary.json。",
        "SA 占比是采用检索候选的 decode 请求行比例，不是端到端请求命中率。",
        "不把吞吐变化直接解释为模型准确率变化。完整输出文本对照见 output-parity.json。",
        f"文本对照共 {sum(r['requests'] for r in parity)} 对请求，其中 {sum(len(r['mismatch_indices']) for r in parity)} 对文本不同。",
        "",
        "repeat-warm 的 seed_inclusive_tok_s_median 将 16 个建池请求和 80 个测量请求一起计入。",
        "模型启动、Graph capture、JIT 和通用预热不计入吞吐；每轮负载前 flush，暖池场景随后再建池。",
        "原三组顺序执行，随后补 split-off 控制；每组内部随机排列场景与并发顺序，温度、功耗、时钟记录在每点 diagnostics。",
        "summary.json 的 global_pool_gain_pct_* 比较相同 Graph 模式下开/关全局池，避免把 split 自身的收益归于共享历史。",
        "这是固定配置与固定长度的吞吐对照，没有搜索最佳参数，也没有测生产请求分布或任务准确率。",
        "",
        "完整启动命令及配置见 plan.json、*-command.json、*-config.json。",
        "官方原始结果为 *-r*.jsonl；每请求延迟、文本、错误和缓存统计均保留。",
        "",
    ]
    if plan.get("interruption"):
        lines += [
            "补充 split-off 控制曾因容器进程重置中断；中断前的 11 个点另行保留并排除，完整三轮重新测量。",
            "原三组 54 个点在中断前已完成；GPU UUID、驱动、源码和参数均核对一致，中断原因未确认。",
            "",
        ]
    (root / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
