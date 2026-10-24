# Qwen3.5 AWQ: split-PDMux TTFT wait attribution

Measured on 2026-10-24 after runtime commit `4bb35cecb3`. **No runtime candidate
was selected.** Keep the existing P88, ReplaySSM, HiCache and split configuration.
The clean control prevented an apparent cold-TTFT improvement from being
mistaken for a code improvement.

## Strongest signal

The second group submits a complete prefill. The split-PDMux group returns to
decode preparation between prefill segments. That preparation calls
`FutureMap.resolve_seq_lens_cpu()`, which waits on a private D2H stream for the
previous verify's sequence-length mirror. The scheduler cannot submit the next
prefill segment while blocked there.

A bounded CPU profile of **30 actual decode submissions** recorded **494 ms**
inside `Stream.synchronize`, reached through `resolve_seq_lens_cpu`, out of
729 ms sampled elapsed time: about **16.5 ms per sampled call**. Both target and
draft CUDA graphs replayed in that window. This is a host blocking location,
not a measurement of pure CPU compute or GPU kernel time. The wait can include
GPU production of the lengths and transfer-stream/copy-engine dependencies;
those components were not separated with a GPU timeline.

FlashInfer planning still consumes the CPU lengths. Removing this synchronization
without replacing that planning contract would risk stale lengths and incorrect
attention. The evidence identifies a significant wait on the split scheduling
path; it does not assign the entire TTFT difference to that one wait.

Existing records from the earlier matched 64K comparison also show:

| Group | Cold queue p50 (ms) | Cold prefill span p50 (ms) | Repeated queue p50 (ms) | Repeated prefill span p50 (ms) |
| --- | ---: | ---: | ---: | ---: |
| DFlash + HiCache + ReplaySSM | 0.71 | 116.8 | 0.92 | 108.2 |
| Same + split PDMux | 1.08 | 179.4 | 1.05 | 119.2 |

The prefill span runs from server admission into forward work to prefill result
processing. It includes scheduling, GPU work and result handling. These medians
are not additive components of the client TTFT median. Queue tails still matter:
the earlier cold maximum queue delay was 168.1 ms with PDMux versus 30.0 ms
without it.

## SM and stream-group controls

Lightweight CPU spans introduced no CUDA synchronization. Each diagnostic used
one cold/incremental and one repeated replay, with the same frozen SWE prefixes.

The original P88 repeated phase recorded six same-group drains totaling about
**0.1 ms**, and 18 actual group-change drains totaling about **0.3 ms**. Skipping
same-group drains was correct in the targeted ordering tests but did not remove
the main latency source. A separate cold transition waited approximately 204 ms;
that tail is retained in the artifacts, rather than generalized to every switch.

Holding the same-group-drain change and all model/cache/split settings fixed:

| Diagnostic | Repeated TTFT p50 (ms) | Repeated TPOT mean (ms) | Repeated output tok/s |
| --- | ---: | ---: | ---: |
| P88 | 154.1 | 6.701 | 392.5 |
| P104 | 152.3 | 6.696 | 392.5 |

The extra prefill SMs did not materially improve the repeated phase. Cold p50
varied in these single diagnostic trials, so this does not rule out an SM-cap
effect on cold prefills. Decode continued to cover all 108 SMs in both cases;
the prefill and decode SM sets overlapped.

## Clean performance controls and rejected candidates

The following runs have **no CPU timing probe or profiler**. Concurrency four,
two trials per phase, 18 requests per trial, 128 output tokens per request.
Values are medians of trial metrics; quantiles are not pooled across trials.

All use the same target/draft revisions, 24 Mamba slots, memory fraction 0.9,
65536-token KV limit, HiCache size 16, ReplaySSM ring 32, draft block eight,
P88/full-device decode, token-layer budget 32768 and max split layers 16.
Both prefill token budgets remain 32768 and request context remains 16384.
The short-prefill candidate bypasses the configured layer cap only below its
128-token threshold; the early-submit candidate preserves the per-round cap.

| Configuration | Cold TTFT p50 (ms) | Repeated TTFT p50 (ms) | Repeated TPOT mean (ms) | Repeated output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Unchanged runtime control | 200.0 | 143.4 | **6.871** | **390.4** |
| Complete prefill in one call when uncached tokens <=128 | 229.5 | **127.3** | 7.242 | 382.6 |
| Submit up to two prefill layers before a pending CPU-length wait | 205.8 | 144.2 | 6.944 | 385.8 |

The short-prefill exception reduced repeated TTFT, but increased TPOT by 5.4%
against the clean control and exceeded the prior 7.04 ms reference. It was rejected.
The two-layer candidate preserved the original total layer budget per scheduling
round and retained the length synchronization. It did not improve TTFT against
the clean control and slightly reduced throughput. It was also rejected.

The unchanged runtime's earlier cold TTFT was 236.4 ms; it measured 200.0 ms in
this clean rerun. Thus comparing the two-layer candidate's 205.8 ms only against
the earlier result would have incorrectly attributed baseline variation to the
patch. The previous 7.036 ms repeated TPOT reference and this rerun's 6.871 ms
belong to the same runtime/configuration. The source of the time-window variation
was not isolated.

The second group's earlier repeated TTFT reference remains 127.9 ms; it was
reused, not remeasured contemporaneously with these final controls. No claim is
made that the third group now beats it on TTFT.

## Validation and retained baseline

Focused CPU checks covered unchanged-group versus actual-group synchronization,
HiCache pumping, pending-result ordering, early-submit layer-budget accounting
and the original uneven-DP loop. The candidate paths completed the frozen SWE
replays; generated IDs, completion/retraction counters, launch settings and
source snapshots were retained. No broad kernel suite, deterministic numerical
matrix or GPU profiler trace was run. The earlier AR token-equivalence gap is
unchanged.

All experimental runtime and test edits were reverted. Runtime behavior and the
[selected configuration](qwen35_dflash_overlap_replayssm_a100.yaml) remain at
`4bb35cecb3`. Further optimization should target the CPU-length planning handoff
and its GPU/copy dependencies while preserving FlashInfer's length contract,
rather than assuming more prefill SMs or fewer segment boundaries are a free win.

The workload SHA256 remains
`687c751bd2ad026771bc31ee737e19a02fc17605bf9d5c764a58eda1a535b123`.
Per-trial metrics, host timing summaries, acceptance/completion counters and
artifact hashes are in [the result JSON](qwen35_awq_scheduler_ttft_results.json).
