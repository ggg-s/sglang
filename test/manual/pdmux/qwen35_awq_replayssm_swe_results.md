# Qwen3.5 AWQ: ReplaySSM capacity and PDMux TTFT follow-up

Measured on 2026-10-24 on the same A100 PCIe 40 GB, AWQ target, unquantized
DFlash draft and frozen SWE prefixes as the [previous results](qwen35_awq_swe_results.md).
Speculative ReplaySSM now runs with DFlash's GDN direct commit, CPU overlap,
PDMux split prefill and HiCache. It remains opt-in.

## Capacity

With four running requests, 24 Mamba slots and block size eight, the allocated
per-draft SSM snapshots fell from **2.34 GiB to zero**. The compact 32-token rings
and their low parts cost approximately **0.11 GiB**, saving **2.23 GiB net**.
Persistent FP32 SSM state remains 1.46 GiB; ReplaySSM does not remove that pool.
Both target and draft successfully allocated **65536 KV tokens**, up from 32768.
Both prefill token limits remain 32768 and the request context limit remains 16384.

The speculative flag is `--enable-linear-replayssm-spec`. The ordinary decode
flag `--enable-linear-replayssm` requires `no_buffer` and is a different control.
This lane retains `extra_buffer`, including tracked prefix checkpoints for HiCache.

HiCache size 16 reserves 16 GiB across the target KV and Mamba host pools;
the draft adds approximately 8.23 GiB. Target/draft host KV capacities were
335069/335070 tokens. The machine had approximately 723 GiB of available RAM
before the probes. This run did not encounter host-memory exhaustion.

## Matched 64K comparison

Concurrency four, two trials per phase, 18 requests and 128 output tokens per
request per trial. Values are medians of trial metrics; quantiles are not pooled.
All three groups use a 65536-token KV limit, four running requests, 24 Mamba
slots, memory fraction 0.9, and CPU overlap. Both DFlash groups enable ReplaySSM
with ring length 32 and HiCache size 16. PDMux keeps P88 and full-device decode,
with a 32768 token-layer split budget and a 16-layer maximum.

| Group | Cold output tok/s | Repeated output tok/s | Cold TTFT p50 (ms) | Repeated TTFT p50 (ms) | Cold TPOT mean (ms) | Repeated TPOT mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AR baseline | 257.0 | 301.2 | 257.6 | 210.9 | 8.100 | 7.857 |
| DFlash + HiCache + ReplaySSM | 289.1 | 375.7 | 155.0 | 127.9 | 7.517 | 7.367 |
| DFlash + HiCache + ReplaySSM + PDMux P88 | 286.7 | 379.9 | 236.4 | 138.8 | 7.636 | 7.036 |

PDMux retains a 4.5% repeated TPOT advantage over ordinary DFlash. Its repeated
throughput lead is only 1.1%, with overlapping trial ranges; two short trials
do not establish a robust throughput win. Ordinary DFlash still has lower TTFT
and slightly higher cold throughput. Against AR, PDMux improves repeated
throughput by 26.1% and lowers repeated TPOT by 10.4%.

Compared with the previous P88/4-GiB/32K result, the combined candidate lowers
cold TTFT from 468.2 to 236.4 ms and repeated TPOT from 7.225 to 7.036 ms.
ReplaySSM and KV expansion were validated together; the end-to-end difference
does not isolate their individual performance contributions.

All 216 requests in this matched comparison completed with 128 tokens and zero
retractions. PDMux's repeated draft acceptance was 24.13%, with 2.689 committed
tokens per verify and 40.02% zero-draft verifications. Ordinary DFlash's values
were 24.68%, 2.727 and 39.32%, respectively. Acceptance remains a limiting factor.
PDMux's repeated phase recorded 46080 host-hit tokens across both trials.

## TTFT diagnosis

The bounded probes kept the P88 SM cap unchanged:

| Configuration | Trials | Cold TTFT p50 (ms) | Repeated TTFT p50 (ms) | Repeated TPOT mean (ms) |
| --- | ---: | ---: | ---: | ---: |
| Previous HiCache 4, 4096/8 split, 32K KV | 2 | 468.2 | 155.3 | 7.225 |
| HiCache 16, 4096/8 split, 32K KV | 1 | 549.6 | 165.1 | 7.197 |
| HiCache 16, 32768/16 split, 32K KV | 2 | 209.7 | 143.2 | 7.529 |
| HiCache 16, 32768/16 split, ReplaySSM, 64K KV | 2 | 236.4 | 138.8 | 7.036 |

Increasing host cache alone did not improve this replay. Increasing the split
budget substantially reduced TTFT while holding the SM cap constant, but cost
TPOT without ReplaySSM. The combined candidate restores the TPOT advantage.
This supports overly fine splitting as a contributor; it does not isolate the
remaining effect of the P88 cap itself. The prefill token budgets stayed 32768
throughout; the split budget controls layers between decode submissions.

## State correctness and reproduction

DFlash previously bypassed the GDN circular commit helper, and its request-pool
builder explicitly rejected GDN ReplaySSM. The direct path now commits accepted
records once per request, materializes tracked prefix states, and rolls back
convolution windows using physical destinations. The generic speculative path
shares that helper. Replay verify also rounds sigmoid beta through the activation
dtype, matching recurrent verify and prefill.

One focused A100 GPU regression passed, covering distinct request/state slots,
translated track slots, ring wrap, capacity folding, deferred active state and
convolution rollback against an independent state oracle. Targeted lint and
format checks passed. No profiler or broad kernel/numerical suite was rerun.
The earlier complete generated-token agreement with independent AR remains
unresolved. Successful requests and the state test do not establish that parity
or a SWE task-resolution score.

Use the ReplaySSM command in the [runbook](qwen35_dflash_overlap_runbook.md) and
the [companion configuration](qwen35_dflash_overlap_replayssm_a100.yaml).
For the matched controls, use the same command with `PDMUX=0` for ordinary
DFlash, or `DFLASH=0 HICACHE=0 PDMUX=0 REPLAYSSM_SPEC=0` for AR.
The workload SHA256 remains
`687c751bd2ad026771bc31ee737e19a02fc17605bf9d5c764a58eda1a535b123`.
Per-trial metrics, pooled acceptance counters, settings and raw artifact hashes
are retained in [the result JSON](qwen35_awq_replayssm_swe_results.json).
