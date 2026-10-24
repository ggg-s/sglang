# Qwen3.5 DFlash with CPU overlap, PDMux split prefill and HiCache

This lane supports text-only Qwen3.5 MoE with DFlash, TP=1 and PP=1.
CPU overlap is enabled by default; omit `--disable-overlap-schedule`.
Other speculative algorithms and PDMux `standard` prefill still require the
synchronous scheduler. Grammar and penalties consume the pending CPU result
before preparing their next decode.

On one A100 40 GB, use an FP16 GPTQ-Marlin target and the unquantized draft:

```bash
export TARGET_MODEL=/path/to/Qwen3.5-35B-A3B-GPTQ-Int4
export DRAFT_MODEL=/path/to/Qwen3.5-35B-A3B-DFlash
bash test/manual/pdmux/launch_qwen35_dflash_overlap.sh
```

The launcher sets both `--chunked-prefill-size` and `--max-prefill-tokens`
to 32768. These are batch token budgets; the per-request context limit remains
16384 to reproduce the SWE replay. The companion YAML caps prefill at 32 SMs,
allows decode to use all 108 SMs, and computes the split interval from a
4096 token-layer budget with an eight-layer upper bound. Without active decode,
prefill executes all remaining layers in one call.

The model's draft block defaults to 16. Set `BLOCK_SIZE=4` to reproduce the
earlier four-token draft experiment. Keep this parameter identical across A/B
runs. Append `--disable-overlap-schedule` to run the synchronous CPU control.
Target and draft decode use full CUDA graphs; prefill graphs are disabled.

Replay the frozen six SWE-rebench OpenHands trajectories as three sequential
assistant prefixes per trajectory, preserving the complete recorded histories.
Use outer concurrency 1 and 4, 128 generated tokens per request, greedy
sampling, and `ignore_eos=true`. Flush before each cold/incremental trial, then
repeat the same histories without flushing. Check host cache hits separately
from device prefix hits. Tokenization and tool execution are outside this
inference-only replay; it does not measure SWE task resolution.

Compare CPU overlap on/off under the same 32K prefill limits, then compare with
ordinary DFlash plus HiCache. Record generated token IDs, completion reasons,
retractions, TTFT, output tokens/s, normalized inter-token latency and stream
push gaps. Speculative responses can contain several tokens per push, so
normalized inter-token latency differs from the raw push gap.

Runtime completion and an accepted-token check against the actual target verify
logits do not establish numerical equivalence to an independent AR run. This
FP16/GPTQ setup already showed repeat-run and cross-path token differences;
retain those mismatches and the AR self-repeat control when evaluating changes.
