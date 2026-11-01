# Remote graph bucket padding

The update based on `623ad55ad1` belongs to the existing
`codex/remote-attn-pre-layerwise` branch. It changes plugin boundaries and probes;
the native graph runner and graph backends are unchanged.

## Selection and execution

Remote eligibility selects the smallest **actually captured** Remote bucket at
least as large as the live decode batch. Default policy is 1/2/4/8/16; native
deployments capturing only 1/2/4 retain that range. Batch 0, batches above the
Remote range, non-decode, embedding overrides and non-single-token rows remain
ineligible. Disabling CUDA graph padding retains exact-size eligibility.

The execution context temporarily supplies Remote buckets to native `load_batch`,
then restores the full bank's original bucket list on success or failure. This
prevents a native intermediate bucket such as 14 from diverging from the Remote
selection of 16. Replay verifies the selected key before launching segments.
The backend prohibits concurrent/nested execution of that runner.

After staging a padded batch, replay rebuilds eager attention metadata from the
original live batch. Qwen attention consumes only real Q/K/V rows, including
wholly Local layers while a progressive lease is still active. DSV metadata uses
the live batch; SWA/carry/indexer state updates consume sliced projection inputs,
and attention preserves shared K/V identity. Attention outputs are zero-padded
back to the captured bridge shape. Native output processing trims logits and
hidden states to real rows. Local full graphs retain native bucket selection.
DSV candidate attention variants continue sharing one Remote bank per bucket.

## Execution counters

Counters live at `model_runner._remote_attn_graph_stats`, independently per rank:

- `rank`: actual TP rank from the runtime parallel context.
- `steps`, `paths`: successful executions of `full`, `remote-segmented`, `eager`.
- `batches`: counts by `path:raw_bs:bucket`; eager has bucket `None`.
- `fallbacks`: reasons for successful eager executions, including unavailable
  Remote buckets, unsupported rows/embeddings, native eligibility rejection,
  compatibility windows and disabled Remote segments.
- `failures`: failed executions by path, excluded from successful step ratios.

`SGLANG_LOG_DECODE_GRAPH_KEY=1` emits one `[remote-attn-graph-step]` line per
execution with rank, path, raw size, bucket, reason and failure status. Successful
counts are emitted every 256 steps as `[remote-attn-graph-stats]` and at Remote
backend cleanup. Eligibility probes never increment these counters. The older
`path=... all_local=...` message still records verdict transitions only, and its
line counts must not be used to calculate execution percentages.

For a benchmark pass, use counter differences at its boundaries, exclude failed
steps, and use one TP rank to avoid counting the same step once per rank. Disable
per-step logging for timing measurements.

## Validation

Local validation passed 209 CPU contracts, including real two-process Gloo.
Source preflight passed 48 hooks and 115 core interfaces. The local harness uses
real repository methods and PyTorch CPU, with the heavyweight test utility
wrapper replaced by `unittest.TestCase`; model/CUDA boundaries remain fixtures.
The repository-selected Ruff rules and formatting checks passed.

CPU contracts cover padded and exact replay, interleaved batch sizes, restoration
after failure, no RPC/KV/carry work for dummy rows, native `load_batch` staging,
native logits/hidden-state trimming, sampler wrapper isolation, candidate variants,
Local retirement and eager fallback. The two-process Gloo test checks identical
bucket mappings and routing across TP ranks without adding per-step collectives.

```bash
python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
python test/manual/remote_attn/segmented_graph_probe.py --dtype bfloat16 --logits-output
python test/manual/remote_attn/segmented_graph_probe.py --dtype float16 --logits-output
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/dsv41_segmented_graph_probe.py --source-root . --output /tmp/remote-padding-dsv.json
```

Both CUDA probes now include 3/5/6/7/13/14/15 and repeated size changes. They must
be rerun on CUDA before claiming graph-capture validation for this update. The
current local host has no CUDA; configured remote GPU access was unavailable
(authentication failure and host-key verification failure). No new CUDA numerical
or c=16 performance result is claimed here. The c=16 workload must still compare
actual Remote graph coverage, output correctness, TPOT, E2EL and throughput after
this change; a return to 7–8 ms TPOT is a hypothesis, not an acceptance result.
