# Progressive P radix cache — 2026-10-30

Scope: plugin-only P device radix read leases. SMG and SGL core have no new
changes. D cache and both roles' scheduler overlap restrictions remain intact.
Native optimistic prefill, chunk sizing, early cached-prefix send and transport
batch sizing are retained. Startup commands: [single pair](../SINGLE-PAIR-ZH.md).

## Ownership

Before reserving the final chunk, the plugin runs the native cached-prefix early
send. An accepted source takes an independent native tree lock on its cached
prefix. Its suffix remains request-owned: instance-local checkpoint/finish hooks
prevent checkpoint deduplication or row remapping while transfers/readers live.
Ordinary requests use the native cache methods immediately.

Existing source retirement already requires native transfer success, a generation
matched D completion, RPC registry tombstoning with all GPU reader references
drained, and the TP retirement vote. Only then does the plugin run native
checkpoint/release, finish the attempt and unlock the independent read pin.
Layerwise sources with a cache also defer release until native completion.
Cancellation suppresses final insertion; a changed attempt, reset during a lease,
unknown release completion, transport failure or timeout fails closed and retains
ownership rather than recycling possibly live pages. Failure quarantine still
requires operator recovery; it is not an automatic failed-transport drain proof.

Supported owners: built-in RadixCache and device-only FULL UnifiedRadixCache,
without HiCache, external linker or session refs. Session requests fall back to
native PD. The plugin does not publish a newly computed final tail into the radix
tree until retirement; simultaneous identical tails may recompute and temporary
KV occupancy can rise.

## Verification

- 77 CPU contracts passed (9.320 s), including real two-process TP2/Gloo on
  loopback. The local lightweight harness source-loads native classes to avoid
  CUDA/Triton imports; it does not start an SGL server.
- Real native RadixCache tree/checkpoint/eviction, BasePrefixCache locks,
  request-row allocator, token allocator and release logic: a shared prefix stays
  valid through a tree split and eviction pressure, and private source indices
  remain live while a competing request caches the same suffix. Native transfer
  and RPC readiness are delayed separately. After retirement, deduplication,
  eviction and request release recover all 32 token slots with no duplicates.
- Abort, late cancellation after an earlier success-release request, stale attempt,
  reset and release-error cases preserve ownership; release errors reject retries.
- The actual release hook and Source finalization/retirement generate lease journal
  fields that the report reads. Native cached-prefix send precedes final source
  reservation and does not send the prefix twice.
- TP2/Gloo: source pins on both ranks remain `[2, 1, 0]` as two sources drain at
  different speeds. A cache-policy mismatch rejects startup on both ranks. Local
  graph steady-state still adds zero per-forward CPU reductions.
- Regression counterexample: bypassing only checkpoint deferral makes the
  concurrent-suffix test fail: the competing request reuses source suffix indices
  `[3, 4]`, demonstrating premature publication before the source lifetime ends.
- Source preflight: 24 hooks / 61 core API and state contracts. Missing cache
  checkpoint/finish parameters fail preflight. Ruff, isort and diff whitespace
  checks passed.

Reproduction in a complete SGL development environment:

```bash
PYTHONPATH=python:plugins/remote_attn python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

This is not a GPU PD validation or production performance result. The native
RadixCache lifecycle executes on CPU; UnifiedRadixCache is covered by public API,
state and backend-policy checks, not a full model run. Before production, use the
native SMG/P/D commands, warm a repeated long prefix, require `--require-p-radix`
and `--require-remote`, compare greedy token IDs with the baseline, then exercise
pressure/cancellation and measure TTFT/throughput/KV occupancy. No new graph
capture or inference accuracy result is claimed by this change.
