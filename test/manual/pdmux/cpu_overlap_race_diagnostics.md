# TP2 CPU overlap: H100 follow-up and diagnostic workflow

## Evidence and current status

The following deployment results were supplied by the user on 2026-10-27.
The H100 logs and databases have not been read in this local session.

| CPU overlap | HiCache | ReplaySSM | Concurrency | Chunk | Additional setting | Reported outcome |
| --- | --- | --- | ---: | ---: | --- | --- |
| on | on | on | 32 | 16384 | none, launch 1 | GDN/FLA index assert around request 96 |
| on | on | on | 32 | 16384 | none, launch 2 | KV slot assert and watchdog around 331 |
| on | on | on | 32 | 16384 | invariant checks | first reported invariant: sampler logits NaN, around 163 |
| on | off | on | 32 | 16384 | invariant checks | crash around 135 |
| on | off | off | 32 | 16384 | invariant checks | crash around 119 |
| on | off | off | 32 | 8192 | invariant checks | crash around 147 |
| on | off | off | 8 | 16384 | invariant checks | crash around 952 |
| on | off | off | 32 | 16384 | CUDA_LAUNCH_BLOCKING=1 | 2246 requests without an observed exception; incomplete run |
| off | on | on | 32 | 16384 | control | all 4219 processed: 4218 successful, one failure shared with the other group; original broadcast timeout did not recur |

The user confirmed these ablations **do not contain `4b797d693e`**, which adds
allocator index-storage lifetime protection and HiCache pump ordering fixes.
Those changes therefore remain untested on this deployment workload. The
single-4060 reproduction in that commit demonstrates a real index-storage bug;
it does not establish that the reported H100 crashes have the same cause.

The completed overlap-off control is evidence that the earlier NCCL BROADCAST
600-second timeout did not recur in this trial. CPU-overlap-on remains outside
deployment acceptance. A watchdog following a CUDA assert must be interpreted
with the first CUDA error, rather than counted as another clean reproduction
of the original broadcast stall.

Launch blocking and the different failure locations strongly support an
ordering/lifetime problem. They do not identify a writing kernel or a particular
shared buffer. Normal usage counters and legal allocator bookkeeping do not
exclude use-after-free, stale input reads or missing readiness dependencies.
Disabling HiCache/ReplaySSM rules them out as necessary conditions for the
observed crash. Lower concurrency delays the failure and does not eliminate
concurrency as a contributing factor. The c=8 row required roughly 952
requests, so a 150-request diagnostic budget is not guaranteed to reproduce it.

## Run the patched workload before instrumentation

Record `git rev-parse HEAD` and confirm it contains `4b797d693e`. Start a fresh
server for each trial and repeat the original frozen SWE 4219 request stream,
order, concurrency and chunk size 16384. Keep shadow checks disabled and keep
the normal asynchronous launch configuration. Preserve the resolved server
settings and the identity/reason of the one control-matching failed request.

This is the direct test of the patch. Then, if a failure remains, run the two
diagnostic modes separately, retaining the same request IDs and server recipe.

## Additional reproduced scratch-buffer defect

Code inspection found that DFlash prefill context injection and decode commit
both call the same `FusedKVMaterializeHelper`. Its old implementation reused
one projection/K/V workspace set across streams. Keeping these buffers alive
does not protect against the other lane overwriting them before a queued KV
consumer reads them.

A real CUDA test with two layers and concurrent plain/green-context streams
reproduced incorrect K values: the original helper disagreed with its serial
reference on all 32768 elements in the first comparison. The patched helper
keeps scratch tuples per actual execution stream and uses the selected tuple
as a local value throughout materialization. The same test then matched the
serial K and V references exactly. Weights and RoPE data remain shared reads.
Scratch memory can grow for each stream that uses the helper.

This establishes another unsafe shared buffer, not the cause of the supplied
H100 crashes. The patch and the allocator fix both need the original replay.
The synthetic regression is:

```bash
python -m pytest -v test/manual/pdmux/test_dflash_fused_kv_streams_cuda.py
```

## Index shadow checks

`SGLANG_PDMUX_INDEX_SHADOW=1` enables checks only when PDMux is enabled. It is
off by default and independent of `SGLANG_INVARIANT_CHECK`.

The hooks observe:

- Target/draft logical forward `out_cache_loc` and `req_pool_indices`.
- Actual decode CUDA Graph static index buffers, after load/metadata preparation
  and before replay, followed by checks after replay is enqueued.
- Persistent split-prefill indices and length tensors across segments, plus
  GDN/linear `query_start_loc`, `mamba_cache_indices` and metadata object identity.

For example, a device assertion can name `batch.out_cache_loc` at
`before-segment-8`, or `graph.req_pool_indices` at `after-graph-replay`. Messages
include target/draft role, P/D lane, TP rank, forward pass, stream-group index,
tensor address and shape. A Python binding-change error identifies metadata
replacement before the next segment. An async assert may surface at a later
synchronization point; its message identifies the observation, not the writer.

The implementation queues clones, comparisons and `torch._assert_async` on the
consuming stream. It adds no cross-lane wait or explicit host synchronization.
It does add memory traffic, check kernels and retained references, and can hide
a lifetime/timing failure. It observes selected buffers, not all intermediates;
a mutation report also needs checking against any intentional backend writes.
A clean shadow run is not proof that the uninstrumented race is gone.

The wrapper accepts the exact existing server command as an argument array:

```bash
# Populate SERVER_COMMAND with the original complete launch command/arguments.
PDMUX_RACE_LOG_DIR=/workspace/race-shadow \
  bash test/manual/pdmux/run_overlap_race_diagnostics.sh shadow -- "${SERVER_COMMAND[@]}"
```

Launch the frozen benchmark client separately. The wrapper records the commit,
command, selected flags, GPU identities, server output and exit status. It sets
shadow mode and defaults invariant checks to strict (2); the normal server exit
status survives the log pipe. It preserves `CUDA_LAUNCH_BLOCKING` and records
its value; use the normal nonblocking configuration for race diagnosis.

## Compute Sanitizer

```bash
# Optional: export COMPUTE_SANITIZER=/path/to/compute-sanitizer
# For split binary/library installations, also set:
# export COMPUTE_SANITIZER_INJECTION_PATH=/path/to/injection/libraries
PDMUX_RACE_LOG_DIR=/workspace/race-memcheck \
  bash test/manual/pdmux/run_overlap_race_diagnostics.sh memcheck -- "${SERVER_COMMAND[@]}"
```

This selects `--tool memcheck --target-processes all`, covering spawned TP
workers as well as the parent, and requests error exit code 86. Shadow checks
are disabled in this mode. The sanitizer version and tool log are saved along
with server output; missing tools and tool failures remain nonzero exits.
Use a toolkit/driver-compatible sanitizer with its injection libraries installed.
`COMPUTE_SANITIZER_INJECTION_PATH` supplies an explicit `--injection-path` when
the binary and injection libraries are installed in separate directories.
On the 4060 workstation, the isolated CUDA 13 sanitizer ran the two healthy
application tests but returned 86 with WDDM debugger initialization and
unsupported-device errors. This is a tool/platform failure, not a clean
memcheck result. The native H100 deployment still needs the sanitizer run.
Do not combine these two diagnostic modes for the first comparison.

Reduce concurrency to 8 if needed for sanitizer cost, while retaining chunk
16384 and the frozen failing request stream. Keep enough requests to cover the
observed c=8 failure point. Memcheck can identify an invalid access and its
address; an in-bounds logical overwrite or a timing-sensitive race can remain
undetected. Its instrumentation also changes timing.

Collect the first violation and preceding logs from every TP rank, not just the
eventual watchdog. Use the allocation address from a shadow report to correlate
with memcheck when both runs report a related buffer. First establish whether
the failing indices were already invalid at capture or changed afterward.

## Acceptance and performance

Two independent uninstrumented full replays should process all 4219 requests,
matching the control's 4218 successes and the same known failure, with no CUDA
assert, NaN invariant failure, collective timeout or watchdog termination.
Compare generated outputs and benchmark performance on successful full runs.
The partial launch-blocking run and diagnostic runs cannot establish this gate.

| Workload | CPU overlap | req/s | TTFT p50 | TPOT | E2EL | Accept rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SWE 4219 | off | 3.584 | 3.52 s | 19.78 ms | 8.86 s | 0.669 |
| SWE 4219 | on | unavailable: crash | — | — | — | — |
| LongCodeBench c16 | on | 0.2827 | 50.6 s | 29.67 ms | 56.4 s | 0.654 |
| LongCodeBench c16 | off | 0.2737 | 52.3 s | 33.98 ms | 58.8 s | 0.657 |

The completed LongCodeBench trial reports about +3.3% throughput and -12.7%
TPOT with CPU overlap. It does not establish a SWE benefit or long-run stability.

The user's original artifacts are reported at
`pd-deploy/bench/qwen36-35b-swe/REPORT-2026-10-27-cpu-overlap-race.md`,
`q35swe-rc-20261027.tgz`, and
`/workspace/validation-runs/qwen36-35b-swe4219-overlap-20261027/` on the deployment
host. The host is not connected to this session; the 4060 diagnostic checks do
not replace that H100 replay.
