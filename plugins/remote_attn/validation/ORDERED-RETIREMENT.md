# Ordered D retirement — local verification, 2026-10-30

D scheduler overlap now uses local notifications and ordered transition commands.
No ordinary Remote retirement poll calls a TP collective. P source retirement,
native admission and model collectives keep their existing protocols; the D
non-overlap control path is unchanged. No SGL core or SMG files changed.

## Protocol

Each admitted target receives a TP-common monotonically increasing admission ID.
A startup-broadcast epoch isolates sessions; each rank retains its own transport
generation. Local observation prunes completed GPU results and publishes an IO
completion/release/error notice only on a state edge. GPU readiness is advisory:
there may be another forward after a notice. Notices are local Unix datagrams,
versioned against duplicate/late delivery; send backpressure retries the same
pending version without marking it delivered.

Rank 0 collects notices. It proposes only targets with reports from every rank
and a uniform release class (or proposes an abort on an error). The intent rides
one existing native request broadcast through a frozen receiver subclass that
preserves every native dataclass field. It does not use the generic experimental
control handler, whose result agreement would add another collective.

Input processing arms the intent. The existing Remote serial gate consumes the
previous result before the next launch. Hooks after result processing, before
run_batch, and after sampling for idle iterations finalize at a common scheduler
boundary. A command—not a local readiness branch—determines whether ranks join
the final vote. Vector shape and order come from the shared target manifest.

The final vote checks epoch/command sequence/admission identity, each rank's own
generation, current GPU/IO proof and all-release-or-no-release. Success retires
the same targets and optional native frees across ranks. A stale GPU snapshot or
release mismatch retains ownership; re-notification and a new command sequence
are required. Pending GPU state cannot cause a retry vote on every loop.
Invalid commands or failed proofs fail closed. Partial allocator release is never
retried. Local empty-target paths do not query proofs, poll/send mailboxes, or
add collectives/fences.

Retraction/shutdown consume any earlier armed intent first. At these already
TP-symmetric native boundaries, rank 0 broadcasts an explicit drain command;
ranks wait locally for the relevant proof, then vote once. Waiting itself has no
polling reductions. A cancel that already freed the request prevents the native
retraction backup/free from running again.

The transport uses epoch-scoped `/tmp` sockets; D TP workers must share the host
and `/tmp`. This preserves the current same-host scope and adds no environment
variables. It is not multi-node TP support.

## Executed local checks

- 90 CPU tests passed in 11.251 s using the lightweight native-source harness.
- Actual TP2/Gloo with native receiver broadcast plus real Unix datagrams:
  24 pending-IO rounds, 8 single-ready-rank rounds and 8 release-mismatch rounds
  produce zero retirement votes. A stale GPU snapshot causes one NACK; 16 further
  unchanged rounds add zero votes. Completion then commits under a new sequence.
- Both ranks record exactly `[[1,5], [2,5], [3,5], [4,5], [5,8]]` for
  `(command sequence, vote vector length)`: stale attempt, completion, explicit
  retraction, an already-armed intent consumed by retraction, and two targets
  coalesced into one final vote. Ordinary native receiver broadcasts continue.
- 32 Local rounds add no retirement reduction or mailbox work. A failed notice
  send retries the same version. All registrations are pruned after retirement.
- Native receiver fields and ordinary/local-abort messages are preserved. Intake
  arms without freeing; an idle common boundary can commit. GPU proof arriving
  after a READY notice, cancel/retraction competition, stale epoch/generation,
  reordered/duplicate commands and old notices retain safety.
- Native radix/token allocator, CUDA-event stand-ins, delayed sampling, CC copy
  errors, partial forward/release, overshoot and memory-pressure cases remain in
  the suite. These are CPU control/lifecycle tests, not CUDA measurements.

```bash
PYTHONPATH=python:plugins/remote_attn python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

Restoring the pre-change `a7549f1dee` retirement method in the TP2 worker makes
both ranks fail the zero-polling-vote assertion; the current implementation passes.
Ruff, isort and diff whitespace checks passed.

Source preflight: 29 hooks / 70 core API and state contracts. This change has no
GPU, model PD, NCCL or throughput result; only local testing was requested.
