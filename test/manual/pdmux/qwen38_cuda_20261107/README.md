# Qwen3.8 PDMux CUDA regression — 2026-11-07

Status: **component CUDA checks passed; full-model acceptance is pending**.
No new numerical or lifecycle failure was observed in the exercised cases. This
run added a reproducible CUDA harness and dtype coverage; it did not require a
new production-code fix.

## Host and source

- Connection: `ssh ubuntu`, through the configured Windows SSH jump host.
- GPU: NVIDIA GeForce RTX 4060 Laptop, 8,188 MiB, SM89; driver 592.82.
- Python: `/workspace/conditional-graph-20261019/venv/bin/python`.
- PyTorch: 2.14.1+cu130.
- Remote snapshot/results: `/workspace/qwen38-pdmux-cuda-20261107/`.
- Snapshot is isolated from existing source trees. It contains the current
  uncommitted Qwen3.8 adaptation on top of merge commit `f1ceeda326`.
- `qwen38-cuda.json` records hashes of the model loop and its tensor fixture.

## Results and their scope

| Check | Result | What it exercises |
| --- | --- | --- |
| Qwen3.8 model loop, FP32/BF16, 1/65/513/2048 tokens, 1/2/4 layers per segment | 24 configurations passed | Real CUDA tensor operations on two streams through the production model loop, with small layer collaborators |
| Repeated prefill/decode interleaving | 768 prefills, 1,792 decodes; exact output equality | Per-batch HC/PLE state, normalization handoff, one PLE prepare/commit, cleanup and temporary allocation pressure |
| PLE prefetch gate, FP8 backend policy, MTP hidden-state handoff | Passed | Control-path fixtures; these do not execute real FP8/GDN/QSA kernels |
| Singleton NCCL lanes with decode CUDA Graph | Passed | Real singleton NCCL groups, lane routing, broadcast and graph replay |
| Same-group selection / stream-group transition | Passed | Existing group does not drain; switching groups drains/rebinds correctly |
| Two-GPU NCCL test | Skipped | One GPU is present; distinct-rank TP was not simulated on one GPU |

The CUDA harness uses the same model loop as the CPU fixture, loaded via AST to
avoid importing the full serving dependency tree. The layer/PLE collaborators
are deliberately small. It does **not** instantiate the checkpoint's real
GDN, QSA, HyperConnection or FP8 operators. The test streams are ordinary CUDA
streams, so production green-context SM partitioning is not validated here.

`harness_seconds` includes Python fixture submission and is **not** a model
throughput, TTFT or TPOT measurement. No serving-performance claim follows from
these numbers.

## Reproduction

From the isolated remote `source/` directory:

```bash
/workspace/conditional-graph-20261019/venv/bin/python \
  test/manual/pdmux/validate_qwen38_pdmux_cuda.py \
  --rounds 32 --output ../qwen38-cuda.json

/workspace/conditional-graph-20261019/venv/bin/python -m pytest \
  test/manual/pdmux/test_pdmux_tp_overlap_cuda.py -v \
  --junitxml=../tp-cuda.xml
```

Local checks also passed: four CPU Qwen3.8 fixture cases, targeted Ruff checks,
and `git diff --check`.

## Still required for full-model acceptance

1. A host with enough aggregate GPU memory for the full FP8 checkpoint, runtime
   buffers and KV/linear-state caches. This single 8-GiB GPU is insufficient.
2. The target checkpoint/tokenizer and a SGLang installation matching this
   branch, including compatible compiled kernels. The selected regression venv
   has PyTorch/Triton but no installed `sglang` or `sgl_kernel`; no target cache
   was found in the common cache locations checked.
3. Same-flags PDMux-off, standard and layer-split serving runs: serial and mixed
   concurrent requests, chunk boundaries, prefix-cache reuse and native MTP.
4. Real-operator numerical comparisons and task accuracy evaluation. Keep
   same-token logprob/cache-state diagnostics separate from answer accuracy and
   FP8 batching noise; do not infer model accuracy from fixture equality.
5. Real multi-GPU TP/NCCL and green-context tests, then sustained mixed-load
   stress and a CUDA trace confirming overlap. Measure throughput, TTFT/TPOT
   distributions and memory on the same workload/settings as the baseline.
