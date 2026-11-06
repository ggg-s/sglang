# DSV D2D 冻结前预留：CPU / TP2 / 4060 验证

日期：2026-11-06。分支：`codex/remote-attn-pre-layerwise`，基于 `612c3d9d13` 的未提交修改。

## 结果与范围

- 完整 CPU 合同：315 项通过，100.739 秒；含真实 loopback socket 和双进程 TP2 Gloo。
- 4060：8 组新预留路径通过，两个既有 CUDA / Mooncake 故障回归各运行一次并通过。
- 静态检查：Ruff F401/F821/F823/UP037 与 `git diff --check` 通过。
- 最终 GPU 快照：`/tmp/d2d-reserve-required-4060.8MN6ZP`。机器通过既有 `ubuntu` SSH 别名访问。
- [机器可读结果与准确源代码 SHA256](dsv-d2d-reservation-4060.json)。所有记录的执行源文件与当前工作树逐项核对一致。

本轮仅完成后端预留、冻结约束和资源消费。P4 Gateway、SSE 输出接续、重启恢复和 SMG 尚未接入。
GPU 探针使用真实 CUDA tensor、native KV layout/state 定义、稀疏算子、pinned memory、CUDA event、Mooncake TCP；
scheduler 与 allocator admission 仍为显式 native ABI 夹具。`reserved_row_consumed` 证明控制路径复用同一 KV record，
不代表已在整模原生 allocator 中完成容量争抢或请求接纳验收。原生 `_pre_alloc` 的窗口范围参数使用 source-loaded
方法与 allocator spy 验证，真实 allocator、DSV 权重、原生 HTTP 流接续、H20 TP4 和 RDMA 吞吐仍需目标部署验收。

## 本轮语义检查

1. 默认冻结缺少预留即拒绝；已有计划即使开启诊断兼容入口也不能绕过预留。
2. 源端 plan 不停算；目标先检查并分配 FULL/SWA/request-row，再同步 GPU 初始化完成事件。
3. SWA 预算覆盖所有允许冻结点的窗口并集。例如 lower=33、upper=48、window=8 时覆盖 `[24,48)`。
4. import 只消费预留，不二次 prealloc；裁剪完整多余尾页，保留半页和较早的 SWA eviction cursor。
5. 已消费的预留不能再由 reserve_cancel 释放；取消先到时留下 tombstone，禁止后到请求复活 epoch。
6. 分配/释放中途异常、CUDA 完成证明失败时保留内存，不重试不确定的 free。
7. 已预留 manifest 强制要求活跃预留；目标不能在丢失预留后静默重新分配。
8. 真实 TP2 中仅 rank 1 缺容量时，两边都拒绝预留、源端保持运行；正常预留后继续验证渐进 Local 与安全退休。
9. 原有 bulk KV Mooncake + 独立 D2H staging、pending token、尾页保护和逐 entry TP retirement 回归通过。

## GPU 矩阵

运行时：Python 3.12.3，Torch **运行时** 2.13.0+cu130，CUDA 13.0，Triton 3.7.1，
`mooncake-transfer-engine-cuda13` 0.3.13.post1，RTX 4060 Laptop GPU（8188 MiB）。

所有用例都先 plan/reserve，再 freeze/import/commit；bulk KV 始终使用 `mooncake-host-v1`。
下表 Q/O 表示 Remote Attention 的 query/output 通道。误差为稀疏数学参考的最大绝对误差，
均通过原有 dtype 容差；state bytes、尾部和 padding 单独要求精确一致。

| 压缩比 | 布局 | Q/O 通道 | 原生 Mooncake 调用 | 最大误差 | 结果 |
|---|---|---|---:|---:|---|
| C2 | v4 | socket | 8 | 0.0 | PASS |
| C2 | v4 | Mooncake GPU | 10 | 0.0 | PASS |
| C2 | v41 | socket | 8 | 0.0 | PASS |
| C2 | v41 | Mooncake GPU | 10 | 0.0 | PASS |
| C4 | v4 | socket | 10 | 0.00390625 | PASS |
| C4 | v4 | Mooncake GPU | 12 | 0.00390625 | PASS |
| C128 | v4 | socket | 9 | 0.125 | PASS |
| C128 | v4 | Mooncake GPU | 11 | 0.125 | PASS |

故障回归：

- `retry-cancel`：bulk 完成前断连，重试次数为 0；目标接收区仍注册并被隔离，abort 被阻止，源 resume 等待真实 reader，原生写入最终完成。
- `d2h-input`：真实 D2H 未完成时注入异常；engine 仍强引用 GPU 输入和 Host 存储，原始输入未被复用，拷贝结果为原始内容。

这两项故障使用显式 `allow_unreserved=true` 的旧诊断路径，验证传输安全回归；预留取消与部分分配故障在 CPU 合同中注入。

## 复现命令

在隔离快照内使用既有 4060 venv：

```sh
export PYTHONPATH=python:plugins/remote_attn
/workspace/remote-attn-cuda-20261028/venv/bin/python test/manual/remote_attn/dsv_d2d_cuda_probe.py --source-root "$PWD" --output reserved-positive.json
/workspace/remote-attn-cuda-20261028/venv/bin/python test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py --case retry-cancel --source-root "$PWD" --output retry-cancel.json
/workspace/remote-attn-cuda-20261028/venv/bin/python test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py --case d2h-input --source-root "$PWD" --output d2h-input.json
```

CPU 测试位于 `test/registered/unit/plugins/test_remote_attn.py`；本地通过既有轻量 unittest wrapper
运行，避免导入完整模型依赖。TP2 worker 位于 `test/manual/remote_attn/dsv_d2d_tp_worker.py`。
