# P4 DSV：编排、SSE、重启与 4060 验证

日期：2026-11-06。分支 `codex/remote-attn-pre-layerwise`，基于 `612c3d9d13` 的未提交修改。

## 结果

- 完整 CPU 回归：**339 项通过，141.152 秒**；保留原有 Qwen/DSV 用例，新增 24 个 P4 测试方法。
- 真实 CLI 进程：在 commit 已生效、回复尚未返回时发送 **SIGKILL**，重新启动相同 P4 main/Handler；
  从磁盘日志继续 COMMIT，未恢复 Source、未重复提交原生 PD、未重复接纳 Decode。
- 真实 HTTP/SSE：同一连接跨 Source→Target 保持 seq 连续；重启后携带 Last-Event-ID 补发，
  包括服务器可能已写出但客户端尚未确认的前缀 token。
- 真实 TP2 Gloo：仅一个源 shard 确认 commit 时所有目标 shard 均未接纳；确认齐全后仅接纳一次。
- 4060：**8 组 P4 + CUDA + Mooncake 正向通过**，两个既有传输安全故障回归各一次通过。
- Ruff F401/F821/F823/UP037 与 `git diff --check` 通过。
- [机器可读结果和准确源代码 SHA256](p4-dsv-gateway-4060.json)。记录的 81 个执行源文件逐项与工作树核对一致。

## 验证边界

CLI SIGKILL 用例使用真实 P4 入口、journal/flock、HTTP/SSE 和 loopback 控制代理；
tokenizer、原生 scheduler/allocator 为明确 ABI 夹具，后端 D2D registry/socket/复制协议实际执行。
P 的真实 HTTP 请求跨 Gateway 重启存活另有本地 HTTP server 验证；未运行带模型权重的 P forward。

GPU 矩阵使用实际 CUDA tensor、native KV/state layout、sparse kernel、pinned memory 和 Mooncake TCP；
通过 P4 状态机 plan/reserve/freeze/import/commit，主动丢失 commit 回复，再销毁并重建协调器对象、
从磁盘日志恢复。GPU 这部分的重启是对象级；进程级 SIGKILL 由上述 CPU CLI 用例单独验证。
GPU scheduler/allocator admission 仍为 native ABI 夹具，因此 `native_decode_admissions=1` 证明接口仅被调用一次，
不代表已做完整 DSV 模型调度与采样验收。整模 HTTP PD、H20 TP4、RDMA 和吞吐/延迟性能仍待目标部署验证。

## 关键安全与恢复覆盖

1. 所有意图阶段 SUBMIT/SOURCE/PLAN/RESERVE/FREEZE/IMPORT/COMMIT/TARGET 均可从日志重建。
2. submit/reserve/freeze/import 的回复丢失保持相同 epoch，避免重复入队和分配。
3. COMMIT 意图落盘后只 roll forward；丢回复或部分 rank 确认不允许 resume Source。
4. 原生接纳失败后 TP 集体 quarantine，保留 KV/注册区/reader/callback，不重放不确定接纳。
5. 预提交取消写 Target tombstone，再排空 IO 并 resume Source；Source tombstone 拒绝迟到 freeze，
   重复 resume 不重复构造 Decode batch。
6. worker UUID 由恢复查询和每条变更命令共同验证，覆盖查询与执行之间 worker 重启的竞态。
7. 独占 journal lock 与损坏日志拒绝恢复；COMMIT journal 写入失败时未发出 commit RPC。
8. P→D Remote 未退休时不开始 D2D；未消费的真实预留会阻止 native idle allocator reset。
9. drain 先封锁 Source admission；所有源 rank 原生队列和插件资源排空前不报告 SAFE_TO_STOP。
10. STOPPED 标记跨 Gateway 重启持久化，保持路由 fence，并停止依赖已停止 Source 的轮询。
11. 终态 token 历史持久化，Source worker 和 Gateway 均重启后仍可按游标补读。
12. 生产后台模式使用独立角色状态轮询，已覆盖实际后台状态机执行。

## GPU 环境与矩阵

RTX 4060 Laptop GPU，8188 MiB；Python 3.12.3；Torch **运行时** 2.13.0+cu130；
CUDA 13.0；Triton 3.7.1；`mooncake-transfer-engine-cuda13` 0.3.13.post1。
最终隔离快照：`/tmp/p4-dsv-final-4060.DAh70U`，通过现有 `ubuntu` SSH 别名访问。

每组均证明：预留先于冻结、同一 KV record 被消费、仅一次 Decode admission、commit 回复丢失后
恢复为 Target owner、bulk 未完成前可 Remote 计算、state bytes 精确、尾页/padding 精确、
延迟 forward 阻止回收、Source free 恰好一次、最后回到 Local。

Q/O 通道与 bulk KV 通道独立；bulk 始终为 `mooncake-host-v1` 和独立 D2H staging。
误差是 sparse attention 数学参考的最大绝对误差，均通过原有 dtype 容差；不是整模 token 误差。

| 压缩比 | 布局 | Q/O 通道 | Mooncake 调用 | 最大误差 | 结果 |
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

- `retry-cancel`：bulk 未完成即断连，RPC 重试为 0；目标仍被注册并隔离，prepared abort 被拒绝，
  源 resume 等待真实 reader，原始原生写入最终完成。
- `d2h-input`：真实 D2H 未完成时注入异常；engine 保留 GPU 输入和 Host backing，输入未复用，
  最终拷贝仍来自原始内容。

这两项沿用显式 unreserved 诊断入口，仅验证传输回归；新 P4 正向矩阵强制先 reserve。

## 复现

```sh
export PYTHONPATH=python:plugins/remote_attn
/workspace/remote-attn-cuda-20261028/venv/bin/python test/manual/remote_attn/dsv_d2d_cuda_probe.py --p4 --source-root "$PWD" --output p4-positive.json
/workspace/remote-attn-cuda-20261028/venv/bin/python test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py --case retry-cancel --source-root "$PWD" --output retry-cancel.json
/workspace/remote-attn-cuda-20261028/venv/bin/python test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py --case d2h-input --source-root "$PWD" --output d2h-input.json
```

CPU 用例：`test/registered/unit/plugins/test_remote_attn.py` 中的 `test_p4_dsv_*`；
CLI 子进程：`test/manual/remote_attn/p4_dsv_gateway_worker.py`；TP2：`dsv_d2d_tp_worker.py`。
本地完整测试使用既有轻量 unittest wrapper，避免导入全部模型依赖。
部署与接口见 [P4 DSV 使用说明](../P4-DSV-GATEWAY-ZH.md)。
