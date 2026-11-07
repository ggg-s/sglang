# D2D 控制面：提前准备与空闲唤醒

日期：2026-11-07。基线：`165575c97e`，分支 `codex/remote-attn-pre-layerwise`。

## 修复

热缓存短请求在 cut 后才串行执行 plan/reserve/freeze，容易在处理 freeze 时已经
结束。Gateway 现在在首个已采样 token 可见且可安全规划时就执行 plan/reserve，
使用持久化 `WAIT_CUT` 等待阈值；到 cut 时只需发 freeze。准备与源生成重叠，
不冻结正在准备的请求。plan 无需等 native Decode 边界，freeze 仍保留安全边界。

提前 plan 的证书必须覆盖准备期间的长度增长。Gateway 至少按剩余输出上限准备，
worker 按实际采样上限/context 截断，import 消费冻结长度并按原生页规则退还
多余尾部。取消或请求结束会显式撤销预留；同一 epoch 支持重启，不让 ticket
隐式失效。代价是 D1 更早占用 row/KV，未增加内存容量，容量不足仍走原有回滚。

当前分支的控制命令是在原生 request broadcast 前注入，再由调度器线程执行，
不是在 `_recv_requests` 尾部直接执行。原生 receiver 本身非阻塞，但空闲入口的
`IdleSleeper` 可在 zmq Poller 上等待 1000 ms。新 watcher 每 5 ms 检查控制文件名，
通过 pipe 唤醒该 Poller；不读命令内容、不访问请求/KV、不执行 Torch/TP。所有
命令仍由 scheduler 按 TP 顺序执行。Rust ring 入口保留原路径，将空闲等待上限
限制为 10 ms。退出关闭 watcher/FD，并恢复原 poller。

Gateway status 遇到已排队的修改命令让行，保留上一份 facts，不把锁忙误记成
worker 错误。D2D 文件回包轮询间隔从 20 ms 改为 5 ms。

没有引入后台线程冻结/搬运 KV，也没有合并跨 worker 的三段事务。此修改不能
抢占正在执行的 GPU forward，不能承诺 H20 完整模型的 freeze 或 handoff 时延。

## 归因与指标

- `request_finished` 为结构化、不可重试的窗口丢失：不创建 Source、不冻结请求；
  Gateway 撤销预留，记录 `window_missed_request_finished`。
- 语义不可迁移单独记录 `semantic_rejection`；native Decode 边界未到仍为可重试
  `migration_deferred`。已预留但在 cut 前完成记为 `completed_on_source_before_cut`。
- 每请求 journal / router / SSE / metrics 行提供 `migration_skip_reason`、
  `prepare_latency_ms`、`handoff_latency_ms`，沿用原有 deferral 与 freeze_pending。
- `/v1/router/status` 的 `control_latency_ms` 是按 role/command 的累计桶直方图，
  记录角色锁等待与 RPC 总时长，包含失败尝试，进程重启清零。
- `handoff_latency_ms` 从首次发起 freeze 至目标 commit/admit 成功，包含退休、
  import、重试和控制等待；由 journal 中成功样本恢复。`migration_skip_reasons`
  汇总 journal 中当前条目的跳过原因。各桶为累计计数，不是独立区间计数。

## 验证结果

Ubuntu / RTX 4060 Laptop 8GB / Torch 2.13.0+cu130，独立源码快照。

真实文件协议、nonce、ready 文件、原生 IdleSleeper 与 zmq Poller 的 8 次空闲控制
往返：`[10.74,10.62,10.79,11.55,5.91,5.82,5.97,5.76] ms`。每次先确认 worker 已
进入 idle poll，再发布命令。只有命令处理器的业务体使用 echo 边界；文件命令
领取、结果发布和调度线程执行路径真实。断言所有处理均发生在 scheduler 线程，
每次小于 500 ms。它不是完整推理 worker 的服务延迟测试。

c=4/8 CUDA 小池验证通过 Gateway 首 token prepare → WAIT_CUT → cut=32 →
部分 Remote freeze → 原生 PD 退休 → D2D import/commit → 目标续 token → Local：

| c | 提前 plan/reserve | cut 前 freeze | 实际迁移/目标续 token | 源 KV 释放 | D1 退还多余 token slots |
|---:|---:|---:|---:|---:|---:|
| 4 | 4/4 | 0 | 4/4 | 每请求一次，共 4 次 | 256 |
| 8 | 8/8 | 0 | 8/8 | 每请求一次，共 8 次 | 512 |

每档一个共享 D0、一个共享 D1、不同请求行。真实 native pinned receive/scatter、
CUDA event、有序退休与 Mooncake TCP；冻结时原 P→D0 尚未排空，未发布 D2D
服务。原 PD 退休后才导出；D2D reader 被阻塞时源保留 KV，解除后仅释放一次。
SSE 在 cursor=31 后仅给出目标 seq=32 的后续 token。该探针的 Gateway driver 是
进程内命令分发，采样/原生分配是 ABI 夹具；不能拿它衡量文件 RPC 延迟或模型收益。

机器结果和被测源码 SHA-256：
[d2d-control-c4-c8-4060.json](d2d-control-c4-c8-4060.json)。

单元回归共 **488 项通过**：主套件 376、native contracts 23、wire 83、native ABI 6。
覆盖提前准备后重启、等待 cut、取消回收、错过窗口回滚、status 给修改让行，以及
原有 TP2 不对称退休/commit 隔离。定向 Ruff 与 51 hooks / 131 core interfaces
预检通过。原始日志：`/private/tmp/d2d-control-latency-results/results.tar.gz`。

额外的单 rank finished → TP2 一致归因用例未同步执行：自动审批拒绝向测试机继续
发送文件，因此未将该补充用例纳入本次提交；现有 TP2 回归与单 rank 窗口归因回归
均已通过。

补充 TP2 窗口归因验证已完成：真实双进程 Gloo 中，分别仅令 rank 0 或 rank 1
在 freeze 预检前发现请求已结束。两个进程均返回相同的 `request_finished`、
`retryable=false`；没有创建 Source、修改 batch membership、冻结请求或释放源 KV。
该用例扩展既有 TP2 回归，总测试项数不变；定向执行 1 项通过（4.659 s）。

负对照在独立快照中仅将 D2D `handle` 恢复为 `165575c97e` 的版本，其余代码和
新增用例保持一致；同一用例失败，复现泛化的 `D2D owner or needs native Decode
boundary` / `peer rejected` 报错。验证使用 CPU 小池和真实 Gloo collective，
不等同于两张 GPU 或跨机 TP 的验收。结果及源码哈希见
[d2d-control-tp2-finished-window.json](d2d-control-tp2-finished-window.json)。

没有 H20 内网访问能力，未测 H20 完整模型、跨机 RDMA、冷热缓存下 c=4/8 的
实际迁移窗口/TTFT/吞吐。H20 复测应先全量预热，再分别记录 c=4/8 的 plan/reserve
是否在 cut 前完成、实际目标续 token 数、request_finished 次数，以及 control 与
handoff 直方图；不能用自然排空或仅建立 ticket 计为成功迁移。

## 复现

```bash
PYTHONPATH=python:plugins/remote_attn \
python test/registered/unit/plugins/test_remote_attn.py

PYTHONPATH=python:plugins/remote_attn \
python test/registered/unit/plugins/test_remote_main_contracts.py

PYTHONPATH=python:plugins/remote_attn:test/manual/remote_attn \
python test/manual/remote_attn/dsv_d2d_control_cuda_probe.py \
  --output /tmp/d2d-control-c4-c8.json
```
