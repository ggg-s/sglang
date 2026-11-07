# Issue B：部分 Remote 请求的冻结与 D2D 迁移

基线：`5240abdb8b`，分支 `codex/remote-attn-pre-layerwise`。验证日期：2026-11-07。

## 行为与所有权

已准入的原生 progressive PD 请求不再需要运行至 `all_local()` 才能 plan/freeze。
目标预留后，D0 在 native Decode 边界固定输出与 KV 长度，移出运行批次，保留
原 receiver/ring/metadata、页表、KV 和 P Remote read lease。其余请求继续调度。

冻结不意味着旧 P 写入已完成。控制命令返回 `pending_retirement=true`，让调度
循环继续处理 CHUNK_READY、deferred scatter 和 TP retirement；不能在命令内
同步 drain，否则会阻塞 scatter 所需的调度循环。旧 lifetime 完成 canonical
Success、scatter/GPU consumer 和 TP 退休后，才发布 D0→D1 manifest/读服务。
即使 Success 先于最后一个 chunk 到达，也不能导出或释放 KV。

D2D 继续使用现有预留、native state/index barrier、bulk copy、Remote/Local 切换
及 reader retirement。没有新增 D1→D0→P 的链式 Remote 路径。commit 前取消可
恢复 D0；旧 PD 未关闭时，恢复必须保留原 receiver/RPC。接收或 GPU/TP 证明
失败则冻结并保留原 KV，拒绝导出与不安全的 resume。源 KV 仅在目标复制完成、
GPU consumers/reader lease 全部退休后释放一次。

Gateway 在 FREEZE 阶段沿用同一 epoch 重试，支持 freeze 回包丢失及重启恢复。
`freeze_pending` 表示冻结后的旧 PD 排空；`source_partial_remote` 表示冻结时仍有
Remote entry。worker 还报告 `source_remote_entries`、`frozen_kv_len` 与 `prepared`。
这些状态和冻结前的 `migration_deferred` 分开，后者仍覆盖 native boundary 等待。

约束仍为两个 remote Decode worker、D radix cache 关闭。非原生/未登记的 Remote
owner 不会因本改动被放行。冻结到导出仍需等待 P→D0 排空，不保证迁移延迟收益。

## 验证

Ubuntu / RTX 4060 Laptop 8GB / Torch 2.13.0+cu130，独立源码快照。
每档均为同一个 D0 worker 上的并发请求及一个共享 D1 worker，使用不同原生行和
非零、不同物理页数据。冻结时至少一个 global KV entry 尚未接收。

| c | 部分 Remote 时已冻结 | 实际 D2D import/commit/Local | 源 KV 释放 | Mooncake 传输 |
|---:|---:|---:|---:|---:|
| 4 | 4/4 | 4/4 | 每请求一次，共 4 次 | 32 |
| 8 | 8/8 | 8/8 | 每请求一次，共 8 次 | 64 |

- 入站 P 字节在 pinned ring 边界注入；使用原生 StagingAllocator、
  DecodeStagingHandler、H2D scatter、CUDA stream/event 和有序退休协议。
- 出站 D0→D1 是真实 Mooncake TCP、D2H/H2D、socket 控制与 reader lifetime；
  验证 native state/index/global 字节、不同目标行、token history、commit 和源释放。
  bulk copy 阻塞时目标已 commit/admit，源仍持有 KV。
- 强制 Success 早于 CHUNK_READY：所有请求保持冻结、没有 manifest/读服务，
  原 ring allocation、metadata 与 KV 均未提前释放；到达并 scatter 后才导出。
- 真实 Gloo TP2：rank 1 的 GPU consumer 证明未完成时，两个 rank 均不导出；
  证明完成后才共同退休及发布。既有 TP 部分 commit、隔离和释放测试保留。
- 冻结中取消恢复保留旧 PD receiver/RPC/行；接收失败拒绝导出和 resume。
- Gateway freeze 回包丢失、重启后仍使用同一 epoch/一次尝试，最终切至 dst；
  SSE 从已有 cursor 继续，没有重复输出。

CPU 主套件 371 项通过（182.2 s），wire 83、core contracts 21、native ABI 6，
共 481 项通过。新增 c=4/8 回归在原 `5240abdb8b` D2D 实现上失败：plan 仍被
`remote_entries_pending` 推迟；修复后通过。定向 Ruff、空白检查、50 hooks / 131
core interfaces 预检通过。

机器可读 CUDA 结果和源码 SHA-256：
[dsv-d2d-partial-remote-4060.json](dsv-d2d-partial-remote-4060.json)。
原始日志归档：`/private/tmp/dsv-d2d-partial-results/results.tar.gz`。

复现小池 CUDA 验证（在可用 CUDA/Mooncake 环境中，从仓库根目录）：

```bash
PYTHONPATH=python:plugins/remote_attn:test/manual/remote_attn \
python test/manual/remote_attn/dsv_d2d_partial_cuda_probe.py \
  --output /tmp/dsv-d2d-partial.json

PYTHONPATH=python:plugins/remote_attn \
python test/registered/unit/plugins/test_remote_attn.py -k partial_remote

PYTHONPATH=python:plugins/remote_attn \
python test/registered/unit/plugins/test_remote_attn.py -k dsv_d2d_tp2
```

## H20 边界与后续复测口径

H20 为不可访问的内网机器，本次没有运行 H20 全模型、HTTP 压测或跨机 RDMA。
模型执行、采样与 native allocation 是测试的显式 ABI 边界，不能把上述迁移计数
当作 H20 c=4/8 压测结果，也不能据此宣称 TTFT/TPOT 或主动迁移性能改善。

内网复测沿用公平预热与 completion_tokens 口径。对 c=4/8、cut=32 记录：
`source_partial_remote`、freeze_pending 时间、实际 dst owner 切换次数、
first_target_ms、token gap、KV/metadata/ring 退休及源释放计数。仅 plan/freeze
不计作迁移；只有 commit 后输出 owner 切至 dst 才计入。
