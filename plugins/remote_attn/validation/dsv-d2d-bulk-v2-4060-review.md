# D2D bulk v2：4060 CUDA 验证与安全问题复现

日期：2026-11-05（Asia/Shanghai）。分支 `codex/remote-attn-pre-layerwise`，基线
`cc0cc820ade5a4ec206a2a977cfa216c24d24b46` 加当前未提交修改。

结论：8 组正向 CUDA 迁移用例通过；review 中的两个 P1 安全问题各复现 3/3 次。
这是修复前快照的验证结果，不是修复后验收。当前修复与复测见
[修复后报告](dsv-d2d-bulk-v2-4060-fixed.md)。

[完整 JSON 与源码 SHA256](dsv-d2d-bulk-v2-4060-review.json)。

## 环境与范围

- NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 592.82，WSL2。
- 实际加载 torch `2.13.0+cu130`、Triton `3.7.1`；Mooncake CUDA13 wheel
  `0.3.13.post1`，强制 Mooncake TCP。报告分别记录运行版本和安装包元数据，
  因既有 venv 的部分 dist-info 版本与实际加载代码不同。
- D0/D1 是同一 GPU 上独立小池；使用真实 CUDA stream/event、pinned memory、
  Mooncake engine 和 TCP socket。分配、scheduler admission 使用显式 native ABI
  夹具，没有加载完整 DSV 模型、验证 HTTP/SSE 接续或 RDMA 性能。
- 使用新的隔离目录 `/tmp/d2d-bulk-v2-4060.LFYoLb`，未覆盖已有服务。

## 正向结果

每组都使用新增的 `mooncake-host-v1` bulk 通道；Q/O 分别覆盖 socket 和 Mooncake。

| 压缩 / 布局 | Q/O socket | Q/O Mooncake | 每组 Mooncake 传输次数 |
|---|---|---|---|
| C2 / V4 | 通过 | 通过 | 8 / 10 |
| C2 / V41（global FP4） | 通过 | 通过 | 8 / 10 |
| C4 / V4 | 通过 | 通过 | 10 / 12 |
| C128 / V4 | 通过 | 通过 | 9 / 11 |

覆盖：SWA/state/index 字节一致；global copy 阻塞时 D1 已接管并使用 Remote Attention；
目标历史 prefix 置零后输出仍与独立 reference 一致；尾页/padding 未被覆盖；延迟
CUDA forward 未完成时不释放 KV；退休后恢复 Local；源端 free 计数为 1。

数值沿用探针 `atol=0.015625, rtol=0.01`。C128 BF16 最大绝对误差为 0.125，
在该相对容差内，不能仅用绝对误差评价该组结果。

## 安全故障结果（失败，不计作正向通过）

### P1：重试与取消导致接收槽位提前注销——3/3

在第一次 bulk 原生发送调用入口设置闸门，注入连接中断让 `TargetClient` 重试；
第二次请求已获得 reader 引用后执行源端 resume，再放行第二次请求。它收到
`ReadRejected`，目标 worker 退出并注销真实 pinned 接收区，但第一次 bulk 调用
仍未完成。

三次均观察到：`receiver_closed=true`、`receiver_quarantined=false`、
`destination_registered=false`、`original_bulk_call_unfinished=true`。

探针额外保留接收 tensor，并在首次发送即将访问失效注册区时拦截调用，
`stale_write_prevented_by_probe=true`；没有故意向失效指针提交 DMA。

### P1：D2H 输入临时显存被提前复用——3/3

真实调用 `cudaMemcpyAsync(D2H)`，在独立 copy stream 上加入延迟，并在提交后注入
“完成证明不可用”的 Python 异常。没有向 CUDA 发非法指令，也没有破坏 CUDA context。

三次均观察到：拷贝仍 pending、源 tensor 不再被强引用、原地址被 allocator 重新
分配。用 231 填充新分配后，最终 Host 数据全部为 231，而非原始的 17。

`sender_quarantined=true`、`host_storage_retained=true` 仍未保护源 GPU 临时 tensor；
三次均为 `copied_original=false`、`copied_replacement=true`。

## 探针与复现

[故障探针](../../../test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py) 输出观察结果；
`unsafe_reproduced=true` 表示安全失败，不代表实现通过。正向探针保留原有断言。

以下命令对应修复前隔离源码快照及既有 4060 CUDA venv；当前探针已加入安全断言：

```bash
PYTHONPATH=python:plugins/remote_attn python test/manual/remote_attn/dsv_d2d_cuda_probe.py \
  --source-root "$PWD" --output positive-all-final.json

PYTHONPATH=python:plugins/remote_attn python test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py \
  --source-root "$PWD" --case retry-cancel --output retry-cancel.json

PYTHONPATH=python:plugins/remote_attn python test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py \
  --source-root "$PWD" --case d2h-input --output d2h-input.json
```

首次 GPU Q/O 用例在测试清理阶段出现 EOF：夹具先关闭 Hub，再发送 `te_close`。
本次仅修正测试关闭顺序为 Q/O transport → Hub → bulk slots，并更正报告里的 bulk
transport 标签，随后八组全部通过。这两项属于测试夹具修正，不修复上述 P1。

完整原始 JSON/log 归档位于本地 `/private/tmp/d2d-4060-validation/results.tar.gz`；
远端 GPU 在验证结束后恢复到 0 MiB 使用量。
