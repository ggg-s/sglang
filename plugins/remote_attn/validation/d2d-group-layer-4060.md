# D2D group-layer：RTX4060 验证

日期：2026-11-06。分支 `codex/remote-attn-pre-layerwise`，基线
`5884814306731be4485f3fa7ef3666e8fbe1979e` 加本次未提交修改。
配置说明见 [D2D group-layer](../D2D-GROUP-LAYER-ZH.md)。
原始结果、环境和 83 个实际测试源码的 SHA256 见 [JSON](d2d-group-layer-4060.json)；
逐文件核对后全部与本地最终源码一致。

## 环境与范围

用户的 RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130、Mooncake TCP；使用隔离目录
`/tmp/d2d-layer-groups-4060.F6ilxd`。Qwen 使用真实独立 source/target 进程、CUDA
stream/event、pinned Host staging、Mooncake native batch 和 FlashInfer。
DSV 使用真实 CUDA sparse kernel、原生 layout/state 字节与 bulk Host 传输。
模型、scheduler 和 allocator admission 采用明确的 ABI 夹具，未加载整模权重；
不是生产 SMG、跨机 RDMA 或性能验收。

FlashInfer 使用既有 CUDA13 工具链：

```bash
export CUDA_HOME=/workspace/conditional-graph-20261019/cuda13
export PATH="$CUDA_HOME/bin:/workspace/mixed-adapt-20261025/venv/bin:$PATH"
export PYTHONPATH=python:plugins/remote_attn
```

## 结果

28 个 CUDA 用例全部通过：

| 范围 | 数量 | 关键检查 |
| --- | ---: | --- |
| Qwen grouped 搬运与异常 | 9 | FP16/BF16、同步/异步、组宽 2/3/4、组尾、relay slot 背压、RemoteDrop、取消 |
| DSV group=4 正向/P4 | 8 | C2 V4/V4.1、C4、C128；Q/O socket/GPU 两种模式；预留、丢失 commit 回复后重启恢复、状态/尾页一致 |
| DSV group=1/2/3 | 3 | 默认路径兼容、组边界与剩余 entry |
| DSV 同组独立 Local | 2 | V4/V4.1 第一 entry 已完成可 Local，第二 entry 阻塞仍 Remote，源 KV 尚未释放 |
| bulk 故障注入 | 6 | 断连取消与 D2H 输入证明丢失，各独立运行 3 次 |

Qwen 6 层 probe 记录真实 native batch 的 descriptor 数：组宽 2 为 `[4,4,4]`，
组宽 3 为 `[6,6]`，组宽 4 为 `[8,4]`。组宽 4、目标仅 1 个 relay slot 时实际
回落为 `[2,2,2,2,2,2]`。各正常搬运后的 GPU KV 逐字节一致，目标注册全部回收。
RemoteDrop 后恰有 5 层 Local、被丢弃层仍 Remote，3 次真实 RPC attention
输出误差均小于 0.0003；取消用例证明目标 GPU 可安全复用、Host 注册最终回收。

DSV group=4 的小池中，一帧实际打包 2 或 3 个 entry。所有正向用例均在 bulk
阻塞期间执行 Remote Decode；copy 完成后仍受真实 pending GPU forward event
限制，最终恢复 Local，源 KV 只释放一次。state、尾页和 padding 字节精确一致。
稀疏 attention 的最大绝对误差：C2 为 0，C4 为 0.00390625，C128 为 0.125，
均通过既有 BF16 容差检查；这不是跨模型精度指标。

断连故障中仅一次 bulk read、零 RPC 自动重试；未完成原生写入期间 receiver
保持注册并 quarantine，abort 被拒绝、resume 返回 pending reader。D2H 证明
丢失时，GPU 输入及 Host slot 保持强引用，原始 allocation 未复用，最终读取到
原始字节。这些故障是注入的，实际 CUDA 拷贝、pinned memory 和 Mooncake 写入均执行。

完整 CPU 合同 348 项通过；另以 `HANDOFF_D2D_GROUP_SIZE=4` 运行真实 TP2 Gloo
非对称完成/异常隔离合同通过。Ruff 检查、`git diff --check` 通过。

## 探针修正

首次运行发现测试环境 PATH 缺少 Ninja；补上既有工具链后 Qwen 全部通过。
DSV 旧 probe 在 bulk 前启动固定 GPU 延迟，延迟可能在 copy 完成前耗尽；现在
在 copy 排空后建立真实 pending forward event，并显式检查 event 尚未完成。
新增 runner 在加载 ABI 夹具前初始化 PyTorch assert helper，避免 DTensor 懒加载
导致重复注册。这些修正只涉及探针，没有放宽搬运、退休或回收断言。

分组可减少控制和 native submit 次数，但会改变首层就绪时间及 staging 内存；
本次没有测量整模吞吐提升，不能将 batch 数减少直接当作性能结论。
