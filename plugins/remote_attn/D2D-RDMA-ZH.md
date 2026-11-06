# D2D Mooncake backend：TCP 与 GPU staging RDMA

适用范围：Qwen3-8B 的 `layer` / `remote` 在线迁移，以及 DSV4 / DSV4.1 的
状态安装和 bulk KV。沿用 [layer group × token range](D2D-LAYER-TOKEN-RANGE-ZH.md)
的逻辑计划、原生预留和逐 entry 完成证明；不改变模型 allocator 或 P4/SMG 路由语义。
Remote Attention 的 Q/O/LSE 现在也复用同一 engine/完成契约，见 [Q/O 数据面](REMOTE-MOONCAKE-ZH.md)。

## 配置与能力协商

两端升级到同一版本。在原有原生 PD/Mooncake 启动参数上设置：

```bash
export MOONCAKE_PROTOCOL=rdma
unset MC_FORCE_TCP
export HANDOFF_D2D_BACKEND=auto       # auto / tcp / rdma，默认 auto
export HANDOFF_D2D_GROUP_SIZE=4
export HANDOFF_D2D_TOKEN_RANGE=1024
```

| 策略 | 行为 |
| --- | --- |
| `auto` | 双方均具备已验证的本地 GPU 注册、注销和接收可见性能力时选择 `rdma-gpu-v1`；否则双方选择 `tcp-host-v1` |
| `tcp` | 只提供 Host staging + TCP；原生 PD engine 是 RDMA 时，共用单独的进程内 TCP fallback engine |
| `rdma` | 只提供 GPU staging RDMA；本地能力不可用或双方不兼容则拒绝，不降级 |

`mooncake_backend.py` 统一原生 wrapper / Python binding 的接口与 engine 身份。
Qwen 从原生 rank-local PD engine 获取 binding，不再为在线 D2D 硬编码创建 TCP engine。
DSV 通过同一适配器使用原生 manager engine。无原生 PD runtime 的独立探针使用
共用 TCP engine。能力判断读取初始化后从 engine 自身 classic P2P metadata 验证的
实际协议，不能仅凭 `initialize(..., "rdma"/"tcp", ...)` 参数或 GPU 注册成功判断。
metadata 查询失败、未知格式或多 transport 描述都关闭 GPU 能力。

TCP fallback 在共享初始化锁内临时设置 `MC_FORCE_TCP`，完成初始化并核实 metadata
为 TCP 后恢复环境。原生 SGLang 初始化及 HCA filter 选择使用同一把锁，避免并发初始化
误读覆盖值；已经建立的原生 RDMA engine 不重新初始化。

GPU backend 当前限定经典 RDMA transport。`MC_USE_TENT`、`MC_USE_TEV1`、
`MC_FORCE_SHM` 只要存在（包括 `0`、空字符串），或 engine 没有实际协议验证证明，
就不会开放 GPU RDMA。`MC_FORCE_TCP` 同样按是否存在判断；`0` 不表示关闭。强制 TENT/SHM 会覆盖新 engine 的
初始化，因此同一进程不能用它们伪装 TCP fallback；需在启动前关闭这些选项。
握手升级为 `d2d-mooncake-v2`，必须两端升级；拒绝 v1 peer，避免旧发送端把内部
重试成功误当作所有 DMA 已完成。旧诊断 full-copy/file wire 不在此协商范围。

GPU 注册探针每个 engine/device 只做一次：查询可见性支持 → 分配小 GPU buffer →
注册 → 注销。注册或注销结果不明时，探针 buffer 保留到进程退出，GPU 能力判定失败。
这只证明本地能力，不能证明 RDMA 网络连通性或远端 HCA 正确配置。

## 跨机地址

原生 engine 沿用原生发布地址。TCP fallback 默认采用原生 engine hostname，可用
`HANDOFF_MOONCAKE_HOST` 指定本机对端可达的 IP；它必须是本机地址，不是目标 IP。
独立探针默认 loopback。

Qwen 的 bulk 控制 listener 仍默认 loopback。跨机需分别设置本机地址：

```bash
export HANDOFF_TRANSFER_BIND_HOST=0.0.0.0
export HANDOFF_TRANSFER_HOST=<本机对端可达的IPv4地址>
export HANDOFF_MOONCAKE_HOST=<本机对端可达的IPv4地址>
```

Remote Attention 的 RemoteHub/mesh 地址仍按 [多机部署](MULTI-PD-ZH.md) 配置。
上述变量不替代 RemoteHub、原生 bootstrap 或 Gateway 地址配置。

## 数据路径与完成顺序

TCP 路径保留独立 D2H Host staging、连续 payload/footer、目标 H2D 和 ACK：

```text
原生 GPU KV → gather → D2H → 注册 Host → TCP → 注册 Host → H2D → scatter
```

GPU backend 使用独立、已注册的 GPU staging：

```text
原生 GPU KV → gather → GPU staging → GPUDirect RDMA → GPU staging → owned GPU copy → scatter
```

GPU staging 不直接别名原生 KV 页，目标独立 copy 保持原有 slot 与 scatter 生命周期。
DSV 每个 slot payload 上限 4 MiB；Qwen 每层 slot 的 token 数同时受 4 MiB 上限限制。
CPU snapshot 诊断输入仍需要一次 H2D，不能作为全 GPU 数据路径的性能样本。

严格顺序：

1. 等待源 producer event，GPU staging 拷贝在独立 stream 上执行并证明完成。
2. Mooncake 使用 `batch_transfer_async_write` 单次提交整个矩形，等待同一个 batch 的
   `get_batch_transfer_status` 成功，才发送控制完成帧。Qwen 的 `HANDOFF_FAST_ASYNC=0/1`
   和 DSV bulk 都使用此契约，不再调用会隐藏内部重试的 native sync convenience API。
3. 目标验证控制会话、sequence、矩形和 backend；GPU 路径按设备能力执行 RDMA visibility
   flush，然后才提交 CUDA 消费。TCP 路径仍检查 Host footer。
4. 完成目标 copy/scatter 和对应 CUDA event，才允许相关 slot 复用。
5. entry 的全部 token ranges 覆盖后才发布 Local-ready；DSV 继续由 retirement
   等待 forward/reader 证明。group 的提交/返回不能代替 entry 的完整覆盖。

GPU 路径不使用“看到尾部 footer”证明整个 RDMA batch 已到达；它依赖原生 **所有描述符**
成功完成后的控制通知，加接收端可见性处理。异步 batch status 的绑定接口本身会等待。
依据：[Mooncake Python batch API](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/api-reference/python/transfer-engine.md)、
[NVIDIA GPUDirect ordering](https://docs.nvidia.com/cuda/archive/13.0.0/gpudirect-rdma/index.html)。

## 预留、回退与回收

原有 plan → native reserve → source confirmation → freeze 流程保持不变。
目标消费原预留 row/KV，不因选择 RDMA 而重新分配；页对齐、部分尾页、FULL/SWA
窗口和状态资源的核验仍由原有原生适配器负责。

TCP fallback 只在数据提交前协商。已经提交的 RDMA 如果状态不明，不重传到 TCP，
也不切换 engine。源 staging/MR、目标 staging/MR，以及被迁移请求的相关所有权继续隔离。
断开连接、epoch 变化或超时都不是 DMA 完成证明。失败 batch 可能仍有 sibling DMA；
即使取消后迟到写完成，也不能仅凭 CUDA stream 同步或后续成功结果解除原隔离。

Qwen 注册 owner 绑定本次选中的 engine，正常释放和延迟 reaper 都调用同一个 engine
的 unregister，避免降级后误用原生 RDMA engine 注销 TCP region。GPU/Host 注册失败
没有返回 owner 时，仍保留 backing；后续 completion ticket 不能解除这种隔离。

## 验证与当前边界

见 [P1 修复验证报告](validation/d2d-rdma-p1-fix-4060.md)。真实 TCP、CUDA streams、
GPU staging 与模型原生小池 fixture 已验证；4060 上的 RDMA NIC 与外部写可见性是
明确的模拟边界，不能据此宣称真实 GPUDirect RDMA 验收或吞吐提升。

当前没有新增矩形之间的多 slot 流水线、原生 KV 页直写或 NIC QoS 调度；Qwen 仍在
一个矩形 native completion 和目标 ACK 后推进下一矩形。真实 RDMA 双机验收应使用
相同 Mooncake 版本，覆盖两种模型布局、GPU/NIC 拓扑、混合能力 TCP 回退、取消、
完成通知丢失和注销失败，并记录迁移期间的 Decode P99 与首次 Local-ready 延迟。
