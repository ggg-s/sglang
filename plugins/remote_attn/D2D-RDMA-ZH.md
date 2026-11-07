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

## 控制面与迁移窗口

Gateway 在首个已采样 token 可见、请求可安全规划时就执行 plan/reserve；
不再等 `cut` 才开始三跳串行控制。预留成功进入持久化 `WAIT_CUT`，源继续生成，
到 `cut` 或显式请求迁移时才进入 FREEZE。同一 epoch 支持等待、重启恢复和取消。
plan 不要求 native Decode 边界；freeze 仍必须在调度器线程的安全边界执行。

提前预留需覆盖源继续生成期间的长度变化，因此预留上界至少覆盖剩余输出上限，
由 worker 的采样上限与原生 context 容量截断。相比临近 cut 才预留，这会更早占用
D1 row/KV；目标容量不足时沿用原有失败/回滚流程。源提前结束或取消会撤销目标
预留，不能让 ticket 隐式过期。实际 import 只消费冻结长度，多预留的尾部按原生
页对齐规则退还。

空闲入口的原生 zmq Poller 增加管道唤醒：后台 watcher 每 5 ms 只检查控制文件名，
不读取请求、不执行命令、不访问 KV 或 TP。命令仍经原生 request broadcast 由
所有 TP rank 的调度线程顺序执行。Rust 入口保留原生 ring 等待，空闲等待上限
缩为 10 ms。该机制唤醒空闲 poll，不能抢占正在运行的 GPU forward。
Gateway 的 status 轮询遇到排队中的修改命令会让行，驱动收取 D2D 回包的间隔为 5 ms。

`/v1/router/status` 增加累计桶直方图：`control_latency_ms` 按 role/command 记录
角色锁等待与 RPC 总耗时（含失败尝试，进程重启清零）；`handoff_latency_ms` 记录
freeze 首次提交至目标 commit/admit 成功的耗时，从 journal 恢复。每请求还暴露
`prepare_latency_ms`、`handoff_latency_ms` 和 `migration_skip_reason`。
worker 发现请求已结束时返回结构化 `request_finished`，Gateway 撤销预留并报告
`migration_outcome=window_missed_request_finished`；语义拒绝单独报告
`semantic_rejection`，可重试的 native 边界等待仍使用 `migration_deferred`。
`migration_skip_reasons` 汇总当前 journal 中的跳过原因。

验证与 H20 复测边界见 [控制面窗口验证](validation/D2D-CONTROL-LATENCY.md)。

## 数据路径与完成顺序

DSV Remote 请求可在仍依赖 P 的部分 global KV entries 时进入 D2D plan。
只接受已通过 indexer/SWA/carry/metadata barrier、持有原生 progressive PD receiver
并登记有序退休协议的请求；Decode radix cache 仍必须禁用。

目标预留成功后，D0 在原生 Decode 边界提交待处理采样结果，冻结输出历史和 KV
长度，并将请求移出 runnable batch。其余请求可继续执行。此时原 P→D0 的
receiver、ring、页表和 Remote RPC lease 全部保留，原生循环继续处理到达的
chunk 和 deferred scatter。不能在调度器控制命令里同步等待，否则可能阻塞
完成这些 scatter 所必需的调度循环。

只有 canonical Success、所有 scatter/GPU consumer 以及 TP 有序退休均完成，
旧 receiver/metadata/ring 和 P Remote read lease 已退休后，D0 才发布 D2D manifest
及读服务。D2D 从冻结的 D0 页读取；不会让 D1 链式读取仍由 P 写入的页。
因此不再等待运行中的请求自然进入 `all_local()` 窗口，但冻结至导出仍需等待
P→D0 真正排空；本改动不承诺迁移延迟或性能收益。

等待旧 PD 退休时，`d2d_freeze` 幂等返回 `frozen=true, pending_retirement=true`。
Gateway 保持同一 epoch 的 FREEZE 阶段，重试不会再冻结或消耗尝试次数。
journal、router status、drain status、SSE status 暴露 `freeze_pending` 和
`source_partial_remote`；worker migration status 还提供冻结时的 Remote entries、
KV 长度与 prepared 状态。Gateway 重启或丢失 freeze 回包后沿用同一 epoch。

commit 前取消：先取消目标预留/import，再撤销 D2D 读服务并恢复 D0；旧 PD
尚未退休时，保留原 receiver/RPC，继续正常接收。接收、GPU 或 TP 证明失败时
冻结请求保留 KV，不导出、不通过 resume 绕过失败。commit 后只有 D1 生成后续
输出，D0 要等全部 D2D reader 和复制完成证明才释放一次原生 KV。

不具备安全冻结条件时（例如 native Decode 边界未到或非原生 Remote owner），
worker 仍报告 `migration_deferred_reason`；Gateway 的 `migration_deferred`、
reason、count、wait_ms 继续表示**冻结之前**的等待。TP plan 返回 `deferred=true`
时不会创建 plan、预留或冻结，也不消耗迁移重试次数。未进入准备阶段就在源
结束仍标记 `migration_outcome=completed_on_source_before_migration`；准备后在 cut
之前结束标记 `completed_on_source_before_cut`，错过冻结窗口则报告
`window_missed_request_finished`。实际切至 D1 才计为迁移。

小池验证结果与 H20 复测边界见
[部分 Remote D2D 验证](validation/DSV-D2D-PARTIAL-REMOTE.md)。

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
