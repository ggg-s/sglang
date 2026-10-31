# Remote Q/O/LSE：原生 Mooncake GPU 数据通道

在 P/D 全部 TP ranks 的既有环境块中增加：

```bash
export HANDOFF_REMOTE_GPU_TRANSFER=1
```

使用原生 SMG/P/D 启动命令，保留已有 graph、overlap、缓存、optimistic prefill、chunk
和 KV/state staging 配置。同机/跨机多 P/D 继续使用 [网络 journal](MULTI-PD-ZH.md)。
默认值为0，保留旧 TCP payload 对照；设1后，Remote RPC 不自动回退 TCP 张量传输。
P/D 能力不匹配的请求不接管，保留原生 PD 传输，不能据此判断已走 GPU Remote。

## 数据路径

- **Q、O、LSE**：使用该 rank 原生 KV manager 已持有的 `MooncakeTransferEngine`，
  通过注册 GPU 缓冲区及 `batch_transfer_sync` 搬运；不另建 engine，不加入 KV 指针清单。
- **TCP**：RemoteHub 只交换认证、GPU region、序号、shape/dtype、sparse positions 与
  `te_open/te_attn/te_ack/te_close` 控制消息，不发送 Q/O/LSE 字节。
- 首先支持逻辑 batch 1/2/4；MHA 按 P endpoint 分组，每帧最多4行；DSV 保留逐行 sparse
  RPC。大组按最多4行拆分，未增加 graph buckets，其他 shape 的图策略保持原样。
- Q 为 BF16/FP16，O 保持原计算 dtype，LSE 为 FP32；不改变 attention 或 merge 数学。
  MHA 和 DSV executor 直接消费 GPU Q、返回 GPU O/LSE。动态 sparse、RPC、merge 仍在图外；
  所有租约退休后恢复 native full graph/overlap。

每个 D/source endpoint 使用一个串行复用 lane，两端各注册约6.02 MiB GPU slab：
Q 2 MiB、O 4 MiB、LSE 16 KiB。每 rank 最多16个 peer/server lanes，故障隔离也有上限。
GPU region/TE session 只在经过 native PD peer 认证的连接建立时交换；RPC 只引用该
连接的固定 region，逐行检查 handoff token/epoch 与 native D session，不接受任意写地址。

## 完成、取消与回收

1. D 将 Q 复制到注册 slot，记录并等待生产 event，然后提交原生 Mooncake 写入 P。
   同步返回成功后才发送 `te_attn`，不会把仅排队的 CUDA copy 当成数据就绪。
2. P 建立接收可见性，持有 registry reader 引用，使用原 RemoteStreamPool 执行 attention。
   O/LSE 写入注册区并经过 GPU event 完成后，原生 Mooncake 回写 D；成功后才发送结果通知。
3. D 建立接收可见性，把 O/LSE 复制为独立 GPU tensors 后交给 merge/图桥。记录最后一次
   slot读取 event；下一次复用、关闭/注销均等待该 event。图内不会引用可被下次 DMA 覆盖的 slot。
4. `te_ack` 确认上一序号结果已接收；P 只从 `returned` 转回 `idle`，未完成、重复或跳号
   均拒绝。第一版显式确认，尚未优化成 piggyback ACK。
5. GPU 传输账本并入 D 的 `gpu_ready` 和 Target.close。Native IO、GPU forward/sampling、
   transport consumer 等全部证明就绪后，才走原有有序 TP 退休。P cache read lease 和
   registry inflight 引用保留至 GPU reader/结果传输已证明完成。
6. 取消中的请求不会越过这些条件。超时、断线、传输失败或完成不明时，相关 GPU region
   **隔离并保持注册/引用，不按 epoch 直接复用**；失败请求不自动重试。新请求只能用不同
   地址的新 lane，达到隔离上限则报错；不强制释放、重置共享 engine 或影响其 KV 注册。
7. 注册、注销、写入沿用 manager 的发送互斥，GPU event 等待在锁外；不套用 KV 的
   host-staging 包装或占用 KV FIFO slot。后台工作不调用 TP collective。

网络与 GPU 数据通道配置进入既有 TP 启动共识。运行期仍由有序切换命令做最终共识；
稳定 Remote、Local 和 P middle→final 不增加逐层/逐 token TP 投票。

## NVLink / RDMA

底层传输沿用原生 Mooncake 配置及硬件能力，不由插件把一种链路伪装成另一种链路。
RDMA 的接收端查询 GPU writes-ordering 能力；必要时执行
`cudaDeviceFlushGPUDirectRDMAWrites`。缺少可证明的可见性能力则拒绝启用，不能仅靠
普通 CUDA event 假定第三方 DMA 已对 GPU 可见。NVLink 的 allocator、注册和完成语义
同样需要用实际安装版本及目标硬件验证。

本次4060使用 **Mooncake 自身的 TCP 后端** 读写 GPU地址；RemoteHub socket 已无张量
payload，但不能据此宣称物理数据走 NVLink/RDMA 或证明对应链路的性能收益。

## 确认与验证

启动日志含 `[remote-gpu] native Mooncake GPU payloads session=...`。
实例的 `controller.gpu_data_plane.stats()` 提供传输次数、字节数、lane 和隔离数量；只看
启动日志不代表已有请求实际走过此路径。需要结合实际 transfer 计数和 RPC/退休证据。

[本地、TP2、4060 graph 与 GPU 生命周期证据](validation/remote-gpu-transport-4060.md)。
该验证尚未覆盖完整模型、双物理 GPU 节点、NCCL 或生产 TTFT/ITL/吞吐。
