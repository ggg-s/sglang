# Remote Attention Q/O/LSE：协商式 Mooncake 数据面

适用 Qwen3-8B 与 DSV4、DSV4.1。原生 PD identity、KV allocator、预留后冻结、
D2D layer group × token range 与逐 entry Local/retirement 规则保持不变。
本实现覆盖 Remote Attention 的 Q 请求及 O/LSE 返回，复用 D2D 的已验证 engine
身份、能力交集与单次提交契约；RemoteHub socket 只发送控制元数据。

## 开关与部署

```bash
export HANDOFF_REMOTE_GPU_TRANSFER=1  # 启用此数据面；0 保留旧控制/张量路径
export HANDOFF_REMOTE_BACKEND=auto   # auto / tcp / rdma，默认 auto
```

`HANDOFF_REMOTE_BACKEND` 独立于 D2D bulk 的 `HANDOFF_D2D_BACKEND`：

- `auto`：双方都有实际 RDMA transport、GPU 注册/注销与外部写可见性能力才选 GPU；否则使用注册 Host staging + Mooncake TCP。
- `tcp`：只提供 Host staging，GPU 指针不交给 TCP engine。
- `rdma`：只允许 GPU staging RDMA，没有共同能力就在提交前拒绝。

两端必须一起升级。新的 Q/O wire 为 `remote-mooncake-v1`，内含
`d2d-mooncake-v2` 的能力描述；旧的未协商 `te_open` 会被拒绝。
仅支持已验证的 classic 单 transport 描述。TENT/TEV1/SHM 或无法确认的协议
不会获得 GPU 能力，详见 [engine 初始化与回退](D2D-RDMA-ZH.md)。

## 连接与身份

1. `hello` 建立 RemoteHub 控制连接。
2. `te_negotiate` 携带 route/token、原生 `native_session`、能力和每个 backend 的 session；源端先验证 native PD peer，再选择能力交集。
3. 选定 backend 后才分配/注册双方 slab；`te_open` 固定该 backend/session 和注册区。
4. 每条 `te_attn` row 继续验证 token/epoch 和原生 session。TCP fallback 的 transport session 不冒充原生 PD 身份。

连接中禁止重协商或替换 endpoint。任何原生提交开始后都不切换 backend、不以 TCP
重传；新请求可以建立新连接/新 slot，旧请求的不确定状态不会因此解除。

## 数据、stream 与完成证明

每个连接的 Q/O/LSE 工作区独立于原生 KV 页。GPU RDMA 注册 GPU slab；TCP 注册
pinned Host wire slab（CPU 契约测试使用普通 Host tensor），另保留 GPU 工作区。

```text
GPU RDMA: producer event → GPU staging → 单次 batch → status → control
          → GPUDirect visibility → GPU 消费

TCP:      producer event → 独立 copy stream D2H → event完成 → 注册 Host
          → 单次 batch → status → control → 各 region footer
          → 独立 copy stream H2D完成 → GPU 消费
```

Q 是一个 descriptor；O 和 LSE 是两个 descriptor，但同属一个单次 batch。
每个 TCP descriptor 连续包含 payload 和 32-byte footer，绑定接收方 nonce、sequence、
region、长度。发送方完成不能代替接收方到达证明；收到全部对应 footer 后才读 Host。
RDMA 以所有 descriptors 的成功状态及接收端 visibility 为证明，不依赖 footer。

两条路径都通过 `batch_transfer_async_write` 单次提交，等待同一 batch 的
`get_batch_transfer_status`；不调用可能隐藏内部重试的同步 convenience API。

目标返回 owned O/LSE，避免别名到可复用 slab。下一次 Q 和 close 等待该 owned copy
事件；ACK 只推进连接状态，不能代替 CUDA 读完或原生 DMA 的证明。
CUDA 拷贝异常时，`copy_owner` 持有 query、输出或 owned clone，避免提前释放输入。

## 取消、异常与 Local

- 注册前将 slab 放入 engine 级持久 owner；部分注册失败、Plane/Client 替换或 GC 都不会丢失 backing。
- 地址已发给对端，即使 `te_open` 回复丢失，也保留注册和 backing。协商前失败没有已发布 slab，新请求可重新协商。
- submit/status 不明、CUDA fence 失败、返回期间断线/取消：slot quarantine，RequestIO poison；禁止复用/注销/跨 backend 重试。
- 返回 DMA 失败时源 Job 保持 registry reader 引用，阻止源 KV 回收。成功完成源 CUDA reader 与返回 DMA 后可释放该引用，因为返回位于独立 slab。
- Qwen 与 DSV 的逐 entry Local/retirement 仍检查已有 reader/forward/RequestIO 证明；网络故障、ACK、epoch 改变均不构成额外完成证明。
- 未知完成的隔离不会因为观察到一次迟到写或同步 CUDA stream 自动解除；没有 NIC drain 证明时持续保留。

配置的容量上限与请求 IO 规则沿用现有数据面；没有新增多请求 slot 流水线或修改模型
计算。性能收益必须在目标 GPU/HCA 与真实负载下测量。

## 验证

CPU：`test_remote_attn.py`、`test_d2d_mooncake_backend.py`、
`test_remote_mooncake_data_plane.py`。
GPU：`qo_mooncake_cuda_probe.py`，另复验 D2D range 与 DSV 逐 entry Local 探针。

最终结果见 [4060 验证报告](validation/qo-mooncake-4060.md)。真实 TCP 和 CUDA
执行覆盖与模拟 NIC 故障边界分开记录；4060 不能替代真实双机 HCA/CQ、GPUDirect
外部写可见性、多 GPU 推理等价性或性能验收。
