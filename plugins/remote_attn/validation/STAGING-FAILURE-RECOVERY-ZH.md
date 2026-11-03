# TCP staging 请求级故障热修

2026-11-03，原分支 `codex/remote-attn-pre-layerwise`。未修改 Mooncake C++。

## 修复范围

原来的非零 TE 返回会 poison 共享发送 slot；后续发送抛通用异常，导致 native
transfer worker 退出。P/D 插件又将请求的 `KVPoll.Failed` 当作进程级错误。

现在分三层处理：

1. TCP 发送按 descriptor 数量分批，而不只按 token/字节限制。默认每批最多 256
   个 descriptor，并限制到显式 TCP queue/admission 上限的一半，给控制流留下余量。
   可通过 `SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_DESCRIPTORS` 调整正数上限；只在
   `MC_FORCE_TCP=1` 时生效。任一批失败就停止剩余批次，不额外重试。
2. 当前失败 slot 不注销、不释放、不覆盖。后续发送另取 slot；退役最多 8 个，
   默认每个 128 MiB，共 1 GiB pinned storage。达到上限后明确失败 staged 请求，
   不再分配 slot，也不改用 GPU 源地址。`StagingTransportError` 只失败当前 room，worker
   继续处理后续 room；CUDA copy 错误、布局错误等仍保留原有 fatal 检查。
3. P/D 在已有 TP 一致边界将已知 transport failure 隔离为请求级 503，保留该请求
   的 KV、cache pins、RPC 状态、ring allocation、metadata row 和延迟释放 callback。
   不把隔离伪装成成功退休或 Local 就绪，不改变后续请求的 Remote/layerwise 配置。
   D 的未知写入请求从 native 超时释放队列中移出，防止 metadata
   行或 ring 槽位被迟到写入覆盖。

未知完成状态不能靠重发相同字节证明安全：旧写入可能晚于重试成功和 CHUNK_READY，
落到已复用的 ring 位置。Peer probe 只用于原生 session 恢复，不作为旧 DMA 排空证明。
正常取消仍按已有 GPU/IO drain 规则处理，不把已排空的取消当成网络故障。

## 运行边界

这版按局部热修收敛：专用异常、单请求失败、有界 slot 退役、worker 继续服务。
已撤销之前版本的自动 GPU 源回退与整实例原生 PD 降级。原生 session probe 保留，
但不是旧写入排空证明。当前 TE 的通用非零返回不能区分未提交与超时，因此没有照搬
“所有错误都原地重试三次”；描述符分批用于减少 queue-full。

8-slot 上限约束 P pinned slot；失败请求持有的 KV/cache/ring 资源另计，数量由故障
发生时的插件在途请求决定。隔离资源继续占用容量，D ring 也可能因未决 reservation
阻塞后续分配；不能承诺只损失一个请求后就完全恢复容量。预算耗尽、ring 无法推进时
需要重启相关进程回收。扩大 TCP 队列不能代替这些完成/释放约束。

## 验证

- CPU 契约 228 项通过，覆盖 worker 遇到 staging 异常后继续发送另一 room、不重试
  未知写入、旧 slot 内容/注册保持、八次失败后拒绝额外发送、descriptor 分批、请求隔离、
  保留释放 callback，以及正常取消。
- 真实两进程 Gloo：只有一个 rank 发现传输失败，其他 rank 仍在等待 callback；
  最终用同一 quarantine command、一次固定形状投票隔离请求，无成功退休或 KV free。
- RTX4060 / PyTorch 2.13.0+cu130：真实 CUDA D2H 和 Mooncake TCP，两个 TE endpoint。
  queue/admission=8，2048 descriptors 分成 512 次调用，每次最多 4 个，数据逐字节一致。
  TE 调用边界注入 8 次错误返回，退役存储内容不变，预算内后续真实传输正确；耗尽后
  不再调用 TE，不回退 GPU 源。
  探针使用 64 KiB slot，8 块共 512 KiB；没有声称实测了默认 1 GiB 的内存压力。

[GPU 原始结果与源码 SHA256](staging-recovery-4060.json)。运行方式：

```bash
python test/registered/unit/plugins/test_remote_attn.py
python test/manual/remote_attn/staging_recovery_cuda_probe.py --sglang-root . --output /tmp/staging-recovery.json
```

探针的错误返回是注入的，不是实际制造 Mooncake queue-full 后的迟到 DMA。
H20 SWE 300/C=4/4P4D、真实队列拥塞、故障后的长期容量与服务性能仍需目标部署验证。
