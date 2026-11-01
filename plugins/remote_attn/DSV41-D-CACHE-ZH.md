# DSV4.1 D 端 radix cache

复用公共 `destination_cache.Guard` 和原生增量 PD，无新增 SMG 或 SGL 核心修改。
纯文本、DSpark 关闭、同 TP、非 unified FlashMLA 池以及既有 compression ratio
0/1/2 限制保持。支持设备内 FULL 或 FULL/SWA `UnifiedRadixCache`；HiCache、
外部 linker、session refs、其他混合缓存组件仍不开放。

## 原生启动

沿用 [DSV layerwise](DSV41-LAYERWISE-ZH.md) 或 [Remote](DSV41-REMOTE-EAGER-ZH.md)
中的 P/D/SMG 命令和公共插件环境。**只在 D 命令**删除 `--disable-radix-cache`，
加入原生参数：

```bash
--disaggregation-decode-enable-radix-cache
```

只删除 disable 参数不会打开原生 D cache；无需新插件变量。保留线上 optimistic
prefill、chunk、传输 Batch、P 缓存、P/D overlap 和原生 graph 配置。Remote 图仍
只捕获原生桶中已有的 1/2/4/8/16，动态 sparse/state、RPC 和 merge 留图外；全部
租约退休后恢复原生 full graph。D Remote 仍串行，最终 Local 才恢复 native overlap。
关闭 D overlap 的对照仍保留 GPU 完成证明和有序退休。

## 各资源的归属

| 资源 | 缓存与传输 | 保护和释放 |
|---|---|---|
| global KV | FULL 页命中 `[0,H)`；仅传输 `[H,N)` | 等待及活动租约各自持有 FULL pin；完整完成前禁止 checkpoint/dedup/rebind/free |
| indexer payload/scales | 与 global 使用同一逻辑页归属，命中 H 内部分不重复传输；缺失尾部先于 Remote 准入完成 | 复用 FULL pin 和 native allocator；native top-k 仍在 D 执行 |
| SWA | 复用原生准入对 H 的窗口上限，完整活动窗口重新分配或使用请求独占 request-window | 不将旧 SWA 缓存页纳入 D 的额外 pin；当前窗口随请求行保护，退休前不得回收 |
| carry / 压缩 pending state | 原生 final state 接口按本次请求行传输，屏障之后仅由 D 更新 | 与请求行共享生命周期，canonical final marker 不重复覆盖；不是共享的缓存快照 |

H 必须是原生逻辑页大小的整数倍，并在发送 destination metadata 前经过 TP 前缀
共识。容量重试复用固定证书和 pin，不重新匹配或增加逐轮 collective。request-window
模式的 SWA destination 使用原生 `RequestWindow.page_indices` 按 D 请求行生成，
不能从共享 FULL 页映射推导；carry 继续保留原生 state indices。

P 的 sender cursor、总页数和 receive ring 容量判断均扣除 H。wire staging cursor
相对 suffix，scatter 的逻辑位置加回 H；global 部分末页仍只写完成的压缩行，保留
D 新增行、carry 和 padding。indexer 只写缺失尾部。sideband 和 remote-service
屏障携带同一 H，避免跨缓存尝试接纳。

Remote sparse 处理保留 native top-k 的顺序和重复项：命中前缀内的选中位置由 D
读取，未传完的 suffix 位置交给 P，decode 新增行由 D 读取。某 global entry 的
尾部 scatter 完成并经有序 TP 共识后，该 entry 整体切 Local；这一步不发布缓存。

## 发布、取消与 TP 退休

复用 Qwen D 的公共保护层：只有 native receiver、scatter、Remote GPU 数据面和
forward/sampler/readback 都完成，并确认各 rank 发布/ABORT 意图一致，才调用原生
checkpoint/dedup/release。活动请求也会在最终退休发布完整 prompt；不发布尚未计算
的输出。原生缓存操作后的结果共识成功，才允许恢复 Local full graph/overlap。

取消保持 sticky ABORT，晚到 SUCCESS 不得插入尾部。retraction 在 native backup
之前排空，再 reset 并推进原生 cache request handle，旧匹配和 ABORT 不污染新尝试。
若 DMA 完成未知或缓存发布部分失败，保留相关所有权并停止恢复 Local，不能假装回滚。

逐资源 Local 及最终退休沿用既有有序命令。缓存最终发布多一次命令内的执行结果
投票；等待证明和 Local 稳态不增加逐步 all_reduce。固定匹配、独立 pin 及延迟发布
会增加短时 KV 占用，也可能错过等待期间新增的更长前缀，需在目标负载量化。

## 核对与验证范围

同一 D 重复发送共享长前缀，检查 native `decode_prefix_len=H>0`、实际 suffix
页数、`sparse-ready.json` / `remote-service.json` 的 H，以及各 rank 的 `done.json`
中 `d_cache_guard_closed`、`d_cache_tp_committed`、命中长度和 ABORT 状态。
仅看到服务启动或首 token 提前返回不构成缓存正确性证据。

本次仅本地 CPU 与真实 Gloo TP2。GPU event、TE 和 graph 执行在边界使用替身，
没有完整 DSV checkpoint 推理、真实 CUDA/RDMA 或性能结论。投产前需在目标 GPU
对比 cache off/on、overlap off/on 的 greedy output_ids，覆盖多请求共享前缀、
逐资源 Local、淘汰、取消和 retraction，并验证页容量恢复、TTFT / ITL / 吞吐。
本地证据见 [验证记录](validation/DSV41-D-CACHE-LOCAL.md)。
