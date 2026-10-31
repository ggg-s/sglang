# DSV4.1 Remote sparse：提前接纳与逐资源 Local

`HANDOFF_PRE_LAYERWISE=remote` 现在允许 D 在 **global KV 尚未传完**时开始 decode。
先完成 indexer、SWA/carry 和 metadata 屏障，再由 P 提供缺失 global 前缀的 sparse
attention；每个 physical global 资源复制完成并经过 TP 有序共识后独立切 Local。
生产代码只改插件，SMG/SGL 核心无新增修改。仍是 eager 参考实现，未证明生产性能收益。

## 数据和请求链路

1. P 原生 middle chunk 及其传输 FIFO 保留。最终 chunk 完成 forward 后，插件通过
   原生 `send_kv_chunk` / `CommonKVSender.send` 获取 authoritative state indices、
   metadata 和已预留的 final page slice；没有复制模型的 SWA/carry planner。
2. 最终 chunk 先发送全部 indexer payload/scale 条目。所有对应 native TE callback
   完成后，在仍计入 native outstanding/IO ledger 的 worker callback 内，调用原生
   `maybe_send_extra` 和 `send_aux` 发送 SWA、compression/indexer carry、metadata。
   不能以“最后一个编号的 chunk 完成”替代全部 indexer 完成集合。
3. P 发布本代 `sparse-ready.json`，再发送 global KV，并保留 canonical final Success。
   final 不重发已交给 D 更新的 SWA/carry，也不再次 DMA metadata。P 的 HTTP leg
   可以用 response-only snapshot 提前返回：需要各 TP rank 的同代屏障；不改变 native
   Req、sender 状态或 KV 所有权，没有新增 response all_reduce。
4. D 必须同时看到：本代 P state/metadata 屏障、**全部 indexer/scales 的 native scatter
   CUDA event 完成**、匹配的 metadata bootstrap room、有效 receiver 和未取消请求。
   既有 TP admission MIN 一致通过后，调用 native `_commit_transfer_to_req(...,
   keep_receiver=True)`，从 transfer queue 移出请求，保留 receiver 和 metadata slot。
5. D 继续原生 compressor/indexer、candidate/top-k 和 SWA 更新。选中的 physical
   sparse slots 按请求 FULL page map 分成 P logical 前缀与 D 新增行，顺序和重复项
   保留。P 只读 `< floor(prompt_len / ratio)` 的完整压缩行；ratio-2 未完成 half-pair
   及其接续始终归 D。sink 仅由 D 计一次。
6. global KV 通过 native receive ring/scatter 写入 D。完整 immutable 页可整体写入；
   最后一页仅复制 P 的有效压缩行，分别处理 native data/scale 地址，保留 D 新行、
   carry 接续及末页 padding。indexer 在接纳前复制一次，接纳后不覆盖 D 的更新。
7. 每个 global physical entry 的 GPU 完成事实只产生本地通知。既有 request receiver
   广播同一有序退休命令；各 TP rank 对命令规定的同形向量做最终 MIN 后，才把资源
   标为 Local。共享消费者通过 native `source_layer_of` 复用同一生产者资源。
   稳定 Remote 观察不做逐轮退休投票，Local 稳态不增加同步。
8. Remote 期间 D 串行运行；全部资源 Local、canonical Success、native scatter 完成、
   forward/sampling/readback GPU proof 和取消 release 状态一致之后，最终退休清理
   receiver/metadata、关闭 RPC、写 `remote-done.json`，再恢复原生 overlap/full graph。
   重牵引在 TP 对称边界先 drain 资源、再 drain 请求 lifetime，之后才能 backup/free/reset。

P registry 保留 token/epoch、attempt/row、cache pin 和真实 GPU reader ref 保护。D
结束不等于 P 可立即释放：tombstone 先禁止新读，最后一个 `RemoteJob.finish` GPU fence
与 read ref 归零后，才能沿用 P overlap 的最终 TP 退休回收 FULL/SWA cache lease。
DMA timeout、scatter/forward proof 失败均保留页面，拒绝不安全复用；没有跨进程故障恢复。

## 原生启动

沿用 [原生 SMG/P/D 参数数组](DSV41-LAYERWISE-ZH.md)。保留 P FULL/SWA cache、P/D overlap、
optimistic prefill、chunk 和 transfer batch 的线上原值；D radix cache 保留既有限制。
纯文本、同机单 pair、同 TP、PP=DP=CP=1，关闭 DSpark/HiCache；使用 `flashmla`，
不启用 mega/TRTLLM、LoRA、unified KV、whole-prefix checksum 或 mixed chunk。

两端公共环境：

```bash
export HANDOFF_PRE_LAYERWISE=remote
export SGLANG_DISAGG_STAGING_BUFFER=0  # 原生 MHA staging 开关保持关闭
# HANDOFF_PD_DIR 指向同一份本机目录，每轮启动换新目录。
```

P 继续原生命令：

```bash
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=1
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=0
python -m sglang.launch_server "${P_NATIVE_ARGS[@]}" \
  --language-model-only --dsv4-attn-backend flashmla
```

D 继续原生命令：

```bash
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=0
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=1
python -m sglang.launch_server "${D_NATIVE_ARGS[@]}" \
  --language-model-only --disable-radix-cache --dsv4-attn-backend flashmla
```

backend 参数已有时不重复追加。SMG 保留原生 PD router 命令，不加载插件变量，无 JSON
包装器。P send staging 可关闭。D receive host staging 关闭时仍需要 **GPU receive ring**：
progressive 不能直接把整页 DMA 到正被 decode 修改的末页。ring 使用原生
`SGLANG_DISAGG_STAGING_POOL_SIZE_MB`，需计入 GPU/host 内存预算，不是无额外内存方案。
若一个 final-chunk physical entry 超过 ring 容量，请求在接管前回退完整 native PD，
不发布 early-ready；不会把 unsafe direct-page write 当作 progressive fallback。

短 prompt、bootstrap/资格不匹配等继续原生 PD；旧 wire/version 的 peer 也不进入本接管路径。
Remote 不捕获 DSV 分段 graph。带 Remote 的混合 batch 全部行用 eager sparse union 计算，
避免 native whole-batch kernel 误读尚未就绪的 global 页；SWA store 只执行一次。
全部 Local 并退休后恢复原生计算。此参考路径有 CPU 序列化与同步开销。

## 确认路径

- `remote-service.json`：`boundary=progressive-sparse`，同代 route/token/epoch。
- `sparse-ready.json`：P 的 indexer TE + native SWA/carry/metadata 屏障。
- `admitted.json`：D 的 indexer GPU/metadata/TP 屏障已通过；不能只凭 P ready 认定已接纳。
- `resource-*.json`：P 的 native TE 事实；不能替代 D GPU/TP Local 证书。
- `local-<physical-entry>.json`：`completion=native-scatter-gpu`、`tp_agreed=true`。
- `remote-done.json`：最终退出与 RPC 计数；`done.json`：D receiver/metadata 清理及 GPU proof。
- `released.json`：P native IO、reader fence、cache pin 和 TP 退休全部完成。

验收需证明 **实际 RPC 发生在 canonical transfer Success 之前**，再证明逐项 Local 后
对应 producer 不再 RPC；只看到 `remote_calls > 0` 不足以证明提前 decode。

[本地/TP2/4060 验证记录](validation/dsv41-progressive-4060.md) 区分真实运行部分与 fixture。
4060 不支持完整模型的 SM100 FlashMLA/indexer/compressor kernels；尚未运行完整 checkpoint、
SMG/P/D、多 GPU NCCL，也未证明 token/logprob 一致性或生产 TTFT/吞吐收益。
