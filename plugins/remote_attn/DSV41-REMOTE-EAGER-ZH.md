# DSV4.1 Remote sparse：提前接纳与逐资源 Local

P 缓存前缀（Qwen3/DSV4.1）：[统一 Remote progressive 与 A/B 对照](PREFIX-REMOTE-PROGRESSIVE-ZH.md)。

`HANDOFF_PRE_LAYERWISE=remote` 现在允许 D 在 **global KV 尚未传完**时开始 decode。
先完成 indexer、SWA/carry 和 metadata 屏障，再由 P 提供缺失 global 前缀的 sparse
attention；每个 physical global 资源复制完成并经过 TP 有序共识后独立切 Local。
SMG 无新增修改。D Remote 支持 batch 1/2/4/8/16 分段 CUDA graph；稳定 QKV 投影通过少量
SGL 模型接口进入图内，动态状态更新、sparse、RPC 和 merge 留图外。
动态 sparse、RPC、merge 仍在图外；默认使用 GPU 分区/融合计算与异步批量 RPC。
参考路径保留为对照；4060 算子验证不代表 H20 整模性能收益。
Remote capture 显式使用 `thread_local`，Marlin reduction workspace 按图和调用在
warmup 预分配，保留 bs=1；见 [H20 捕获修复及验证边界](validation/H20-REMOTE-CAPTURE-FIX.md)。
低压缩率 index 池现按完整逻辑 guard 页分配，解决 staging 尾条不对齐；
见 [分配、末页传输及边界验证](validation/DSV41-STAGING-ITEM-ALIGNMENT.md)。
Q/O/LSE 可启用 [Mooncake GPU 数据通道](REMOTE-GPU-TRANSPORT-ZH.md)，控制消息继续走 TCP；
默认仍保留旧传输作为对照。

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
optimistic prefill、chunk 和 transfer batch 的线上原值；D radix cache 可按 [专用说明](DSV41-D-CACHE-ZH.md) 显式开启。
纯文本、同机单 pair、同 TP、PP=DP=CP=1，关闭 DSpark/HiCache；使用 `flashmla`，
不启用 mega/TRTLLM、LoRA、unified KV、whole-prefix checksum 或 mixed chunk。

H20/Hopper 保留原生 `KVLayout.V4` 池，两端使用 `SGLANG_DSV4_KV_LAYOUT=auto`
或 `v4`，不要强制 `v41`。Remote sparse reader 支持 V4 的 448 维 FP8 NoPE +
64 维 BF16 RoPE：按原生页地址读取七个 per-64 UE8M0 scale，忽略第八个 padding
字节和页尾 padding。V41 的 FP8/FP4 reader 继续保留；TRTLLM uniform-FP8 不在此
适配范围。P/D 实际 wire schema 必须一致，不能混用 V4 与 V41 池。

Hopper 的状态更新沿用核心现有实现：C1 非融合写入、C2 独立 pair-pooling/ring
kernel，indexer 使用非 SM100 融合写入分支。分段图只捕获投影，SWA、C2 carry、
indexer/top-k 仍在图外用 live metadata 更新，裁掉 graph bucket 的虚拟行。
实际 KV layout 和 compressor fused/gate 策略进入 TP 启动共识；不支持的 layout
在原生图后端构建前拒绝。V4 支持不代表已经完成 H20 整模或生产性能验收。

不依赖 SM100 的 CUDA 正确性探针（可在 4060/Hopper 上运行）：

```bash
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/dsv41_hopper_cuda_probe.py \
  --source-root . --output /tmp/hopper-kernels.json
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/dsv41_remote_cuda_probe.py \
  --sglang-root . --kv-layout v4 --progressive --output /tmp/hopper-remote.json
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/dsv41_segmented_graph_probe.py \
  --source-root . --kv-layout v4 --serial-prepare --output /tmp/hopper-graphs.json
```

[V4/Hopper 本地与 4060 验证记录](validation/dsv41-hopper-v4-4060.md) 含原始结果和源码哈希。

第一个探针运行原生 V4 Triton quantize/store/dequant 和 C2 ring kernel；后两个
探针验证真实 RemoteHub/CUDA graph，但模型投影、norm/store 以及 indexer 规划仍有
小型 fixture 边界，不替代 H20 checkpoint、FlashMLA、SMG/P/D 或多 GPU NCCL 验收。

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
  --language-model-only --disaggregation-decode-enable-radix-cache --dsv4-attn-backend flashmla
```

backend 参数已有时不重复追加。SMG 保留原生 PD router 命令，不加载插件变量，无 JSON
包装器。P send staging 可关闭。D receive host staging 关闭时仍需要 **GPU receive ring**：
progressive 不能直接把整页 DMA 到正被 decode 修改的末页。ring 使用原生
`SGLANG_DISAGG_STAGING_POOL_SIZE_MB`，需计入 GPU/host 内存预算，不是无额外内存方案。
若一个 final-chunk physical entry 超过 ring 容量，请求在接管前回退完整 native PD，
不发布 early-ready；不会把 unsafe direct-page write 当作 progressive fallback。

短 prompt、bootstrap/资格不匹配等继续原生 PD；旧 wire/version 的 peer 也不进入本接管路径。
D Remote 分段图的启用要求见下节。带 Remote 的混合 batch 全部行用 eager sparse union 计算，
避免 native whole-batch kernel 误读尚未就绪的 global 页；SWA store 只执行一次。
全部 Local 并退休后恢复原生计算。此参考路径有 CPU 序列化与同步开销。

## D Remote 分段 CUDA graph

复用插件现有的 native full + BreakableCudaGraphBackend 双 bank，D 设置：

```bash
export HANDOFF_REMOTE_SEGMENTED_GRAPH=1  # 默认开启；设 0 并重启可做 eager 对照
# D_NATIVE_ARGS 保留原生 full decode backend，capture buckets 包含 1/2/4。
# 使用本分支原生 --cuda-graph-backend-decode full 或等价 cuda-graph-config。
```

P 不需要开启分段图。原生 graph 若已关闭，仍走 eager；启用分段图的 D 需要原生 full
backend、1/2/4 buckets、cuda-python，且不使用 torch.compile、debug-eager、memory saver
或自定义 full backend。Remote 8/16 需要原生列表包含相应桶；插件不扩大原生 capture
范围。其他更大 buckets 继续保留给 Local full graph。

- 只为原生列表中已有的 batch **1/2/4/8/16** 额外捕获；Remote batch 3/12/32 等回退 eager，不 padding 到其他 Remote
  bucket。Local 继续原生 full bank 的 buckets/padding 策略。不是每个请求重新捕获。
- 原生 `candidate_all/candidate_c2_all/candidate_unfiltered/candidate_filtered` full variants
  全部保留。同一 batch 的 Remote native keys 复用一套分段图；图外按本轮原生标签设置
  执行作用域，并在回放结束或异常时恢复。variant 标签和 limits 纳入启动 TP 共识。
- 图内包含 QKV 线性投影、Q norm/RoPE、attention 后投影、norm、MLP/MoE 等稳定计算。
  图外为 native decode metadata raw→full 升级、KV norm/quantize/SWA 写入、
  compressor/indexer/carry 更新、attention/sparse 选择、RPC 和 merge。所有动态写入读取
  本轮 live batch，避免固化捕获时的请求、页地址和 carry。
- SGL `MQALayer` 提供三个清晰边界：`prepare_qkv` 保留原生串行/多流路由；
  `project_decode_qkv` 仅返回 Q、q_lora、原始 KV tensors；`update_decode_state` 消费这些
  tensors 更新本轮状态。原有 KV helper 仅新增可选 `projected_kv`，复用原 KV 精度开关，
  融合 wqa/wkv 不重复计算；量化 wrapper 不跨 graph/eager 桥。
- 插件每层只绑定一次 `prepare_qkv`，替换原先三个 private prepare 绑定；保留 backend 的
  metadata/attention 绑定，沿用原生模型循环。Remote 分离路径的 state 更新串行执行；
  Local eager/full graph 的串行及多流分派保持原样，不修改 serving 或 scheduler 配置。
  预热/捕获以 Local 数据初始化桥，不发 RPC；K/V 仍以单一 tensor 传递。
- metadata 一个 break，每层 state + attention 两个 break。L 层是 **2L+1 个 break**，
  通常 **2L+2 个 CUDA segment**，每个 bucket 一套；不是整层各捕获完整模型一次。
- rank 启动时先交换开关、buckets、模型/压缩配置及原生方法 ABI，再进入额外 capture
  barriers。投影融合模式和 KV 精度开关也纳入启动共识。捕获实际边界纳入既有 graph manifest 共识。每个 token 固定资源状态，检测
  中途 row/attempt/Local mask 变化；图路由不新增逐 decode collective。
- 部分资源 Local 或刚完成拷贝时继续 Remote bank；全部租约经 GPU/native PD/TP 退休，
  才恢复 full bank；原有退出边界消费最后一份结果后恢复 scheduler overlap。不重新捕获图。
- Remote bank 使用独立于 full bank 的 pool，每个 bucket 保留独立 output bridge。
  eager/graph 结果 shape、dtype、device type 必须相同。RPC 失败不重跑同一 token；native
  prepare 异常可能缺失侧流 join，标记 unsafe、保留 graph storage，拒绝仅凭主流 event 回收。

日志可用原生 `SGLANG_LOG_DECODE_GRAPH_KEY=1`：Remote 看到
`path=remote-segmented` / `replay bank=remote batch=1/2/4/8/16`；全部退休后为
`path=graph-eligible all_local=True` / `replay bank=full`。仅有 full graph 日志不证明 overlap
已经恢复，还需原有 scheduler policy 日志。

[分段图本地/TP2/4060 证据](validation/dsv41-segmented-graphs-4060.md) 明确区分 fixture 和整模验证。
新增 QKV 入图的接口、融合/精度矩阵和回归结果见
[QKV graph 验证记录](validation/dsv41-qkv-graphs-4060.md)。此版本需要本分支配套的
SGL 核心接口，不能只把新插件复制到旧 SGL；启动前先执行核心接口预检。

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

## Remote 热路径优化（2026-11-04）

默认 `HANDOFF_DSV41_FAST=1`，包含以下行为：

- 一层的所有 live rows 都经 TP 确认其 physical producer 为 Local 后，该层直接调用
  原生 attention（包含原生 store-cache 行为）。共享消费者按 source-layer 映射判断，
  不以本 rank 的 DMA 完成代替 TP 共识。
- 请求自己的 FULL 物理页逆表在 GPU 增量构建，top-k 分区与本地 SWA 索引留在 GPU。
  每层只回读一份合并后的 Remote logical positions 与错误标志；保持重复项、
  D 缓存前缀和 ratio-2 half-pair 的原始归属。
- P 的选中行读取、反量化和 partial attention 融合为 Triton kernel；D 的本地
  SWA/global partial 与 LSE merge 同样在 GPU 完成。支持原生 V4、V41、V41 FP4
  的独立/交错布局，动态 sparse 长度不引入逐长度 JIT 编译。启动时使用全 masked
  索引按池几何预热；预热不读取有效 KV、不写 KV。
- 同 endpoint、layer、query shape/dtype 的请求每组最多 4 行，复用一个 RPC。
  每 worker 最多 2 个执行线程、16 个已接纳 batch。IO 持有在提交线程中建立，
  Q producer event 在 D local partial 前记录；线程流等待该 event，主计算流等待
  reply event 并登记结果 tensor 的 stream lifetime。
- 前缀传输默认允许 2 个有界 payload 在途；Remote GPU Q/回复优先争用原生
  Mooncake send lock，连续 4 次优先任务后允许已排队的 bulk 前进。TE 访问仍串行，
  staging slot、native credit 和完成回调继续限制实际并发。

`HANDOFF_PREFIX_INFLIGHT=1..4` 可调整前缀流水线，默认 2。
P/D 的同 TP ranks 必须使用相同配置，启动共识会拒绝不一致。
`HANDOFF_DSV41_FAST=0 HANDOFF_PREFIX_INFLIGHT=1` 可对照旧 sparse 计算与单批流水线；
公平 send lock 仍生效。GPU payload 通道仍由 `HANDOFF_REMOTE_GPU_TRANSFER` 单独控制，
不会因开启 FAST 自动切换传输模式。

失败或 quarantine 不凭 timeout 释放 IO：未知 DMA 使请求 IO 保持 poisoned，
对应 pipeline 停止新增提交并保留未证实完成的 ticket。只有自己的完成证明才能
退还 credit；请求结束仍经过原有 overlap、source reader、checkpoint/TP 退休门槛。
这些优化没有绕过 source-cache 的窗口预算、checkpoint 回放或 allocator。

`remote-done.json` 增加 worker 累计的 sparse/RPC 统计，包括 native-layer 次数、
索引回读次数、batch/row 数、队列等待及 RPC 时间；GPU plane 统计包含 bulk/urgent
send-lock 等待时间。这些 worker 累计值不能当作单请求 TTFT。

验证见 [4060 优化验证记录](validation/dsv41-fast-4060.md)。

## 最后 chunk 完整计算

Remote P 不安装逐层生产者 forward hook；最后 chunk 完整计算后，通过统一的
forward event 进入现有 indexer/state/global 交接。串行路径保留原生 bounded-replay
尾窗 CUDA graph，overlap 路径继续使用自己的完整 forward 证明。独立 `layerwise`
模式仍保留逐层 hook。Remote window 内的传输分组、RPC 与逐资源 Local 不变。
见 [4060 完成事件验证](validation/remote-final-forward-4060.md)。
