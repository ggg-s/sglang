# DeepSeek-V4-Flash-0731 Remote

模型 `deepseek_v4` 现进入 DSV PD 控制器，支持压缩率 0/4/128；原 C1/C2 路径保留。
采用原生非 unified FlashMLA 池与 V4 字节布局。模型权重加载、专家量化与通信后端
继续由核心提供；本适配处理 PD KV/状态及 Remote，不修改权重格式。

## 计算与交接

- P 的最后一个 token chunk 完整 forward 后才进入该 chunk 的交接，使用统一 CUDA
  event 覆盖 KV 写入；历史前缀可继续传输。原生 chunk/overlap 调度和缓存 pin 保持。
- C4 每层独占 compressed KV 与 indexer；C128 每层独占 compressed KV，没有 C128
  indexer。资源 schema 按 native sources_by_ratio 和实际 buffer/stride 建立，
  不要求 Flash 具备 V4.1 的 kv_source_layer_ids。
- C4 使用 D native top-k；C128 使用 native 全历史压缩行列表，Remote 上限为
  floor(prompt_len/128)，不受 index_topk=512 限制。FULL 页与压缩页对应，逻辑
  position 经请求自己的 FULL 页表转换成 P slot。重复 top-k 不去重。
- 原生 planner 提供 indexer、SWA、C4 overlapping compression/indexer 状态页、
  C128 request-scoped 状态。C128 online 传单行，offline 传未完成块所在 ring 页；
  整块边界没有 pending 状态。插件复用 native state_indices/maybe_send_extra，
  不实现另一份压缩算法或另算状态索引。
- 完整 indexer scatter 和 SWA/carry/metadata 屏障满足且 TP 同意后，D 可在 global
  尚未传完时 Decode。接纳后不再 DMA 覆盖 D 的可变状态；canonical final 保持在尾部。
- global scatter 只覆盖完整压缩行，最后未完成 C4/C128 block 属于 D；数据、scale、
  padding 分别按 native layout 处理。native V4 的未初始化 padded TP query heads
  先置零，避免 Remote 校验或计算读取 NaN；有效 heads 不变。

Flash 的 C4/C128 native prepare 现已提供 split projection/state ABI。
`HANDOFF_REMOTE_SEGMENTED_GRAPH=1` 启用 Remote 分段图；设为 0 可显式使用 eager。
全部 Local 并通过 GPU/PD/TP 退休后恢复 native full graph。
启动的 TP 签名包含实际 model_type、压缩率和布局。

## 配置与边界

沿用现有 P/D remote 参数：HANDOFF_PRE_LAYERWISE=remote、共享控制目录、同 TP、
PP=DP=CP=1、--language-model-only、early destination allocation、native FlashMLA。
保持模型的压缩率和 SWA 窗口，不要为了通过校验改为 C1/C2。
关闭 speculative/DSpark、unified KV、mega/TRTLLM、HiCache、LoRA 与 mixed chunk。
FULL 页需能按压缩率划分，遵循 native 模型 page-size 校验。
GPU payload 通道仍由 HANDOFF_REMOTE_GPU_TRANSFER 独立选择。

取消、quarantine、overlap 仍使用现有有序退休：未知 DMA 不构成释放证明；P 等待
native completion、Remote GPU reader 和 TP，D 等待 scatter/forward/IO/TP；allocator
部分失败后拒绝重复释放。正常路径缓存 checkpoint 回放与 pin 回收继续执行。

## Split Prefill（实验）

P、D 两端显式设置 `HANDOFF_HYBRID_SPLIT_PREFILL=1` 和
`HANDOFF_PRE_LAYERWISE=remote`，使用原生 Mooncake、同 TP、early allocation，
保持 `HANDOFF_EXPERIMENTAL_PD_REMOTE=0`，并关闭 DSpark。两端再设
`HANDOFF_SPLIT_ADAPTIVE=1` 可启用 decode 时间预算闭环，缺省使用静态策略。

D 先按可用 SWA 页和后缀预算协商最早切点。P 匹配缓存后选择页对齐切点，
传输已有压缩 KV、C4 indexer 及切点处的 SWA 窗口和 C4 压缩状态。
切点落在完整的 C128 块边界，由 D 续算后缀并采样首 token。中间块也按页对齐，
末块使用实际剩余长度。自适应时，D 按 TP 协商后的速率和步时派生每块等待量。
控制器 cap 地板包含尾页对齐余量，P 仍要求实际后缀至少为 2,048 token。

D 已有前缀命中、SWA 预算不足、无合法切点或对端无法协商时使用原生 PD。
本地 CPU 契约覆盖静态和自适应 Bootstrap、状态页重映射、原生回退及页回收。

2026-11-15 在 Ubuntu RTX 4060 Laptop 上运行 CUDA 探针，输入 9,215 token，
KV 页大小 256。每种策略分别跑 C128 online 和 offline。两组均验证 Mooncake TCP
搬运、C4 attention 和 indexer 压缩、C128 后缀及首步 decode。回收时逐页核对
释放的页 ID、已退休映射清零和活跃映射保留。

| 策略 | 切点 | D 后缀各块（token） | 每例回收 SWA 页 | 续算最大绝对误差 | 首步 decode 最大绝对误差 |
| --- | ---: | --- | ---: | ---: | ---: |
| 静态 | 7,936 | 1,024 + 255 | 17 | 0 | 0 |
| 自适应 | 6,656 | 1,536 + 1,023 | 14 | 0 | 0 |

自适应探针用 CUDA event 采样 suffix 和 decode，使用协商后的切点及块大小。
验证范围是单卡上的合成张量及压缩状态交接。整模输出一致性、多 TP GPU 和性能仍待实测。

```bash
for adaptive in 0 1; do
  HANDOFF_SPLIT_ADAPTIVE="$adaptive" PYTHONPATH=python:plugins/remote_attn \
    python test/manual/remote_attn/flash0731_split_prefill_cuda_probe.py
done
```

## 验证

2026-11-04：全部 TestRemoteAttn **272 项通过**（含真实 TP2 Gloo），新增 5 项覆盖
模型路由/native backend、full-graph fallback、C4/C128 resource schema、native
metadata 页表、C128 >512 行、native state registration/pending ring、正常与取消
退休。已有真实 checkpoint/插入、SWA、metadata 提前发送和 TP quarantine 回归通过。
49 hooks / 125 interfaces、Ruff E9/F63/F7/F82、git diff --check 通过。

[4060 原始结果](validation/flash0731-4060.json)：C4/C128 × TCP tensor/Mooncake GPU
四项均通过，使用 source-loaded native 布局与 metadata，真实 socket/RPC、GPU
buffer 注册、Triton 计算。C128 选择 521 行，其中 P 提供 520 行历史数据。
D 的不可用前缀被刻意清零，输出仍匹配独立 reader/partial oracle（BF16 容差
atol=0.015625、rtol=0.01）；验证重复 top-k、未完成块、padded heads、尾页精确字节保护。
真实 GPU reader 在途取消时，提前 close 被拒绝，tombstone 等 GPU 完成后才允许
关闭 pin；无 quarantined slot，RPC teardown 后 native registration 仍可使用。

另在 4060 复验完整最后 chunk 的 CUDA event/tail graph 与 P overlap/cache drain，
normal/cancel/revoke 均通过。源码 SHA256 已与最终工作树逐项核对。
4060 物理通道为 Mooncake TCP，**没有加载 Flash-0731 整模权重，也未验证 H20 RDMA、
TP4 GPU、文本精度或整模 TTFT/吞吐**。这是适配层与真实 CUDA/传输生命周期验证。

```bash
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/flash0731_cuda_probe.py \
  --source-root . --output /tmp/flash0731.json
```

## 后续生命周期审阅

已修复混合 C4/C128（及 C1/C2）native guard 容量不同导致的逆页表冲突。
逆页表按 ratio 持有，仍拒绝同 ratio 容量漂移和请求身份变化；成功退休统一清理。
详见 [状态、尾块与 KV 生命周期审阅](validation/flash0731-lifetime-review.md)。


## C4/C128 分段图更新

默认 `HANDOFF_REMOTE_SEGMENTED_GRAPH=1` 不再对 `deepseek_v4` 强制回退 eager。
需要 native FlashMLA、full decode graph、纯 TP、非 unified KV，并使用本分支配套
`project_decode_qkv/update_decode_state` 接口。C4 先写 SWA，再更新 indexer 和独立 attention
compressor；C128 更新 SWA 与 attention compressor，不引入 indexer。
捕获/回放仍保留逐 entry ownership 快照、实际行裁剪和安全回收 fence。

`--language-model-only` 现在允许 V4 与 V4.1 原生 PD，不必再本地放宽校验。
分段图不能消除每层网络往返；需要在目标 H20 上重测 TTFT/TPOT 和实际通信路径。
仅设置 `HANDOFF_REMOTE_GPU_TRANSFER=1` 或看到 RDMA 初始化日志，不证明每次 Q/O 提交
实际使用 GPU RDMA；应同时核对协商 backend 和请求的 `path=remote-segmented`。

```sh
python test/manual/remote_attn/dsv41_segmented_graph_probe.py --source-root . \
  --model-family v4 --kv-layout v4 --gpu-transfer --output /tmp/flash-graph.json
```

这是小型 CUDA/原生方法及 TCP 数据面回归，不是 DeepSeek-V4-Flash-0731 整模性能基准。
主动迁移部署和测量见 [P4 DSV Gateway](P4-DSV-GATEWAY-ZH.md)。
