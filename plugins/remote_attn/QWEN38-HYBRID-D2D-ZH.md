# Qwen3.8-27B-FP8：PD Remote 与混合状态 D2D

本分支通过原生 `Qwen3_5ForConditionalGeneration` / `qwen3_5` ABI 接入
`Qwen/Qwen3.8-27B-FP8`。复用原生模型加载与 FP8 权重计算，不新增模型类。

当前阶段：纯文本、P/D0/D1 同 TP。**P 支持原生 FULL+MAMBA prefix cache；D0/D1 关闭 radix cache**。
关闭 MTP/推测、ReplaySSM、LoRA、HiCache 和 unified memory。完整 attention 使用 page-size=1 的
原生 NHD BF16/FP16 KV；SSM 使用 FP32。**FP8 权重不等于 FP8 KV。**

P→D 先使用原生 `MAMBA` 状态规划和传输安装全部 Conv/SSM 与采样 metadata。
只有该 barrier 在所有 TP rank 完成后，D 才提前接纳；此时 full-attention KV 可以仍在 P，
D 通过 Remote attention 读取不可变前缀，逐 entry/range 搬完后切回 Local。GDN 始终在 D 本地续算，
不走 DSV sparse math 或 DSV staging ring，也不等待整个 KV 传完才建立 target。

P 缓存命中同时恢复 FULL KV 与 GDN checkpoint；原生 CoW 得到请求自己的工作 slot。
P 只计算未命中的后缀，但 D 没有 prefix cache，仍需传送完整 prompt KV 和最终工作状态。
不能把缓存 checkpoint 的 slot id 当作最终工作 slot，也不能用 P 命中长度偏移 D 接收位置。

P 安装支持 `(FULL, MAMBA)` 的读 lease，独立固定缓存前缀、保留请求私有后缀和工作 slot。
当已排队的 KV 都处于原生缓存锁的保护范围内，P 在 chunk 完成时发布对应的原生 GDN checkpoint。
已排队的私有页会阻止会重绑它们的发布。最终回收使用原生 checkpoint 处理当前工作状态。
全部 KV 写、Remote 读、GPU forward 与 TP 退休证明完成后，才调用原生释放/状态 donation。
D 传输中完成或取消也先延期回收，避免工作 slot 或请求行提前复用。
P 当前要求内置、device-only `(FULL, MAMBA)` UnifiedRadixCache；HiCache、
外部 storage/linker、session cache 和第三方 radix backend 不在此支持范围。
P 支持原生 Prefill CUDA Graph 和 overlap schedule。最后一批 forward（含图回放）结束后，
以实际生产流上的完成事件约束 KV、Conv 和 SSM 的发布。
D 的 Remote 窗口支持 overlap schedule 和 Decode 分段图。只有 full-attention 层进入 eager
回调，GDN 和其他算子留在图内。D 在进入或离开窗口时先处理上一批结果，稳定窗口使用原生
overlap。全部资源经 TP 退休后切回原生全量 Decode 图。

## 分配和冻结边界

1. D0 生成 plan：固定请求、目标、epoch、模型/TP/KV/GDN schema，给出冻结长度上下界。
2. D1 使用原生 `_pre_alloc` 真实预留请求行、上界长度的 KV、GDN 工作 slot 和原生需要的
   tracking slots。`ensure_mamba_capacity` 计算工作/追踪 slot，不能只检查 KV 剩余 token。
3. 所有 rank 预留成功后，D0 在原生结果边界冻结。保留最后一个已采样、尚未计算的 token，
   满足 `kv_len = prompt_len + output_len - 1`。混合模型不允许跳过预留的诊断入口。
4. 冻结流显式等待原生 `ModelRunner.forward_stream`，其完成 event 覆盖 KV、Conv 和 SSM。
   Conv/SSM 用原生 `state_ids(...).ids(...)` 定位物理工作 slot，不能用 request row 代替。
5. D1 消费预留，沿用其原生请求行和工作 slot，释放未用 KV 上界部分。先安装全部层的
   Conv/SSM，再同步安装 event，最后清除 `mamba_needs_clear`。否则下一次 forward 会清零导入状态。
6. 全部 GDN 状态完成后才允许 commit/admit。GDN 从该冻结点在 D1 本地续算；
   完整 attention 的历史 K/V 同时按 layer group × token range 搬运。

层映射与原生模型、pool 构造保持一致，读取 `full_attention_layer_ids` /
`linear_layer_ids`，不依赖 HF 的 `layer_types` 属性。启动和 manifest 都检查
完整逻辑层覆盖及 KV/GDN pool 归属，manifest 的 `layer_mapping` 保存两组原生层 ID。
迁移两端必须使用相同 schema；旧 `layer_types` schema 无法通过匹配。

## 数据面与 Local 证明

KV、Conv、SSM 都使用现有 D2D Mooncake bulk 传输。Q/O/LSE 使用共用 Mooncake
单次提交与能力协商。RDMA 使用注册 GPU staging；无共同 GPU RDMA 能力时，在提交前
选择独立 D2H Host staging 的 TCP 路径。已提交但完成未知时，保留原注册和 backing，
不在同一批写失败后换路重试。

GDN 不需要每层 RPC。只有完整 GQA attention 的不可变历史前缀在 D0 计算，D1
计算新 token 的后缀，再按 LSE 合并。物理 KV entry 与逻辑 full-attention layer 显式映射；
同一层的 K、V 两个 entry 均有 GPU/传输/TP 完成证明才切 Local。

`HANDOFF_D2D_GROUP_SIZE` 和 `HANDOFF_D2D_TOKEN_RANGE` 继续控制批量搬运，
group 根据逻辑模型 layer id 分组。每个 entry/range 的覆盖证明保留，不用 group ACK
代替 entry 完成。page-size=1 时 bulk 只写冻结前缀，不覆盖 D1 新生成的 suffix。

取消仍由现有 epoch tombstone、native request cancellation 和有序退休处理。
准备阶段必须 join 复制线程、证明 CUDA 和原生写已排空；完成未知则隔离 KV、GDN slot
及 staging。D0 只有在 commit、copied、远端读 lease 关闭等证明齐全后释放原生资源。

## 启用

按现有原生 PD 命令部署 P、D0、D1，保持同一模型与 TP。新增/确认以下参数：

```bash
--model-path Qwen/Qwen3.8-27B-FP8 \
--language-model-only \
--page-size 1 \
--kv-cache-dtype bfloat16 \
--mamba-ssm-dtype float32
```

- **P** 保留 device radix cache。启用 overlap 时使用 Triton GDN 和
  `--mamba-radix-cache-strategy extra_buffer`（原生 `auto` 在适用时也会选择它）。
  Prefill Graph 使用 `--cuda-graph-backend-prefill breakable`。
- **D0、D1** 添加 `--disable-radix-cache --cuda-graph-backend-decode full`。
  Decode 捕获 bucket 至少包含 1、2、4，Remote 分段图使用其中的 1、2、4、8、16。
  超出已捕获 bucket 的 Remote batch 走 eager。
- 两端使用上述配置时去掉 `--disable-overlap-schedule` 和 `--disable-cuda-graph`。
  `HANDOFF_REMOTE_SEGMENTED_GRAPH=0` 可作为 D 端 Remote eager 的对照。
- 主动 D2D 冻结依赖源端的原生结果队列排空。D2D 独立验证仍使用源 D 的
  `--disable-overlap-schedule`。

不传 speculative/MTP 开关；保留模型原生的 attention/linear-attention 后端。
插件环境：

```bash
case ",${SGLANG_PLUGINS:-}," in
  *,remote_attn,*) ;;
  *) export SGLANG_PLUGINS="${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}remote_attn" ;;
esac
# P/D 共用 PD rendezvous；多机使用现有 PD mesh 配置。
export HANDOFF_PD_DIR=/path/to/shared/pd-rendezvous
export HANDOFF_PRE_LAYERWISE=remote
export HANDOFF_REMOTE_GPU_TRANSFER=1
export HANDOFF_REMOTE_BACKEND=auto
export HANDOFF_D2D_BACKEND=auto
export HANDOFF_D2D_GROUP_SIZE=8
export HANDOFF_D2D_TOKEN_RANGE=1024
```

不要开启原生 disaggregation staging ring；D2D 的私有 staging 由共用数据面管理。
原生 Mooncake transport、bootstrap、SMG、control directory 配置沿用现有部署。
主动在线迁移使用 P4 的 `--backend hybrid`，显式设置 `--src-control-dir` /
`--dst-control-dir`。它复用 `dsv-d2d-v2` 原生状态迁移协调协议、SSE 和持久化恢复，
peer schema 中的 `family=qwen-gdn-d2d-v1` 防止把 DSV 或不同模型状态当作 Qwen 导入。
初始请求仍由该 gateway 的原生 PD 提交通道创建可控制的 rid。gateway 需设置
`HANDOFF_MODEL=Qwen/Qwen3.8-27B-FP8`（或相同本地模型目录）、
`HANDOFF_PD_PREFILL_URL` 及 bootstrap 地址；为这套部署单独设置 `HANDOFF_STATE_DIR`。

## 验证范围

回归与 4060 结果见 [验证记录](validation/qwen38-hybrid-4060.md) 和
[原生配置启动回归](validation/qwen38-native-layer-mapping-4060.md)。8 GB 4060
验证小尺寸原生池、实际 CUDA GDN/FlashInfer 和 Mooncake TCP；不等于完整 27B
权重、TP 多卡或真实 RDMA HCA/CQ 验收。整模型部署仍需目标显存与 TP 配置实测。

P prefix cache 的原生 CUDA/PD 验证见 [专项验证](validation/qwen-hybrid-prefill-cache-4060.md)。

本次 PD Remote、缓存读 lease 与网关 journal 修复的验证见
[专项记录](validation/gateway-journal-hybrid-pd-4060.md)。


## 对比原生 PD 和 Hybrid Remote

两臂使用同一批输入 token、唯一的测量后缀、相同的 chunk 大小及采样设置。分别先预热同样的共享
前缀，再开始计时。每档并发独立预热，测量前清点预热请求已经完成。

- 在 P 上打开 `--enable-metrics`，在测量前后抓取 P 的 `/metrics`。核对完成请求数和
  `sglang:cached_tokens_total` 的增量。先确认两臂实际复用的 token 数相同，再比较延迟。
  共享前缀比例由输入构造决定，实际命中长度由可恢复的 GDN checkpoint 决定。
- Remote 臂的 P 端 `[handoff-src]` 和对应 `ready.json` 记录首 chunk 的 `p_cached_tokens`。
  D 端退休的 `done.json` 提供 `rid`、`hybrid_pd` 和 `remote_calls`。
  统计同一测量窗口的请求，按 `rid` 去重 TP 副本，并分别报告 Remote 实际执行占比和调用次数。
- 对相同并发分别报告 TTFT、E2E 和 ITL 的分位数，并附两臂的实际 P 命中数及 Remote 执行占比。
  命中不匹配或 Remote 调用为零的结果只用于排查配置和缓存。
