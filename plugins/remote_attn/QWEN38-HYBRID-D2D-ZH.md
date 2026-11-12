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

Remote PD 默认开启分段 Prefill。P 在真实 Prefix Cache Match 后决定切点，
所有已支持的 dense MHA 和 FULL+MAMBA 布局共用规则，不按模型或并发分档。
Remote attention 用于 D2D 迁移。P→D Remote 早接纳作为实验功能，需在两端设置
`HANDOFF_EXPERIMENTAL_PD_REMOTE=1`。对比该实验路径时，同时设置
`HANDOFF_HYBRID_SPLIT_PREFILL=0`。Hybrid PD 的实验 Remote 仍使用
`HANDOFF_HYBRID_REMOTE_MIN_CACHE_HIT_RATIO`，默认仅尝试首次实际 Prefill 时
命中率大于 50% 的请求。

### 分段 Prefill

`HANDOFF_HYBRID_SPLIT_PREFILL` 默认 `1`，在 P、D 两端设为 `0` 可做原生 PD 对照。
至少 4096 token 的纯文本请求可参与协商。P 配置切点，D 配置续算批大小。

- P 的 `HANDOFF_HYBRID_SPLIT_PREFILL_CUT_RATIO` 范围为 `[0.5, 1)`，默认 `0.75`。
  默认按实际未命中量分配，D 接手其中 25%，后缀下限为
  `min(2048, max(prompt_tokens // 16, 1))`。有缓存命中或 prompt 短于 32K 时，
  后缀再受 D 声明的 chunk cap 和 prompt 长度的 15% 限制。切点须越过命中前缀，
  满足 D 声明的下界，并留下足够的后缀。没有合法切点时走原生 PD。
- P 可用 `HANDOFF_SPLIT_CUT_BASIS=whole` 对照整段比例。该臂的切点取
  整段比例切点与命中前缀后 1024 token 的较大值。`HANDOFF_SPLIT_SUFFIX_TOKENS`
  默认 `0`，设为正整数可钉住后缀长度，用于标定。两个对照都检查切点边界。
- D 的 `HANDOFF_HYBRID_SPLIT_PREFILL_MAX_OUTSTANDING` 默认 `4`。每次发出 offer
  预留一个名额，涵盖等待 P 传输和 D 续算的请求。D 达到该上限或 decode 批次已满时，
  新请求走原生 PD。续算完成、原生 PD 完成或请求终止清理时归还名额。
  默认上限是压测起点，可按在途 decode 的 token 间隔调整。
- D 的 `HANDOFF_HYBRID_SPLIT_PREFILL_CHUNK_SIZE` 默认 `4096`，以 D 的
  `--max-prefill-tokens` 为上限。D 声明最多承接半个 prompt，分块完成续算，
  并向 P 声明 `max(实际续算块上限, 2048)` 作为 chunk cap。
  这个设置独立于原生 `--chunked-prefill-size`。

沿用同 TP 的原生 Mooncake、early device allocation 和 page-1 KV 配置。
D 命中已有前缀、请求输出 logprobs、结构化生成或对端未协商时走原生 PD。
P 等到 bootstrap 后再为可协商请求启动 Prefill，首次 Forward 前对实际 cut 做
TP 一致性检查。V3 Bootstrap 传递最小切点和 chunk cap，完成 metadata 传递实际切点。
新旧版本协商不匹配时走原生 PD。

P 发出切点前的 KV 和 Conv／SSM 后完成原生传输。D 在已有预留中续算剩余 prompt，
并从完整 prompt 的末尾采样首 token。前缀 KV 在 D 本地读取。每个续算 batch 最多处理
一个请求，已有 decode batch 会在两个续算 batch 之间获得执行机会。
验收时用相同的缓存冷热顺序和并发，对比 TTFT、E2E、吞吐和逐请求 ITL。
同时记录最大 token 间隔超过 100 ms 的请求占比，以及逐请求最大间隔的分位数。
P、D 的 `[handoff-split-decision]` 记录协商与回退原因，可按 bootstrap room 对照。
D 的 `[handoff-split]` 记录切点和总续算量，`[handoff-split-chunk]` 记录每块的起止
位置和剩余 token 数。D 复用同一份 KV 预留和 Conv／SSM 状态。

- P 可用 `--cuda-graph-backend-prefill breakable` 运行 prefill 图。D 同时配置
  `--cuda-graph-backend-prefill breakable --cuda-graph-backend-decode full`，
  让满足原生 replay 条件的续算块走 breakable 图，普通 decode 走 full 图。
  D 未显式开启 prefill 图时，续算走 eager。
- D 开启原生 overlap schedule 时，如果已有请求在 decode，可以先提交下一批 decode，
  再在 CPU 上结算中间续算块的结果。续算和 decode 按顺序提交到原生 forward stream。
  当前块结算后才能重排该请求，最后一块结算后才能加入普通 decode 批次。

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
权重、TP 多卡或真实 RDMA HCA/CQ 验收。完整 27B 的 TP2+TP2 TCP 实测见下节。

P prefix cache 的原生 CUDA/PD 验证见 [专项验证](validation/qwen-hybrid-prefill-cache-4060.md)。

本次 PD Remote、缓存读 lease 与网关 journal 修复的验证见
[专项记录](validation/gateway-journal-hybrid-pd-4060.md)。


## 39 机 Qwen3.8-27B-FP8 PD 实测

当前默认策略的长度、命中率和 ITL 对照见 [分段 Prefill V1 实测](validation/qwen-split-prefill-v1-5090.md)。
以下保留此前构建的实验记录。

39 机使用 RTX 5090，P TP2 + D TP2，开启 CUDA Graph 和 overlap schedule。
Mooncake 使用 TCP（`MC_FORCE_TCP=1`），机器没有 RDMA 或 GPU P2P。
负载为 64K 输入、128 输出、单波次 4 并发。延迟单位为 ms，尾延迟沿用原测量口径。

### 冷缓存（0% 命中，`2b859c206f`）

分段臂在 P、D 两端显式开启 `HANDOFF_HYBRID_SPLIT_PREFILL=1`
（本次测试脚本用 `SPLIT_PREFILL=1`），切分比例由 D 端设置。
分段默认关闭，冷请求在默认门控下走原生 PD。

| 指标 | 原生 PD | cut 0.5 | cut 0.75 |
| --- | ---: | ---: | ---: |
| TTFT 均值 | 28,283 | **20,630（−27.1%）** | 24,281（−14.1%） |
| TTFT 尾 | 40,505 | **30,145（−25.6%）** | 33,785（−16.6%） |
| E2E 均值 | 29,870 | 28,885（−3.3%） | **26,937（−9.8%）** |
| E2E 尾 | 42,315 | **32,313（−23.6%）** | 35,837（−15.3%） |
| ITL p90 | 26.1 | **18.5** | 26.1 |
| 波次墙钟 | 42.2 s | **31.5 s（吞吐 +34%）** | 35.4 s（+19%） |

在这组负载下，cut 0.5 的 TTFT、E2E 尾和波次吞吐更好，cut 0.75 的平均 E2E 更低。
收益主要来自跨请求流水线。C=1 的独立对照中，分段与原生 PD 的 TTFT 分别为
10,027 和 10,052 ms，基本持平。

分段续算会占用 D 的 GPU。冷缓存测量中，ITL p50 从 14.0 ms 升至 17.2 ms，
16 条请求中有 2 条的逐请求 ITL p90 达到约 600 ms。每波最早进入 Decode 的请求
容易遇到后续请求的续算。比较全局 p90 时，同时检查逐请求 ITL 和最大 token 间隔。

### 热缓存（90% 命中，旧构建）

下表为此前构建的 Hybrid Remote 结果。`2b859c206f` 尚未复测热缓存。

| 指标 | 原生 PD | Hybrid Remote |
| --- | ---: | ---: |
| TTFT p50 | 10,460 | **4,233（−59.5%）** |
| E2E p50 | 12,477 | **11,117（−10.9%）** |
| ITL p90 | 16.2 | 88.5 |

热缓存下，Remote 早接纳缩短了首 token 等待，Decode 步内的远程读取增加了 token 间隔。

### 传输边界

39 机上 KV 搬运与 Prefill 的等效速率都约为 6,000 token/s。冷缓存时，普通
Hybrid Remote 仍要等 P 完成整个 prompt 的 GDN 状态，大部分 KV 搬运可被 P 的计算
覆盖。分段臂在切点交接原生 KV 和 Conv/SSM，收益主要来自 P、D 同时处理不同请求。
RDMA 会改变交接和 Remote 读的成本，也可能改变原生 PD 基线。冷缓存 Remote 能否
获得净收益，以及哪种切分比例更合适，需在 RDMA 上复测。

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
  热缓存 Remote 臂若命中不匹配或 Remote 调用为零，应先排查配置和缓存。
  冷缓存门控与分段臂应单独统计，Remote 调用为零符合这两条路径的预期。
