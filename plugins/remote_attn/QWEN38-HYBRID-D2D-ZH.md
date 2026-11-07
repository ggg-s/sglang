# Qwen3.8-27B-FP8：纯文本混合状态 D2D

本分支通过原生 `Qwen3_5ForConditionalGeneration` / `qwen3_5` ABI 接入
`Qwen/Qwen3.8-27B-FP8`。复用原生模型加载与 FP8 权重计算，不新增模型类。

当前阶段：纯文本、P/D0/D1 同 TP。**P 支持原生 FULL+MAMBA prefix cache；D0/D1 关闭 radix cache**。
关闭 MTP/推测、调度 overlap、
ReplaySSM、LoRA、HiCache 和 unified memory。完整 attention 使用 page-size=1 的
原生 NHD BF16/FP16 KV；SSM 使用 FP32。**FP8 权重不等于 FP8 KV。**

P→D 使用原生最终 PD 提交和 `MAMBA` 状态传输；不提前接纳 GDN 状态尚未完整的请求。
P 缓存命中同时恢复 FULL KV 与 GDN checkpoint；使用原生 CoW 得到请求自己的
Conv/SSM 工作 slot。P 只计算未命中的后缀，但 D 没有 prefix cache，因此仍需发送
**完整 prompt KV**，以及后缀完成后的工作 slot 状态。不能把缓存 checkpoint 的
slot id 当作最终工作 slot，也不能把 P 的命中长度用作 D 的接收偏移。

P 保留原生 checkpoint/dedup、缓存锁、早发 cached-prefix 与发送队列退休路径。
Hybrid P 不提供 Remote 读服务，不安装 dense-only `source_cache.CacheGuard`；
Remote 的读 lease 由完成原生 PD 的 D0 在后续 D2D freeze 时建立。
原生 PD 未完成时请求行及工作 slot 保留；成功后才释放，缓存 checkpoint 可供后续请求复用。
P 当前要求内置、device-only `(FULL, MAMBA)` UnifiedRadixCache；HiCache、
外部 storage/linker、session cache 和第三方 radix backend 不在此支持范围。
D0→D1 支持 Remote attention 与逐 entry 渐进 Local。Remote 期间使用 eager，
全部资源经 TP 退休后恢复原生 Local CUDA graph；本阶段不提供混合模型的 Remote 分段图。

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
--disable-overlap-schedule \
--page-size 1 \
--kv-cache-dtype bfloat16 \
--mamba-ssm-dtype float32
```

**P 端不传 `--disable-radix-cache`**。当前 page-size=1、overlap 关闭的默认
`auto` 策略解析为 `no_buffer`，也可显式传 `--mamba-radix-cache-strategy no_buffer`。
**D0、D1 额外添加 `--disable-radix-cache`**；插件仍会拒绝 D 开启 radix。
不要在共用启动参数中给三侧统一关闭 radix，否则 P 也不会有缓存命中。

不传 speculative/MTP 开关；保留模型原生的 attention/linear-attention 后端。
插件环境：

```bash
case ",${SGLANG_PLUGINS:-}," in
  *,remote_attn,*) ;;
  *) export SGLANG_PLUGINS="${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}remote_attn" ;;
esac
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
