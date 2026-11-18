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
保持 `HANDOFF_EXPERIMENTAL_PD_REMOTE=0`。两端再设
`HANDOFF_SPLIT_ADAPTIVE=1` 可启用 decode 时间预算闭环，缺省使用静态策略。

D 先按可用 SWA 页和后缀预算协商最早切点。P 匹配缓存后选择页对齐切点，
传输已有压缩 KV、C4 indexer 及切点处的 SWA 窗口和 C4 压缩状态。
切点落在完整的 C128 块边界，由 D 续算后缀并采样首 token。中间块也按页对齐，
末块使用实际剩余长度。自适应时，D 按 TP 协商后的速率和步时派生每块等待量。
控制器 cap 地板包含尾页对齐余量，P 仍要求实际后缀至少为 2,048 token。

D 已有前缀命中、SWA 预算不足、无合法切点或对端无法协商时使用原生 PD。
本地 CPU 契约覆盖静态和自适应 Bootstrap、状态页重映射、原生回退及页回收。

### 双端 DSpark（实验）

P、D 使用同一分支、相同的 draft 模型与 DSpark 配置，并同时设置
`--speculative-algorithm DSPARK`。仅接入原生 paged FlashMLA 的 SWA-only
draft 布局。Mooncake 在 bootstrap 校验两端的 target/draft 布局。

P 使用原生 DSpark prefill 写入 draft SWA。发生 split 时，target 和 draft
的 SWA 都搬到切点窗口，draft 请求状态沿用原生传输。D 使用原生 DSpark prefill
续算后缀，并用末块结果直接进入 DSpark decode。draft 和 target verify
按原生 graph 准入条件回放。DPA 使用调度器已同步的准入结果，碰到不满足
原生条件的 batch 时走 eager。在 D 设置 `SGLANG_LOG_DECODE_GRAPH_KEY=1`，
可在日志中分别确认 `worker=draft` 和 `worker=target` 的实际回放。
自适应计时覆盖整个 DSpark forward，包括 draft KV 写入或提议与验证。

#### 双端 DSpark 状态交接复测（RTX 4060 Laptop）

基线为 `b6ee6c8520`，叠加本次改动。输入 9,215 token，KV 页大小 256，
使用真实 CUDA 和经 host staging 的 Mooncake TCP。静态和自适应分别跑了
C128 online 与 offline，各策略 22 次传输，四组均通过。

| 策略 | C128 模式 | 切点 | D 后缀各块（token） | target/draft SWA 与 draft 请求状态 | 续算 / 首步 decode 最大绝对误差 |
| --- | --- | ---: | --- | --- | --- |
| 静态 | online、offline | 7,936 | 1,024 + 255 | 通过 | 0 / 0 |
| 自适应 | online、offline | 6,656 | 1,536 + 1,023 | 通过 | 0 / 0 |

逐组验证 target 和 draft SWA 都写入切点对应的目标页，切点后的下一页保留
原值，draft 的 `swa_valid_from` 写入正确的请求槽位。C4/C128 压缩续算
及探针首步 decode 与本地参考一致。日志位于
`ubuntu:/tmp/sglang-flash-dspark-probe.Hm7Vh4/probe-{0,1}.json`。

40 项相关 CPU 契约通过，覆盖双 SWA 重映射、缺失 draft 状态时拒绝 bootstrap，
以及原生 PD 回退。4060 探针使用合成的 draft SWA 和请求状态张量，整模 DSpark
输出一致性、多 TP GPU 和性能仍待验证。

### D 侧后缀 PCP4（实验）

P、D 各用 4 张 GPU，D 启用 `HANDOFF_SPLIT_DECODE_PCP=1`、
`--enable-prefill-cp --cp-strategy interleave --enable-cp-decode-attn-tp`。
后缀 `EXTEND` 走原生 CP，普通 decode 和 DSpark verify 由原生路径切分
attention 权重。Mooncake 仍按 TP4 注册四份接收页表。接收完成和预分配
使用同一个四 rank CPU 组。无切点的请求使用原生 PD 收发和 DSpark。
该模式要求 interleave、复制的 GPU KV、attention DP=1 和原生 Mooncake。

两台各四卡的机器可用直连冒烟。两端使用同一个 checkpoint、同一份代码和相同的
DSpark draft 配置。如果 draft 是独立 checkpoint，两端都设置 `DRAFT`。

```bash
# P 机器
MODEL=/path/to/DeepSeek-V4-Flash P_ONLY=1 P_PCP=1 D_DPA=0 \
  P_GPUS=0,1,2,3 bash test/manual/remote_attn/flash_split_dpa_smoke.sh

# D 机器。P_HOST 填 P 的内网 IPv4 地址
MODEL=/path/to/DeepSeek-V4-Flash P_HOST=10.0.0.10 P_PCP=1 D_DPA=0 D_PCP=1 \
  D_GPUS=0,1,2,3 bash test/manual/remote_attn/flash_split_dpa_smoke.sh
```

同机 8 卡时省略 `P_ONLY`、`P_HOST`，在 D 命令上指定
`P_GPUS=0,1,2,3 D_GPUS=4,5,6,7`。冒烟检查长请求续算、draft 和 target
verify 图回放，以及短请求拒绝切分后仍完成 DSpark。用下面的设备探针在
四张 D 卡上逐卡对比切点处 target/draft SWA 和 draft 请求槽的内容。
探针使用合成页和本地 Mooncake 拷贝，整模跨节点接收由上面的冒烟覆盖。

```bash
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=python:plugins/remote_attn \
    python test/manual/remote_attn/flash0731_split_prefill_cuda_probe.py || exit
done
```

#### 2026-11-18 验证

基于 `c9eee64c81` 叠加 D 侧 PCP4 opt-in，在 Ubuntu RTX 4060 Laptop 上用
PyTorch 2.13.0+cu130 复测。源码预检覆盖 66 个 hook 和 147 个核心接口。
主契约 69 项通过，含 156 个通过的子测试，decode 预算契约 22 项通过。
另在本机用四进程 Gloo 验证
原生接收同步，单个 rank 未就绪时四个 rank 都等待，收到全部 KV 后共同放行，
单 rank 失败也同步给全组。

| 4060 CUDA 探针 | C128 模式 | Mooncake TCP 传输次数 | 切点 | D 后缀各块（token） | target/draft SWA 与 draft 请求槽 | 续算 / 首步 decode 最大绝对误差 |
| --- | --- | ---: | ---: | --- | --- | --- |
| 静态 | online、offline | 22 | 7,936 | 1,024 + 255 | 通过 | 0 / 0 |
| 自适应 | online、offline | 22 | 6,656 | 1,536 + 1,023 | 通过 | 0 / 0 |

日志位于 `ubuntu:/tmp/sglang-flash-pcp-publish.Jt4iNN/` 下的
`test_remote_main_contracts.log`、`test_remote_split_budget.log`、
`probe-0.json` 和 `probe-1.json`。4060 验证的是
单卡合成页，整模四卡 PCP4、DSpark verify 图回放与原生 PD 回退待内网冒烟。

### DPA 与 SMG 的前缀路由（实验）

P、D 两端启用 `HANDOFF_SPLIT_DPA=1`。双端 DSpark 的 CUDA 部署每组
attention TP 为 1，例如两端各用 `--tp 4 --attn-dp-size 4 --enable-dp-lm-head`。
D 设置 `SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1`。沿用上面的原生 Mooncake
和 FlashMLA 布局。P 开 radix cache，D 保持 `--disable-radix-cache`。

也可让 P 使用 `--tp 4 --attn-dp-size 1 --enable-prefill-cp --cp-strategy interleave`，
保留复制的 KV 布局，P 设 `HANDOFF_SPLIT_DPA=0`。D 仍用 DPA4 和
`HANDOFF_SPLIT_DPA=1`。P 的四个 CP rank 对同一 offer 和切点达成一致，
默认由 CP0 发送 KV 和 DSpark 状态。直连时，D 可选择任意 attention DP rank，
P 的 DP rank 固定为 0。冒烟脚本可用 `P_PCP=1` 覆盖此拓扑，并检查 D 的 draft
和 target verify 回放。

P 使用 PCP4 时，D 也可用纯 TP4（`--tp 4 --attn-dp-size 1`，两端均设
`HANDOFF_SPLIT_DPA=0`）。P 收齐四个 D rank 的 offer 后，取最晚的最早切点和
最小的 chunk 上限。Mooncake 按各 rank 的页表写入 target 和 draft SWA。
任何一个 rank 的 offer 缺失或不兼容，请求就走原生 PD。
用 `P_PCP=1 D_DPA=0` 跑冒烟脚本可检查这组拓扑。

采用能从 `/server_info` 读取 `attn_dp_size` 的 SMG。一个 P 和一个 D 各开
DPA4 时，`/workers` 应各显示 4 个 worker。SMG 按 P 的 DP 组匹配前缀，
独立选 D，并把选中的 P 组号传给 D。P 默认按 `bootstrap_room` 分组选路，
SMG 发的随机 room 可能与选中的 P 组不一致。启动 P、D 前需设置
`SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1`，允许 P 注册实际组号；
D 收到 SMG 显式传来的组号时直接使用。

```bash
python -m sglang_router.launch_router --pd-disaggregation --dp-aware \
  --prefill http://127.0.0.1:31116 31997 --decode http://127.0.0.1:31117 \
  --prefill-policy cache_aware --decode-policy power_of_two --port 31100
```

同机多卡冒烟会自行启动 P、D 和 SMG。两条不同尾部的共享前缀请求由 SMG
选组。第二条需要报告实际 KV 命中，两条都需要在 D 日志中确认 split 且返回
`spec_verify_ct > 0`。

```bash
SMG=1 MODEL=/path/to/DeepSeek-V4-Flash \
  P_GPUS=0,1,2,3 D_GPUS=4,5,6,7 \
  bash test/manual/remote_attn/flash_split_dpa_smoke.sh
```

冒烟沿用 32K context 和固定 4,096-token 后缀。长上下文压测应按最长输入、
输出预算和每个 DP 组的实际 KV 容量配置。

### 原有 Flash CUDA 探针

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
