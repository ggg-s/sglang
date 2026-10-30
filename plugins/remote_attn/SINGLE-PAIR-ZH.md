# 原生 SMG + 单 pair P/D 接入

使用原生 `sglang.launch_server` 和 SMG PD router，不使用 JSON 配置或启动包装器。
新增能力通过 P/D 的 `remote_attn` 插件启用；SMG 无需设置插件环境变量。
核心和插件必须来自同一份包含 progressive 扩展的构建，不能只安装插件。
依赖、容量和完整验收见 [部署手册](DEPLOYMENT-ZH.md)。

当前范围是同机 Qwen3 dense NHD MHA BF16/FP16、同 TP、PP=DP=CP=1、page size 1、
early allocation、mooncake_tcp。P 可开启设备内 dense FULL radix cache；D 仍关闭 radix cache，两端仍关闭 overlap scheduler。
CUDA graphs 沿用原生 full decode 配置：D 的 Remote 阶段对 batch 1/2/4 使用分段图，
其他 batch 为 eager；完成 GPU scatter 并经 TP 一致回收后恢复原生 full graph。
P 普通 prefill 保留原生 graph，最终 chunk 的分层 prefill 和 P 端 Remote RPC 仍为 eager。
容器中的 P/D 必须共享同一份本机目录及可互通的 loopback 网络。
以下 TP2 示例使用 GPU 0、1 做 P，GPU 2、3 做 D；线上其他模型、显存和并发参数
继续保留，但与这些限制冲突的能力需要先适配。

## 1. 核对代码与接口

在 GPU 机的 serving Python 环境中，从同一份代码安装 core 和插件。
SMG 复用现有兼容构建，以下使用仓内原生 Python launcher：

```bash
cd /workspace/sglang
python -m pip install -e ./python
python -m pip install --no-deps -e ./plugins/remote_attn
python -m sglang_remote_attn.check
python -c 'import sglang_router; print(sglang_router.__file__)'
```

安装态预检应输出 `hooks: 24`、`core_interfaces: 61`、`send_probe: true`。
它检查实际加载的接口：保留 receiver 的 metadata commit、选层发送与完成回调、
staging 分组、host copy、缓存 checkpoint/lock/finish、字段初始化及 typed env 声明；插件注册时也会执行同样的
核心检查，避免直接启动 P/D 绕过预检。`**kwargs` 不能替代显式参数。
检查失败会给出具体缺失接口；检查通过不等于模型已启动或 GPU 验收通过。

无 GPU 的开发机可以检查 checkout，但不能用它证明部署机安装的 wheel 匹配：

```bash
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

## 2. 保留线上 Prefill 与 chunk 配置

P/D 使用各自已有的 serving 环境。将线上实际值设置为以下 shell 变量，再在两个
终端分别执行公共环境块。这里不提供 chunk 或 optimistic 的新默认值：

```bash
export MODEL_PATH=/workspace/models/Qwen3-8B

: "${ONLINE_CHUNK:?请先设置线上 chunked-prefill-size 原值}"
: "${ONLINE_OPTIMISTIC:?请先设置线上 optimistic-prefill-attempts 原值}"

case ",${SGLANG_PLUGINS:-}," in
  *,remote_attn,*) ;;
  *) export SGLANG_PLUGINS="${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}remote_attn" ;;
esac
export HANDOFF_PRE_LAYERWISE=remote
export HANDOFF_PD_DIR=/workspace/remote-attn/run-001/journal
export MC_FORCE_TCP=1
export SGLANG_DISAGG_STAGING_BUFFER=1
```

每次 pair 重启都换新的 `run-001` 路径，P/D 的 HANDOFF_PD_DIR 指向同一份实际
目录。该目录交换 offer、ready、done，KV 仍走 Mooncake。角色各自的 CTL_DIR
存放 rank readiness；它们不能共用。

`SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES` 沿用线上有效值；如果旧入口通过
`common.sh` 默认设置了 2048，改成直接启动后须显式照录这个有效值，否则会变成
原生未设置时的 0。不要为了接入而重设 Batch，也不要 source 旧实验 common.sh。

层组默认 auto；清除此前实验遗留的 group 覆盖。token chunk 完全沿用 ONLINE_CHUNK。
性能轮确认 HANDOFF_PROFILE 和 HANDOFF_RPC_PROFILE 为 0（默认均为 0）。

## 3. P 原生启动命令

在 P 终端执行公共环境后：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export HANDOFF_CTL_DIR=/workspace/remote-attn/run-001/control/prefill
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=1
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=0

mkdir -p "$HANDOFF_CTL_DIR" "$HANDOFF_PD_DIR" /workspace/handoff_state
python -m sglang_remote_attn.check

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port 31016 \
  --tp-size 2 --page-size 1 \
  --chunked-prefill-size "$ONLINE_CHUNK" \
  --optimistic-prefill-attempts "$ONLINE_OPTIMISTIC" \
  --disable-overlap-schedule \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-decode-allocation-policy early \
  --disaggregation-bootstrap-port 35409
```

将线上原有上下文长度、显存比例、dtype、attention backend 和并发等参数继续放入
同一条命令，避免依赖不同入口的隐式默认。以上只列出与接入相关的参数。

P 的启动命令删除 `--disable-radix-cache` 即可启用，无需新增插件环境变量。
支持原生 `RadixCache` 和默认 `UnifiedRadixCache` 的设备内 FULL 组件；不支持
HiCache、外部 KV linker、session refs 或混合组件缓存。Session 请求走原生 PD。
普通请求与中间 chunk 保留原生缓存和 optimistic prefill 行为；只有已经激活分层
最终块的请求会延迟 checkpoint 和回收。关闭 P 缓存仍可作为原来的对照组。

## 4. D 原生启动命令

在 D 终端执行同样的公共环境后：

```bash
export CUDA_VISIBLE_DEVICES=2,3
export HANDOFF_CTL_DIR=/workspace/remote-attn/run-001/control/decode
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=0
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=1

mkdir -p "$HANDOFF_CTL_DIR" "$HANDOFF_PD_DIR" /workspace/handoff_state
python -m sglang_remote_attn.check

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port 31017 \
  --tp-size 2 --page-size 1 \
  --chunked-prefill-size "$ONLINE_CHUNK" \
  --disable-radix-cache --disable-overlap-schedule \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-decode-allocation-policy early \
  --disaggregation-bootstrap-port 35409
```

D 的其他线上参数同样保留。默认每个 D rank 需要 4 GiB pinned host ring 加
4 GiB GPU shadow，此外还有模型、KV 和 workspace；显存比例须留出空间。
目录 `/workspace/handoff_state` 是现有 runtime 固定诊断路径，服务账户需要可写。

两端保留线上原生 graph backend、capture buckets/max-bs 配置；同一角色各 TP rank
必须一致。插件不要求新增环境变量，不强制扩大原生 capture 范围；分段图在启动时额外捕获，服务中不重新 capture。
若原命令明确关闭了 graph，需先移除关闭配置；本分支原生 CLI 使用
`--cuda-graph-backend-decode full` / `--cuda-graph-backend-prefill breakable` 等逐阶段参数，
也可沿用线上 `--cuda-graph-config`。该原生参数不属于启动包装器。

当前 Remote 分段功能默认开启，D 必须使用 `--cuda-graph-backend-decode full`，
原生实际 capture buckets 须包含 1、2、4，最大运行请求数至少为 4。
沿用线上完整桶列表；需要时补入这三个值，不要把线上列表替换为仅 1/2/4。
若线上 `--cuda-graph-config` 已设置 decode backend，以其中的 full 配置为准。
分段功能需要 `cuda-python`，暂不支持 torch.compile、debug-eager、memory saver 或自定义 full backend。
只在 D 设置 `HANDOFF_REMOTE_SEGMENTED_GRAPH=0` 并重启，可恢复“Remote eager、Local full”对照；
不设置或设置为 `1` 时，在现有 remote 模式下启用分段图。P 无需增加环境变量。

## 5. 原生 SMG 启动

使用独立终端，不加载 P/D 的插件环境。等模型加载完成：

```bash
curl --fail --max-time 10 http://127.0.0.1:31016/health
curl --fail --max-time 10 http://127.0.0.1:31017/health
ls /workspace/remote-attn/run-001/control/prefill/ready.*.json
ls /workspace/remote-attn/run-001/control/decode/ready.*.json

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:31016 35409 \
  --decode http://127.0.0.1:31017 \
  --host 127.0.0.1 --port 31000
```

TP2 两个角色都应有当前进程生成的 ready.0.json 和 ready.1.json；核对 PID、
tp_size 和启动时间。SMG 使用原生分发和 bootstrap，不需要实验 gateway 或 D1。

## 6. 验证真实激活与输出

先通过 SMG 检查流式请求。首 token 由 P 完成全输入 Prefill 后采样，再经 D
PREBUILT 和 SMG 输出；提前元数据可使它不必等待完整 KV 传输。
stream=false 时客户端仍等完整结果，不能用这个响应耗时测首 token。

```bash
curl -N --fail http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain how a KV cache works.","stream":true,"sampling_params":{"temperature":0,"max_new_tokens":32}}'
```

该短请求只证明接口可用。随后运行覆盖实际 chunk 边界、长输入及并发的请求：

```bash
python -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port 31000 --tokenizer "$MODEL_PATH" \
  --dataset-name random --random-input-len 8192 --random-output-len 128 \
  --random-range-ratio 1 --num-prompts 16 --max-concurrency 4 --request-rate inf \
  --output-file /workspace/remote-attn/run-001/bench.jsonl

# 请求及后台回收结束后检查；未回收时稍后重查并排查超时。
python /workspace/sglang/test/manual/remote_attn/pre_layerwise_report.py \
  /workspace/remote-attn/run-001/journal --require-remote \
  > /workspace/remote-attn/run-001/report.json
```

检查所有目标 TP rank：activated_shards > 0、early_admitted_shards > 0、实际
remote_calls > 0，host_staging/native_staging/recv_host_staging 为 true，激活
请求全部 completed/released 且无 error。报告不包含未激活请求的完整分母。

auto 在合格最终块计算量不足 3072 tokens 时，remote 会合成一组；KV 提前到齐时
可以没有 Remote 调用。专门覆盖多层组时，在 P/D 两端统一设置
`HANDOFF_PRE_LAYERWISE_GROUP_SIZE=4`，新目录重启；保持线上 chunk/optimistic/Batch
不变。正式性能轮恢复 auto。只有采用多层组时才另外要求 `--require-overlap`；
控制侧时间不能替代 GPU overlap trace。首 token 变快也不能单独证明 Remote。

完整输出使用相同 input_ids 和 greedy 配置与基线比较，再测取消、故障和回收。
保持相同 GPU、到达率和请求集，对比线上原配置、受限原生 PD、layerwise、remote；
记录 TTFT、第二个实际 token、前若干 token 间隔、E2E、吞吐和错误率。
A'→layerwise 同时包含 staging 改变；layerwise→remote 还包含提前元数据和响应，
需要据此归因。SSE 条目不一定对应单个 token。

### 验证 eager → graph 恢复

诊断轮在 D 启动前设置已有的 `SGLANG_LOG_DECODE_GRAPH_KEY=1`，并保存各 rank 日志。
该开关逐次记录实际 decode replay，正式性能轮关闭。清除旧实验的
`HANDOFF_GRAPH_DISABLE_ANY_STATE=1` 覆盖，确认启动日志已完成 graph capture。
通过 SMG 发足够长的输出请求，让同一请求覆盖 Remote 和 Local 两个阶段：

1. 启动时各 D rank 应有 `captured remote batch=1/2/4 attention_breaks=层数`。
   Remote batch 1/2/4 应出现 `path=remote-segmented all_local=False`，实际执行日志为
   `[remote-attn-graph] replay bank=remote batch=...`；未覆盖 batch 出现 `path=eager`。
2. 最慢 shard 的所有层完成 GPU scatter、原生传输完成且接收租约统一回收后，各 rank 应出现
   `path=graph-eligible all_local=True`；实际执行日志应为 `replay bank=full`。
   `all_local=True` 但 shape 不在原生 capture 范围时仍为 eager，符合预期。
3. 配合 report 的各 shard `remote_calls > 0` 和 GPU trace，确认 segment replay 之间有
   图外 attention/RPC/merge，Local 后为 full replay。再加入未导完的新请求，确认重新
   进入 segmented 或未覆盖 shape 的 eager，完成后再次恢复 full。

Progressive 使用已有提前准入/回收的 TP 投票维护相同的接收租约集合。非空期间全组
segmented/eager，集合统一变空后恢复原生 full graph 判断；graph 判断本身不再增加 collective，
正常 Local decode 也不扫描逐层事件。Remote 期间已有的传输完成轮询同步仍然保留，
不能称为整个生命周期只同步两次。启动时额外做一次配置/capture manifest 一致性检查。
相较于所有层刚就绪就恢复，等待原生完成与统一回收可能多保留一段 eager 时间。

各 TP rank 的原生输入、配置与 capture 范围仍须一致。Local 阶段沿用原生 eligibility，
不再逐步检测原生 predicate 的 rank 间差异。旧手动迁移/import 命令会在全组撤销此
fast path，该进程后续改用逐 forward 的保守投票；普通 status 查询不撤销。
对比同一模型/请求/线上 chunk 和 optimistic 配置下 graph 开关两组的输出、ITL、吞吐。
CPU 已验证状态机和真实双进程一致性；GPU 上的数值与实际 graph replay 仍须执行本节验收。
实现及本地证据见 [CUDA graphs 适配记录](validation/PROGRESSIVE-CUDA-GRAPHS.md)。
分段图的 GPU 探针与验证范围见 [Remote 分段验证](validation/REMOTE-SEGMENTED-GRAPHS.md)。

## 7. 撤流与本地检查

模式没有热切换能力。先撤掉候选 pair 的新流量，等普通请求、Remote RPC 和传输
全部 drain，再由现有 supervisor 停止 P/D/SMG。回原生基线时在 P/D 的启动环境
unset HANDOFF_CTL_DIR，使插件不注册 hooks，同时关闭 HANDOFF_PRE_LAYERWISE、
SGLANG_DISAGG_STAGING_BUFFER、HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING 及
SGLANG_MOONCAKE_HOST_STAGING_ALL，再用新目录启动。超时不构成安全释放证明。

```bash
python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

CPU 合约和接口预检不代表目标 GPU 的完整服务、输出正确性或性能验收。

### 验证 P radix 命中与读租约回收

先用相同长输入执行预热，等待该请求在 journal 中出现 `released.json`，再提交
共享前缀的请求。保留线上 optimistic、chunk、传输 Batch 和 D 配置；使用已有
长输入请求集，避免改变输入分布。不要在源租约仍活跃时测刚生成尾部的命中，
因为这段 KV 要在 Remote 和原生传输完成后才交给缓存树。

```bash
python /workspace/sglang/test/manual/remote_attn/pre_layerwise_report.py \
  /workspace/remote-attn/run-001/journal --require-remote --require-p-radix
```

报告中 `p_radix_leased_shards`、`p_radix_closed_shards` 应与 `activated_shards`
相等；预热后的请求应有 `cached_prefix_tokens > 0`（各 rank 都检查），并记录
`deferred_cache_checkpoints`。这证明锁已接入、命中存在且正常回收；不代替 GPU
正确性验证。单独比较 P cache off/on：相同 input_ids、greedy 输出应一致，测
TTFT、输入计算 token 数、吞吐和 P KV 占用高水位；加并发、淘汰压力和取消请求，
确认回收后容量恢复。D 保留 `--disable-radix-cache`，两端保留
`--disable-overlap-schedule`。

P 的共享前缀持有独立读锁；当前最终块的私有尾部在所有 TP rank 的传输及 RPC
读取排空前不做 checkpoint/去重/释放。正常完成后走原生 checkpoint/release，
再释放独立读锁；取消请求不把未完成的尾部写进缓存。租约超时或释放状态不明时
保留页并报错，不能把该轮计入性能结果。延迟发布尾部会增加短时 KV 占用，也可能
让同时到达的相同尾部重复计算，需在目标负载下量化。
