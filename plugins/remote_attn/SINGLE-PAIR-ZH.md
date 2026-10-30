# 原生 SMG + 单 pair P/D 接入

使用原生 `sglang.launch_server` 和 SMG PD router，不使用 JSON 配置或启动包装器。
新增能力通过 P/D 的 `remote_attn` 插件启用；SMG 无需设置插件环境变量。
核心和插件必须来自同一份包含 progressive 扩展的构建，不能只安装插件。
依赖、容量和完整验收见 [部署手册](DEPLOYMENT-ZH.md)。

当前范围是同机 Qwen3 dense NHD MHA BF16/FP16、同 TP、PP=DP=CP=1、page size 1、
early allocation、mooncake_tcp，并关闭 radix cache、CUDA graphs、overlap scheduler。
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

安装态预检应输出 `hooks: 23`、`core_interfaces` 正数、`send_probe: true`。
它检查实际加载的接口：保留 receiver 的 metadata commit、选层发送与完成回调、
staging 分组、host copy、字段初始化及 typed env 声明；插件注册时也会执行同样的
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
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-decode-allocation-policy early \
  --disaggregation-bootstrap-port 35409
```

将线上原有上下文长度、显存比例、dtype、attention backend 和并发等参数继续放入
同一条命令，避免依赖不同入口的隐式默认。以上只列出与接入相关的参数。

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
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-decode-allocation-policy early \
  --disaggregation-bootstrap-port 35409
```

D 的其他线上参数同样保留。默认每个 D rank 需要 4 GiB pinned host ring 加
4 GiB GPU shadow，此外还有模型、KV 和 workspace；显存比例须留出空间。
目录 `/workspace/handoff_state` 是现有 runtime 固定诊断路径，服务账户需要可写。

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
