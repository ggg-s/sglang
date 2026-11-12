# 3P3D 原生 SMG 接入迁移控制器

SMG 继续注册三个真实 P，三个 Decode URL 指向同一个迁移控制器的三个监听端口。每个端口对应一个逻辑 D。控制器沿用 SMG 提供的 bootstrap host、port 和 room。SMG 发 P 请求，控制器用外部 Prefill 模式接纳 Decode，并持有返回 SMG 的 SSE。

## 应用和启动

本分支提供普通 HTTP 模式和原生 PD Decode 模式。已有引擎启动方式继续使用，D0/D1/D2 使用同版本插件、同模型与 KV 布局。现有文件控制通道要求控制器能读取各 D 的控制目录和检查 worker PID。

```bash
python -m pip install -e ./plugins/remote_attn --no-deps
```

将 [3P3D 配置示例](deploy/pool.3p3d.native-pd.json) 复制为 `/pool/pool-native-pd.json`，替换实际 P 地址、控制目录和模型名。Prefill 的 bootstrap 地址应与 SMG 下发值一致，Decode URL 应与 SMG 的 `--decode` 完全对应。配置里的 `smg_admin_urls` 列出该池的所有 SMG 副本。若 `/workers` 需要认证，在 `native_pd` 下配置可读的 `smg_admin_token_file`，内容为用于查询所有副本的 Bearer token。使用新的持久化状态目录。

在三个 Decode 端口的 `/health` 都可用后启动 SMG。下面假设迁移控制器和 SMG 同主机。

```bash
python plugins/remote_attn/control_plane/migration_controller.py \
  --config /pool/pool-native-pd.json \
  --host 127.0.0.1 --admin-port 31081 \
  --natural-drain-ms 300 --precopy-window-ms 400 \
  --out /pool/state/migration-controller-native-pd/handoff.jsonl
```

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://p0:31016 35409 \
  --prefill http://p1:31016 35409 \
  --prefill http://p2:31016 35409 \
  --decode http://127.0.0.1:31117 \
  --decode http://127.0.0.1:31118 \
  --decode http://127.0.0.1:31119 \
  --health-check-endpoint /health \
  --health-check-interval-secs 1 --health-failure-threshold 2 \
  --host 0.0.0.0 --port 32000
```

控制器用 `/workers` 中已注册 Decode 的 `is_healthy: false` 确认摘流。部署时保留摘流 worker 的注册记录。可用 `curl http://127.0.0.1:32000/workers` 核对六个 P/D 地址。

## SMG 0.3.2 HTTP PD 入口

使用 `POST /generate`，请求体带一维 `input_ids`、`sampling_params.max_new_tokens` 和 `stream: true`。SMG 给 P/D 两路注入同一组标量 bootstrap 字段。控制器返回原生 SSE，`output_ids` 累积增长，`meta_info.id` 在迁移前后保持一致。

PyPI 的 `sglang-router==0.3.2` 在 `/v1/completions` 上接受字符串和字符串数组。传整数数组会在 SMG 返回 HTTP 422。当前控制器的旧 `/v1/completions` 只接受预分词输入，因此该版本的 SMG 验收统一用 `/generate`。本地已用 PyPI 0.3.2 的 arm64 macOS release wheel、模拟 P 和模拟 Decode worker 跑通真实 SMG 1P2D HTTP 转发和迁移续流。Linux/GPU 的 1P2D 和 3P3D 仍需现场验收。

压测前可先使用部署模型生成的 token ID 文件发一个小请求。

```bash
jq -nc --slurpfile ids /pool/bench/prompt-64k.ids.json \
  '{model:"dsv4-flash",input_ids:$ids[0][:16],stream:true,
    sampling_params:{max_new_tokens:8,temperature:0}}' |
  curl -iN --max-time 30 http://127.0.0.1:32000/generate \
    -H 'Content-Type: application/json' --data-binary @-
```

## 确认 Decode 身份

以示例配置为准，D1 的逻辑地址是 `127.0.0.1:31118`，实际控制目录是 `/pool/ctl/D1`。D2 分别是 `127.0.0.1:31119` 和 `/pool/ctl/D2`。引擎启动时设置相应的 `HANDOFF_INSTANCE_ID`。先查控制器加载的标签，再确认 SMG 识别了两个逻辑地址。

```bash
curl -sS http://127.0.0.1:31118/health  # {"ready":true,"decode":"D1"}
curl -sS http://127.0.0.1:31119/health  # {"ready":true,"decode":"D2"}
curl -fsS http://127.0.0.1:32000/workers |
  jq '.workers[] | select(.worker_type == "decode") | {url, is_healthy}'
```

需要核对物理进程时，在控制器所在的 PID 命名空间读取 `/pool/ctl/D1/ready.0.json` 和 `/pool/ctl/D2/ready.0.json` 的 rank、TP 和 PID。

## 验收 D1 → D2

先通过 SMG 发送 `/generate` 流式请求。第一版覆盖单条预分词纯文本流，支持 `max_new_tokens`、`temperature`、`top_p` 和 `ignore_eos`。

确认 D1 有流式输出后，从控制器所在环境下发：

```bash
curl -fsS http://127.0.0.1:31081/v1/drain \
  -H 'Content-Type: application/json' \
  -d '{"src":"D1","dst":"D2"}'

curl -fsS http://127.0.0.1:31081/v1/drain/status
```

1. `CORDONING` 期间 D1 的逻辑地址返回健康检查 503。已经配对的请求仍可进入 D1。
2. 所有 SMG 副本确认摘流后，控制器持久化源端封锁、成员清单和自然排空截止时间。启用 `--natural-drain-ms` 时，`NATURAL_DRAINING` 期间源端继续 Decode。成员全部结束且源端空闲时可提前进入 `SAFE_TO_STOP`。此后迟到的已配对请求沿用原 room，进入显式指定的可用目的 D。
3. 比较同一请求的 `meta_info.id`、累积 token 前缀和 `[DONE]`。用下方命令确认该轮源/目的和迁移结果。至少一条成员同时出现 `owner=D2` 和 `migration_outcome=migrated`，才说明有在途请求实际迁到 D2。成员为空只证明摘流。只有 `SAFE_TO_STOP` 才能停物理 D1。控制器和逻辑 D1 监听进程继续服务迁走的流。
4. 再验证 P leg 失败导致 SMG 关闭 Decode HTTP 连接。确认首 token 前的 D 请求也会取消，原生 KV 走正常回收。

```bash
curl -fsS http://127.0.0.1:31081/v1/drain/status |
  jq '{state, src, dst, moved: [(.members_state // {}) | to_entries[] |
    select(.value.owner == "D2" and .value.migration_outcome == "migrated") |
    {rid: .key, phase: .value.phase}]}'
```

自然排空窗口结束后，控制器为仍在生成的请求执行 PLAN 和 RESERVE。目标全部 rank 确认 RESERVE 后才开始单请求 Pre-copy 窗口。窗口到期后收尾已开始的帧。请求等待切换额度时继续在源端 Decode，自然结束就走现有回滚，释放目标端预留。两个窗口的 CLI 默认值都是 `0`，分别关闭对应等待和限时 Pre-copy。上面的 `300/400 ms` 用于首轮验证。

- `--max-concurrent-migrations` 接受 `1/2/4`，默认 `1`，限制从 Freeze 到目标接管的请求数。接管后释放切换额度，后台回收仍占总在途额度。
- `--max-preparing-migrations` 接受 `1–4`，默认与切换额度相同，限制 PLAN、RESERVE 和 Pre-copy 请求数。两项均为 `1` 时，总在途为 `1`，保留整条迁移串行。其他组合的总在途为两项之和，最多 `8`。
- worker 支持异步 handoff 和控制 RPC 打包时，控制器按物理 Decode 和 TP 实例身份组批。切换波次先于 Pre-copy 波次发送，每个 Pre-copy 波次最多搬两个 COPY 帧。每个请求仍按分配器实际可用容量预留。

## 撤销迁移和摘流超时

控制器配置 `native_pd.smg_cordon_timeout_s` 默认 **15 秒**，从发出摘流起计。所有配置的 SMG 副本都确认源 Decode 的 `is_healthy: false` 后才开始迁移。超过期限时，控制器自动撤销该轮摘流，健康的源 D 恢复接流。状态为 `ABORTED`，`abort_reason` 为 `cordon_timeout`，`cordon_error` 保留最后观察到的管理查询错误或仍健康的副本。

手动撤销时提交当次操作的 ID。重复提交同一轮的撤销可查询进度，过期的 ID 返回 HTTP 409。

```bash
admin=http://127.0.0.1:31081
drain_id=$(curl -fsS "$admin/v1/drain/status" | jq -er '.id')
jq -n --arg id "$drain_id" '{drain_id: $id}' |
  curl -fsS "$admin/v1/drain/abort" \
    -H 'Content-Type: application/json' --data-binary @-
curl -fsS "$admin/v1/drain/status" |
  jq '{id, state, abort_reason, cordon_error, members_state}'
curl -fsS http://127.0.0.1:31118/health
```

- 摘流确认期间撤销直接进入 `ABORTED`。源 D 的健康探针恢复后，SMG 按原健康检查周期重新将其加入路由。
- 迁移开始后撤销会停止选择新请求，并恢复健康源 D 的准入。提交前的迁移释放目的端预留，再由源端继续生成。已记录提交意图的迁移继续完成提交和源端回收。
- `ABORTING` 表示仍有迁移资源在清理。此时可查看成员阶段和错误，清理完成后可重新发起 drain。只有 `SAFE_TO_STOP` 允许停源 D，已完成的 drain 不接受撤销。

## 四条长请求全迁和长短混跑

在隔离的 3P3D 池上使用同一批引擎、输入 ID 文件和预热。每轮使用独立的控制器状态目录，上一轮 SSE 全部结束后再重启测试控制器和 SMG。测试 SMG 在上面的命令中加入 `--prefill-policy round_robin --decode-policy round_robin --disable-retries`，并将 `--health-failure-threshold` 设为 `1`。测量批次从三个 Decode 均健康开始，期间只发送本轮基准请求。

先跑四条长请求全迁，设置 `natural_ms=0`、`precopy_ms=0`，分别跑 `k=1/2/4`。再跑长短混跑，设置 `natural_ms=300`，对选定的 `k` 分别跑 `precopy_ms=0/400`。每轮在终端一启动控制器，记下打印的状态目录。

```bash
mode=all-long
k=1
natural_ms=0
precopy_ms=0
# 混跑示例：mode=mixed，natural_ms=300，precopy_ms=0 或 400
variant="${mode}-k${k}-precopy${precopy_ms}"
state_dir=$(mktemp -d "/pool/state/native-pd-${variant}.XXXXXX")
echo "$state_dir"
jq --arg state_dir "$state_dir" '.state_dir = $state_dir' \
  /pool/pool-native-pd.json > "$state_dir/config.json"
python plugins/remote_attn/control_plane/migration_controller.py \
  --config "$state_dir/config.json" \
  --host 127.0.0.1 --admin-port 31081 \
  --max-concurrent-migrations "$k" --max-preparing-migrations "$k" \
  --natural-drain-ms "$natural_ms" --precopy-window-ms "$precopy_ms" \
  --out "$state_dir/handoff.jsonl"
```

启动本轮 SMG，确认三个 Decode 在 `/workers` 均为健康。终端二设置本轮的 `variant` 和控制器打印的 `state_dir`。输入文件是一份由部署模型 tokenizer 生成的一维 token ID JSON 数组，复现 64K 用例时提供 65,536 个 ID。脚本通过 SMG 发起流式请求，观察到每条流有文本后，从本机管理端口下发一次 D1→D2。传入同轮 journal 可按请求记录初始 Decode，区分源、目标和池中其他前台请求。

四条长请求全迁时发 12 条长请求，三个健康 Decode 轮询各接四条。脚本在此模式设置 `ignore_eos: true`。1P2D 下用 `--per-group 8`。

```bash
variant=all-long-k1-precopy0
state_dir='/pool/state/替换为终端一打印的目录'
python test/manual/remote_attn/bench_native_pd_mixed_drain.py \
  --smg http://127.0.0.1:32000 --admin http://127.0.0.1:31081 \
  --model dsv4-flash --prompt-ids /pool/bench/prompt-64k.ids.json \
  --mode all-long --src D1 --dst D2 --long-tokens 512 --per-group 12 \
  --journal "$state_dir/dsv_migration_journal.json" \
  --output "/pool/bench/${variant}.json"
```

长短混跑时发六条短请求和六条长请求，允许 EOS 自然结束。1P2D 下可用 `--per-group 4`。

```bash
variant=mixed-k2-precopy400
state_dir='/pool/state/替换为终端一打印的目录'
python test/manual/remote_attn/bench_native_pd_mixed_drain.py \
  --smg http://127.0.0.1:32000 --admin http://127.0.0.1:31081 \
  --model dsv4-flash --prompt-ids /pool/bench/prompt-64k.ids.json --mode mixed \
  --src D1 --dst D2 --short-tokens 192 --long-tokens 1024 --per-group 6 \
  --journal "$state_dir/dsv_migration_journal.json" \
  --output "/pool/bench/${variant}.json"
```

```bash
jq '{mode, valid_run, source_safe_to_stop_ms, activate_to_safe_to_stop_ms,
  peak_cutovers, control_wave_sizes, short_natural, moved,
  moved_precopy_bytes, requests: [.requests[] |
  {group, initial_decode, content_chunks, finish_reason, max_chunk_gap_ms,
   owner: .drain_member.owner, attempts: .drain_member.migration_attempts,
   remaining_bytes: .drain_member.remaining_copy_bytes.tp_sum,
   descriptor_to_import_ms: .drain_member.handoff_timing.descriptor_to_import_ms,
   cut_to_first_token_ms: .drain_member.handoff_timing.cut_to_first_token_ms}]}' \
  /pool/bench/all-long-*.json /pool/bench/mixed-*.json
```

两种模式都要求所有流收到一次结束原因和 `[DONE]`。全迁样本需要 D1 四条全部迁到 D2，源和目的端均有控制 wave，且 Freeze 意图到目标接管的重叠峰值达到本轮切换额度。混跑样本需要 D1 至少有一条短请求以零次迁移尝试自然完成、一条长请求实际迁到 D2。源端请求提前 EOS 或分流导致样本不满足条件时，脚本仍输出结果并返回非零。若 512 的输出不足以维持四条在途，成组用同一个更长输出上限重跑，并记录实际配置。

`source_safe_to_stop_ms` 从 POST 发出前计时，含 SMG 摘流。 `activate_to_safe_to_stop_ms` 从控制器确认摘流并封入口的边界计时，包含自然等待、迁移和源端回收。Pre-copy 对照同时看 `moved_precopy_bytes`、冻结剩余字节和切换时间。按 `initial_decode` 看未迁请求携带新增 token 的 SSE chunk 最大间隔，单 token ITL 以引擎计时为准。

本地回归覆盖三个回环监听端口、bootstrap 透传、同一 SSE 的 D2D 续流、摘流撤销、四路控制 RPC 打包、阶段额度及固定 Pre-copy 窗口。真实 3P3D GPU/RDMA 性能在目标环境验收。
