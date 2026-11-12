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

先通过 SMG 发送 `/v1/completions`，请求体使用当前模型 tokenizer 生成的一维 `prompt` token ID 列表，设置 `stream: true`、`n: 1`。第一版覆盖预分词纯文本流，提供 `temperature`、`top_p` 和 `max_tokens`，高级采样、chat、批量、logprobs 和 usage 扩展留待接入。

确认 D1 有流式输出后，从控制器所在环境下发：

```bash
curl -fsS http://127.0.0.1:31081/v1/drain \
  -H 'Content-Type: application/json' \
  -d '{"src":"D1","dst":"D2"}'

curl -fsS http://127.0.0.1:31081/v1/drain/status
```

1. `CORDONING` 期间 D1 的逻辑地址返回健康检查 503。已经配对的请求仍可进入 D1。
2. 所有 SMG 副本确认摘流后，控制器持久化源端封锁、成员清单和自然排空截止时间。启用 `--natural-drain-ms` 时，`NATURAL_DRAINING` 期间源端继续 Decode。成员全部结束且源端空闲时可提前进入 `SAFE_TO_STOP`。此后迟到的已配对请求沿用原 room，进入显式指定的可用目的 D。
3. 比较同一请求的 SSE ID、文本和 `[DONE]`。用下方命令确认该轮源/目的和迁移结果。至少一条成员同时出现 `owner=D2` 和 `migration_outcome=migrated`，才说明有在途请求实际迁到 D2。成员为空只证明摘流。只有 `SAFE_TO_STOP` 才能停物理 D1。控制器和逻辑 D1 监听进程继续服务迁走的流。
4. 再验证 P leg 失败导致 SMG 关闭 Decode HTTP 连接。确认首 token 前的 D 请求也会取消，原生 KV 走正常回收。

```bash
curl -fsS http://127.0.0.1:31081/v1/drain/status |
  jq '{state, src, dst, moved: [(.members_state // {}) | to_entries[] |
    select(.value.owner == "D2" and .value.migration_outcome == "migrated") |
    {rid: .key, phase: .value.phase}]}'
```

自然排空窗口结束后，控制器每次只为一条仍在生成的请求执行 PLAN 和 RESERVE。目标全部 rank 确认 RESERVE 后才开始单请求 Pre-copy 窗口。窗口到期后收尾已开始的帧，并按现有协议 Freeze。期间请求自然结束时走现有回滚，释放目标端预留。只有上一条请求的源端资源回收完成，才启动下一条。两个窗口的 CLI 默认值都是 `0`，分别关闭对应等待和限时 Pre-copy。上面的 `300/400 ms` 用于首轮验证。

## 长短请求混跑 A/B

在隔离的 3P3D 池上使用同一批引擎、同一个输入 ID 文件和相同的预热。两轮自然排空均为 300 ms。A 直接 Handoff，B 在 RESERVE 后增加 400 ms Pre-copy。对照轮使用独立的控制器状态目录，上一轮 SSE 全部结束后再重启测试控制器和 SMG。测试 SMG 可在上面的命令中加入 `--prefill-policy round_robin --decode-policy round_robin --disable-retries`，并将 `--health-failure-threshold` 设为 `1`，两轮使用相同参数。

每轮在终端一启动控制器，先设置该轮变量。下面是 A，B 使用注释里的变量。

```bash
variant=A
precopy_ms=0
# B: variant=B; precopy_ms=400
state_dir=$(mktemp -d "/pool/state/native-pd-ab-${variant}.XXXXXX")
jq --arg state_dir "$state_dir" '.state_dir = $state_dir' \
  /pool/pool-native-pd.json > "$state_dir/config.json"
python plugins/remote_attn/control_plane/migration_controller.py \
  --config "$state_dir/config.json" \
  --host 127.0.0.1 --admin-port 31081 \
  --natural-drain-ms 300 --precopy-window-ms "$precopy_ms" \
  --out "$state_dir/handoff.jsonl"
```

启动本轮 SMG，确认三个 Decode 在 `/workers` 均为健康。终端二运行以下命令。输入文件是一份由部署模型 tokenizer 生成的一维 token ID JSON 数组，复现 64K 用例时提供 65,536 个 ID。脚本通过 SMG 并发发起两组流式请求，观察到每条流有文本后，从本机管理端口下发一次 D1→D2。

```bash
variant=A # B 轮改为 B
python test/manual/remote_attn/bench_native_pd_mixed_drain.py \
  --smg http://127.0.0.1:32000 --admin http://127.0.0.1:31081 \
  --model dsv4-flash --prompt-ids /pool/bench/prompt-64k.ids.json \
  --src D1 --dst D2 --short-tokens 192 --long-tokens 1024 --per-group 6 \
  --output "/pool/bench/mixed-${variant}.json"
```

```bash
jq '{valid_mixed_run, source_safe_to_stop_ms, short_natural, moved,
  moved_precopy_bytes, requests: [.requests[] |
  {group, content_chunks, finish_reason, max_chunk_gap_ms,
   owner: .drain_member.owner, attempts: .drain_member.migration_attempts,
   remaining_bytes: .drain_member.remaining_copy_bytes.tp_sum,
   cut_to_first_token_ms: .drain_member.handoff_timing.cut_to_first_token_ms}]}' \
  /pool/bench/mixed-A.json /pool/bench/mixed-B.json
```

有效样本需要 D1 至少有一条短请求以零次迁移尝试自然完成、一条长请求实际迁到 D2，且所有流收到一次结束原因和 `[DONE]`。短请求在摘流确认前已经结束或 SMG 分流导致 D1 缺少任一组时，脚本记录结果并返回非零。用相同参数重跑，或调整两组长度后成对重跑。A 的 `moved_precopy_bytes` 应为 0，B 应大于 0，才适合比较 Pre-copy 效果。对比冻结时剩余字节、切换时间、源端安全释放时间和未迁流的最大文本 chunk 间隔。SMG SSE 的文本 chunk 间隔用于客户端侧观察，单 token ITL 以引擎计时为准。

本地回归覆盖三个回环监听端口、bootstrap 透传、同一 SSE 的 D2D 续流、自然排空截止时间恢复、串行迁移和固定 Pre-copy 窗口。SMG 原生进程与真实 3P3D GPU/RDMA 性能在目标环境验收。
