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
2. 所有 SMG 副本确认摘流后，控制器持久化源端封锁和成员清单，逐条迁移。此后迟到的已配对请求沿用原 room，进入显式指定的可用目的 D。
3. 比较同一请求的 SSE ID、文本和 `[DONE]`。用下方命令确认该轮源/目的和迁移结果。至少一条成员同时出现 `owner=D2` 和 `migration_outcome=migrated`，才说明有在途请求实际迁到 D2。成员为空只证明摘流。只有 `SAFE_TO_STOP` 才能停物理 D1。控制器和逻辑 D1 监听进程继续服务迁走的流。
4. 再验证 P leg 失败导致 SMG 关闭 Decode HTTP 连接。确认首 token 前的 D 请求也会取消，原生 KV 走正常回收。

```bash
curl -fsS http://127.0.0.1:31081/v1/drain/status |
  jq '{state, src, dst, moved: [(.members_state // {}) | to_entries[] |
    select(.value.owner == "D2" and .value.migration_outcome == "migrated") |
    {rid: .key, phase: .value.phase}]}'
```

本地通过 10 项控制器测试和 8 项原有状态回归，覆盖三个真实回环监听端口、原生 bootstrap 透传、worker 侧仅入队 Decode、首帧前断连、所有 SMG 确认后封入口和同一 SSE 的 D2D 续流。SMG 原生进程、真实 3P3D GPU/RDMA 和线上请求兼容性仍待目标环境验收。
