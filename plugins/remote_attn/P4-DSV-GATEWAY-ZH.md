# P4 Gateway：DSV 原生 PD / D2D / SSE / 重启恢复

入口为 `control_plane/p4_gateway.py --backend dsv`（或 `HANDOFF_GW_BACKEND=dsv`）。
Qwen 默认路径保持原协议。DSV 模式使用 `p4_dsv_gateway.py`，不会把 DSV 请求转入
旧 Qwen `reserve/import_wave/admit_decode`，也不代表已接入原生 SGLang Model Gateway（SMG）。

## 部署与启动

先满足 [DSV D2D 前置条件](DSV-D2D-REMOTE-ZH.md)：D0/D1 使用相同模型和 TP/wire
schema，Decode 关闭 radix cache，并启用本分支的原生 PD Remote 插件。控制目录和
nonce/TP 广播沿用 `p2d_driver.send`，worker 应分别在 `<CTL>/src`、`<CTL>/dst` 发布
ready/result 文件。控制文件可访问性与原 P4 部署相同；没有新增远程控制文件同步服务。

以下地址和模型目录应替换为实际部署配置。Bootstrap host/port 指向 P 的元数据服务，
`HANDOFF_PD_PREFILL_URL` 指向 P 的 HTTP 服务。

```sh
export HANDOFF_MODEL=/workspace/models/your-dsv-model
export HANDOFF_STATE_DIR=/workspace/handoff_state_dsv
export HANDOFF_CTL_DIR=/workspace/handoff_ctl_dsv
export HANDOFF_PD_PREFILL_URL=http://127.0.0.1:30000/generate
export HANDOFF_PD_BOOTSTRAP_HOST=127.0.0.1
export HANDOFF_PD_BOOTSTRAP_PORT=35409
python plugins/remote_attn/control_plane/p4_gateway.py \
  --backend dsv --port 31080 --poll-ms 10 \
  --out /workspace/handoff_state_dsv/phases.jsonl
```

Gateway 使用 `HANDOFF_MODEL` 的 tokenizer 编码文本；也可以直接传 `prompt_ids`。
`/v1/completions` 沿用 P4 的 token/seq SSE 格式，不是完整 OpenAI 文本增量兼容层。

```sh
curl -N http://127.0.0.1:31080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"rid":"demo-001","prompt_ids":[1,2,3],"max_new_tokens":128,"cut":32,"growth_tokens":128}'
```

默认 `cut=max_new_tokens`，常规请求自然运行；指定较小 cut 或调用 drain 才触发迁移。
采样参数通过 `sampling_params` 传入；默认 greedy、`ignore_eos=true`。后端仍只支持
既有纯文本、无 grammar/custom logits/logprob 状态迁移的范围。

## 原生 PD 与在线迁移

1. **SUBMIT**：先落盘提交意图、内部请求 ID、bootstrap room 和 P 配置，再发送
   `d2d_submit_pd`。请求进入原生 Decode prealloc/transfer 队列；P 的 HTTP 请求由
   **D worker 的线程**持有，Gateway 重启不会顺带断开它。同一提交 epoch 禁止重复入队。
2. **SOURCE**：从原生请求历史取 token。P→D Remote 尚未退休或不在可冻结 Decode
   边界时继续等待，不抢占已有 Remote 窗口。
3. **PLAN → RESERVE → FREEZE**：D0 提供长度增长范围，D1 真正占住 FULL/SWA/状态行；
   所有 TP rank 预留成功后才冻结 D0。SWA 覆盖所有允许冻结点的窗口并集。
4. **IMPORT**：消费已有预留，同步恢复 SWA/压缩器/indexer 状态，启动 Mooncake bulk
   KV 后台搬运。沿用独立 D2H Host staging、尾页保护、逐 entry TP 退休和渐进 Local。
5. **COMMIT**：在发送第一条 commit 前 fsync 不可逆的提交意图。后端先集体确认所有
   源 shard 的 ownership ACK，再接入任何目标 Decode batch；仅部分 ACK 时所有目标
   shard 均保持未接纳。原生接纳中途失败则 TP 集体 quarantine，不重放不确定的接纳。
6. **TARGET → LOCAL**：commit 的所有 rank 返回 collective confirmation 后，SSE 输出
   owner 才改为 D1。无需等完整 KV 拷贝；Source read lease 与 KV 仍由原有 GPU/IO/TP
   fence 自动退休，Gateway 不调用强制 free。

每次阶段变化前落盘必要意图和对应元数据。源/目标状态由独立角色线程轮询，避免一个
worker 的慢状态调用阻断另一个 worker 的输出轮询；`--out` 记录阶段 JSONL。

## SSE 接续

每个 token 带稳定的索引，示例：

```text
id: 32
data: {"seq":32,"token_id":123,"owner":"dst","epoch":"move-...","rid":"demo-001"}
```

- 同一连接只发送 `seq > last_sent` 的 token；D1 的完整历史包含冻结前输出，不重发前缀。
- 断线默认保留请求，允许恢复连接；显式取消使用 `/v1/cancel`。
- 重启后客户端携带 **实际已收到的** `Last-Event-ID`，或使用 `after` 查询参数：

```sh
curl -N 'http://127.0.0.1:31080/v1/attach?rid=demo-001' \
  -H 'Last-Event-ID: 31'
# 等价：/v1/attach?rid=demo-001&after=31
```

TCP 写出不等于客户端收到，不能靠服务器日志猜测送达位置；重新连接会按客户端游标
补发。每个请求只允许一个活跃 SSE reader；旧连接尚未退出时 attach 返回 409。
网络或 ownership 证明不足时保持 owner，不借机恢复源端；阻塞错误通过 `event: status`
报告。正常结束发送 `[DONE]`。完成且 KV 已安全释放后，最终 token 历史写入 journal，
即使源 worker 随后停止，也能继续补读已结束请求。

## 取消与重启恢复

```sh
curl -X POST http://127.0.0.1:31080/v1/cancel \
  -H 'Content-Type: application/json' -d '{"rid":"demo-001"}'
```

- commit 意图落盘前：先在目标写取消 tombstone、停止并证明 prepared import 的 IO
  已排空，再在源端撤销 reader 并 resume；之后通过原生取消路径结束源请求。
- commit 意图已经落盘：只 roll forward 到目标，再走目标原生取消。回复丢失或部分
  rank 已 commit 不能成为恢复源端的理由。
- 取消 tombstone 同时防止迟到的 import/commit 和迟到的 freeze；重复 resume 不重复
  构造 Decode batch。部分分配、未知 DMA 完成、释放中途失败保留内存。
- `dsv_gateway_journal.json` 通过临时文件、fsync、原子替换持久化；独占 flock 防止
  两个协调器同时重放同一 journal，损坏/不兼容日志不自动清空。
- 每个 TP rank 有 worker 实例 UUID。恢复和每条变更命令都校验 UUID；重启 Gateway
  可重放同一 epoch，重启 worker、lease 超时或 quarantine 则停止自动恢复并保留资源。
  重启不能恢复已经断开的 TCP 连接，客户端仍须 attach。

## Drain 与路由封锁

```sh
curl -X POST http://127.0.0.1:31080/v1/drain \
  -H 'Content-Type: application/json' -d '{"mode":"d2d","targets":["dst"]}'
curl http://127.0.0.1:31080/v1/drain/status
curl http://127.0.0.1:31080/v1/router/status
```

先原子保存 Source admission fence 与成员快照，再开始迁移；后续新请求从第一次提交
起就路由到 D1。预提交失败可安全回滚；drain 对回滚成员最多自动尝试三次，之间退避，
随后保留 Source 正常执行并报告状态，不强行搬走或释放。

只有成员均到 Local/终态、源 read lease/KV 已释放，而且所有源 rank 的原生队列及插件
资源均为空时才报告 `SAFE_TO_STOP`。未接管的原生队列也会阻止这一判断。所有新流量
应通过 P4；本模式不封锁用户绕过 P4 直接访问原生端口的并发新请求，也不自动接管它们。

`POST /v1/drain/stop` 只在 `SAFE_TO_STOP` 后持久化 `STOPPED` 标记，保持 Source admission
fence，并停止轮询随后可被外部停止的源 worker；不发送进程 kill。Gateway 重启后继续
保持 fence，已 Local 的目标请求不再依赖源 worker 在线。

## 验证边界

见 [P4 DSV CPU / SIGKILL / 4060 验证](validation/p4-dsv-gateway-4060.md)。
真实 CLI/HTTP/SSE、进程 SIGKILL、TP2 共识、CUDA sparse kernel 和 Mooncake TCP 分别验证；
模型 scheduler/allocator admission 使用明确 ABI 夹具。尚未加载完整 DSV 权重验证整模
HTTP PD，也没有将这些结果作为 H20 TP4/RDMA 性能或原生 SMG 接入验收。

## 连接已有 D0/D1 与主动迁移测量

现有 Decode worker 已发布 ready 文件时，不必为了 gateway 重建 `src/dst` 目录。
指定两个实际目录（包含 `ready.0.json` 等文件），或使用环境变量
`HANDOFF_SRC_CTL_DIR` / `HANDOFF_DST_CTL_DIR`：

```sh
HANDOFF_MODEL=/workspace/models/your-dsv-model \
HANDOFF_STATE_DIR=/workspace/p4-active-migration \
HANDOFF_PD_PREFILL_URL=http://127.0.0.1:31016/generate \
HANDOFF_PD_BOOTSTRAP_HOST=127.0.0.1 HANDOFF_PD_BOOTSTRAP_PORT=35409 \
python plugins/remote_attn/control_plane/p4_gateway.py --backend dsv \
  --src-control-dir /workspace/control/D0 \
  --dst-control-dir /workspace/control/D1 --port 31080
```

Bootstrap 端口需使用 P 实际发布的端口。控制目录、ready 文件的 PID 与所有 TP rank
必须在 gateway 所在机器可见；这不是跨机文件同步机制。
Driver 为每条命令生成 nonce，并在对应目录完成逐 rank 控制协议和 flock。
不要直接写无 nonce 的 `cmd.0.json`。

通过该 gateway 提交新请求，可显式指定公开 `rid`；gateway 保存它与内部 source/target
rid 的映射。已有 router/OpenAI 请求不自动成为 P4 托管请求。`cut` 触发的完整流程包含
PLAN → RESERVE → FREEZE → IMPORT → COMMIT，不应直接跳过预留执行 freeze。
所有迁移测试流量需改发 gateway；双 router 停止投放只测自然排空。

```sh
python test/manual/remote_attn/bench_p4_dsv_migration.py \
  --gateway http://127.0.0.1:31080 --prompt-ids /workspace/prompt-32k.json \
  --concurrency 4 --cut 32 --max-new-tokens 512 --output /workspace/d2d-c4.json
```

`prompt-32k.json` 是同一模型 tokenizer 生成的输入 token ID 列表。输出长度应足够让迁移
在源请求结束前完成。探针逐请求要求实际看到 `owner=src` 后切到 `owner=dst`、seq 连续、
不切回源、最终收到 DONE。自然完成或只看到源输出会标为失败，而不是迁移成功。
输出 TTFT、首个目标 token 时间、E2EL 和最大 token 间隔；这些不等于 bulk-copy 耗时，
也不证明 SAFE_TO_STOP，后者仍需读取 `/v1/drain/status` 和源 lease 退休状态。
