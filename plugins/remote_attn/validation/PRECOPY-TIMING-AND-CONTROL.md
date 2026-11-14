# Precopy 对照与逐帧账本

## 对照开关

DSV/hybrid gateway 新增 `--disable-precopy`；默认保持开启。两组使用独立、干净的 gateway state，除该开关外保持相同的 cut32、c、输入和 max_new。分别跑 max_new=128/512，保留完整 `--out` JSONL。

关闭只禁止新帧：在 cut 前等待，到 cut 后直接 FREEZE。若恢复的 journal 已有 pending 帧，仍会完成它并将 receipt 纳入 freeze；所以旧 journal 恢复后并不保证 precopy_bytes=0。`/v1/router/status` 返回 `precopy_enabled`。

N3 比较每请求的 `freeze_ack_latency_ms`、`handoff_latency_ms`、D 首 token、源端 safe-to-stop 和 E2EL。不能把 D 首 token 延迟当作 freeze 本身。此次不修改 freeze 或迁移决策策略。

## 日志与计时

`--out` 新增 `type=precopy_stage` 行，以 rid/epoch/frame.chunk/stage 关联。状态接口的 `precopy_frame_events` 仅保留最近 256 条；完整实验使用 JSONL，避免窗口截断。

每次阶段尝试使用同一进程的 monotonic clock：

- `tick_wait_ms`：上一阶段处理完成到本阶段重新获得调度的时间。
- `dispatch_wait_ms`：yield 命令后到回复交还生成器，扣除 RPC 和控制槽等待；包含 wave 分派及回复分发等待。
- `slot_wait_ms`：gateway 获取该角色控制槽的等待。
- `rpc_ms`：driver.send 墙钟时间，包括 worker 排队、执行、通信。
- `journal_ms`：状态更新路径，包括 payload 存储、序列化、fsync/replace；无变化的更新也计入该路径耗时。
- `processing_ms`：本阶段其余处理。
- `active_ms`：上述五项（不含 tick_wait）的总和，可能有浮点舍入差。

首个 PREPARE 记录包括可用页查询及帧创建。首次调用没有 tick_wait；恢复或旧 journal 缺少进程内时间戳时，tick_wait=null、timing_recovered=true，不跨进程拼接 monotonic 时间。

`wave_envelope=true` 时 rpc_ms 是共享 envelope 耗时；可用于解释单帧关键路径，不能跨 wave 成员相加当 worker 总时间。`ok=false` 的尝试也记录，按 chunk 聚合时必须包含它们。旧 `precopy_frame_recent_latency_ms` 保留原定义（持久化帧创建到 APPLY 完成），与包含 SELECT 的新记录总和不严格相同。日志输出本身在 active 计时之外。

## 字节与页单位

- frame.entry/start/count、available_pages、applied_pages_before 均以该 entry 的 native pool page 为单位；available_pages_by_rank 保留各 rank 值，available_pages 是其最小值。
- reservation_upper_pages 是预留上界，不是可复制范围。
- page_bytes_by_rank 是 schema 的每页 payload 字节。
- RDMA 多 entry 帧用 frame.segments 列出每个 entry/start/count。其 payload_bytes_by_rank 是对应 shard 各段的 count × page_bytes 之和，单 entry 的页级标量只在单 entry 帧上记录。
- payload_bytes_tp_sum 是所有 TP shard 的计划 payload 之和，每帧只传一次。
- COPY 成功回包的 copied_payload_bytes_by_rank 是本次新发送并确认完成的 payload；copy_replayed_by_rank=true 时为 0，proof 仍保留原字节数。
- precopy_bytes / applied_bytes_tp_sum_after 是已 APPLY receipt 的跨 TP payload 总和，不能作为单 shard 字节或预留容量。

逐帧只在成功 APPLY 时统计一次覆盖，以 epoch/chunk 去重；PREPARE/COPY/APPLY 三行不能各算一次字节。payload 不含 transport footer、协议头或底层重试流量。回包丢失、未确认完成的 DMA 不能从 gateway 得出精确物理流量；copied 字段缺失表示旧 worker/未知，不能当 0。COPY RPC 尝试次数可以由同 chunk 的 COPY 行及 rpc_attempts 统计。

U3 应先比较同一页范围的 byte/page、唯一覆盖和各 rank 数据，再判断是真复制扩大还是统计口径差异。本次不预先认定 3.3× 的根因。

## Q/O 私有发送锁

Q/O 的提交和完成等待持有 Plane 私有 mutex，不再排队等待 KV 推送；同一 Plane 的 Q/O 仍互斥。Slab 注册/注销保留 manager 共享锁及 priority 路径。source_return_slot_wait 因而测量私有发送锁；旧 send_lock_wait_ns 仍来自 manager，共享锁指标含义没有变。

本地 Mooncake classic 源码的 send_probe 走 metadata socket 握手；这不是对部署版本并发安全的实机证明。须在目标机器复跑 128K，并单独验证频繁 peer 注册/注销与 Q/O 并发。此前提供的 15918 ms 结果不是本补丁保留共享注册锁后的新测量。

## 验证范围

CPU 真实 tensor/控制 socket、显式 fake NIC 的合约测试覆盖：128/512 相同可用页复制量、sync/wave 计时对账、关闭后 pending 帧完成、恢复时间戳未知、COPY 重放 0 新字节、Q/O 不等待 KV 锁、Q/O 仍串行、注册/注销持共享锁。轻量入口绕过 SGLang 顶层初始化和 CI test base；不等于 CUDA/Mooncake 注册风暴或真实性能验证。

## 从 cut 到 D1 首 token

状态行和 `/v1/router/status` 中每个请求的 `handoff_timing` 记录以下 gateway
时钟上的事件，缺失的阶段为 null。D1 首 token 可见时额外写一条
`type=handoff_timing`，按 rid/epoch 关联。

| 字段 | 口径 |
|---|---|
| `cut_observed_at` | gateway 首次通过 TP 共同输出观察到请求达到 cut 或显式迁移条件，含状态轮询延迟。请求重试时保留这个起点 |
| `freeze_requested_at` | 首次持久化 FREEZE 意图。实际 RPC 提交时间看对应 epoch 的 `d2d_control_slot.submitted_at` |
| `freeze_descriptor_ready_at` | 所有 rank 返回完整 descriptor 且冻结 token 历史和 KV 长度一致。pending retirement ACK 不计 |
| `import_ready_at` | 所有 rank 确认 import 成功。异步 pending 回包不计 |
| `target_admitted_at` | 全部目标 rank 确认接纳与源端确认，gateway 切换输出所有权 |
| `dst_first_token_observed_at` | TARGET 或 LOCAL 状态推进时，gateway 首次观察到 D1 的 TP 共同输出在冻结前缀之后新增 token。包括轮询和调度延迟 |

相邻阶段和端到端差值以 ms 给出。时间点是 gateway Unix 时间，随 journal 持久化。
缺少任一端点或时间倒退时差值为 null。旧 journal 没有记录过的端点不会补造。

`remaining_copy_bytes.by_rank` / `tp_sum` 是完整 descriptor 冻结时计划的
剩余唯一 payload。按各 rank 的冻结 KV 页数减去封存的 precopy prefix 页，
再乘 schema 每页字节，并加请求状态的 rows × stride。它涵盖后续 import
和 bulk materialization，不含协议头、对齐、重传或远程 attention 流量。
缺少字节计算所需 descriptor 字段时为 null。

`type=d2d_control_slot` 对每个非 status RPC 记一次 envelope，成员用 cmd/epoch
关联请求。每个 `holders` 项给出等槽期间
重叠的 dst 持槽命令、wave 成员及 `overlap_ms`。status 也可出现在持槽者列表。
未提交的超时尝试 `submitted_at` 和 `rpc_ms` 为 null。记录使用有限的近期持槽
历史，未归因的等待包含线程调度与记账边界。precopy 阶段的 `dst_slot_waits`
带同一份归因。多个成员共享 wave 的等待与 RPC 时间，分析吞吐时只计算一次。

跑 m128/m512、c=8、cut32 的热缓存 ON/OFF 对照时，按完整请求对比上述阶段、
剩余字节、D1 首 token 和 safe-to-stop。按 epoch 关联 PREPARE 的持槽记录，
再分辨它在等 precopy、IMPORT/COMMIT 或 status。


## 控制器阶段推进 A/B

在 H20 使用相同构建、负载、`--poll-ms`、迁移并发和 Pre-copy 配置。A 组沿用周期扫描，B 组给迁移控制器加 `--event-driven-reconcile`。两组都设置 `--out`，各轮用独立 state 目录。B 组的快照和已持久化阶段变化只唤醒相关请求，周期扫描仍会检查重试和等待期限。

按 `epoch` 对齐以下 JSONL 记录。

- `type=d2d_reconcile` 的 `phase_wait_ms` 从 journal 完成阶段写入计到该阶段开始推进。恢复后没有进程内时间戳时为 null。B 组的 `event_wait_ms.snapshot` 单独记录相关快照发布到推进的等待。
- `type=d2d_control_slot` 的 `slot_wait_ms` 和 `rpc_ms` 分别记录控制器等 worker 锁、以及拿到锁到 RPC 返回的耗时。目标端的 `holders` 可定位占锁命令。
- `worker_wait_before_exec_ms` 从 worker claim 计到执行开始，`worker_exec_ms` 是 worker handler 的执行时间。两者取各 TP rank 最大值，可能来自不同 rank，均包含在 `rpc_ms` 内。控制 wave 按 envelope 统计一次。

同时比较 `PLAN→FREEZE`、`FREEZE→首 token`、`activate_to_safe_to_stop_ms` 和未迁请求的最大 chunk 间隔。阶段推进时间和 worker 锁等待分开归因。

## Scheduler 状态快照 A/B

A 组沿用同步状态 RPC。B 组在每个 Decode worker 启动前设置
`HANDOFF_D2D_STATUS_SNAPSHOT=1`，并给迁移控制器加 `--worker-status-snapshots`。
两组保持 `--event-driven-reconcile`、并发、Pre-copy 和压测负载一致，都设置
`--out`。SMG 用同一配置。

每个 TP rank 在 Scheduler 安全边界按 20 ms 节流采样。后台只保留最新待写快照，
在现有控制目录中原子发布 `d2d-status.<rank>.json`。控制器先用 RPC 建立实例身份，
然后读取所有 rank 的快照。任一 rank 缺失、超过 250 ms、PID 或实例身份变化时回退
RPC。同一份快照反复读到也不会续期。按请求找终态和 drain 最终空闲确认继续用同步 RPC。

控制 RPC 的 `d2d_control_slot` 增加以下字段。worker 时间取各 TP rank 的最大值，
与控制器的 `slot_wait_ms`、`rpc_ms` 和 `phase_wait_ms` 分开对照。

- `worker_publish_to_claim_ms` 从 driver 开始发布命令文件到 rank 0 完成 claim 和解析。
- `worker_wait_before_exec_ms` 从 claim 到 handler 开始，`worker_exec_ms` 是 handler 耗时。
- `worker_tp_confirm_ms` 从 handler 返回到 TP 对账完成并开始写回包。
- `worker_reply_observed_ms` 从开始写回包到 driver 读到该 rank 的结果，含文件写入和 driver 轮询。

先看 `GET /v1/drain/status` 中 `worker_status_reads` 的常规 RPC 和快照读取次数、
迁移命令的 `slot_wait_ms` 和未迁请求 ITL。状态读取次数按每轮开始和结束的计数差比较。
再看 `PLAN→FREEZE` 与 `activate_to_safe_to_stop_ms`。如果 RPC 频繁回退，先核对
控制目录、各 rank 快照的采样时间和实例身份。该旁路依赖控制器和 worker 的墙钟同步。

## H20 满载迁移 A/B

两组用同一提交和同一套 SMG。保持 H20 拓扑、模型和 Mooncake 后端、SWE 24 条会话、
8 用户、max_new=1024、打流 4 分钟后 D1→D2、压测种子和迁移配置一致。
准备上限 4、切换上限 2，固定自然排空和 Pre-copy 窗口沿用现场参数。
每轮用新的控制器 state 目录和日志，恢复 D1、D2 后再回放。两组均重启 Decode，
避免对照组继承已初始化的传输资源。

| | 控制器 | 每个 Decode 的进程环境 |
|---|---|---|
| A | `--event-driven-reconcile --worker-status-snapshots` | `HANDOFF_D2D_STATUS_SNAPSHOT=1 HANDOFF_D2D_WARM_TRANSPORT=0` |
| B | 与 A 相同，加 `--chain-drain` | `HANDOFF_D2D_STATUS_SNAPSHOT=1 HANDOFF_D2D_WARM_TRANSPORT=1` |

仓库启动入口默认是 `optimized`。把现有 Decode 启动命令完整放在 `decode`
之后，把现有控制器参数放在 `controller` 之后。A/B 分别设置
`D2D_PROFILE=baseline` / `optimized`。

```bash
export D2D_PROFILE=optimized
launch=plugins/remote_attn/deploy/run_native_pd_d2d.sh
# 在内网脚本的每条 Decode 启动行，保留原有参数及日志重定向。
# "$launch" decode bash start_pd.sh <现有 Decode 参数>
"$launch" controller --config "$POOL_CONFIG" \
  --max-preparing-migrations 4 --max-concurrent-migrations 2 \
  --natural-drain-ms "$NATURAL_MS" --precopy-window-ms "$PRECOPY_MS" \
  --out "$RUN_DIR/handoff.jsonl"
```

`optimized` 默认每 rank 预注册 2 个 64 MiB RDMA 源端发送槽。
`HANDOFF_D2D_WARM_RDMA_SLOTS=0` 保留 catalog 和 worker 预热，关闭发送槽预注册。
两组都用 `--out`。
都用同一个 admin 请求触发迁移。

```bash
curl -fsS "$ADMIN/v1/drain" -H 'Content-Type: application/json' \
  -d '{"src":"D1","dst":"D2"}'
curl -fsS "$ADMIN/v1/drain/status" | jq '{
  id, state, source_release_ms,
  cordon_ms: (if .activated_at then 1000 * (.activated_at - .started_at) else null end),
  members_state, migration_limits, active_cutovers, controller_features,
  control_wave_sizes
}'
```

- 发压前确认 `controller_features`。两组的 `event_driven_reconcile` 和
  `worker_status_snapshots` 都应为 true，B 的 `chain_drain` 应为 true。
  迁移时确认 `migration_limits.preparing=4`、`migration_limits.cutover=2`，
  `active_cutovers` 包含尚未恢复的冻结回滚，峰值应 ≤2。
- `cordon_ms` 仍包含 SMG 的探针等待。控制器持久化请求后立刻开始查，
  每个 SMG 副本均须确认摘流。sglang-router 0.3.2 的
  `--health-check-interval-secs` 接受整数，沿用 1 秒。
- 主指标用 `source_release_ms`（POST 接受至 SAFE_TO_STOP），另比较
  `1000 * (source_released_at - activated_at)` 和摘流耗时。至少做 3 轮配对，
  同时记每轮源端成员数、实际迁移数、源端自然完成数和回滚数。
- SSE 记录请求成功率、完整终止、token 前缀连续性及客户端 ITL。
  对已迁与未迁请求各报 p50/p95/p99、最大间隔和大于 300 ms 的次数。
- `handoff_timing` 记录 `freeze_request_to_descriptor_ms`、
  `descriptor_to_import_ms`、`import_to_target_ms`、`target_to_first_token_ms`。
  `source_transport_setup_ms` 从传输能力选择计到批量传输 worker/stream 就绪，
  `source_route_publish_ms` 记录请求路由注册，用各 TP rank 最大值。
- 按 epoch 对照 `d2d_reconcile.phase_wait_ms` 和
  `d2d_control_slot.{slot_wait_ms,worker_publish_to_claim_ms,worker_exec_ms,worker_tp_confirm_ms}`。
  发给同一 worker 的 wave envelope 只计一次。
- `rdma_sender_prewarmed_ranks` 记录实际命中预热发送槽的源端 rank 数。
  预注册槽耗尽时沿用按需分配。目的端接收槽注册、路由 attach 和实际 peer 写入
  仍依赖每次迁移。两组均在 Import 成功后先写 journal，再执行 COMMIT。
- `worker_snapshot_sample_ms` 记录控制器接收到的每份快照的 TP 最大采样耗时。
  快照丢失时对 drain 两端的全量状态回退 RPC 按最近成功状态节流到约 100 ms，
  优化组的整条控制 wave 期间全量状态 RPC 也会让路，命令结束后补读。
  已经提交的状态 RPC 会等回包交还角色锁。首次建立身份、终态查证和
  idle 正证仍用 RPC。

## 引擎侧开销定位

先保持 `optimized`、PREP=4、K=2，用四条长请求全迁跑一轮。目标优化前后再用同一份
输入做配对，核对四条请求均接续输出并自然完成。以下字段随成功 Import/Commit 的回包记录，
可从 `/v1/drain/status` 或 `handoff.jsonl` 读取，状态快照保持开启。

```bash
curl -fsS "$ADMIN/v1/drain/status" | jq '
  .members_state | to_entries[] | {
    rid: .key, outcome: .value.migration_outcome,
    timing: .value.handoff_timing
  }'
jq -c 'select(.type == "d2d_control_slot" or .type == "d2d_reconcile")' \
  "$RUN_DIR/handoff.jsonl"
```

- `source_event_record_ms` 记录 Freeze 的事件创建加记录耗时，和传输初始化及路由发布分开。
- `import_timing_ns_by_rank` 分列 `native_prepare`、`setup_stream_create`、
  `setup_stream_dependency`、`attach`、`manifest`、`state_index_copy`、
  `setup_event_record` 和 `setup_event_wait`。原 `setup_gpu_fence` 是记录加等待，
  分析时只算一次。异步 Import 的事件等待发生在后台线程。
- `setup_reads_ns_by_rank.slot_setup` 观测接收槽创建和注册。该对象里的
  `count`、`bytes`、`max_frame_bytes` 分别是次数和字节数，其余时间字段为 ns。
  尚未发生读操作时计数为 0。
- `commit_timing_ns_by_rank.wave_prepare_agree` 是 Commit wave 共享的
  TP 有效性、确认进度和容量投票耗时，取最终准入的那次 wave。
  `source_confirm_rpc` 是每个请求成功确认的 RPC 耗时。
  `wave_native_admit` 和 `wave_admit_agree` 也是共享 wave 耗时。
  旧引擎单独记录 `wave_validity_agree`、`readiness_agree` 和 `capacity_agree`，
  跨版本总耗时用 `d2d_control_slot` 对比，一个 wave 只记一次。
  用 `phase_wait_ms` 与 `worker_*_ms`
  区分阶段等待、发布到认领、认领到执行、执行、TP 对账和回包。
- `*_by_rank` 按目标 TP rank 顺序记录，慢 rank 用最大值看，跨 rank 不求和。
  老 worker 的回包可能没有计时。状态输出仍可通过 `d2d_status` 查单个 epoch
  的引擎状态。
