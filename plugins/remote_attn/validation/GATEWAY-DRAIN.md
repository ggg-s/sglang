# Gateway drain under concurrent traffic

此修复只修改实验控制面 `control_plane/p4_gateway.py`，没有修改 SMG、SGL
核心、原生 optimistic prefill 或 chunk 传输参数。适用拓扑为 P-only + D0 +
D1 + 实验 gateway；原生 SMG 的单 P/D pair 不使用这个 drain 状态机。

## 修复边界

1. `start_drain` 在 admission 锁下写入 `REQUESTED`、成员快照和
   `preflight_pending=true`，持久化后释放锁，再等待 migration 锁。已经开始的
   cutover 可以完成，后续请求立即看到 D0 为 DRAINING。重复 drain 返回同一实体。
2. 目标 status、容量等待和 reserve RPC 全部在 admission 锁外。预检期间新 D1
   请求只等待 journal 中的 pending 标记；预留完成后按目标剩余容量领取普通
   ticket，不等待整个迁移结束。`max_running=8`、一个迁移成员时，第八个短请求
   仍可能等待实际槽位释放，这是必要的容量排队。
3. submit 前在 admission 锁下重新检查路由。即使请求在 drain 前读到 READY，
   drain 后也必须领取 D1 ticket，不能落回 D0 或无票占用目标容量。
4. reserve、freeze、import 使用同一个 drain epoch。预检 RPC 完成但 journal
   尚未记录完成时重启，reconcile 会用相同 epoch 重试预检，预检完成前不冻结。
5. 真正容量不足仍在 freeze 前 ROLLED_BACK，D0 恢复 READY。已导入、可能提交的
   迁移不能因为目标不响应而推断取消成功，也不能把 imported 降回 frozen。

## 目标失联的有限失败

`HANDOFF_DRAIN_TARGET_TIMEOUT_S` 默认 **30 秒**（最小 1 秒），从连续 status
失败被确认后计时；实际返回还受当前有界 RPC/轮询周期影响。

- 导入前、可证明源仍持有请求时，保留已有安全回滚路径。
- 导入后失联超时，drain 持久化为 **FAILED_UNSAFE**，状态返回
  `terminal=true`、`requires_intervention=true`、`source_safe_to_stop=false`。
- 原有 SSE 和 attach 返回 `target_unreachable_timeout` 错误并结束 SSE；不记录
  `finished=true`，不据此恢复源、取消导入或释放仍可能被读取的 KV。
- D0 保持停投；重复 drain、重启 gateway、排队的旧 cutover 都不能解除这个隔离。
  `stop_drain` 拒绝下线。终态不再不断向失联目标发送 reconcile RPC。
- 目标在期限内恢复，仍由原 reconcile 路径继续；自然 drain 不依赖目标健康。

**FAILED_UNSAFE 是失败终态，不是成功迁空，也不承诺自动恢复。** 必须先保留证据，
确认并隔离旧目标、取得停止读 KV/取消或提交的证明，再人工处理保留的资源；不要
通过清空 journal 或重发普通请求来恢复同一条在途请求。

外部 bench/缩容控制器应在 `terminal=true` 时停止轮询并保存结果；只能在
`source_safe_to_stop=true` 时下线 D0。若旧 driver 只识别 SAFE_TO_STOP/ROLLED_BACK，
需将 FAILED_UNSAFE 加入失败出口。本仓库不包含问题描述中的短流量 bench/start_d
脚本，不能在这里修改它们。

## 状态与故障证据

`drain_status` 使用 reconcile 的缓存事实，不会因每个状态轮询额外插入 status RPC。
它现在返回：

- `admission_fenced_at`：开始停投时间。
- `preflight_started_at` / `preflight_finished_at` / `preflight_ms`：可区分迁移锁等待
  与整个预检耗时。
- `slot_gate`：各 TP rank 中最大 load、最小 max_running、成员数和 gate elapsed_ms；
  容量等待仍由 `HANDOFF_DRAIN_SLOT_WAIT_S` 控制，默认 10 秒。
- 每个普通请求 journal 的 `admitted_at`、`admission_worker`、
  `admission_drain_id`、`admission_drain_state`：用于关联 D1 的两个短请求是在
  停投前还是停投后接纳。
- 超时的 `failure_evidence`：wave、成员所有权、源/目标事实、最近健康目标事实、
  失联时长及采样时间。它不能说明 D1 为什么失联。

每轮 GPU 验收使用独立 run/日志目录，在下一次 `start_d()` 前保存 P/D0/D1/gateway
日志、完整 controller_journal、driver 输出和失败时间附近的 dmesg。不要让下一轮
覆盖 D1 日志；D1 崩溃、被杀、网络失联仍需这些外部证据区分。

## 本地验证与 GPU 验收

在 `test/registered/unit/plugins/test_remote_attn.py` 的既有 CPU suite 中增加 8 个
测试，直接执行真实 Gateway 方法，只替代 Role RPC 和 HTTP 输出边界：

- remote/layer/copy 三臂：旧 cutover 阻塞期间启动 drain，再加入 8 个短请求；
  成员维持 1、先预留再普通 admission、短请求全部走 D1；继续经过 freeze/import、
  Local 完成事实与源释放，最终 SAFE_TO_STOP。
- 在 admission 读到 READY 后精确插入 drain，确认提交前重读状态且携带目标 ticket。
- 预检中断重启沿用 epoch；reserve 等待期间 admission 锁可用。
- 全 TP 容量检查、冻结前安全拒绝并恢复 READY；上一轮 rollback 的迟到写入不能覆盖新 drain。
- imported 波次目标失联：有界 FAILED_UNSAFE，原 SSE/attach 退出、重启保留隔离，
  没有 resume/abort/release，也不允许排队旧 cutover 继续。
- 暂时失联恢复、导入前可回滚、自然 drain 不误报目标失联。

本地使用 Python 3.12 / CPU PyTorch，98 项测试通过（12.119 秒），包含真实双进程
Gloo 检查；源码预检的 29 个 hook / 70 个核心接口，以及静态和差异格式检查通过。
新增竞态和失联用例对修改前 Gateway 失败，对修改后通过。CPU 边界测试不代表
CUDA、TE、真实 TP 或 GPU 延迟已验收，也不定位本次 D1 失联根因。

GPU 上仍需按原命令重跑 32K/128、token 32 drain、256/16 短流量，三臂至少各
3 轮，与 `--short 0` 对照。保存成员快照、短请求路由和 TTFT/E2E、drain 时长、
源所有 rank 资源归零证明。另分别在导入前、导入后断开 D1，确认安全回滚与
FAILED_UNSAFE 两种结果，且 driver 总会产出失败结果文件。

## Source completion during the capacity wait

Refresh migration demand as well as destination load on every slot-wait tick.
Keep `members` as the complete admission-fence snapshot for reconciliation,
source resource release and outcomes. `migration_members` records only unfinished
source-owned rows at preflight; only these rows need migration reservations.

If 5 of the original 9 rows finish during the wait, a destination with load 1
and max_running 8 now admits the remaining 4 (`1 + 4 <= 8`). The 5 completed
rows remain in the drain's cleanup bookkeeping. Demand that still exceeds
capacity after HANDOFF_DRAIN_SLOT_WAIT_S safely rolls back before freeze.

The new CPU regression exercises 9-to-4 demand shrinkage, four-row reservation
and freeze/import, release of all nine source rows, and SAFE_TO_STOP. Additional
cases preserve cleanup records for rows completed or target-owned before
preflight. Worker RPCs are simulated; this does not establish GPU/TE latency
or throughput results.
