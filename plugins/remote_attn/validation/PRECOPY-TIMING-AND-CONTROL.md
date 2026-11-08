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
- payload_bytes_by_rank = count × page_bytes；payload_bytes_tp_sum 是所有 TP shard 的和。这是帧计划 payload，不是每个阶段都实际发送一次。
- COPY 成功回包的 copied_payload_bytes_by_rank 是本次新发送并确认完成的 payload；copy_replayed_by_rank=true 时为 0，proof 仍保留原字节数。
- precopy_bytes / applied_bytes_tp_sum_after 是已 APPLY receipt 的跨 TP payload 总和，不能作为单 shard 字节或预留容量。

逐帧只在成功 APPLY 时统计一次覆盖，以 epoch/chunk 去重；PREPARE/COPY/APPLY 三行不能各算一次字节。payload 不含 transport footer、协议头或底层重试流量。回包丢失、未确认完成的 DMA 不能从 gateway 得出精确物理流量；copied 字段缺失表示旧 worker/未知，不能当 0。COPY RPC 尝试次数可以由同 chunk 的 COPY 行及 rpc_attempts 统计。

U3 应先比较同一页范围的 byte/page、唯一覆盖和各 rank 数据，再判断是真复制扩大还是统计口径差异。本次不预先认定 3.3× 的根因。

## Q/O 私有发送锁

Q/O 的提交和完成等待持有 Plane 私有 mutex，不再排队等待 KV 推送；同一 Plane 的 Q/O 仍互斥。Slab 注册/注销保留 manager 共享锁及 priority 路径。source_return_slot_wait 因而测量私有发送锁；旧 send_lock_wait_ns 仍来自 manager，共享锁指标含义没有变。

本地 Mooncake classic 源码的 send_probe 走 metadata socket 握手；这不是对部署版本并发安全的实机证明。须在目标机器复跑 128K，并单独验证频繁 peer 注册/注销与 Q/O 并发。此前提供的 15918 ms 结果不是本补丁保留共享注册锁后的新测量。

## 验证范围

CPU 真实 tensor/控制 socket、显式 fake NIC 的合约测试覆盖：128/512 相同可用页复制量、sync/wave 计时对账、关闭后 pending 帧完成、恢复时间戳未知、COPY 重放 0 新字节、Q/O 不等待 KV 锁、Q/O 仍串行、注册/注销持共享锁。轻量入口绕过 SGLang 顶层初始化和 CI test base；不等于 CUDA/Mooncake 注册风暴或真实性能验证。
