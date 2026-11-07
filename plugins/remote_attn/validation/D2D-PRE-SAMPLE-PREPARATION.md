# D2D：将容量准备移至首 token 之前

日期：2026-11-07。基线：`b86de9fc1b`，分支 `codex/remote-attn-pre-layerwise`。

## 两个不同的时间窗口

基线已在首 token 后 plan/reserve，不是全部等到 cut 后才开始。但 planning_blocker
和 Gateway 都仍要求 output_ids 非空，因此此前还不能利用 Prefill 窗口准备 D1。
对热缓存、128-token 输出而言，这个限制仍可能让 freeze 在请求结束之后才落地。

必须赶在源自然结束前完成的是 **freeze**。源冻结并移出 runnable batch 后，
不会继续生成至 finished；旧 PD 退休、D2D import、目标 commit/admit 可以随后
完成。因此不能直接用完整 PLAN→COMMIT 的 2.4 s 推导“Decode 小于 2.4 s 必失败”。
这段总耗时既可能包含冻结前等待，也可能包含冻结后的用户可见停顿，必须拆开。

本轮时间线：

```text
源 D0 接纳 native PD 请求 → PLAN → D1 RESERVE → WAIT_CUT
                            与 P Prefill、D0 接收并行
P 首 token 到达 → 源 Decode 至 cut/显式迁移 → FREEZE（需赶在源结束前）
                                            ↓ 源停止生成
                        原 PD 退休 → D2D IMPORT → COMMIT/目标续 token
```

没有把目标容量准备做成 P 启动的阻塞门槛，也不保证所有极短/全缓存请求都有足够
准备时间。目标容量不足、准备期间请求结束仍显式回滚，不能伪造迁移成功。

## 行为与所有权

- planning_blocker 不再要求采样输出；migration_blocker/freeze 仍要求至少一个
  真实 sampled token、原生 Decode 边界及已有 Remote 安全冻结条件。
- 未采样计划的 lower 是完整 prompt 长度 N，而非 N-1；upper 仍由增长预算、
  原生 context 和采样上限共同限制。与 native pre_alloc_fill_len 一致。
- D1 ticket 的容量占位保留 pending-token 位置，包括 max_new_tokens=1 的边界。
  占位仅存在于未准入 ticket，从不插入源输出，也不作为实际模型 token 发给客户端。
- 目标 import 消费该原生 row/KV/GDN slot，使用冻结时的真实输出及状态，退还多余
  页；不重新分配一个目标 row。部分 Remote 的原 PD 退休与 reader 释放顺序保留。
- source plan 绑定同一 Req、cache handle 和 retraction_count；freeze 另核验 prompt、
  已知输出前缀和 sampling。提前准备期间 attempt/history 改变会拒绝旧证书。
- 源结束/取消时取消目标 ticket；never-frozen resume 移除 source plan 并留下 epoch
  取消 tombstone，拒绝迟到 plan 复活该 epoch。已冻结/已 commit 的恢复规则不变。
- 补齐 reservation 的外部 owner 记账：未消费且未取消的 ticket 占用 D1 原生容量，
  必须参与 row/FULL/SWA 检查；consume 后由 Target 接管，cancel 后退出。
  native checker 按 KV record 去重，不把共享记录或 ticket/Target 切换计为两份。

## 可观测性

journal、router/SSE status、metrics 行新增：

- plan_before_first_token：各 TP rank 的 plan 输出快照都为空；不等同于已完成
  reservation，也不等同于成功迁移。应结合 WAIT_CUT/实际目标续 token 判断。
- freeze_ack_latency_ms：从首次 dispatch freeze 到首次收到成功冻结确认。
  包含排队、重试、回复之前的准备；它是冻结落地延迟的上界，不是 worker 内纯执行
  时间。回复丢失时沿用同一 epoch 与最初 timestamp，不重置窗口测量。

原 prepare_latency_ms 记录准备到 reserve 完成；handoff_latency_ms 记录首次
freeze 请求到目标 commit/admit，仍包含冻结后退休和 import 等停顿。
应同时看按命令拆分的 control_latency_ms，不能用 status RPC 的 4 ms 替代
reserve/freezing/retirement/data-plane 的实际耗时。

## 验证

4060 小池 c=4/8：源输出为 [] 时完成同一 epoch 的 plan/reserve 并停在 WAIT_CUT；
随后在采样 ABI 边界推进至 cut=32，执行真实 native receive/scatter、CUDA event、
PD 有序退休与 Mooncake TCP D2D。目标续 token 后，源等 reader 退休再释放一次。

| c | 首 token 前准备 | 已记账的预留 owner | 实际迁移/目标续 token | 源释放 |
|---:|---:|---:|---:|---:|
| 4 | 4/4 | 4 | 4/4 | 4，每请求一次 |
| 8 | 8/8 | 8 | 8/8 | 8，每请求一次 |

CUDA 探针使用共享 D0/D1 小池、不同请求行，Gateway driver 为进程内命令分发，
采样与源生成是 ABI 边界；不是完整 32k/128 模型请求、跨机 RDMA 或延迟收益验证。
机器结果与实现 SHA-256：
[d2d-pre-sample-4060.json](d2d-pre-sample-4060.json)。

CPU 覆盖首 token 前预留、完整 prompt 下界、pending-token 容量、同 epoch 重启、
cut 等待、取消/提前结束回收、attempt/prompt/sampling 改变拒绝旧证书。
真实双进程 Gloo TP2 验证提前预留可行但未采样不能冻结；Hybrid 原生 row/KV/GDN
allocator 验证早期 working slot 占用与取消返还。原生 FULL/SWA/row checker 验证
reservation-only owner 计账、consume 后去重及 cancel 移除。

独立负对照：恢复旧 plan/gateway 实现后，状态机停在 SOURCE /
sampled_output_pending，无法到达未采样 WAIT_CUT；恢复旧 owner 回调后，reservation
单独持有的 FULL/SWA 占用被错误算为 (0,0)，实际应为 (12,4)。

最终主套件 381 项通过（201.809 s），native contracts 24 项通过（18.184 s），
native ABI 6 项通过（0.481 s），共 **411 项**；5 项定向与 2 项 owner 用例是
其中子集，不重复计数。51 hooks / 145 core interfaces、Ruff 与空白检查通过。
原始日志归档：`/private/tmp/d2d-presample-results/results.tar.gz`。

## 短请求是否值得迁

准备提前解决的是迁移机会，不会消除冻结后的数据搬运成本。
若目标只是单请求延迟，可用近似判据：

```text
迁移停顿 + 剩余 token 数 × 目标 TPOT < 剩余 token 数 × 源 TPOT
```

这里的停顿应测最后一个源 token 至第一个目标 token 的间隔，剩余量按冻结点
估计；max_new_tokens 是上限，不是 EOS 前实际还会生成的数量。
在目标 TPOT 没有显著改善且迁移停顿大于剩余本地工作时，128-token 请求应优先
自然完成；将输出改成 256/512 只能扩大实验窗口，不代表短请求迁移已优化。
若目标是腾空源 worker，则应比较 D2D 真正释放源 KV 的时间与自然排空时间，
不能只用 freeze 或 commit 作为 drain 完成证明。

H20 复测保持 128 输出、先统一预热、分别 c=1/4/8；对短请求优先从 cut=1 或
显式迁移开始测，以免 cut=32 额外花掉 31 个 token 的窗口。记录准备是否在首 token
前完成、window_missed 次数、首次真实目标 token、停顿、源 KV 释放与自然排空时间。
Gateway/D0/D1 应一起更新；旧 D1 对未采样计划会在校验阶段拒绝，不能混版验收。
本轮没有 H20 内网访问，未给出 H20 TTFT/E2EL/吞吐改善或固定毫秒数承诺。
