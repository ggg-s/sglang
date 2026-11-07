# 网关大 journal 与 Hybrid PD Remote 修复验证

代码基线：`171a8d351b`，分支 `codex/remote-attn-pre-layerwise`。
机器可读结果及源码 SHA-256 见 [JSON](gateway-journal-hybrid-pd-4060.json)。

## DSv4 并发问题的修正定位

用户补充的 H20 数据来自 DeepSeek-V4-Flash-0731，并非 V4.1 或 Qwen。
PLAN 后约 6.57 s 没有再被服务，最终才记录 `request_finished`；其后的 5.376 s
是请求完成后的收尾等待，不能解释为 Remote 安全条件阻挡冻结。
本次不放宽 `can_quiesce`、采样结果或 KV 所有权条件。

根因是每个 entry 状态跃迁都会 deepcopy、序列化并持久化包含所有长提示词的 journal。
现在将 input_ids、plans、reservations、transfers、admission_plans 和 precopy_receipts
写成按内容寻址的不可变 payload，热 journal 仅保存引用和状态。
payload 先 fsync/rename/目录 fsync，之后才发布 journal 引用；每次控制意图仍持久化后再发命令。
v1 journal 在恢复命令前迁移为 v2，并验证引用内容；缺失/损坏拒绝继续。
保留 `dsv_payloads/` 与 journal 一起备份，不在仍可能有 owner 引用时回收 payload。

同时，worker 常规 status 不再重复返回历史迁移的完整 transfer manifest；单 epoch 查询保留原行为。
后台 tick 轮转起点，在 entry 之间按 50 ms 预算让出，不能抢占已执行中的单个 RPC。
metrics 提供近期 tick 延迟、journal 字节数和版本。

原生 submit 同一条调度器命令内可创建容量 plan，无需再等一次 worker status/首 token；
网关接着预留目标，freeze 仍等待 cut、至少一个采样结果以及原生安全边界。
重试沿用原 submit/plan epoch，不重复入队；回滚取消预留并废弃 admission plan。

## 同机 journal 对照

`bench_p4_dsv_journal.py` 使用同一 CPU/目录，27 个请求各 32768 输入 token，
10 次标量状态更新；旧实现取自基线提交。没有提交模型请求。

| 指标 | 原实现 | 本次实现 |
|---|---:|---:|
| 标量更新中位数 | 188.84 ms | 4.44 ms |
| journal 大小 | 5,919,629 B | 27,918 B |

更新中位数约快 42.5 倍。这仅衡量网关序列化/持久化，不代表 H20 TTFT、握手或迁移停顿提升。
回归检查另验证 27×32768 输入的热更新不再调用 payload 编码，v1 重启迁移与输入恢复无损。

## Hybrid PD 实现

问题确实存在：原 Controller.prepare/offer/early_targets 没有建立 P→D Remote。
修复增加独立的原生 hybrid PD 路径，而非去掉 hybrid 守卫后复用 DSV sparse 计算。

1. P 用原生状态规划获取 Conv/SSM 的物理工作 slot，等待最终 forward 和索引 event。
2. 先传全部 GDN 状态和 metadata，所有 TP rank 确认该 barrier 后才提前接纳 D。
3. D 在本地续算 GDN，full-attention KV 未完成时由 P 服务 Remote attention。
4. KV 按 entry/range 接收并经 GPU/传输/TP 退休证明后切 Local；canonical success 在完整覆盖之后。
5. 结束/取消均延期 native free；Remote reader 关闭、原生发送和 GPU proof 齐全后才释放资源。
6. P 的 FULL+MAMBA 缓存使用独立读 pin，保留 native checkpoint/CoW；延期 checkpoint 不会
   把最终可变状态发布到较早的前缀。原生最终 release 保持 donation/free 语义。

`handoff-src` / `handoff-target` 日志现在标明 hybrid PD，remote-service 边界为
`hybrid-gdn-state`。仍保留 hybrid 对 DSV ring、sparse math 和 Decode radix 的隔离。
已开始且未纳入 range 证明的原生发送不会中途切入 Remote，避免缺失前缀覆盖导致悬挂。
部署约束与环境见 [Hybrid 说明](../QWEN38-HYBRID-D2D-ZH.md)。

## 必要验证

复用现有探针；只新增两个网关回归和一个含正常/取消子场景的 PD 生命周期回归，没有重跑全量矩阵。

| 检查 | 结果 |
|---|---|
| 网关大输入、准入 plan、各阶段重启、丢 commit 回复、取消、写 journal 失败、部分 Remote 恢复 | 7 项通过 |
| 真实两进程 Gloo TP2 非对称完成/拒绝/隔离 | 通过 |
| 既有 Hybrid 启动、P 缓存策略、GDN 安装、D2D 取消 | 4 项通过 |
| Hybrid PD CUDA 正常/传输中取消 | 通过；确实执行 P Remote RPC，未等 KV 完整接收 |
| 原生 UnifiedRadixCache / CUDA Conv+GDN / Mooncake TCP | 命中 8、10 token；CoW 不改旧 checkpoint，命中后续算与冷算完全一致 |
| source preflight / Ruff / Python AST / whitespace | 51 hooks、146 core interfaces 及静态检查通过 |

Startup CPU fixture 在作用域替身安装前预加载原生 `torch._dynamo`，避免 lazy import 的重复注册；
未改共享环境或生产代码来绕过依赖。

Hybrid CUDA：Conv 与 SSM 输出最大误差均 0；Remote GQA 对 BF16 参考最大误差
0.0009765625，Local GQA 最大误差 0，18 次原生 Mooncake 提交，P/D 工作 slot 分别为 1/2。

DSV 并发探针使用共享 CUDA 小池、原生接收 ring/scatter、有序退休和 Mooncake TCP，
采样及分配入口为明确的 ABI fixture，控制命令为进程内分发。

| c | 实际迁移 | 部分 Remote 冻结 | D1 首 token | 源端释放次数 |
|---:|---:|---:|---:|---:|
| 4 | 4/4 | 4 | 4 | 4 |
| 8 | 8/8 | 8 | 8 | 8 |

验证源端先完成原 PD 退休再导出，reader 退休前不释放；预留所有权被正确消费，未用容量返回。
现有 c=4/8 探针补齐了 producer stream 与预复制快照的 durable WAIT_CUT 步骤。

## 目标机器仍需复测的部分

未运行内网 H20 完整 DSv4、5090 TP2 完整 Qwen3.8-27B、跨机 RDMA 或 HTTP 端到端性能。
不能把上述小池 c=4/8 成功当作这些完整部署已经验收。

H20 复测沿用 32k/128、统一预热后的 c=4/8，同一起点比较迁移与自然排空；保留
phases.jsonl、tick 延迟、journal_bytes、admission_plan_at、每个请求的 cut_attempted、
freeze_pending、迁移 outcome 与 D0 真正释放时间。检查长提示词留在 payload 文件中，
而非每次 journal 重写；确认不再有整轮 tick 跨越请求生命周期。
Hybrid 复测同时检查 handoff-src/target、Remote RPC 次数、确定性输出及取消后的行/slot 回收，
不能仅凭插件已加载或性能差值判断 Remote 生效。
