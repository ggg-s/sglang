# D2D：layer group × token range

Qwen3-8B 的 layer/remote 在线迁移与 DSV4/DSV4.1 bulk KV 统一使用
`layer-token-range-v1` 逻辑计划。每个搬运单元绑定迁移 epoch、逻辑层组和半开区间
`[token_start, token_end)`；资源适配器将它映射为实际 K/V 或压缩 entry 的页行。
不将不同模型的物理 KV、SWA 和状态布局合并成同一种 allocator。

```bash
export HANDOFF_D2D_GROUP_SIZE=4       # 默认 1，必须为正整数
export HANDOFF_D2D_TOKEN_RANGE=1024   # 默认 1024，必须为正整数
```

配置在源 Decode 实例。冻结时捕获配置，目标消费 manifest 中的参数；两端必须
同时升级。此配置独立于 P→D 的 pre-layerwise 分组，P4 不需要额外启动包装器。

GPU staging RDMA 与 TCP fallback 的选择、跨机地址和完成语义见
[共用 Mooncake backend](D2D-RDMA-ZH.md)。下文 Host staging 描述对应 TCP backend。

## 先预留，再冻结

两种后端遵循相同顺序：

1. 源端在完整 Decode 边界规划 `[lower, upper]`，其中 pending sampled token
   尚未写 KV，故 `KV_len = prompt_len + output_len - 1`。`upper` 由允许增长量、
   sampling 最大输出数和原生请求行容量约束；它是明确的容量上界，不是预测冻结长度。
2. 目标调用原生 Decode preallocation，实际占住请求行和 KV；DSV 还实际占住
   SWA 和请求状态资源。容量判断使用原生分页预算，Qwen 同时计入普通 admission
   尚未兑现的软 lease。不能用 token 计数或一个软执行槽代替实际请求行。
3. 等待预分配初始化/映射的 CUDA event，所有 TP rank 确认后，源端才接受预留凭据。
4. 冻结时再次检查真实 committed KV、pending-token 历史、请求身份和预留区间。
   DSV 先排空已有 Decode overlap result，再集体验证长度；Qwen 沿用非 overlap
   的完整 Decode 边界约束。任一 rank 不满足时，均在冻结前拒绝。
5. 导入消费原预留的原生 Req/KV record，不重新 alloc。冻结点早于上界时，只释放
   从 `ceil(KV_len / allocator_page_size)` 开始的完整尾页，保留有效半页。
   DSV 保留原预留的较早 SWA cursor，由原生窗口回收路径释放多留的窗口页。

Qwen Gateway 的 C1、wave/batch 和 drain 都通过 source plan → native reserve →
source confirmation 进入冻结，取消携带持久化的 reservation epoch。旧的
`prompt + cut + 64` / `+512` 不再用于实际预留。增长预算由
`HANDOFF_D2D_RESERVE_GROWTH` 控制，默认 64；超过区间必须取消旧预留、换新 epoch
重新规划，不能先冻结再扩容。DSV 继续使用 `growth_tokens`。

预留没有隐式 TTL。取消先于 reserve 到达时保留 epoch tombstone；已消费的预留
拒绝 `reserve_cancel`，必须走目标请求 abort/cancellation。部分分配、初始化证明
或释放状态不确定时保留存储，不能因 coordinator 失联而复用。

## 二维搬运与完成证明

| 后端 | 逻辑区间到物理资源 | 传输和 Local 条件 |
| --- | --- | --- |
| Qwen3-8B | 每个逻辑层的 K/V，按冻结 token map gather | 独立 source copy stream 等待 producer event，D2H 至有界注册 Host slot；Mooncake 每 slot 一个连续 KV+footer descriptor。目标验证序号和到达 footer，完成 H2D/scatter 后 ACK |
| DSV4 / 4.1 | 唯一 KV producer 的 native page-row，压缩和共享布局保持原样 | 同一层组、同一 token 区间内打包 entry segments，复用原有独立 D2H Host staging、Mooncake 到达证明和目标 scatter event |

逻辑层组固定为 `[0,G)`、`[G,2G)` 等。Qwen relay slot 少于组宽时，只在原组内
拆 batch，不改变层组边界；源/目标均保留有界 Host slot。每个 Qwen slot 的 KV
payload 不超过 4 MiB，必要时缩小 manifest 的实际 token range。组宽越大，源
Host staging 上限越大，不是整段前缀大小的 staging。

DSV 的 token 区间按 native FULL page 对齐：实际步长为
`max(page_size, floor(T/page_size)*page_size)`，末段截止真实冻结长度。压缩 entry
保留原生 row stride 和 ratio，最后页只 scatter 已完成的压缩行，禁止覆盖 D1 的
新尾部、scale 或 padding。共享 producer 只发一次。每帧仍最多 4 MiB / 64 个
segment，一个矩形可以拆成多帧。SWA、compressor/indexer 状态与 indexer 历史仍
在 commit 前完成 setup，不伪装成普通 token KV。

统一 Qwen 路径当前使用一条 source copy stream，按矩形完成 native batch 和 ACK
后复用 slot；没有实现组间双缓冲流水。旧 `HANDOFF_FAST_BUFFERS`、
`HANDOFF_FAST_COPY_STREAMS`、`HANDOFF_TRANSFER_PIPELINE` 不改变此路径的队列深度，
`HANDOFF_FAST_ASYNC=0/1` 均使用单次 async submit＋同 batch status，避免原生同步
接口隐藏内部重试；不改变队列深度。性能需要在目标负载上单独测量。

覆盖记录逐 entry 拒绝缺口、重复和越界。完成一个 range 只推进该 entry 的覆盖；
只有其全部冻结前缀完成 scatter 和 GPU event 后，才发布 Local 就绪证明。DSV
继续经过有序 TP retirement，Qwen 继续使用原有逐层 Local/Remote 窗口。一个 entry
可先 Local，而同组其它 entry 继续 Remote；不是整组一起切换。

## 取消、异常与回收

- Host slot 复用同时受传输完成、payload 到达和目标 GPU 读取证明约束。Qwen
  range ACK 带匹配序号，不能把旧 slot ACK 当成下一次写入的证明。
- 目标 receiver 先停 IO、排空实际 CUDA copy，再允许原生 KV row 释放；仍在
  Remote 使用的源 prefix 继续受 registry reader/forward 与原有退休规则保护。
- 原生 submit/status 不确定、到达未证明、CUDA fence 失败均保留相应输入、Host
  backing 和注册。未知注册结果即使没有返回 owner handle，也必须保留 backing，
  不能通过后来的 completion ticket 提前释放。
- 全部 reader、copy、forward 和 ownership 证明满足后才回收一次。普通成功或
  resume 的安全终态清理规划 metadata 引用；不确定分配/释放保留诊断记录。

Qwen 的既有 file/socket/full-copy 对照接口保留；通过新 P4 预留的 full-copy 导入
同样消费原生预留，不回退到临时 token alloc。本文二维 wire 协议针对在线
layer/remote 与 DSV bulk 路径，不改变原生 P→D 部署与模型 forward 策略。

## 手工 Qwen 控制

通常由 P4 自动执行。手工控制时必须广播到所有 TP rank，不能独立修改某一个 rank。

```json
{"cmd":"plan","epoch":"move-001","items":[{"rid":"source","target_rid":"target","growth_tokens":64}]}
```

将源端各 rank 返回的 `plans` 列表按 rank 排列，发目标：

```json
{"cmd":"reserve","epoch":"move-001","plans":[["rank 0 plans"],["rank 1 plans"]]}
```

上例字符串要替换成原始 plan 对象；TP1 外层也只有一项。所有 rank 完成后，将目标
各 rank 的 `reservations` 对象按 rank 排列回传源端：

```json
{"cmd":"reservation_ready","epoch":"move-001","certificates":[{"target":"rank 0 reservation 对象"},{"target":"rank 1 reservation 对象"}]}
{"cmd":"freeze","epoch":"move-001","rid":"source","target_rid":"target","transfer":"remote"}
```

之后沿用 import / admit / Local / source release 的现有编排。DSV 控制命令见
[DSV D2D 协议](DSV-D2D-REMOTE-ZH.md)。两种后端保留控制命令命名空间，统一的是
逻辑搬运单元、真实预留顺序、区间校验与资源完成规则。

验证入口：`range_layer_cuda_probe.py`、`qwen_reservation_cuda_probe.py`、
`dsv_d2d_cuda_probe.py`、`group_layer_cuda_probe.py --suite dsv-local` 和
`dsv_d2d_bulk_cuda_review.py`。完整记录见
[二维搬运 4060 验证](validation/d2d-layer-token-range-4060.md)。
