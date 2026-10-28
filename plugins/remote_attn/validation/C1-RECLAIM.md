# C1 reserve/epoch 与异常 host 缓冲区回收

## C1：同一次迁移复用预留 epoch

旧 gateway 的 drain pre-reserve 使用 drain epoch，单请求 `_cutover` 又生成随机 epoch。
目标 `reserve` 正确拒绝了第二个 epoch，因此请求没有进入 freeze，C1 被错误回滚。

修复后的单请求、逐成员 wave、batch cutover 在属于活跃 drain 时复用其 reserved epoch。
reserve 与 freeze 之间先持久化相同 epoch。原先的严格 epoch/大小校验保留，不允许新 epoch 抢占旧 reservation。
网关增加独立迁移顺序锁，使 start_drain 的 snapshot/pre-reserve 与已开始的 legacy cutover 按顺序执行。
这把锁不占用 admission 锁等待 cutover 的 backend I/O，避免阻断普通请求 admission。

还修复了相邻的回滚问题：`_take_reserved` 发现 KV 预留不足时，原来先 free indices，
随后 `_reservation_free` 又 free 一次。现在只由 reservation owner 释放一次。

控制面变更是对原 gateway 的针对性修补，不能再称其与原包逐字节相同；driver 和原始来源 manifest 保留。
初次提交的 CPU 回归验证实际 reserve/cutover 控制逻辑；后续内网已反馈 Copy / Layer / Remote
三模式完整模型 C1 均 SAFE_TO_STOP、96/96 exact，见 [5090 新提交复验](5090-new-commits.md)。

## Host：完成、CUDA fence、native unregister 三个条件

旧代码把 `import_done_at` 和 `error=None` 当作 host 回收条件。
这将“所有层都已本地化”和“源端不再写 host 缓冲区”混为一谈；LOCAL Drop 可以继续 Remote，
却不再达到该回收条件，造成不必要的长期隔离。

现在分别判断：

1. 已收到源端 `{done:true}` 或成功的 Copy ACK，或读到源端完成凭据，证明远端写入已结束。
2. 目标 copy stream 已排空，证明没有 GPU 继续读取 host storage。
3. native memory unregister 成功后，才释放 Python tensor 强引用。

因此 LOCAL Drop 即使保留 error、缺失层继续 Remote，也能回收 host 注册内存。
无法同步的 GPU stream 仍不允许回收，缺层 event 也不会因此被标记为 LOCAL。

## 断连后的完成凭据与重试

源端在完成所有 TE batch 和 source CUDA fence 后，原子写入：

```text
HANDOFF_CTL_DIR/transfer-completion/<transfer-token>.json
```

metadata 中携带该路径和 transfer token。目标仅接受共享控制根目录下、名称/token 匹配的记录。
原连接断开后，目标仍保留 storage；scheduler housekeeping 读取完成凭据并重试注销。
目标 ACK 与源端发布支持任一顺序，正常完成后删除配对记录，避免文件长期累积。

native unregister 返回错误时，registration manager 保留 stale range，tensor 同时留在回收队列。
后续重试成功才丢弃强引用；不能仅因为调用过 unregister 就复用该内存。
相同规则应用于 Copy、CPU layer snapshot、Fast Freeze source/target 和已暴露地址的 setup rollback。

需要所有角色共享同一个 `HANDOFF_CTL_ROOT`，并升级网关、源和目标。
旧端不发布凭据时，收到可信完成消息的路径仍可回收，真正断连的路径会保持保守隔离。
TTL、连接断开和过期 epoch 都不是远端写完成证明。源端失联且没有完成凭据，或 GPU fence 失败时，
仍需人工确认/进程重启；本修复不声称能安全强制释放这些状态。

## 验证

- 修复前 C1 测试无法到达 freeze：第二次 reserve epoch 冲突；修复后复用同一 allocation 和 epoch。
- 确定性双线程回归：drain 在 cutover 之后取 snapshot；等待时 admission 锁仍可用。
- 原不足预留回滚后 allocator available 错增为 36，实际容量为 32；修复后为 32，free indices 无重复。
- full-mirror 和 bounded-slot 的 Drop/EOF 回归：Done 可回收，未知 EOF 保留，缺层仍走 Remote。
- token 不匹配或 GPU 不安全时不可回收；匹配完成凭据后可回收。
- native unregister 首次失败时 weakref 证明 tensor 仍存活，重试成功后 storage 才释放。
- 源发布与目标 ACK 的两种顺序均清理凭据。

完整 CPU/Gloo 回归为 **36 项**。真实 CUDA/Mooncake、Remote Drop RPC、取消后回收及
版本/hash 见 [4060 C1/回收结果](4060-c1-reclaim-results.json)。历史 5090 报告和先前 4060 结果保留，不覆盖。
