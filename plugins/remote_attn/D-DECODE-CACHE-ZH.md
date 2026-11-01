# Qwen3 D 端 radix cache 与 progressive / overlap

此接入复用原生增量 PD：D 先匹配并锁定已缓存的前缀 `H`，只为 `[H,N)` 分配和
传输新 KV。SMG、SGL 核心没有新增修改；公共保护层和接入均在插件内。
当前支持 Qwen3 dense NHD MHA、page size 1、同 TP 的既有部署。DSV4.1 D 缓存
仍被启动检查拒绝，其 indexer、SWA/carry 需要另行适配。

## 原生启动差异

沿用 [单 pair 的 P/D/SMG 命令](SINGLE-PAIR-ZH.md)。只在 Qwen 的 **D 命令**中
删除 `--disable-radix-cache`，加入原生参数：

```bash
--disaggregation-decode-enable-radix-cache
```

只删除 disable 参数不会打开原生 D cache。无需新增插件环境变量。P 保留原有
FULL 前缀缓存配置；原有 graph、optimistic prefill、token chunk、传输 Batch
和 SMG 路由不变。D 可以保持 scheduler overlap 开启：Remote 串行，Local 恢复
原生 overlap。需要串行对照时给 D 加回 `--disable-overlap-schedule`，缓存完成
证明和 TP 退休仍生效。P 的 overlap 支持范围沿用原有模型限制。

支持内置 `RadixCache`、设备内 FULL `UnifiedRadixCache`；不支持 HiCache、外部
linker、session refs 或混合缓存组件。新增预检总计 46 hooks / 113 core interfaces。

## 前缀、页面与发布边界

1. 在原生匹配后、发送 PD destination metadata 前，各 TP rank 确认相同的请求、
   attempt、缓存 key 和前缀长度。独立 pin 保住匹配结果。若容量不足、原生准入
   解锁并重试，复用同一份证书和 H；不重复 lookup 或进入 collective。成功发送
   metadata 后撤下等待 pin，原生已分配请求的 lock 继续保护 H。
2. 已命中的 `[0,H)` 直接成为各层的有效覆盖；native sender/scatter 的 suffix
   cursor 按 H 换算。早接纳后再取得独立活动 pin。新尾部保持 request-owned；
   checkpoint/finish 暂存意图，禁止传输或 GPU 读取期间去重、重绑行或释放页面。
3. 各层 progressive Local 只表示该层可以本地读取。缓存发布还需 native receiver
   成功、host scatter 完成、Remote Q/O/LSE 生命周期完成，以及参与过的 forward /
   sampling / readback GPU 完成证明。overlap 模式复用 native `copy_done`；串行
   模式在实际 forward stream 记录 event。不能以 `kv_committed_len` 证明 DMA 完成。
4. 有序退休命令携带 H、key、待发布长度、ABORT 和 finish 意图。各 rank 复查证明及
   意图，一致后调用原生 checkpoint / dedup / release。即使请求还在 decode，也会
   发布已经完整到达的 prompt；长度被实际 owned KV 裁剪，不发布尚未计算的输出。
   随后仅在此转换点确认原生操作在所有 rank 成功，再允许 Local full graph 和
   原生 overlap。Local 的后续缓存行为回到原生路径。

等待中的固定匹配可能错过稍后出现的更长前缀，pin 也会暂时减少可淘汰容量。
这是保持 admission 重试无额外 TP 同步的代价；原生队列在首个容量不足请求处
停止准入。活动租约延迟发布也会增加短时 KV 占用和相同尾部的重复存储。

## 取消、retraction 与失败

取消把同 attempt 标成 sticky ABORT；后到的 SUCCESS/checkpoint 不能把它改回可
发布状态。仍等待已有 PD 写入与 GPU 工作安全排空，再由原生 release 释放私有页，
最后解除独立 pin。这里复用已有完成协议，没有新增快速 transport cancel ACK。

retraction 在原生备份 KV 之前进入 TP 一致 drain，禁止读取半传输的行；清理完毕
后再 reset，并通过原生公共接口推进 cache request handle，旧 ABORT/匹配不能污染
新 attempt。活跃或等待中的租约阻止 cache reset。

未知 DMA 完成、超时、损坏的 attempt/row/lock 都保留页面并报错。若原生 cache
发布在某一 rank 失败，转换后的结果共识让所有 rank 停止恢复 Local，并隔离 cache；
不能回滚已经发生的原生树修改，需人工处理故障实例。

Remote 轮询仅更新本地证明和状态变更通知；collective 次数、形状及顺序来自已广播的
有序命令。首次 prefix 匹配有一次准入共识，缓存退休有原有证明投票及一次执行结果
投票。稳定 admission 重试、Remote 等待和 Local decode 不新增逐轮 all_reduce。

## 核对与验收

对同一 D 重复发送共享长前缀，核对原生缓存命中、PD metadata 的 `decode_prefix_len`
与实际传输尾部。early-admitted shard 退休后的 `done.json` 新增：

- `d_cached_prefix_tokens`：本次 D 命中的 H。
- `d_cache_guard_closed`、`d_cache_tp_committed`：租约关闭且各 rank 发布成功。
- `d_cache_aborted`：取消时为 true，该 attempt 不插入尾部。
- `d_cache_retained_tokens`：退休后原生保护的长度；ABORT/free 后可能归零。

这些字段需逐 rank 检查，只能证明相应协议边界。投产前还要在目标 GPU 对照 cache
off/on、overlap off/on 的 greedy output_ids，覆盖并发相同前缀、淘汰压力、取消与
retraction，再量化 TTFT / ITL / 吞吐 / KV 高水位。此次仅本地 CPU 与 Gloo TP2
验证，未运行真实 GPU PD 或性能测试；详见 [验证记录](validation/D-RADIX-CACHE-LOCAL.md)。
