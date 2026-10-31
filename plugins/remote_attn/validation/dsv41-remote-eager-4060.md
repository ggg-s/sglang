# DSV4.1 eager sparse Remote：本地与 RTX4060 验证

日期：2026-10-31。接入见 [启动与归属说明](../DSV41-REMOTE-EAGER-ZH.md)。
边界是 **native PD 完成后，首轮 D decode 的 eager RPC**；不是 early admission 或
传输未完成时 decode。生产代码只改插件，SMG/SGL 核心无修改。

## 实现范围

- 继续原生 global/indexer/SWA/carry PD；D 原生 top-k/candidate 选择是唯一权威。
  按 request FULL page map 将其分为 P 已完成前缀与 D 新增行，保留顺序和重复项。
  ratio-2 pending half-pair 排除在 P 可读集合之外；SWA/carry 更新由 D 原生模型负责。
- P 复用 RemoteHub TCP、token/epoch attach、registry acquire/tombstone/release_ref、
  RemoteJob GPU fence。read lease 保护请求行、FULL/SWA cache pin 和源 KV。
  只读选中的 native V41 fp8/fp4 条目，不全量导出 KV、不重做 top-k。
- D attention 在对应 backend 实例内绑定，检查 native sparse ABI，不额外全局加载
  DSV attention backend 到 Qwen。SWA-only 层走 native；混合 batch 保留普通请求的
  native output，不重复 store。带 Remote 的行使用 FP32 Torch eager partial/LSE
  合并；sink 仅在 D 计入一次。
- native PD commit 的既有 TP MIN 建立首轮窗口；同一轮 model forward 完成后写
  remote-done 并恢复 Local full graph/overlap。所有 KV 已经本地，转换不依据 rank-local
  copy readiness。Remote graph bank 暂不捕获。
- P 的实际 GPU reader 未释放前禁止退休。P overlap 继续原有事件驱动最终 TP 共识；
  不增加逐 chunk/逐 decode CPU all_reduce。DSV 模式/staging flags 纳入原有启动共识。
  cancel/retraction 关闭旧 attach、使用旧 generation；陈旧 row/attempt 拒绝读取。

## 本地

**136 项完整插件回归通过，含 38 项 DSV4.1**。新增六个测试覆盖：

1. V41/V41_NEW/V41_FP4/V41_FP4_NEW 的选中 slot，与仓库独立 quant reference
   逐 BF16 bit 对照；异构 scale、页内地址、padding、重复 slot、越界/空集合。
2. 稀疏联合 attention 等价性、sink 只计一次、ratio-2 奇数边界、共享 producer/consumer、
   native top-k 不修改。混合 batch 的普通行保持 native 输出，store 不重复。
3. registry tombstone 后拒绝新 reader，最后一个 read ref 之前不释放 FULL/SWA cache；
   stale generation 不能退休新租约，GPU fence unsafe 标志阻止回收。
4. native PD commit 后首轮 eager，再恢复 native graph；target row 变更拒绝读取。
5. Remote P overlap 保留、D backend 实例绑定幂等、Local native fallback，mega backend 拒绝。
6. Qwen P overlap 限制仍由模型启动校验拒绝，没有因 DSv Remote 而误放开。

真实双进程 TP2 Gloo 在原有测试中扩展覆盖：

- rank 间 Remote/layerwise 模式不一致，由既有启动 all_gather 一致拒绝。
- Remote 稳定观察 32 轮不增加 all_reduce；慢 rank 仍持有 reader 时两端缓存均不释放。
- 最后一份 reader 关闭后，同一条有序命令产生 **一次 5 元素最终投票**，两端各释放一次。
- Local 32 轮图判定不新增同步。原有 63 稳定 chunk 0 接管投票、staging、取消/retraction
  和缓存读租约回归继续通过。

源码预检 **40 hooks / 88 core interfaces**；native D sparse method 的关键参数还会在
实例绑定时检查。isort、ruff format/选定错误检查和 `git diff --check` 通过。

## RTX4060

在已授权 `ubuntu` 工作机 RTX4060 Laptop 8GB / PyTorch 2.13.0+cu130 上，独立目录
`/tmp/dsv41-remote-sparse-20261031`，复用已有 venv，无包安装或部署变更。

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
/workspace/remote-attn-cuda-20261028/venv/bin/python \
  test/manual/remote_attn/dsv41_remote_cuda_probe.py \
  --sglang-root . --output remote-result.json
```

结果：

- 四种布局的真实 GPU 读取均与独立 CPU reference **逐 bit 一致**，scale 非均匀。
- 使用真实 RemoteHub/TargetClient TCP、原生 registry 和 CUDA streams，三次 sparse RPC
  覆盖 c1、c2 及共享 source 的后续消费层。与独立 union-softmax reference 的最大绝对
  输出误差 **3.814697265625e-06**，在用例容差内。
- native PD 未完成时的 RPC 被拒绝；请求 P 读取位置 16 的 ratio-2 pending half-pair
  被拒绝；有效 prefix positions 0/15 正常读取。top-k 和 P KV/SWA canary 保持不变。
- 首轮窗口结束，native graph predicate 恢复，32 轮 Local 检查正常。
- 实际 GPU reader 在等待未完成 CUDA event 时关闭目标连接：P tombstone 拒绝新读，
  FULL/SWA pin 仍保留；GPU reader fence 完成、ref 归零后才可最终释放，两类 pin 均归零。
- 回归现有 D receive staging：真实 Mooncake TCP、1 MiB native pinned ring，P staging
  on/off 两种模式字节一致、回绕、取消 drain、无释放后写入均通过。
- 回归 P overlap + prefix cache CUDA probe：indices/event、四个 plan、normal/cancel/revoke
  回收通过，稳定接管和空轮询没有新增投票。其 IO 是队列边界，不计作模型服务 E2E。

[完整结果与源码哈希](dsv41-remote-eager-4060-results.json) 分别包含 Remote、receive、
overlap 三组证据；校验 **25 / 30 / 20** 份源码 SHA256，均与提交前本地源码一致。

## 尚未验证

没有加载完整 checkpoint 或运行完整 SMG/P/D、多 GPU NCCL。4060 无法验证模型所需的
SM100 FlashMLA/indexer/compressor kernel；使用 source-loaded native pool/metadata/accessor
和真实 native-byte 格式，PD Success/top-k 输入及 SWA-only compute 是小型 fixture。
carry 的 wire/奇偶接续在原有 native CPU planner 用例中覆盖，未跑完整 GPU compressor。

此结果不证明 token/logprob 一致性、生产吞吐、TTFT 收益或网络/进程故障恢复。当前 eager
参考路径包含 CPU 序列化、同步与额外计算，混合 batch 还保留 native batch 计算以保持
普通行输出；不适合作为性能优化结论。下一阶段须先在支持完整模型的 GPU 上建立精度
基线，再接传输未完成时的 D 接纳、全局资源 readiness 和逐资源 Local 切换。
