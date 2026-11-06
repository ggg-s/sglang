# DSV FlashMLA full → remote 捕获边界修复

日期：2026-11-06。分支 `codex/remote-attn-pre-layerwise`，本地基线
`5884814306731be4485f3fa7ef3666e8fbe1979e`，包含该基线之后的未提交工作。
用户的 H20 故障报告对应更早的 `cc0cc820ad`；本报告不将两者混为同一验证版本。

## 代码核查与修复

native bank 的顺序是 `warmup → hook → warmup → hook → capture`。DSV 在
full capture 中由 Raw 升级 Full，并可持有本次 capture 才分配的 FlashMLA
调度张量。capture 记录初始化内核，不能证明其已经执行。原先 full 返回后直接
开始 remote bank 的首次 eager warmup，此时还没有该 bank 的 post-warmup hook。

`RemoteGraphBackend.capture_one` 现在在 full 返回、remote bucket/key 检查通过后，
开始 remote session/warmup 前调用已有 `post_warmup_hook`。DSV hook 重新创建
适用的 c0/c1/c2/c4/c128 FlashMLA metadata wrapper（同时丢弃其
`tile_scheduler_metadata` 和 `num_splits` 的 host 缓存引用），并恢复该 bucket
的 `_current_capture_raw`。remote 首次 eager forward 因而重新构造并实际填充
metadata；之后继续使用原生的两次 warmup 与 capture 顺序。

不原地清零或修改旧图 tensor。full graph 的 capture allocations 仍由 graph
pool 保活；remote bank 使用自己的 pool。无 hook 的 backend 和非 remote bucket
保留原行为，DSV 多 candidate variant 仍只为每个 size 捕获一次 remote bank。
该修改发生在启动捕获阶段，不改变 Q/O、D2D、逐 entry Local 或回收协议。

## 对故障报告的限定

- `2 → 1` 是 H20 报告中的稳定复现组合；代码缺口位于同 bucket 的
  `full capture → remote eager warmup`，不能推断其他组合永远安全。
- Blackwell fast-path gate 不是需要放开的地方。Hopper 使用 FlashMLA 原生
  scheduler；只需避免用非空 metadata 跳过尚未执行的初始化。
- “清空”应指重建 wrapper/恢复 Raw，不能将 tensor 全零作为正确性修复。
  全零调度并不能证明 partition、split、batch 范围或输出正确。
- 本修复不修改外部 FlashMLA 内核。负索引防护可另作上游防御，但不能替代
  调用方初始化和生命周期的修复。

## 验证

| 验证 | 结果 |
| --- | --- |
| 既有 CPU/socket/TP/P4/graph 回归 | 361 项通过 |
| 新 metadata 生命周期回归 | 8 项通过 |
| 4060：`1`、`2→1`、`4→1`、`8→4→2→1` | 四组通过，18 个 CUDA graph 捕获与重放 |
| 4060 负对照：跳过新增边界 hook | 首次 remote eager 读取确定性拒绝 poison |
| 选定 Ruff 检查、新文件格式、`git diff --check` | 通过 |

CPU 回归使用实际运行时代码的 AST 加载、模拟 capture-only 未初始化数据；
覆盖所有适用 ratio 的两类调度张量、Raw 恢复、旧 tensor 内容/地址、非 remote
bucket、无 hook、多个 candidate variant、异常状态恢复与重试。

4060 probe 使用真实 CUDA graph 录制 fill producer，再由 eager consumer
检测未初始化状态。为确定性复现，probe 在 capture 外预分配并毒化小型张量；
它不复现 FlashMLA 的具体 allocator 残值。保留旧 wrapper 验证未被原地改写，
并验证旧图 replay 的输出和地址；此探针不单独证明 allocator 自动保活行为。
运行时代码仍使用既有 full/remote graph pool 所有权。

验证机为 RTX 4060 Laptop GPU，Torch `2.13.0+cu130`，隔离目录
`/tmp/flashmla-capture-4060-20261106`。三份实际 GPU 测试源码 SHA-256 与本地一致。
[JSON 结果及源码哈希](flashmla-capture-lifecycle.json)；本地日志在
`/private/tmp/flashmla-capture-fix`。

```bash
python test/registered/unit/plugins/test_flashmla_capture_lifecycle.py
python test/manual/remote_attn/flashmla_capture_lifecycle_probe.py
```


4060 生命周期 probe 不加载模型权重，不运行 SM90 FlashMLA、H20 TP4 或 RDMA。
尚需在报告中的 H20 环境以 `SGLANG_REMOTE_BATCHES=1,2` 和完整 bucket 集合
重新启动捕获，并验证 remote/full replay 与实际请求输出。不能将本地回归通过
表述为 H20 原场景已验收。
