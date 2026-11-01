# Qwen3 D radix cache：本地验证

日期：2026-11-01；基线 `7fce6c0fee`。本次仅本地测试，没有上传源码或运行 GPU。
运行时只改插件；SMG 和 SGL 核心无新增修改。启动方式见
[D 缓存接入](../D-DECODE-CACHE-ZH.md)。

## 实现范围

公共 `destination_cache` 保护层复用原生 prefix match、lock、checkpoint、finish、
request handle 和增量 PD metadata。Qwen D 在 native radix opt-in 下接入；DSV4.1
D cache 显式拒绝。P 缓存、原有 graph、optimistic prefill 和 chunk 配置保持。

固定 H 的等待证书覆盖容量不足时的重试，不重复 TP 匹配共识。早接纳取得活动 pin，
缓存 checkpoint/finish 延迟到传输与 GPU 完成后的有序退休。命令包含发布/ABORT 意图，
旧命令遇到取消不得发布；缓存执行后的额外结果投票只发生在此转换点。单 rank 原生
发布失败会阻止所有 rank 恢复 Local；未知 DMA 完成保留页面，不能视作安全取消。

D scheduler overlap 可以同时开启：Remote 串行，在最终结果消费、缓存发布及 TP
共识之后恢复原生 Local 流水。关闭 overlap 的对照路径仍强制安装有序退休，并在
真实 forward stream 边界记录 GPU event。Remote 等待、稳定 admission 重试以及
Local 稳态均不新增逐轮 collective。

## 测试证据

**184 项插件测试 + 7 项共享 BCG 输出测试，共 191 项通过**。插件测试包含
10 项 D cache 用例、62 项 DSV 用例，以及真实双进程 Gloo TP2；不是 191 次模型推理。
源码预检 **46 hooks / 113 core interfaces**；isort、Ruff、diff whitespace 通过。

- 真实原生 RadixCache 树、checkpoint、锁、token/request allocator 和 release：
  命中 H=2 的共享前缀，私有尾部在 PD/GPU 未完成时不发布、去重或回收；并发相同
  尾部成为 canonical 后，只在安全转换点重绑。释放与淘汰回收全部32页，无重复页。
- 真实原生 Decode prefix match / `_pre_alloc` / prefix key 构建：仅分配尾部2页，
  `[0,H)` 地址保持不变，metadata 必须承诺相同 H。8次容量重试复用固定匹配，
  不重复 native lookup 或 TP 共识；未分配取消解除等待 pin 和 native lock。
- cached-prefix-relative sender cursor 与 host scatter 覆盖：层完成为 `[H,N)`，
  已缓存区间另有独立证明；改变 H/attempt/lock 或错误 wire end 拒绝执行。
- 自动在 Local 转换发布完整 prompt，不依赖 decode 期间先发生一次 checkpoint。
  单独延迟传输、GPU completion、native result 后，才能退休。最终 done 字段验证
  命中2、保留4、guard closed 和 TP committed。
- 取消的 ABORT 优先于已排队 SUCCESS、晚到 checkpoint 与 release；旧发布命令
  无效后等待新状态边沿。未知 DMA、native cache 部分失败和 reset 均不误回收。
- 真实 native `Req.reset_for_retract` 不自动推进 cache handle；插件在排空、备份、
  reset 后调用 native 公共 advance 接口，新 attempt 不继承旧 ABORT/匹配。
- source-load 原生 PD overlap 调度循环：先消费之前的 Local 结果，再串行运行
  两个 Remote batch；缓存发布及执行结果投票在首个 Local launch 前完成；随后
  Local batch 恢复一前一后流水。16次后续 Local launch 禁止额外 proof query /
  all_reduce。overlap off 用例缺少 native copy_done，也必须等已记录的 event。
- Gloo TP2：前缀长度不一致与单 rank native match 异常均在 metadata 前拒绝且清理
  pin；稳定等待重试禁止 collective。缓存意图不一致不发命令，一 rank GPU 未完成
  不退休，广播后 ABORT 使旧命令失效。最终 shared command 的投票形状依次为
  `[5,5,5,1,5,1]`，包括缓存执行结果票。32次 Local 轮询禁止任何 collective。
  注入一 rank cache 发布异常时，即使另一 rank 已插入成功，两 rank 都停止恢复
  Local，失败 rank 的 pin 保留。
- 负对照：仅绕过 checkpoint 暂存，原生树立即匹配到4而非2，用例失败；仅换回
  基线 `Target.refresh`，H=2 的合法 suffix scatter 被误判成未覆盖 final chunk，
  用例失败。新实现对相同用例通过。

标准完整依赖环境中可执行：

```bash
PYTHONPATH=python:plugins/remote_attn python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

本机使用轻量 CPU harness source-load 上述原生逻辑以避开 CUDA/Triton 导入；
CUDA event、网络 receiver 和 TE 在边界用替身，Gloo TP2 使用真实本机进程。
没有实际 CUDA capture/replay、DMA、完整 Qwen checkpoint、NCCL、SMG PD 端到端
或延迟/吞吐结果。UnifiedRadixCache 的范围由公开接口与后端策略检查约束，尚无完整
模型验证。真实 GPU 上仍须做 cache off/on、overlap off/on 输出一致性、淘汰压力、
取消及 retraction，并确认未增加不可回收页和稳态同步后再验收收益。
