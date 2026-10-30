# Progressive CUDA graphs 适配与验证

本页记录 Remote eager → Local full 及 Local 稳态优化的验证。
最新 D 端 batch 1/2/4 Remote 分段图实现与 GPU 探针见
[Remote 分段验证](REMOTE-SEGMENTED-GRAPHS.md)；该路径取代下文的 Remote eager 默认行为。

2026-10-30，初版基于 `ba7b2891296b`，Local 稳态优化基于 `ae4e4b92fa`，
目标分支 `codex/remote-attn-pre-layerwise`。

## 行为与改动范围

仅修改 `remote_attn` 插件、已有 CPU 测试/双进程 fixture 及文档；本次没有新增
SGL 或 SMG 核心改动，也没有新增环境变量、启动包装器或 graph capture 实现。
继续使用现有 23 个函数边界 hooks 和分支已有的 progressive 核心扩展。

- 删除两个启动预检中“必须关闭 graphs”的限制，保留 radix cache、overlap scheduler、
  模型、KV layout、并行拓扑等原有约束。
- 启动时一次 TP `all_gather_object` 检查 resolved graph 配置、两个 runner 的类型、实际
  capture buckets 和 max-bs。不一致时全组拒绝启动；任一 rank 已有旧 target 时全组
  使用保守路径。检查成功才启用 progressive fast path。
- Progressive 复用 `Controller.early_targets` 已有的 TP 准入/退休投票。只有统一准入
  才添加接收租约；只有最慢 shard 的 GPU scatter 与原生传输全部完成、统一完成资源
  回收后才删除租约。即使本 rank 已经 Local 或已经关闭，也保留租约到全组同意。
- `can_run_graph` 只检查一致的租约集合：非空就 eager，空则交回原生 eligibility。
  不再扫描本地逐层事件或每步 all-reduce；新请求准入在下一次 forward 前重新阻止 graph。
  其他请求持有租约时，已 Local 的 batch 也保守地 eager。
- Local 恢复需要等统一回收，可能比最后一层刚可读稍晚。无需重新 capture；Native TP
  的输入/配置一致性约束继续适用，不再逐步检测 native eligibility 的 rank 间差异。
- 旧手动 import/resume 控制会在广播后的执行边界全组撤销 fast path，进程后续继续
  对 native eligibility 和 target readiness 逐 forward 做 MIN 投票。Wrapper 安装也校验
  target 所有权，防止未受 progressive 生命周期管理的 state 绕过保护。Layerwise 等待、
  失败唤醒和 `HANDOFF_GRAPH_DISABLE_ANY_STATE=1` 的保守语义在旧路径中继续保留。
- P 最终 chunk 由已有 `SPLIT_PREFILL` 路径执行，仍为 eager；普通中间 chunk 保留原生
  graph dispatch、chunk 发送及 optimistic prefill 配置。P 提供 RPC 的计算路径不变。
- 原生 graph runner 仍负责 capture buckets、padding、加载 live batch 和实际 replay。
  插件只记录 verdict 变化；`graph-eligible` 日志本身不代表实际执行了 replay。

Progressive graph 判断不增加 collective，Local 稳态只增加少量 Python 检查；Remote 期间
已有的完成轮询投票仍然存在，不能称为整个生命周期仅同步两次。尚无 GPU 性能收益数据。

## 本地证据

本地为 macOS CPU 环境，使用临时隔离 Python 3.12 / PyTorch 2.15 和源加载的实际
`CustomTestCase` harness，绕过无关 server/eval 导入。没有启动 GPU server。

| 检查 | 结果 |
| --- | --- |
| 完整 `test/registered/unit/plugins/test_remote_attn.py` | 68 项通过，最终轮 3.877 秒 |
| 旧路径真实双进程 Gloo fixture | 两种 runner 的九阶段结果均为 `T,F,F,T,F,T,F,F,T`，两个 rank 一致 |
| Progressive 实际准入/回收双进程 Gloo | 八阶段租约数 `0,1,2,1,1,0,1,0`，各阶段连续 16 轮 Decode/Prefill 判断均一致，forward 新增 all-reduce 为 0 |
| 启动一致性 | capture buckets 不同则两 rank 均拒绝；只有一个 rank 预存 target 也使两 rank 均进入保守路径 |
| 原生 `ModelRunner._forward_raw` 源加载分支 | Remote→Local→新 Remote 对 Decode/Prefill 均为 eager→graph→eager；split-prefill 不走 graph |
| GPU scatter 完成边界的 CPU 合约 | host 接收及未完成 scatter 不可 Local；全部 scatter 完成后可 Local |
| 预修复函数回放 | 三个选定用例失败：layer 等待被绕过、Prefill graph 不恢复、TP graph 启动被拒绝 |
| 稳态优化预修复回放 | `ae4e4b92fa` 对新增两个用例失败：仍扫描/投票、未受管 state 与 import 控制未撤销 fast path |
| 源码预检 | 23 hooks、40 项核心接口/状态检查通过 |

预修复回放分别将上述基线的相关函数加载进同一测试 fixture，保持测试和其余依赖
不变。当前版本的启动用例同时确认 graph-on 可用且 overlap 限制继续生效。

在完整 serving Python 环境可直接运行：

```bash
python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

CPU 测试的 CUDA/模型边界为替身，断言的是实际状态与原生调度分支，不验证 CUDA
capture/replay、NCCL、模型数值、真实 Mooncake DMA 或 SMG 端到端行为。

## GPU 验收

使用[原生单 pair 命令](../SINGLE-PAIR-ZH.md)，保留线上 graph、chunk、optimistic prefill
和传输 Batch 配置。验证方式见其中“验证 eager → graph 恢复”：各 D rank 的 verdict
日志、原生 `SGLANG_LOG_DECODE_GRAPH_KEY=1` 实际 replay 日志、分 shard Remote 调用
报告及 GPU trace 需要相互印证。覆盖 TP1/TP2、跨 chunk 边界、混合长度、连续准入、
取消和故障；比较完整 greedy 输出及 graph 开关时的 ITL/吞吐。

GPU 数值、真实 eager→graph 恢复及收益验收尚未执行。
