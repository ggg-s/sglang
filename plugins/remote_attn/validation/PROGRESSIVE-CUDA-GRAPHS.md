# Progressive CUDA graphs 适配与验证

2026-10-30，基于 `ba7b2891296b`，目标分支 `codex/remote-attn-pre-layerwise`。

## 行为与改动范围

仅修改 `remote_attn` 插件、已有 CPU 测试/双进程 fixture 及文档；本次没有新增
SGL 或 SMG 核心改动，也没有新增环境变量、启动包装器或 graph capture 实现。
继续使用现有 23 个函数边界 hooks 和分支已有的 progressive 核心扩展。

- 删除两个启动预检中“必须关闭 graphs”的限制，保留 radix cache、overlap scheduler、
  模型、KV layout、并行拓扑等原有约束。
- Decode/Prefill `can_run_graph` 先计算原生 eligibility，再对 eligibility、Local readiness
  做一次 TP CPU-group MIN all-reduce。没有 target 的 rank、原生不满足 capture 范围的
  rank 也必须投票；TP1 不发 collective。
- 任何 rank 存在缺层、失败或 aborted target，整组走 eager。Layerwise 等待也受保护，
  防止 graph 绕过 Python 的逐层等待。检查范围为该进程全部 target states，较 batch
  局部判断保守，因此其他活跃 Remote 请求也会暂时阻止本批 graph。
- 全部 rank 的 target 层都完成 GPU 可读证明，且原生 eligibility 均为 true，恢复原生
  replay。无需清除 target 对象或重新 capture；新 Remote 请求会再次使整批 eager。
  `HANDOFF_GRAPH_DISABLE_ANY_STATE=1` 是保留的实验覆盖，验收默认路径时应清除。
- P 最终 chunk 由已有 `SPLIT_PREFILL` 路径执行，仍为 eager；普通中间 chunk 保留原生
  graph dispatch、chunk 发送及 optimistic prefill 配置。P 提供 RPC 的计算路径不变。
- 原生 graph runner 仍负责 capture buckets、padding、加载 live batch 和实际 replay。
  插件只记录 verdict 变化；`graph-eligible` 日志本身不代表实际执行了 replay。

TP>1 每次 eligibility 检查增加一次 CPU collective，Local 稳态也包含这个成本。
没有性能收益数据，不能以恢复 graph eligibility 直接推断净收益。

## 本地证据

本地为 macOS CPU 环境，使用临时隔离 Python 3.12 / PyTorch 2.15 和源加载的实际
`CustomTestCase` harness，绕过无关 server/eval 导入。没有启动 GPU server。

| 检查 | 结果 |
| --- | --- |
| 完整 `test/registered/unit/plugins/test_remote_attn.py` | 66 项通过，最终轮 8.020 秒 |
| 真实双进程 Gloo fixture | 两种 runner 的九阶段结果均为 `T,F,F,T,F,T,F,F,T`，两个 rank 一致 |
| 原生 `ModelRunner._forward_raw` 源加载分支 | Remote→Local→新 Remote 对 Decode/Prefill 均为 eager→graph→eager；split-prefill 不走 graph |
| GPU scatter 完成边界的 CPU 合约 | host 接收及未完成 scatter 不可 Local；全部 scatter 完成后可 Local |
| 预修复函数回放 | 三个选定用例失败：layer 等待被绕过、Prefill graph 不恢复、TP graph 启动被拒绝 |
| 源码预检 | 23 hooks、40 项核心接口/状态检查通过 |

预修复回放将 `ba7b2891296b` 的相关函数加载进同一测试 fixture，保持测试和其余依赖
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
