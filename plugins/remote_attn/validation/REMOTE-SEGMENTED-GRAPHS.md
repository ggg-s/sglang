# Remote 分段 CUDA graphs

2026-10-30，基于 `b136e62de4`，目标分支 `codex/remote-attn-pre-layerwise`。

## 实现

本次只修改插件、测试和文档，SGL/SMG 核心没有新增改动。新增后端工厂 hook，
共 24 个 hook；核心接口/状态预检扩展到 47 项，包括 breakable API 与 pool 字段。

`segmented_graph.py` 组合原生 FullCudaGraphBackend 和 BreakableCudaGraphBackend。
full 图沿用线上 capture buckets；额外只为 batch 1/2/4 捕获分段图。Qwen3 dense MHA
每层 RadixAttention 都是一个固定 eager 断点，通常每桶为 `层数+1` 个 segment。

```text
Graph：Norm/QKV/RoPE
图外：KV 写入、Local/Remote attention、P/D RPC、O/LSE 合并
Graph：O projection、TP 通信、MLP、下一层 Norm/QKV/RoPE
……
```

- 工厂在启动 capture 前安装 wrapper，full capture 保持原生 attention；Remote capture
  使用 dummy Local attention 分配桥接输出，不发送 RPC。分段图由原生 breakable primitive
  捕获及重放，没有复制模型层循环或 graph runner。
- replay 中 Q/K/V 地址来自捕获图，实际请求、epoch、长度和 Remote mask 来自本轮 live
  ForwardBatch。每个 token 开始冻结 mask；图外结果由原生 breakable copy 写入固定输出
  bridge，下一段消费相同地址。分段池与 full 池分开，退出后恢复原生 pynccl pool 选择。
- 仅 exact batch 1/2/4 支持 Remote graph；3、8 等其他 batch 进入原有 eager，不能把
  dummy/padding 行发送给 RPC。embedding override 等不支持的输入也在执行前回退。
- 继续复用统一的接收租约：租约非空选 segmented/eager，全部 scatter/传输完成并统一
  回收后恢复 full。Local full 不新增 CPU collective，也不新增 CUDA event。旧手动 import
  撤销 fast path 后仍使用旧保守判定，不启用 Remote segments。
- TP 在 capture 前校验模式、开关、buckets、模型/层数及相关配置，避免一端退出而另一端
  进入额外 capture barrier；capture 后再检查分段边界 manifest。每层始终保留相同断点，
  不因 rank 本地 Local/Remote 状态改变 segment 或模型 TP collective 顺序。
- 每次 Remote replay 结束/异常都记录当前流 event；cleanup 等完成后释放两套 bank。
  RPC 失败不会重跑 token。记录 event 失败时标为 unsafe，保留 bank 并拒绝继续执行/释放。

默认在 `HANDOFF_PRE_LAYERWISE=remote` 的 D 端开启；P 端无新增 capture，普通 prefill
保持原生，最终分层 chunk 和 P 的 Remote prefix RPC 继续 eager。
可在 D 设置 `HANDOFF_REMOTE_SEGMENTED_GRAPH=0` 重启作 Remote eager 对照。

D 要求 native full decode backend，实际原生桶包含 1/2/4；不修改线上其他 buckets。
需要 cuda-python，首版不支持 torch.compile、debug-eager、memory saver、自定义 full backend。
额外 graph exec、bridge 和私有 pool 会增加启动时间与显存，真实模型开销及净收益待测。

## CPU 与双进程验证

完整现有 CPU 文件覆盖 70 项用例，包括：捕获时零 RPC、固定输出地址、多个后续请求
使用 live metadata、未覆盖 batch 回退、Local 恢复 full 且不增加 event、异常清理和未证明
完成时保留 bank。核心预检覆盖 graph API 缺参及 pool 字段缺失。
最终轮 70 项在 3.890 秒内通过；回放 `b136e62de4` 的旧 graph 判定时，新增用例在
Remote batch eligibility 断言失败，当前实现通过。

真实双进程 Gloo 继续覆盖实际准入/回收的八阶段生命周期；另增加分段开关/buckets
不一致在 capture 前两端共同拒绝，以及 Local→Remote→Local→Remote 的 batch 路由矩阵：

| 阶段 | batch 1/2/3/4/8 的 graph eligibility |
| --- | --- |
| Local | T / T / T / T / T（native 条件允许） |
| Remote | T / T / F / T / F |

这些判断不增加逐 forward all-reduce；原生 capture 与模型 collective 仍保留。

```bash
python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

## 实际 CUDA 探针

GPU：NVIDIA RTX 6000D，PyTorch 2.12.1+cu130。使用任务独立目录，未停止或修改既有
服务。探针源加载本分支原生 full/breakable backend 与 primitive，实际执行 CUDA
capture/replay、GEMM、主机往返、GPU 合并和 stream event；模型为 3 层、宽度 8。

| 项目 | BF16 / FP16 结果 |
| --- | --- |
| Remote batch 1/2/4 | 各 4 个真实 CUDA segment |
| Remote replay | 各 12 次，输出与 eager reference 一致 |
| 全 Local full replay | 各 4 次，输出一致且没有 RPC callback |
| capture RPC | 0 |
| 图外断言 | callbacks 中无 CUDA capture，且读取本轮 marker |
| 层状态变化 | 全 Remote、部分 Local、全 Local mask 均通过 |
| 故障 | 第二层注入 RPC 失败，后续层不执行，live frame 清理，event 后安全释放 |
| 峰值 allocated | 各 67,142,144 bytes；不是实际 Qwen3 图显存需求 |

原始结果：[BF16](remote-segmented-6000d-bfloat16.json)、[FP16](remote-segmented-6000d-float16.json)。
结果附带插件、探针和原生 backend/primitive 的 source SHA256，已核对与本次提交源码一致。

**验证边界**：该 GPU 环境没有完整 SGL/sgl_kernel。探针用模拟 attention/RPC、TP1
协调器及关闭的 pool borrowing/precarving 边界；通过显式 `--allow-strong-refs` 保留强引用，
没有验证生产 weak tensor view 的内存回收行为。实际分段 graph 与新插件调度代码执行了，
但没有运行真实 Qwen3 attention/LSE、Mooncake、TP2/NCCL 或 SMG PD 服务；不能据此宣称
端到端数值、生产可用性或性能收益已经验收。

```bash
python test/manual/remote_attn/segmented_graph_probe.py --dtype bfloat16
python test/manual/remote_attn/segmented_graph_probe.py --dtype float16
# 缺少 sgl_kernel 的独立 GPU 环境，显式接受上述强引用验证边界后：
python test/manual/remote_attn/segmented_graph_probe.py --dtype bfloat16 --allow-strong-refs
```

部署机使用[原生单 pair 命令](../SINGLE-PAIR-ZH.md)，确认各 rank 捕获 1/2/4；诊断时
设置已有 `SGLANG_LOG_DECODE_GRAPH_KEY=1`，实际 Remote/Local replay 分别应有
`replay bank=remote` / `replay bank=full`。Trace 中 RPC/merge 应在 segment 之间；比较
与 `HANDOFF_REMOTE_SEGMENTED_GRAPH=0` 相同输入、greedy 输出、chunk/optimistic/Batch 的
TTFT、Remote ITL、Local ITL、E2E、吞吐和显存。正式性能轮关闭逐步日志。
