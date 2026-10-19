# 原生 SWITCH 接入与 RTX 4060 对照

**已接入并验证：一个 SWITCH 可以替代整棵 IF 树。** 同一轮小 GEMM
调度测量中，B=32/N=16 的 GPU dispatch 从 IF 的 **78.78 µs 降到
30.18 µs（降低约 62%）**，完整轮 kernel 从 35 个降到 14 个。
前一轮测得 78.91/30.27 µs，主要设备开销下降在两轮中一致。
这是相对于 IF 实现的收益，不能直接当成真实模型的吞吐收益。

## 接入与启用

在已有 split 配置中设置：

```json
{
  "draft_graph_mode": "split",
  "conditional_draft_graph": true,
  "conditional_draft_graph_dispatch": "switch"
}
```

完整模板：[device-draft-split-switch.json](../../configs/device-draft-split-switch.json)。
默认仍不启用 conditional；已经开启 conditional 而没有指定 dispatch 的
配置继续使用 IF。详细要求见 [CONDITIONAL_GRAPH.md](../../CONDITIONAL_GRAPH.md)。

- 用一个 native setter kernel 读取 GPU int32 `num_neural`，设置 SWITCH
  handle；0 是空分支，1..B 对应精确行数，保留原始动态压缩语义。
- 每个 tail 由 PyTorch 捕获为 `CUDAGraph(keep_graph=True)`，保留 Graph
  对象与 private pool，再作为所选 SWITCH body 的 child graph node。
  不在 replay 热路径调用 Python 分支，也不调用嵌套 `graph.replay()`。
- 小型 CUDA JIT 适配器在 parent 捕获中插入 setter/SWITCH 并更新依赖。
  PyTorch 继续负责 parent、leaf 的捕图和内存池生命周期；parent 先 reset，
  再释放各 leaf。尾段为已有的确定性 greedy 推理，不支持随机采样 child。
- 统计、`live_bs.fill_`、prefix 和行压缩均保持原逻辑，便于单独比较 dispatch。
- SWITCH 与 IF 的离线耗时表指纹不同，需要按新 dispatch 重跑表；
  默认 IF 和默认关闭仍兼容之前的表。诊断快照会标出 dispatch。

CUDA SWITCH 自 CUDA 12.8 提供，按整数条件选择一个 body；见
[NVIDIA 官方说明](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)。
此实现需要 runtime/nvcc >=12.8、Ninja、兼容 C++ 编译器与 raw graph API，
在 Torch 2.14.1+cu130 / CUDA 13.0 实测。CUDA 12.8 的兼容编译分支尚未实测。
捕图失败报错，不允许 TP rank 独立回退；TP2 collective 仍需双卡验收。

## 环境与正确性

`ai-workstation` / RTX 4060 Laptop 8 GiB / WSL2，驱动 592.82；
Python 3.12.3，Torch 2.14.1+cu130，CUDA 13.0，Triton 3.8.0。
沿用此前隔离 venv / CUDA 13 编译环境，在独立目录
`/workspace/conditional-switch-20261019/repo` 测试。

| 测试 | 通过数 | 证据 |
| --- | ---: | --- |
| CUDA compact / 动态行压缩回归 | 300 | [日志](cuda.log)、[JUnit](cuda.xml) |
| 组件、动态 verify、recurrence、SWITCH 生命周期 | 359 | [日志](components-final.log)、[JUnit](components-final.xml) |
| 首次 native SWITCH 专项 | 4 | [日志](native.log)，是上行的子集 |

两组最终合计 **659 个不同用例通过，0 失败、0 跳过**。CUDA 组包含
20 项新增 SWITCH 用例，覆盖 triton、dsv4、EAGLE3、hot-token EAGLE3、
STANDALONE 的 B=1/8、非二次幂 B=3、source caps、无 tail、缩批与 padding。
这些模型计算为合成实现，原生 worker recurrence、metadata、topk、行压缩
和 CUDA Graph 路径保留，不代表完整权重服务测试。

新增 native 专项在 B=1/3/8/32 上反复切换每个行数，检查空分支、精确大小、
中间临时 tensor 的 pool 保护及 histogram/mixed 计数。独立 GEMM 探针
每次测量前再校验 **290 轮**不同 tail 行数、前/后行排列、raw batch 和 padding；
两轮共 580 次校验，输出及统计与独立参考一致。

首次组件回归为 357 通过/2 失败：已有测试匹配原错误提示文本，另一项手工
构造 backend 的测试 fixture 缺少新增 dispatch 字段。恢复提示中的绑定说明、
补齐 fixture 后，最终 359 项通过；未修改 CUDA 计算来绕过测试。
[初次日志](components-initial.log)、[初次 JUnit](components-initial.xml) 保留。

本地 CPU 回归 117 通过、4 GPU 项跳过（以上 GPU 项已在 4060 实跑）。
生产 Python、独立测试和诊断脚本通过 Ruff；已有大测试文件仍有原有 lint
告警，未做无关重排。wheel 构建通过，确认包内含 `switch_graph.cu`。
测试脚本：[run-regression.sh](run-regression.sh)。

## GPU 调度耗时

单位 µs，中位数，第二轮结果。B 是捕图容量，N 是实际 neural 行数。
**排除公共 prefix 和行压缩**；生产路径包含 histogram/mixed 统计及 fill。
direct 提前知道所选 tail，只是诊断下界，不能直接用于动态服务。

| B / N | direct | IF 树无统计 | IF 完整路径 | SWITCH 无统计 | SWITCH 完整路径 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 0 | 1.02 | 8.19 | 19.46 | 7.17 | 18.43 |
| 1 / 1 | 8.06 | 15.49 | 24.48 | 14.46 | 23.55 |
| 3 / 1 | 8.13 | 34.02 | 43.30 | 14.59 | 23.94 |
| 4 / 2 | 9.28 | 35.36 | 44.83 | 15.81 | 25.25 |
| 8 / 4 | 9.63 | 45.12 | 54.82 | 16.45 | 25.82 |
| 16 / 8 | 10.72 | 55.97 | 65.63 | 17.70 | 27.33 |
| 32 / 16 | 12.93 | 69.22 | 78.78 | 20.48 | 30.18 |

B=32 时剩余成本可以拆为：直接 tail 12.93 µs；SWITCH 控制/child graph
执行额外约 7.55 µs；统计约 7.62 µs；fill 约 2.08 µs。
这里的相邻中位数差是消融估计，不是可移植的硬件常数。

SWITCH 主要去掉了随 IF 深度增长的串行控制开销，B=1 的 IF 本来只有一个
节点，因此收益很小。并没有额外跳过 neural 计算，也没有减少有效输出。

## Kernel 表

B=32/N=16，完整轮 profiler 的 5 轮归一化次数：

| 类别 | IF kernel 数/轮 | SWITCH kernel 数/轮 |
| --- | ---: | ---: |
| 公共 prefix、行压缩、tail | 6 | 6 |
| IF predicate 比较 | 11 | 0 |
| condition handle setter | 11 | 1 |
| histogram / mixed 统计 | 6 | 6 |
| live_bs.fill_ | 1 | 1 |
| **总数** | **35** | **14** |

原始 trace：[IF](measurements/b32-n16-production.trace.json.gz)、
[SWITCH](measurements/b32-n16-switch_production.trace.json.gz)。
PyTorch 的 `set_conditional_handle_kernel` 变为 native `set_switch_handle`，
不是把 setter 全部消掉。共享分析 helper 的 `set_handle_kernels_per_step`
只匹配前者；新字段 `switch_handle_kernels_per_step` 补充计数，SWITCH 为 1。
无统计 SWITCH 的完整轮为 7 个 kernel。统一分析器原始三表见
[triage.txt](triage.txt)，单 trace 的 Python 归属 unresolved 不用于强行归因。

## 重叠/空闲表

| 观测 | 证据 | 限制 |
| --- | --- | --- |
| IF/SWITCH 均无中间 `.item()` | 两者 5 轮 aten::item 均为 0；普通路径为 5 | 最终 verify 读回不在此探针中 |
| 图启动仍为两次 | 非零 tail 每 5 轮均 10 次 cudaGraphLaunch | prefix 尚未合图 |
| 零 tail 仍需 dispatcher | SWITCH 2 次启动、10 个 kernel/轮；普通 1 次、2 个 | SWITCH 不会让 skip 分支免费 |
| 控制流成本下降可复现 | 两轮 B32 GPU 78.91/30.27、78.78/30.18 µs | 不是 TP/NCCL 或整模型 overlap 证据 |

## 融合表

| 项目 | 状态 | 收益与边界 |
| --- | --- | --- |
| IF 树替换为一个 SWITCH | 本次已实现并实测 | 分发结构减少；不是模型 kernel 融合 |
| 融合 GPU 统计 | 尚未实现 | 仍有 6 个统计 kernel，B32 消融约 7.62 µs |
| prefix/dispatch 合图 | 尚未实现 | 启动数保持原样，收益需要另测 |

没有已确认的新模型融合模式；没有同时改统计或启动策略来扩大 SWITCH 收益。

## 与普通 .item() 路径的关系

完整一轮 wall（第二轮，µs，包含末尾同步）：

| B / N | 普通 split | IF | SWITCH |
| --- | ---: | ---: | ---: |
| 1 / 0 | 89.68 | 79.25 | 80.77 |
| 1 / 1 | 117.38 | 98.99 | 101.58 |
| 4 / 2 | 129.32 | 123.54 | 113.97 |
| 8 / 4 | 125.92 | 138.41 | 107.24 |
| 16 / 8 | 122.89 | 160.08 | 99.04 |
| 32 / 16 | 133.58 | 197.60 | 120.05 |

第一轮 B32 普通/IF/SWITCH 为 117.28/196.88/119.40 µs：SWITCH 显著
改善 IF 回退，但相对普通路径是持平到小幅领先，host/WSL 噪声足以改变
较小差异。**不据此声称持续优于普通 split，更不换算服务吞吐。**
真实模型、TP2 collective、启动显存与生成吞吐仍待验证；默认开关未改变。

## 证据与复现

- [compare_dispatch.py](compare_dispatch.py) 复用前一轮
  [ablate_dispatch.py](../conditional-graph-overhead-4060-20261019/ablate_dispatch.py)
  及 [dispatch_probe.py](../conditional-graph-4060-20261019/dispatch_probe.py)，
  对同一 tail 与相同统计做对照；不替换 serving 导入。
- 31 组随机 CUDA event 样本，start event 前排入 2M-cycle sleep，让 CPU
  提交先完成；wall 为 50 次交错预热、7 组随机排列 × 每组 100 轮，每轮
  末尾同步。profiler 只用于次数，不从 profiled kernel 时长外推 event 延迟。
- 第二轮 [comparison.json](measurements/comparison.json)、
  [首轮 comparison.json](measurements-initial/comparison.json)，各自 trace
  均保留；[运行脚本](run-benchmark.sh)、[日志](benchmark.log)。
- [metadata.json](metadata.json) 的 12 个源码/测试哈希与本地逐项一致；
  辅助脚本也校验哈希，证明测试的是当前代码。
- [post-test-gpu.txt](post-test-gpu.txt) 记录结束后 GPU 0 MiB、0%。
