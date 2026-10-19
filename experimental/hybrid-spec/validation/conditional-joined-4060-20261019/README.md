# Prefix / 行压缩 / SWITCH 合图：RTX 4060

**已接入并通过变长验证。** 固定 B=8 的完整轮微测相比当前融合 SWITCH
降低约 **33%～37%**，相比原 `.item()` 降低约 **58%**；B=32 分别降低
约 **33%**、**48%～49%**。循环变长 B=8 也分别降低约 **33%～34%**、
**56%～57%**。范围来自两次测量，不是统计置信区间。

这里的整轮仍是合成 **prefix + 行压缩 + 小 GEMM tail + 统计 + 输入更新 +
末尾同步**，不包含真实模型权重、target verify、NCCL 和请求调度。
本次没有用 kernel-only 时间代替 wall，也没有漏算每轮 raw_bs 更新。

## 接入与变长参数

在已有 SWITCH 配置中额外设置 `"conditional_draft_graph_joined": true`；
默认 false，分段融合 SWITCH 保持可用。完整模板：
[device-draft-split-joined.json](../../configs/device-draft-split-joined.json)。

```text
CPU：更新 batch 节点的整数参数 → 一次 Graph replay
GPU：batch 写入 → prefix child → 读取 GPU raw_bs 的行压缩 → SWITCH → tail
```

- prefix 使用 `keep_graph=True`，作为新主图中的 child；行压缩在新主图中
  捕获，SWITCH 在同一主图中重新创建，继续使用已捕获的固定形状 leaf。
  不把旧 SWITCH 图直接克隆为 child：CUDA 不允许克隆含条件节点的图。
  [CUDA Graph 管理文档](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__GRAPH.html)
- `raw_bs` 放在固定 GPU int32 地址。主图开始是一个 4-byte memset 节点，
  每次 replay 用 `cudaGraphExecMemsetNodeSetParams` 更新其值。节点 template
  保留到 cleanup；已排队/运行的 launch 不受后续参数更新影响。
  没有复用 pinned-host scalar 的覆写竞态，也没有额外的 eager fill launch。
  [CUDA 参数更新语义](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__GRAPH.html)
- 行压缩读取 GPU 实际 raw_bs，以容量 B 限制扫描范围，仅选择有效请求行。
  SWITCH 和 mixed 计数仍使用实际 batch；零 neural 行跳过全部 tail。
- PyTorch Graph 对象、prefix private pool、leaf private pool 和 template/node
  handle 保留到主图 reset 后释放。仅用于已有确定性 greedy tail。
- `joined=true` 进入离线成本指纹，拒绝分段 SWITCH 的表；默认关闭时继续
  兼容当前融合 SWITCH 的表。诊断快照会标出 joined 状态。

## 固定 batch 完整轮 wall

单位 µs。绝对时间采用 seed=911；降幅范围覆盖 seed=327/911。
每个 mode 都是同一 prefix、mask、tail、输出与输入顺序。

| B / neural 上限 | .item() µs | 融合 SWITCH µs | 合图 µs | 相比融合 SWITCH 降低 | 相比 .item() 降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 / 2 | 121.29 | 73.87 | 52.81 | 28.5%～32.2% | 56.5%～58.3% |
| 8 / 4 | 125.32 | 78.41 | 52.65 | 32.8%～36.7% | 58.0%～58.0% |
| 16 / 8 | 121.30 | 85.96 | 53.61 | 33.0%～37.6% | 54.3%～55.8% |
| 32 / 16 | 135.28 | 104.65 | 69.88 | 33.2%～33.3% | 48.3%～48.7% |

B=8 第二轮融合 SWITCH 78.41→合图 52.65 µs，原 `.item()` 125.32 µs；
B=32 为 104.65→69.88 µs，原 `.item()` 135.28 µs。
比较的是耗时降低，不能直接把百分比当成吞吐增加。

## 循环变长完整轮 wall

每个容量 B 固定 neural mask 上限 N，循环 raw_bs 为
`[B, 1, max(1,B-1), min(3,B), max(1,B/2)]`；实际 neural 行数随有效请求变化。
三条路径使用同一循环，合图即使 raw_bs 未变也每次更新参数，不缓存跳过更新。
所有循环值均在计时前预热，排除分段基线首次遇到该标量特化时的 Triton 编译。

| B / neural 上限 | .item() µs | 融合 SWITCH µs | 合图 µs | 相比融合 SWITCH 降低 | 相比 .item() 降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 / 2 | 115.34 | 70.64 | 49.22 | 30.3%～35.7% | 57.3%～59.8% |
| 8 / 4 | 116.18 | 73.67 | 49.67 | 32.6%～34.1% | 56.4%～57.2% |
| 16 / 8 | 127.11 | 94.35 | 57.03 | 33.5%～39.6% | 54.6%～55.1% |
| 32 / 16 | 145.20 | 108.65 | 71.79 | 33.8%～33.9% | 50.5%～50.6% |

两次测量的方向一致。仍受 WSL/host 噪声影响，不外推真实服务器负载。

## Kernel / API 表

B=32、neural=16，5 轮 trace 归一化：

| 项目 | 原 .item() | 分段融合 SWITCH | 合图 SWITCH |
| --- | ---: | ---: | ---: |
| 计算 kernel / 轮 | 6 | 7 | 7 |
| 图内 batch memset / 轮 | 0 | 0 | 1 |
| cudaGraphLaunch / 轮 | 2 | 2 | 1 |
| 独立行压缩 kernel 的 host 提交 | 1 | 1 | 0（已在图内） |
| cudaGraphExecMemsetNodeSetParams / 轮 | 0 | 0 | 1 |
| aten::item / 轮 | 1 | 0 | 0 |

合图没有进一步削减计算 kernel，还增加了图内写入节点。收益主要来自 host
提交减少：无 host 提交间隙的 GPU event 整轮，B32 为 28.54→26.94 µs、
28.67→27.01 µs，仅约 1.6 µs；完整 wall 则省约 35 µs。不能把这两个
测量口径的差值精确分配给某一个 CUDA API。

Trace：[原 .item()](seed-911/b32-n16-host.trace.json.gz)、
[融合 SWITCH](seed-911/b32-n16-fused.trace.json.gz)、
[合图 SWITCH](seed-911/b32-n16-joined.trace.json.gz)。
统一分析器原始三表见 [triage.txt](triage.txt)。

## 重叠/空闲表

| 观测 | 证据 | 边界 |
| --- | --- | --- |
| 主图一次启动 | 5 轮 cudaGraphLaunch 从 10 降为 5 | 输入 API 更新仍为 5 次，计入 wall |
| 没有中间 host scalar 读回 | fused/joined 的 aten::item 均为 0 | 最终 target verify 不在探针范围 |
| 变长 launch 保留各自参数 | 连续排队、GPU 延迟后读取累计行计数和 histogram | 不在每个 replay 后同步来掩盖竞态 |
| 不能据此推断模型/NCCL overlap | 单卡合成 workload、单 trace，无 mapping/formal 双跑 | 完整模型/TP2 待测 |

## 融合表

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| prefix + 行压缩 + SWITCH 同一主图 | Confirmed | 一次 graph launch，依赖按顺序连接 |
| setter 统计与 batch store | 沿用前轮融合 | 每步精确计数，没有低频抽样 |
| 模型 GEMM/attention kernel 融合 | 未改动 | 小 GEMM 与其他路径相同 |

## 正确性

| 组别 | 通过 | 证据 |
| --- | ---: | --- |
| CUDA compact / scan / joined 模型路径 | 330 | [日志](cuda.log)、[JUnit](cuda.xml) |
| 组件 / 动态 verify / recurrence / queued raw_bs | 370 | [最终日志](components.log)、[JUnit](components.xml) |
| 最终 joined 模型专项 | 20 | [日志](joined-final.log)，是上面 330 的子集 |
| 首次 queued native 专项 | 5 | [日志](native.log)，是上面 370 的子集 |

共 **700 个不同用例通过，0 失败、0 跳过**。新增 20 项 joined 模型路径
覆盖 triton、dsv4、EAGLE3、hot-token EAGLE3、STANDALONE、B=1/3/8、
source caps、零 tail 和 padding；模型计算合成，原生 metadata/worker 保留。
queued 专项 B=1/3/8/32/257 在 GPU 延迟后，连续排队 18 次不同 raw_bs、
中间无 host fence，检查每行执行累计、prefix 恰好一次、histogram 和 mixed。
容量 257 还覆盖跨 block 与大于单字节的 batch 值。

独立探针每次测量前，对每条路径校验全部 tail 大小、前/后行排列和缩批，
每个 mode 每轮 290 次，两轮共 1,740 次输出校验，两个 GPU dispatcher 的
统计也与独立参考一致。固定和变长性能测量在这些校验后进行。

本地 CPU 子集 119 通过、13 GPU 项跳过（GPU 项已在 4060 实跑）。
Ruff、wheel 构建和 CUDA 源码打包检查通过。700 项包含最终 prefix pool
所有权和诊断字段更新后的组件复测；首次组件也通过，日志另保留。
TP2 测试已扩展 joined case，但工作站只有一张卡，未实测 collective。

## 环境、方法与证据

- `ai-workstation` / RTX 4060 Laptop 8 GiB / WSL2，驱动 592.82；
  Python 3.12.3、Torch 2.14.1+cu130、CUDA 13.0、Triton 3.8.0。
- [.item 基线](../conditional-switch-4060-20261019/README.md) 源自 `8f9d113455`；
  [融合 SWITCH 基线](../conditional-switch-fused-4060-20261019/README.md) 为
  `0f6240e1ba`。仅重定向旧 row-kernel import 的模块名，以便同进程加载。
  共享 prefix 重捕获为 keep_graph=True，执行与输入不变。
- 100 次交错预热（变长时覆盖所有循环值）；15 block×100 iteration，
  每个 iteration 随机排列三条路径，逐次记录包含末尾同步的 wall，取 block
  均值的中位数。seed=327/911。每个 mode 每个 case/负载每轮 1,500 次。
- GPU event 记录 fused/joined 的整轮（含 prefix/scan/输入写入），start
  event 前排入 2M-cycle sleep，31 组随机样本；不是此前 dispatch-only 口径。
  profiler 与无 profiler 计时分开。
- [测量脚本](compare_rounds.py)、[启动脚本](run-benchmark.sh)、
  [回归脚本](run-regression.sh)、[seed327](seed-327/comparison.json)、
  [seed911](seed-911/comparison.json)。
- [metadata.json](metadata.json) 的 12 个新源码/测试、8 个基线源码和辅助
  脚本哈希逐项核对。补齐循环值预热前的初次数据见
  [prewarm-initial.tar.gz](prewarm-initial.tar.gz)，不作为本报告最终表格。
- [设备结束状态](post-test-gpu.txt)：0 MiB、0%，无遗留测试负载。

保持 joined 默认关闭。实际部署开启后需重跑该配置的离线成本表；真实模型的
完整 draft+verify 或请求端到端收益仍需要同一模型和负载的 A/B。
