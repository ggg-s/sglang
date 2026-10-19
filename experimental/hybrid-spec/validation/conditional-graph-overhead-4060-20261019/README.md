# Conditional graph 开销拆解：RTX 4060

**当前实现的主要新增成本是多层 IF 分发，不是 `.item()` 没有消掉。**
B=32、16 行进入 tail 时，GPU dispatch 从已知形状的直接 replay 的
12.93 µs 增至 78.56 µs；其中 IF 树增加约 55.94 µs，统计增加约
7.71 µs，`live_bs.fill_` 增加约 1.98 µs。IF 树约占新增 GPU 成本的 85%。
这是合成小 GEMM 的调度实验，不是模型服务吞吐测试。

## 环境与口径

- `ai-workstation`，RTX 4060 Laptop 8 GiB / WSL2，驱动 592.82；
  Python 3.12.3，Torch 2.14.1+cu130，CUDA 13.0，Triton 3.8.0。
- 生产源码仍为 `55444165099a4248e6ea534c0c51171ff06bcb85`，未改动。
  使用上一轮的 [dispatch_probe.py](../conditional-graph-4060-20261019/dispatch_probe.py)
  创建真实生产 dispatcher / 行压缩，tail 为 gather、FP16 256×256 GEMM、
  cast、scatter。各对照使用相同 tail 计算与缓冲区。
- 每个对照先校验输出。`direct`、`single_if` 和预计算 predicate 对照提前
  知道选中的形状，只是诊断下界，不能直接替换动态分发。
- wall：50 次交错预热，7 组随机排列 × 每组 100 轮，每轮末尾同步。
  host/WSL 提交噪声明显，细项归因采用下面的 GPU event 测量。
- GPU dispatch：31 组随机排列，start event 前排入 2M-cycle GPU sleep，
  给 CPU 留出提交时间；记录 start→dispatch→end，结束同步。
  **不包含公共 prefix 和行压缩，包含 production 的 `live_bs.fill_`。**
- profiler 每个对照记录 5 轮完整路径，用于算子与 API 次数；时间表来自
  不开 profiler 的 event 测量，不用 profiler 的 kernel 时长直接相减。
- [metadata.json](metadata.json) 保存源码/脚本哈希与测量口径；
  [运行脚本](run.sh)、[日志](run.log)、[原始结果](ablation.json) 可复核。
  日志中的 empty graph warning 来自 N=0 的直接 replay 诊断下界。

## 逐项消融

单位 µs，GPU dispatch 中位数。N 是实际 neural tail 行数，B 是捕图容量。

| B / N | 已知 tail 直接 replay | 单个 IF | 完整 IF 树 | IF 树 + 统计 | 生产路径：再加 live_bs.fill_ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 0 | 1.02 | 8.19 | 8.19 | 17.41 | 19.46 |
| 1 / 1 | 8.16 | 15.68 | 15.68 | 22.46 | 24.42 |
| 4 / 2 | 9.15 | 16.64 | 35.07 | 42.59 | 44.58 |
| 16 / 8 | 10.69 | 18.11 | 56.06 | 63.49 | 65.50 |
| 32 / 16 | 12.93 | 20.35 | 68.86 | 76.58 | 78.56 |

`tree_stats` 直接 replay 生产条件图，仅省略外部 fill；`production` 使用完整
生产 replay。`tree` 重建相同 IF 结构并删除统计，`single_if` 只留下一个
`count>0` 条件，其分支直接运行已知的目标 tail。N=0 的直接图为空图，
1.02 µs 是参考下界，不代表普通路径会启动这个空图。

### 1. 一层判断实际有两个 IF

生产代码在每个二分点依次捕获 `count <= middle` 和 `count > middle`
两个 IF，根部另有 `count > 0`。只有选中的 body 会运行，但两个兄弟 IF
都需要计算 predicate、设置各自的 condition handle 并完成条件节点判断。
对 B 为二次幂且 N>0 的情况，运行时经过的 IF 数为 `1 + 2*log2(B)`。
B=32 时是 **11 个 IF**，不是 5 个，也不是执行全部 32 个 tail。

每个 IF 在 trace 中都有一个 scalar compare kernel 和一个
`set_conditional_handle_kernel`。所以仅控制逻辑就新增 **22 个 kernel**，
而这些微小 kernel 之间还有依赖和条件图执行转换成本。

### 2. 去掉比较后，大部分 IF 成本仍然存在

第二轮只测 B=32 / N=16，将 bool predicate 在图外预先算好，保留相同
handle 设置与 IF 结构。结果见 [precomputed.json](precomputed/precomputed.json)
及 [脚本](ablate_precomputed.py)。这是固定选择的诊断对照，不能用于动态 mask。

| 对照 | GPU dispatch µs | 完整轮 kernel 数 |
| --- | ---: | ---: |
| 直接 replay | 13.31 | 6 |
| 单 IF | 20.48 | 8 |
| 单 IF，预计算 predicate | 19.46 | 7 |
| 完整 IF 树，无统计 | 66.56 | 28 |
| 完整 IF 树，预计算 predicate，无统计 | 56.32 | 17 |

移除 11 次比较省约 **10.24 µs**，但仍比直接 replay 多约 **43.01 µs**。
因此仅融合点算子不能解决主要成本；需要减少 handle / 条件节点数量。
当前实验没有进一步把 handle kernel 与原生 CUDA 条件节点转换成本分离，
不能把这 43 µs 全部称为 CUDA 调度器开销。

### 3. 统计与启动也是成本，但不是最大项

GPU histogram 和 mixed 计数每轮产生 6 个 kernel，包括 cast、scatter_add、
两次比较、bool AND、add。B=32 时增加约 7.71 µs；`live_bs.fill_` 再增加
一个 kernel，约 1.98 µs。即使统计不读回 CPU，更新本身也有 GPU 成本。

当前 prefix、行压缩、conditional graph 仍独立提交。N>0 时普通路径与
条件路径均为 **2 次 cudaGraphLaunch**，并没有减少 graph launch 次数。
N=0 时普通路径不启动 tail 图，条件路径仍需启动 dispatcher，
graph launch 由 1 次变成 2 次，完整轮 kernel 由 2 个变成 11 个。

## Kernel 表

B=32 / N=16，完整轮；次数由 5 轮 trace 归一化。
trace：[普通路径](b32-n16-host.trace.json.gz)、
[IF 树无统计](b32-n16-tree.trace.json.gz)、
[生产条件路径](b32-n16-production.trace.json.gz)。

| 类别 | 普通路径 kernel 数/轮 | 生产条件路径 kernel 数/轮 | 归属 |
| --- | ---: | ---: | --- |
| prefix、行压缩、tail 计算 | 6 | 6 | 公共工作 |
| IF predicate 比较 | 0 | 11 | 条件分发 |
| condition handle 设置 | 0 | 11 | 条件分发 |
| histogram / mixed 计数 | 0 | 6 | GPU 统计 |
| live_bs.fill_ | 0 | 1 | replay 前元数据更新 |
| **合计** | **6** | **35** | `.item()` 每轮从 1 次降至 0 次 |

统一分析器的原始三表保存在 [triage.txt](triage.txt)。该单 trace 未做
mapping/formal 双跑，Python 归属部分 unresolved；以上归属依靠显式消融
和源码交叉确认。生产 trace 中比较 kernel（包含统计的一次比较）和 handle
kernel 分别占 kernel busy time 的 29.0%、26.3%。不要把这些 busy 占比
等同于无 profiler 的 dispatch 时间占比，也不能将两者差值全归因于图调度器。

## 重叠/空闲表

| 观测 | 证据 | 解释与边界 |
| --- | --- | --- |
| 中间 host scalar 同步确实消失 | 5 轮 aten::item：普通 5，条件 0 | 不是隐藏的同一个 `.item()` 导致回退 |
| 条件节点串行控制链增加 | 无 host 提交空隙的 event 测量；预计算 predicate 对照 | 有设备控制流成本，未独立分离原生节点转换与 setter 成本 |
| 非零 tail 的 graph launch 未减少 | 两条路径均为 10 次 / 5 轮 | prefix 与 tail 尚未合图 |
| 无法证明具体模型 overlap 收益 | 合成 TP1、每轮末尾同步，无双 trace 映射 | 不推断真实模型/NCCL 的流水线重叠或吞吐 |

## 融合表

| 候选 | 证据 | 收益范围与限制 |
| --- | --- | --- |
| 融合/降低统计频率 | 6 个统计 kernel，消融差约 7.71 µs | 尚未实现，低频必须保持计数语义清楚 |
| 融合 predicate 生成 | 预计算对照省约 10.24 µs | 仅去比较仍有约 43 µs 的 handle/IF 成本 |
| prefix、行压缩与 dispatch 合图 | 仍有 2 次 graph launch | 尚未测量其收益；不能消除 IF 树自身成本 |

没有已验证的新模型 kernel 融合；统一分析器也未匹配到有源码证据的融合模式。
优先级最高的是**减少条件节点/分支档位**，其次才是融合零碎算子。
可以评估原生多路分发或较少的 token/batch 档位；需要适配底层绑定并验证
padding、metadata 与 TP collective 一致性，当前 IF 接口不是直接替换即可。

## 为什么去掉 .item() 仍可能更慢

`.item()` 的 host 时间包含等待上游 GPU 完成，移除同步不会删除 prefix、
GEMM 或 NCCL 的工作。实际收益是少一次 CPU 等待/往返、让后续提交更早；
当前实现同时引入上述串行 GPU 控制流与统计，净收益取决于两者的大小。

此前 TP2 真实模型 trace 中约 2 ms 的 `.item()` 区间不能直接当成这里可
节省 2 ms 的证据；模型、设备、通信和测量范围均不同。本实验也不能否定
该真实负载上可能的收益，应在同一模型下做开关 A/B 并比较关键路径。

本轮完整 wall 中位数（普通 / conditional，µs）：B1N0 84.04 / 79.01，
B1N1 117.57 / 87.44，B4N2 120.87 / 117.39，B16N8 143.12 / 164.09，
B32N16 123.47 / 198.30。wall 受 WSL/host 噪声明显影响，不能拿相邻
模式的差值做精细归因。

上一轮 B1N0 的 conditional 174.95 µs **没有复现**，不能将那次短测解释
为 B=1 固有的大幅回退。本轮更可靠的结论是 IF 树增加设备成本、随捕图容量
增长，是否抵消 host 同步收益需要实际负载判断。

生产开关保持默认关闭。本轮只增加诊断脚本和证据，未改变 serving 代码，
未跑 TP2、完整权重服务或生成吞吐；结束后的设备状态见
[post-test-gpu.txt](post-test-gpu.txt)。
