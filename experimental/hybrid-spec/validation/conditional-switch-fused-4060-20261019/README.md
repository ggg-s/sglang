# SWITCH 统计与 batch 写入融合：RTX 4060

**两项融合完成，整轮小 GEMM 微测有稳定改善。** B=8/N=4 相比融合前
SWITCH 降低 **23.8%～25.8%**，相比原 `.item()` 路径降低 **33.3%～35.0%**；
B=32/N=16 分别降低 **19.0%～22.8%** 和 **21.6%～23.0%**。
范围来自两次不同随机种子的交错测试，不是置信区间。

这里的「整轮」是与此前相同的 **prefix + 行压缩 + neural 小 GEMM tail +
分支统计 + 末尾同步**，不包含真实 target verify、模型权重、NCCL 或请求
调度。不是完整模型服务的端到端吞吐/延迟结论。

## 改动

- 把 histogram/mixed 的 6 个独立 GPU kernel 融进已有的 SWITCH handle
  kernel，用 int64 存储的 64-bit 原子加法逐步准确累计；没有降低统计频率。
  统计在分发 setter 中完成，每次 replay 一次，零 tail 也计入 histogram。
- 行压缩 kernel 在写 neural count 的同时写实际 `raw_bs`，SWITCH 的
  生产 replay 不再调用独立 `live_bs.fill_`。直接调用 dispatcher 而未
  经过行压缩时仍保留 fill，防止变长时读取旧 batch。
- prefix/compaction/dispatcher 未合图，IF 和普通 split 行为保持原样。
  现有 `conditional_draft_graph_dispatch="switch"` 直接采用融合实现，
  没有新增开关。图和 private pool 的所有权/释放顺序保持原样。
- 离线成本指纹加入 SWITCH revision=2，拒绝融合前 SWITCH 的耗时表；
  IF 和默认关闭的旧表兼容性保留。启用新实现需要重跑 SWITCH 离线表。

## 整轮 wall 对照

单位 µs。下表为 seed=911 的 block-mean 中位数；百分比是耗时降低，
不是吞吐增加。最后两列列出两次测量的范围。

| B / N | 原 .item() | 原 SWITCH | 融合 SWITCH | 相比原 SWITCH 降低 | 相比 .item() 降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 0 | 82.80 | 87.09 | 57.31 | 29.6%～34.2% | 30.8%～33.7% |
| 1 / 1 | 117.33 | 98.53 | 71.75 | 25.7%～27.2% | 38.8%～39.2% |
| 3 / 1 | 119.81 | 101.72 | 73.50 | 26.4%～27.7% | 38.7%～39.7% |
| 4 / 2 | 129.70 | 110.07 | 78.55 | 26.3%～28.6% | 37.5%～39.4% |
| 8 / 4 | 126.79 | 111.03 | 82.44 | 23.8%～25.8% | 33.3%～35.0% |
| 16 / 8 | 136.85 | 118.40 | 94.04 | 19.9%～20.6% | 26.5%～31.3% |
| 32 / 16 | 136.22 | 131.88 | 106.83 | 19.0%～22.8% | 21.6%～23.0% |

例如 B=8 第二次由原 SWITCH 的 111.03 µs 降至 82.44 µs，省 28.59 µs；
原 `.item()` 为 126.79 µs，省 44.35 µs。B=32 第二次分别省
25.05 µs、29.39 µs。未改变 tail 实际行数或输出。

## 测量公平性与复现

环境沿用前轮：`ai-workstation`，RTX 4060 Laptop 8 GiB / WSL2，驱动
592.82，Python 3.12.3，Torch 2.14.1+cu130，CUDA 13.0，Triton 3.8.0。
目录 `/workspace/conditional-switch-fused-20261019/repo` 独立于此前基线。

- 原 `.item()` 与原 SWITCH 使用 `8f9d113455` 的 compact dispatcher、
  row kernel、conditional wrapper 和 CUDA adapter。为了同一进程加载，
  只将旧 compact 模块的 row kernel import 重定向到另一个模块名；
  没有重写旧路径的分发/计数逻辑，CUDA 源码也未改。
- 三条路径共享 prefix、mask、row buffer、count、GEMM tail 计算及输出，
  分别持有需要的 Graph/private pool；用同一输入顺序执行。
- 每个 case 先校验所有模式的变长、前/后 neural 行、padding 和分支统计，
  每个模式每轮测量前共 290 次校验；两次共 **1,740 次输出校验**。
- 100 次交错预热；15 个 block × 每个 100 次，每次都随机排列三种模式，
  逐个记录完整 wall（包含最终 `cuda.synchronize()`），取 block 均值的
  中位数。每条路径每个 case 每次测量 1,500 轮，seed 为 327 和 911。
  比前轮按模式整块测量的方式更能平衡 host/WSL 噪声，不能把跨轮绝对值
  差异全当成优化收益。
- 无 profiler 的计时与 profiler 分开。GPU event 只测 dispatcher：
  start event 前排入 2M-cycle sleep，让 CPU 提交先完成，31 组随机样本；
  排除公共 prefix 和行压缩。旧 SWITCH 包含 fill，融合路径使用已写 batch。

脚本：[compare_rounds.py](compare_rounds.py)、[run-benchmark.sh](run-benchmark.sh)。
原始结果：[seed-327](seed-327/comparison.json)、[seed-911](seed-911/comparison.json)。
共享辅助来自前轮 [ablate_dispatch.py](../conditional-graph-overhead-4060-20261019/ablate_dispatch.py)
和 [dispatch_probe.py](../conditional-graph-4060-20261019/dispatch_probe.py)。
原/新源码与辅助脚本哈希保存在 [metadata.json](metadata.json)，已与本地
当前源码和 Git `8f9d113455` 基线逐项核对。

首次探针校验发现继承的 Case helper 只给固定计时大小准备普通 tail Graph，
遍历其他大小时原 `.item()` 对照缺 Graph。诊断脚本补齐所有大小后两次
测量通过；没有修改生产代码绕过验证。[初次日志](probe-initial.log) 保留。

## Kernel 表

非零 tail 的完整轮，profiler 5 轮归一化：

| 类别 | 原 .item() | 原 SWITCH | 融合 SWITCH |
| --- | ---: | ---: | ---: |
| prefix | 1 | 1 | 1 |
| 行压缩 | 1 | 1 | 1（同时写 live_bs） |
| GEMM tail：gather/GEMM/cast/scatter | 4 | 4 | 4 |
| SWITCH setter | 0 | 1 | 1（同时计数） |
| 独立 GPU 统计 | 0 | 6 | 0 |
| 独立 live_bs.fill_ | 0 | 1 | 0 |
| **总计** | **6** | **14** | **7** |

零 tail 则为 2/10/3 个 kernel。普通路径的统计由 CPU 完成，属于其完整
wall 测量；没有为了对照而删除原有 CPU counters。

## 重叠/空闲表

| 观测 | 证据 | 解释与限制 |
| --- | --- | --- |
| 融合前后均无中间 scalar 读回 | 各 5 轮 aten::item 为 0，原普通路径为 5 | 没有用新的同步换取收益 |
| 图启动未变 | 非零 tail 各 5 轮 cudaGraphLaunch 均为 10 | prefix 尚未合图 |
| 设备调度段缩短 | seed911 B8：29.15→18.46 µs；B32：33.63→22.75 µs | 不含 prefix/scan，不等同完整 wall 降幅 |
| 整轮 wall 包含 host 提交 | 上表两种随机种子的同进程交错测量 | 不用 kernel busy 时间直接解释全部 wall 差值 |

## 融合表

| 融合 | 状态 | 验证 |
| --- | --- | --- |
| SWITCH setter + histogram/mixed | Confirmed | 6 个独立 kernel 消失，逐步计数与独立参考相同 |
| 行压缩 + live batch store | Confirmed | 独立 fill 消失，poison 旧 batch 后正确覆盖，包含 raw_bs=0 的扫描测试 |
| prefix/dispatcher 合图 | 未实现 | 启动数保持不变 |

原始 trace：[普通](seed-911/b32-n16-host.trace.json.gz)、
[原 SWITCH](seed-911/b32-n16-switch.trace.json.gz)、
[融合 SWITCH](seed-911/b32-n16-fused.trace.json.gz)。
统一分析器三表见 [triage.txt](triage.txt)，单 trace 的 Python unresolved
不用于强行做模型/重叠归因；融合结论由源码、算子消失和语义校验共同确认。

## 正确性与范围

| 组别 | 结果 | 证据 |
| --- | ---: | --- |
| CUDA compact / stable scan 回归 | 310 通过 | [日志](cuda.log)、[JUnit](cuda.xml) |
| 组件、动态 verify、recurrence、SWITCH | 363 通过 | [日志](components.log)、[JUnit](components.xml) |

合计 **673 项通过，0 失败、0 跳过**。新增扫描覆盖是否写 batch、raw_bs=0、
padding、跨 block 稳定行序；SWITCH 覆盖 direct fill 和生产 prepared batch
两种入口，并故意 poison 旧 batch，检查 histogram/mixed 不受容量 B 干扰。
成本表测试确认旧 SWITCH revision 不会被新路径误用。
[run-regression.sh](run-regression.sh) 可复现。

本地 CPU 子集 117 通过、8 GPU 项跳过；GPU 项已在 4060 实跑。
修改的生产 Python、新测试和诊断脚本通过 Ruff，wheel 构建及 CUDA 源码
打包验证通过。最终 [设备状态](post-test-gpu.txt) 为 0 MiB、0%，无遗留测试负载。

TP2 collective 和完整权重服务尚未实测。需要在同一模型、接受率和负载下
比较整轮 draft+verify 或请求延迟，才能得出服务端到端收益。
