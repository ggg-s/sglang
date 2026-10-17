# RTX 6000D：跨请求 SA 并行扫描对照

日期：2026-10-17，分支 `codex/hybrid-suffix-speculation`，基线
`2d53a3d2d2` 加本分支既有全局池实现。仅将串行跨槽位查询替换为并行查询。

## 吞吐结果

**并行扫描减少了查询开销，full 的配对吞吐中位收益为 +2.96%～+5.36%，
split 为 +3.14%～+6.19%。三轮所有配对均为正。** 下表四组都开启全局池。
吞吐列为三轮中位数；收益列为按轮配对比值的中位数，故不一定等于两列中位数的比值。

| 负载 | 并发 | Graph | 串行 tok/s | 并行 tok/s | 配对收益 [三轮范围] |
| --- | ---: | --- | ---: | ---: | ---: |
| repeat-warm | 1 | full | 338.03 | 355.73 | +5.36% [+5.24%, +5.40%] |
| repeat-warm | 1 | split | 397.62 | 422.24 | +6.19% [+5.81%, +6.71%] |
| repeat-warm | 4 | full | 785.46 | 811.80 | +3.35% [+3.32%, +3.37%] |
| repeat-warm | 4 | split | 820.13 | 849.31 | +3.56% [+3.37%, +4.55%] |
| unique-cold | 1 | full | 305.46 | 320.17 | +4.82% [+4.76%, +4.87%] |
| unique-cold | 1 | split | 344.22 | 361.72 | +5.37% [+5.04%, +5.41%] |
| unique-cold | 4 | full | 706.16 | 727.52 | +2.96% [+2.96%, +3.02%] |
| unique-cold | 4 | split | 706.92 | 735.48 | +3.14% [+2.70%, +4.64%] |

48 点共 3,840 请求、491,520 输出 token，全部成功且 KV prefix cache 命中为 0。
串行/并行共比较 1,920 对输出文本，**全部一致**；24 个配对点的 SA 反馈计数和
split 分支计数也逐项相同（不含耗时计数）。两版采用了相同候选路径和接受长度，
本轮收益可以归于扫描执行方式变化，无需通过改变命中标准换取吞吐。

split 两版合计覆盖 3,422 个 mixed batch、12,134 次整批尾段跳过。
并行 split 本身覆盖 1,711 个 mixed batch、6,067 次整批尾段跳过。
本轮没有测新的任务准确率，也不将这些有限输入上的文本一致性外推到所有负载。

SA 采用比例和接受长度保持不变：例如不同代码请求 full 并发 1 的平均接受长度
仍约 7.13。此次减少的是查询成本；跨请求候选质量和 gate 选择仍影响共享历史的净收益。

## 查询微基准结果

| 来源数 | batch | 不同历史：串行→并行 μs | 加速 | 长匹配：串行→并行 μs | 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1 | 2192.78 → 910.78 | 2.41× | 596.08 → 61.57 | 9.68× |
| 20 | 4 | 1972.66 → 888.56 | 2.22× | 608.73 → 62.32 | 9.77× |
| 64 | 1 | 7254.32 → 1086.27 | 6.68× | 1955.33 → 93.62 | 20.89× |
| 64 | 4 | 7111.92 → 1111.99 | 6.40× | 1997.00 → 96.72 | 20.65× |
| 257 | 1 | 29350.10 → 2278.84 | 12.88× | 7939.99 → 185.06 | 42.91× |
| 257 | 4 | 29538.18 → 2321.11 | 12.73× | 8101.92 → 189.92 | 42.66× |

上表为单个 shared-query kernel 的 profiler GPU 持续时间，构造负载和计时方法见下文；
总查询 Event 计时另见 `micro-summary.json`，不能将微基准倍率当作整机吞吐倍率。

## 实现与 split 兼容性

参考 TensorRT-LLM 固定提交
[`7ebcf4debcac29b3f6225cb4affff3ecf05514aa`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomatonKernels.cu)
的 `suffixAutomatonGlobalSearchKernel`，采用每个请求一个 CUDA block、各线程独立
查询来源、共享内存归约的结构。独立编写实现，无 TRT 运行时依赖。

线程数按来源数组长度向上取 2 的幂，再限制在 32–128；更大池由线程跨步扫描。
20 槽位配置使用 32 个线程。所有来源更新先完成，随后查询只读 SA，沿用 forward stream。
未增加 CPU 线程池、CUDA stream、设备回读或 Graph 内分配。

排序保持原串行语义：先过滤不符合完整度策略的来源，优先最长匹配，同长度优先本请求；
外部来源再比较未截断续写长度，完全同分按发布索引选择。发布顺序仍由 generation 排序。
超过一个 warp/block 的候选也显式比较发布索引，归约树不会改变同分结果。

跨请求兼容 `split`：查询位于 prefix Graph 中，后续 neural gate、紧凑尾段和 target verify
沿用原实现。当前 split 仍限定 device 历史、gate=2、K>2、TP=1、greedy、topk=1、
最多 4 个活跃请求、Triton draft attention 等已有支持范围。完成来源的数量不计入活跃并发。
full 仍执行全部 MTP；并行查询只减少 SA 扫描开销。

## 正确性与编译资源

RTX 6000D（SM120）完整包测试：**182 passed、1 skipped，76.60 s**，包含首次编译。
唯一跳过项需要双卡 NCCL；Gloo 1/2/4 rank 通过。CUDA 测试从 19 项增加至 26 项。

新增测试用独立 CPU 文本搜索 oracle 比较 20/64/257 个来源、两种 partial 策略、
随机历史、长匹配、代际失效、失效来源、padding、发布顺序变化和 Graph 重放；
另构造不同线程、同一线程跨步的完全同分候选，检查固定发布顺序。
已有 final-feedback、活跃行更新屏障、停止尾部、full/split/compact 回归全部通过。

`cuobjdump --dump-resource-usage`：新查询 44 registers/thread、3072 B shared memory、
0 B stack、0 B local；原串行查询 40 registers/thread、0 B shared。
布局、Python/C++ 接口和槽位容量均不变。精确源码差异见 `parallel-query.patch`。

## 微基准方法

每个槽位 384 token，外部上下文 128，K=8、min_match=4、H=1024。
20/64/257 槽位，batch=1/4，两个构造负载：随机不同历史；外部来源含完整共同后缀和续写。
随机 token 来自 32 词表，旨在检验扫描开销，不代表真实代码 token 分布。

串行与并行分别加载各自源码和编译缓存，逐项核对输入哈希、候选 token 和 mask。
12 组完全一致。每个 Graph 包含 8 次幂等查询，预热后 5 轮 CUDA Event 计时，
每轮重放 25 次；总查询时间包含 finalize、own-query 和 shared-query 三个 kernel。
另用 Torch Profiler 采集每组 40 次 shared-query kernel 的 GPU 持续时间，报告中位数。
profiler 与吞吐测量分开运行。完整 traces 在证据归档中。

## 真实模型方法

沿用前次同一 GPU UUID、Qwen3.6-35B-A3B-FP8 检查点、Torch 2.14.1+cu130、TP=1。
K=8、gate=2、min_match=4、partial=fallback、4 活跃槽位 + 16 完成槽位，H=4096，
跨请求上下文 128；greedy、ignore_eos=True、统一关闭 KV 前缀缓存。

使用仓库未改动的官方流式 `sglang.benchmark.serving` 客户端。每点 80 请求，
固定输出 128 token，并发 1/4，三轮随机排列场景顺序，seed=43/44/45。
模型加载、JIT、Graph capture 和通用预热不计入吞吐。

`unique-cold`：80 个不同真实代码编辑 prompt，每点 flush 后空池起步。
`repeat-warm`：其中 16 个 prompt 各重复 5 次，flush 后先实际生成 16 个请求建池；
建池时间单独记录，不计入主表。未使用数据集参考答案。
这两个场景分别检查低复用开销与可复用收益，没有重跑 repeat-cold。
四组全局池都开启，目的是隔离查询并行化，不是与关闭池作新一轮比较。

同日执行顺序：串行 full、并行 full、并行 split、串行 split。
相同 Graph 模式按轮配对，报告三轮比例中位数及最小/最大值。
这不是交替逐点 AB 测试；每点温度、功耗、时钟和服务计数保存在 diagnostics。
三个轮次不是统计显著性检验；固定 128-token 前缀也不是完整代码任务评分。

48 点的原始响应、请求延迟、官方结果、命令、配置、种子和源码清单已一并保存。
汇总见 `summary.json`、`summary.csv`；输出文本差异单独保存在 `output-parity.json`。
同输入的 kernel 候选一致性与真实并发服务的输出一致性分别核查，不将吞吐代替正确率。

## 归档与复现

完整原始证据：[本地归档](/private/tmp/sa-parallel-6000d-20261017-evidence.tar.gz)。远端持久副本：
`/root/rivermind-data/validation/sa-parallel-6000d-20261017-evidence.tar.gz`。源码、逐请求原始结果、24 组 profiler traces、
启动命令和参数、种子、日志均在归档内；不含模型权重。两版源码各 5,616 个记录文件
及归档内全部文件哈希已在本地再次核对。详情见 `archive.json`、`verification.json`。

归档 SHA-256：`47ae0f1a7d3beab2ab34a290aa6f1e017ce7e7178d236f3f1de3fbcc581d48b1`。临时服务已停止，最终显存 0 MiB，原远端 checkout 保持干净。

复现顺序：分别设置串行/并行源码 PYTHONPATH 和独立 TORCH_EXTENSIONS_DIR，
先运行 `microbenchmark.py`，再用 `run_comparison.py` 调用归档中的 `run_benchmark.py`，
最后运行 `analyze.py`。精确参数在每个 `*-driver-command.json` 和子目录 `plan.json`。
组合配置示例为 `configs/device-qwen35-mtp-split-global-pool.json`，已核对与实测 JSON 完全一致。
