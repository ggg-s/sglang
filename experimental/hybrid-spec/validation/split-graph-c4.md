# 并发 4：按请求压缩 MTP 尾段的 A100 实验

已实现混合批逐请求跳过：SA 命中行不执行 MTP 尾段。但本次并发 4 对照中，
相对完整 MTP+SA，裁剪版吞吐下降约 **2.4%**、平均 TPOT 增加约 **2.7%**，
未体现额外性能收益。相对原生的约 1% 吞吐涨幅小于重复波动，不能视为稳定加速。
正确性评测按用户要求不运行；下面的 CUDA 回归测试只验证图调度与数据搬运契约。

## 实现

配置 `device-qwen35-mtp-split-c4.json`，K=8、最小历史匹配长度 4、neural 前缀检查 2，
每请求历史上限 16384，最多维护 4 个请求。第一候选来自前一轮原生 draft-extend，
所以整批只需一次 MTP inner forward 就能检查前两个 token。

前缀 Graph 生成全部 SA 候选并计算两个 neural token 的一致性。CPU 每批读取一次最终
选择 mask，使用 runner 的原始请求数排除 padding。未通过检查的行进入紧凑尾段，
通过检查的行保留 SA 候选，不进入尾段模型计算。

| 活跃请求 4 个时的 SA 命中数 | 尾段模型行数 | 后续 inner forward 次数 |
| ---: | ---: | ---: |
| 0 | 4 | 6 |
| 1 | 3 | 6 |
| 2 | 2 | 6 |
| 3 | 1 | 6 |
| 4 | 0 | 0 |

尾段为每个大小分别捕获 CUDA Graph，按动态行号收集 hidden state、第二候选、request
pool index、seq_lens、positions、mRoPE positions 和逐步 KV 写入地址。每种大小独立
维护注意力元数据；输出链写回原始请求行。target 仍验证 8 个草稿加一个 bonus，
原生 draft-extend 与已接受状态提交路径保持运行。此实现不使用独立 SA CUDA stream。

前缀、各尾段均保留私有 Graph pool，并显式持有 capture 外分配的暂存张量。最初模型
预跑遇到 CUDA 越界，补齐缓冲区持有后预跑完成；toy 计算不能替代真实模型的存储生命周期
验证。另修复 EAGLE runner 回放前暂时将 3 请求补齐为 4 行的情况：compact dispatch
必须读取 `runner.raw_bs`，不能读取此时的 `forward_batch.batch_size`。回归测试在恢复
旧表达式时得到 4 而预期为 3，修复后通过。

## 使用范围

```bash
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/device-qwen35-mtp-split-c4.json"
# 在原生 Qwen3.5 MTP 服务命令中设置：
# --max-running-requests 4 --cuda-graph-max-bs-decode 4
# --speculative-draft-attention-backend triton
```

TP=1、纯文本、greedy、topk=1、普通 KV 池，要求本分支 draft split Graph API v2。
最大 capture batch 不得超过 4。多请求路径目前仅支持 Triton draft attention，
不支持 KV canary、memory saver、DSA seed、随机采样和其他已被插件拒绝的组合。
`draft_graph_mode` 默认仍是 `full`；本轮实验不自动开启。
只改 Python Graph 调度，CUDA SA 状态布局仍为 v3，无需重新编译 CUDA 插件；
SGLang 和插件 Python 均需更新并重启。

## 对照协议

同一 NVIDIA A100 PCIe 40GB，TP=1，target FP16/gptq_marlin、MTP unquantized，
K=8、验证宽度 9、seed=42，overlap 和 chunked prefill 保持开启。所有三组统一使用
Triton draft attention，target full attention 为 FlashInfer，GDN 为 Triton。
这一 draft attention 选择不同于此前默认后端的 C1 实验，不能混合两轮绝对耗时。

使用官方 `python -m sglang.benchmark.serving`：相同 50 条 `glaiveai/code_edits_sample`
提示，沿用 sa-spec 原始 completion 模板，输出上限 2048、temperature=0、保留 EOS，
并发 4、预热 4 个请求、测量前 flush cache，每种模式完整重复两次。
原始结果保留每请求输出与流式延时，TPOT 直接使用官方统计。

初版 padding 调度实验保存在 `split-prototype-padded`，正式汇总只使用修正后的 `split`。
原生和 full 不走 compact dispatch，其测量路径不受此修正影响。每轮结果分别保留，
汇总均值不能证明所有负载稳定加速。接受长度由官方读取服务器累计统计，包含预热等
历史请求，不作为精确的单轮接受长度对照；分支计数也明确包含预热。

实验目录：`/root/rivermind-data/validation/split-graph-c4-20261016`。
数据、启动命令、源码快照与最终验证日志随原始证据保留。

## 正式性能结果

下面是两轮独立运行的算术均值；每轮 50 个请求均完成，无请求错误。

| 模式 | 输出 token/s 均值 | 平均 TPOT ms |
| --- | ---: | ---: |
| 原生 MTP | 837.59 | 4.4814 |
| MTP+SA，完整 Graph | 867.38 | 4.2909 |
| MTP+SA，逐请求裁剪尾段 | 846.26 | 4.4057 |

| 模式 | 第 1 轮 token/s / TPOT ms | 第 2 轮 token/s / TPOT ms |
| --- | ---: | ---: |
| native | 807.24 / 4.5888 | 867.93 / 4.3740 |
| full | 857.23 / 4.3050 | 877.53 / 4.2768 |
| split | 855.21 / 4.4605 | 837.30 / 4.3510 |

裁剪版相对原生均值吞吐 +1.04%、TPOT −1.69%；相对 full 吞吐 −2.44%、TPOT +2.68%。
原生两轮吞吐相差约 7.5%，因此不能把约 1% 的差异解释为稳定收益。
所有六轮的输入长度向量逐项一致，原始输出长度与生成文本保存在官方 JSONL 中；
本轮没有验证输出等价或任务准确率。

正式 split 两轮（均包含官方预热）共有 7529 个 decode 批次、28740 个活跃请求行，
其中 11442 行跳过尾段，比例 39.81%；284 批整批跳过尾段，比例 3.77%，也包含收尾小批。
5830 批为混合来源。尾段大小 0/1/2/3/4 的回放次数为 284/1577/2293/2365/1010。
这证明逐请求裁剪确实生效，同时说明约 40% 的行命中不等于约 40% 的批次省掉整段 MTP。

## Trace：为何少算请求行仍没有额外收益

正式 trace 观察到 8 个完整 decode 轮次：6 批运行前缀和尾段，2 批仅运行前缀。
前缀 topk finalize 的 GPU grid 为 4 行，尾段分别出现 1/2/4 行，且每段只有 6 次
后续模型 forward。相应逐轮 row-forward 数为 28、10、10、16、10、16、4、4，
与 4 行完整 MTP 的每轮 28 个 row-forward 不同。3 行尾段在正式计数中也实际运行，
仅未落入此短 trace 窗口。

| 尾段行数 | 样本数 | 尾段 Graph 跨度 ms | 六次 LM head 合计 ms |
| ---: | ---: | ---: | ---: |
| 1 | 3 | 5.5373 | 4.2174 |
| 2 | 2 | 5.7490 | 4.2482 |
| 4 | 1 | 6.1360 | 4.2896 |

以上为带 profiler 的 GPU 时间，不用于重算正式吞吐或 TPOT。短窗口的全命中比例
也不能替代正式负载的 3.77%。LM head 的定位沿用此前 eager 形状映射，结合当前
GEMM 名称与 grid=[3880,1,1]；不能把同名的小矩阵 GEMM 全部计作 LM head。

| Kernel / 路径归因 | 已验证的证据 | 含义 |
| --- | --- | --- |
| MTP LM head `[M,2048] × [2048,248320]`，FP16 | M=1 与 M=4 的六次 head 合计 4.2174 / 4.2896 ms | 缩行 75% 没有同比减少主要矩阵耗时 |
| `_draft_topk1_finalize_kernel` | prefix grid=4、tail grid=1/2/4，每个 tail 6 次 | 真正减小模型 batch，未仅做输出 mask |

| 可重叠项 | 当前观察 | 边界 |
| --- | --- | --- |
| CPU 前缀决定与尾段提交 | 两段之间有暴露间隙；8 轮均摊约 0.505 ms | profiler 有扰动，不等于纯同步成本 |
| SA 查询与 neural draft | 当前仍为主流查询，未增加独立 stream | 隐藏查询也不会消除剩余请求的 LM head 权重读取 |

| 融合模式 | 当前实现 | 本轮处理 |
| --- | --- | --- |
| argmax + position advance + draft write | 复用原生 topk1 finalize kernel | 未重写 CUDA kernel |
| 行 gather、元数据生成、scatter | 在紧凑尾段 Graph 中执行 | CPU 只决定行集合；未验证额外融合收益 |

1 到 4 行都仍需读相同的 LM head 权重，当前 kernel 的 tile 网格也相同；减少行数
主要减少部分算术，并未省掉六次大权重读取。只有整批命中才真正取消六次 head。
这是基于形状、grid 和耗时的解释，未用硬件带宽计数器直接测量 DRAM 流量。
再加上 CPU 决定、gather/scatter 和重新生成元数据的开销，本次 full 对照表现更好。
不能由此推断所有模型、后端或并发数都没有裁剪收益。

启动日志中 split draft Graph 分配约 0.41 GB、full 约 0.03 GB，多约 0.38 GB。
这是当前私有池与独立元数据的代价，不是整个服务显存，也不是 SA 历史状态增长。

## 验证与证据

本地核心/真实 draft_forward 递推函数体 141 passed；A100 CPU/native/Gloo
148 passed、1 skipped（单卡环境不执行双卡 NCCL），最终 CUDA 15 passed，
覆盖所有混合 mask、请求行重排、非同一 request pool 顺序、padding、
完整 SA 链以及连续 skip/fallback 时位置与模型输入状态的恢复。

正式数据见 [split-graph-c4-a100.json](split-graph-c4-a100.json)。保留的原始证据包括
官方客户端命令与输出、服务日志、profile、数据哈希、实际测试源码与最终测试日志。
通用 profiler analyzer 的名称推测（例如误将 GDN sigmoid 归入 MoE）不作为本结论依据；
上述归因来自模型路径、topk grid 和已核实的 LM head 形状映射。

模型基准与最终回归均已结束，测试服务已停止，GPU 显存已释放。

原始证据归档：`/root/rivermind-data/validation/split-graph-c4-evidence-20261016.zip`，
SHA256：`ab5c7e4278bcab1a252ab452b287f73d421fe2914ed51fe239a5af9143eb4231`。本机下载副本已核验相同哈希，模型权重与数据目录软链未纳入归档。
