# 混合投机解码：完整历史记录附录

> 整理日期：2026-10-19。入口：[测试总报告](TEST_RESULTS.md)；逐次表：[JSON 测量数据](TEST_RESULTS_RUNS.md)。
>
> 以下是本地现有报告的逐份快照，保留当时的环境、版本、局限和修正记录。**其中“当前”“服务仍运行”“batch≤4”等均描述原报告当时状态，不是本次整理后的状态或最新能力。** 本次未执行这些历史命令，未重新验证其中的外部链接或上游现状。最新口径及作废结果以总报告为准。
>
> 相对文件链接已转换为相对本附录的位置。所有源报告原样保留；正文内容仅对标题层级和相对链接作展示调整。

> 更新：新增 [DSV4 TP8 汇总专题](DSV4_PERFORMANCE_20261017_19.md) 已纳入总报告，覆盖真实模型 fixed/dynamic compact。下面 32 份保留的是此前历史快照；其中“尚无 DSV4 TP8 compact 结果”等旧说明不再描述最新测试状态。

## 目录

- [记录 1：acceptance-regression.md](#record-1)
- [记录 2：arctic-overlap-4060/README.md](#record-2)
- [记录 3：arctic-suffix-4060/README.md](#record-3)
- [记录 4：arctic-suffix.md](#record-4)
- [记录 5：compact-verify-4060/README.md](#record-5)
- [记录 6：draft-algorithms-split-20261017.md](#record-6)
- [记录 7：draft-algorithms-split-4060/README.md](#record-7)
- [记录 8：dsv4-lease-4060/README.md](#record-8)
- [记录 9：dsv4-lease-lifecycle-20261017.md](#record-9)
- [记录 10：dynamic-compact-ab-4090-20261018/README.md](#record-10)
- [记录 11：dynamic-compact-ab-4090-20261018/report.md](#record-11)
- [记录 12：dynamic-rows-gate-20261017.md](#record-12)
- [记录 13：dynamic-rows-gate-4060/README.md](#record-13)
- [记录 14：dynamic-verify-4060-20261018/README.md](#record-14)
- [记录 15：dynamic-verify-4090-20261018/README.md](#record-15)
- [记录 16：global-pool-6000d-throughput/report.md](#record-16)
- [记录 17：global-pool.md](#record-17)
- [记录 18：gpuhome-gate1-c16-profile-20261018/history-notes.md](#record-18)
- [记录 19：gpuhome-gate1-c16-profile-20261018/report.md](#record-19)
- [记录 20：gpuhome-gate1-concurrency-20261018/report.md](#record-20)
- [记录 21：model-pr-history-notes.md](#record-21)
- [记录 22：qwen3-8b-06b-compact-ab-4090-20261019/README.md](#record-22)
- [记录 23：qwen3-8b-06b-compact-ab-4090-20261019/report.md](#record-23)
- [记录 24：qwen3-native-k3-ar-4090-20261019/README.md](#record-24)
- [记录 25：sa-parallel-6000d-throughput/report-table.md](#record-25)
- [记录 26：sa-parallel-6000d-throughput/report.md](#record-26)
- [记录 27：split-graph-c4.md](#record-27)
- [记录 28：split-profile-6000d/report.md](#record-28)
- [记录 29：gpuhome-gate1-mixed-tail-20261018/report.txt](#record-29)
- [记录 30：gpuhome-gate1-profile-20261018/report.txt](#record-30)
- [记录 31：gpuhome-profile-4090-20261018/report.txt](#record-31)
- [记录 32：gpuhome-split-gate1-20261018/report.txt](#record-32)

<a id="record-1"></a>

## 记录 1：acceptance-regression.md

来源：[acceptance-regression.md](acceptance-regression.md)。源文件 SHA256：`c59b6d4f50d0fe81089de0de30691abb378a6e0c18f18b48fb2a427200b8d497`。

### Qwen3.5 MTP + SA 接受长度回退分析

修改前源码：`588bbaa4239189813eafd64b293e56c91e1d9fc8`。模型为
Qwen3.5-35B-A3B-GPTQ-Int4，单张 A100 PCIe 40GB，TP=1，K=8，原生
MTP 使用 `EAGLE` 算法身份。固定 sa-spec 原始 completion 提示模板、
`glaiveai/code_edits_sample` 数据和 2048 输出上限。

#### 独立检查与同位置证据

对照本地 sa_spec `suffix_automaton.hpp` 和 TensorRT-LLM #10951 MTP 集成：
两者均选择最长重复后缀的最早出现位置，反馈含 bonus，每步所选 token
进入下一步 neural forward。参考 benchmark 也是 K=8。没有发现这些接口的
错位证据；不能把“仍执行 neural forward”当成本实现独有的问题。

使用 model-pr-history-knowledge 查阅 Qwen3.5 #19391、#27869，保留原生
MTP draft-extend、GDN 和量化 dispatch，不以历史 PR 替代当前源码检查。
用 llm-torch-profiler-analysis 分析上一轮 Graph trace：SA 查询 8 次合计
347.3 µs，反馈 10 次合计 67.9 µs；native/hybrid 的 Marlin、GDN、conv
与 attention dispatch 均存在。单条 trace 未提供可确认的 overlap 优化点；
自动 fusion 候选不能作为算子修改依据。这些 kernel 时间不是端到端开销。

在 A100 做 diagnostic-only shadow 实验：SA 在 Graph 内查询，但关闭替换，
让原生 MTP 生成；verify 后读取实际 GPU SA 候选、原生草稿链、target 接受
前缀。20 条正式请求外加官方预热及服务健康请求，共 4,749 次 verify、22 个
请求代际；该插桩实验的延迟不用于性能报告。

原策略（匹配长度 ≥4）命中 2,502 个位置；其中 238 个位置的 SA 候选会在
原生已接受前缀内提前出错，合计比原生少接受 843 个 token。另有可能改善的
位置，因此 843 是这些有害位置的损失，**不是整个实验的净损失**。不足 K
个续写仅 1 次，补零不是这组诊断的主要原因。

实际例子：匹配长度 6，原生本轮接受 9 个 token；SA 复制的是 4 空格缩进的
`articles = []`，target 接受的却是 8 空格缩进。另一例匹配长度 6，SA
续写 ` Request:**`，target 接受 ` Code:**`。重复后缀并不保证其后续代码或
说明相同，较短匹配会覆盖更好的 MTP 候选。

原有“SA 分组接受长度 7.64、neural 分组 4.22”的比较有选择偏差，不能推出
SA 在同一位置提高接受长度，也不能与原生全体均值直接比较。

随后用完整 prompt 与 GPU 接受前缀重建历史，对照独立 Python SA，并编译
未经修改的 sa_spec C++ `suffix_automaton.hpp` 做直接差分：**4,746 次查询
全部一致**（排除 1 个服务健康请求，保留 1 个官方预热请求）。历史版本、
anchor 和原生草稿接受前缀亦全部一致。因此目前没有证据支持修改 SA 查询
偏移或 verify 反馈顺序。

##### 当前上游与历史参考的区别

上述直接差分覆盖 Baseten sa_spec 的 SA 核心和旧版 TRT-LLM #10951 集成，
不能据此声称已对齐当前 TRT-LLM 执行方式。2026-10-16 重新核对当前
[SADraftEnhancer](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/speculative/sa_enhancer.py)：
它在独立 CUDA stream 等待主 stream 的已接受 token，执行更新/查询，记录
event；主 stream 正常运行完整 MTP loop，最后等待 SA event 并统一替换候选。
[MTP 调用点](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/speculative/mtp.py)
也明确在全部 draft layers 完成后替换，使 MTP 层始终接收 neural token。

本实现目前仍在 forward stream 先查询，再在 loop 内逐步替换；backend
显式限制所有状态操作使用同一 forward stream。prefill 的后台 CPU 构建
和 Graph 内 SA 查询都不能称为 decode CUDA side-stream 重叠。此项尚未
实现；它主要处理暴露延迟，不能代替接受长度选路的独立验证。

#### 修复策略与边界

增加 opt-in `num_neural_gate_tokens`：先保持原生 neural 前缀，要求 SA 与前
N 个 token 一致才接管余下候选；任一步不一致后整轮保持 neural。mask 原地
更新，保留 Graph staging/verify 的存储别名；每轮 prepare/stage 恢复资格，
不会沿用上轮的拒绝结果。默认 0 保留原始 sa-spec 语义。

第一轮试验取 N=1、min_match_len=16；随后测试 N=4、min_match_len=4，
后者作为 `configs/device-qwen35-mtp.json` 的实验配置。两者均使用
partial_drafts=fallback。这是模型/负载专用策略，不声称 sa-spec 原实现采用
此前缀检查，也不保证它对其他模型或所有位置都优于原生。后缀状态容量仍是
4×16384，未靠扩大历史解释或修复回退。前缀检查没有跳过任何 MTP forward。

同一组诊断位置上的筛选结果如下；有害位置仅指在原生已接受前缀内
提前出错，不包含可能的收益，不是端到端净接受长度：

| 最小匹配 | neural 前缀检查 | 选中位置 | 有害位置 | 这些位置少接受 token |
| --- | --- | ---: | ---: | ---: |
| 4 | 0 | 2502 | 238 | 843 |
| 4 | 1 | 2401 | 165 | 523 |
| 4 | 4 | 2181 | 72 | 167 |
| 16 | 0 | 2007 | 101 | 343 |
| 16 | 1 | 1984 | 85 | 270 |

提高匹配阈值与新增前缀检查都能过滤部分有害位置，但也会过滤可能有益的
位置；上表不是服务加速证明。不同配置的完整服务 A/B 只能说明组合效果，
不能将全部变化归因给某一项。

CPU/native/Gloo 验证 141 passed、1 skipped；A100 CUDA 验证 11 passed。
覆盖原生 draft_forward 的所选前缀递推、Graph 重放后的资格恢复、反馈
mask 别名及参数边界。多 GPU 模型的新策略验收仍未完成。

#### A100 复测

2026-10-16 完成。使用官方 `sglang.benchmark.serving`，相同 100 条 edits
请求、seed=42、temperature=0、原始 completion 模板、输出上限 2048，
保留 EOS 停止；overlap 和 chunked prefill 开启。下表接受长度为整体统计，
TPOT 来自官方 streaming benchmark，未把 shadow 或 profiler 耗时作为性能。

| 配置 | 并发 | 接受长度 | 输出 token/s | 平均 TPOT ms | 输出 token 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本轮原生 MTP | 1 | 5.6822 | 288.7953 | 3.3612 | 169653 |
| N=1、min_match_len=16 | 1 | 5.8894 | 296.6123 | 3.2708 | 166499 |
| N=4、min_match_len=4 | 1 | 6.0326 | 303.8256 | 3.1968 | 175194 |
| 本轮原生 MTP | 4 | 5.8295 | 847.4537 | 4.5378 | 169390 |
| N=1、min_match_len=16 | 4 | 6.0604 | 854.7020 | 4.4396 | 162221 |
| N=4、min_match_len=4 | 4 | 6.0623 | 878.9112 | 4.3681 | 165630 |

N=4 相对本轮原生吞吐增加约 5.2% / 3.7%，但**不能据单轮对照声称稳定
收益**。上一轮相同输入与主要启动参数的原生 C1 接受长度为 6.0707、
吞吐 309.1450、TPOT 3.1449 ms，反而略好于 N=4。本轮与上轮原生仅
2/100 条输出文本完全相同；确认输入长度向量和数据文件相同，启动参数的
实质差异仅为指向同一 draft 模型内容的路径别名。重复性差异的原因尚未
查明，不能把此前整体回退全部归因于 SA，也不能将不同生成轨迹当作逐 token
等价或无损性的证明。

官方 `sgl-eval` CoT 提示模板、同一 GSM8K 全量 1319 题、输出上限 2048：
原生 1270/1319（96.2851%），N=4 为 1269/1319（96.2092%）；两者请求错误
率均为 0，截断分别为 15 / 11 题。这是单次质量对照，不是严格一致性证明。
另保存了官方默认直接回答模板的结果，不能与 CoT 分数混为同一测量。

原始证据保存在 A100：
`/root/rivermind-data/validation/sa-accept-fix-20261016`（第一轮及原生），
`/root/rivermind-data/validation/sa-accept-fix-g4-20261016`（N=4，链接原生证据）。
上述两组流水线均完成，测试服务已停止。这里记录的是完整 Graph 版本；
其后的并发 1 MTP 提前退出实验见下节。独立 SA stream 与多 GPU 新策略
验收仍未实现或完成。

#### MTP 草稿开销的分阶段 trace 分析

在上述 warmed Graph trace 中，以 GPU stage annotation 的 External id
跨所有实际 CUDA stream 归组 kernel，排除 prefill；另在同一 A100、相同启动
参数下补采原生 C1。C4 原生/混合各包含 8 个完整 decode 轮次，C1 包含 9 个。
每阶段 Graph 跨度取该阶段所有 stream 的首个 Graph kernel 起点至最后一个
终点，另计算 kernel 时长之和与 busy interval 并集，不能混用。

##### 阶段与 kernel 归因

| 阶段或算子 | 原生 C1，ms/轮 | 原生 C4，ms/轮 | N=4 混合 C4，ms/轮 | 证据 |
| --- | ---: | ---: | ---: | --- |
| MTP 草稿 Graph 跨度 | 6.4795 | 7.0139 | 7.1520 | `draft`，每轮 7 次 inner forward |
| 其中 LM head 投影 kernel 时长之和 | 4.9349 | 5.0011 | 5.0038 | 每轮 7 次大词表 GEMM |
| target verify Graph 跨度 | 10.4536 | 14.9483 | 15.0167 | `step[VERIFY bs=1/4]` |
| decode draft-extend Graph 跨度 | 1.0535 | 1.1996 | 1.2377 | `draft_extend`，排除 prefill |
| SA consume/query kernel 时长之和 | — | — | 0.0360 | 每轮 1 次 `consume_query_kernel` |
| 前缀比较与逻辑 AND kernel 时长之和 | — | — | 0.0159 | 每轮 4 组比较 |
| SA 逐步 where kernel 时长之和 | — | — | 0.0152 | 每轮 8 次选择 |

LM head 的 eager 调用点来自 `logits_processor._compute_lm_head`，mm 形状
为 `[M,2048] × [2048,248320]`；大词表 kernel 的准确签名及 grid
`[3880,1,1]` 与 measured Graph 每步一一对应。相同 GEMM 名字还用于较小
矩阵，已通过 grid 区分，未把全部同名 kernel 归到 LM head。MTP 入口是
`qwen3_5_mtp.py` 的 `self.logits_processor(...)`。形状/来源映射置信度较高，
测量时间来自 warmed Graph，未套用 eager prefill 时间。

MTP 占以上三个模型阶段跨度之和约 C1 36.0%、C4 30.3%；LM head 约占
MTP Graph 的 76.2% / 71.3%。这不是完整服务耗时占比。采集中的轮间间隔
还含 profiler 放大的 CPU/metadata 等开销；不能从 CPU `draft` scope 或
`cudaGraphLaunch` 的 host 时长推导 GPU 模型开销。

##### 重叠与跳过机会

| 方向 | 观察 | 依赖与边界 | 判断 |
| --- | --- | --- | --- |
| SA 查询与 MTP 重叠 | C4 SA query 约 0.036 ms/轮 | event/反馈存储及 TP collective 顺序仍需处理 | 单纯隐藏查询的空间很小 |
| 前 2 token 一致后跳过 MTP 尾部 | 原生 C1 尾部跨度 5.5498 ms，C4 6.0178 ms | 首 token 由上轮 draft-extend 提供，因此只需再做 1 次 inner forward 才能检查两个 token | 值得做 C1/整批通过的原型 |
| 前 4 token 一致后跳过 MTP 尾部 | 原生 C1 3.7003 ms，C4 4.0111 ms | 需先做 3 次 inner forward | 保护更严格，可省区间更短 |
| 仅 mask 部分请求 | 混合批仍执行整批 GEMM | 要省矩阵工作须压缩重排，或整批均通过后跳过 | 不能把逐请求命中率直接当批跳过率 |

上述尾部跨度以相应 argmax finalize 结束为分界，跨全部 Graph stream
计算至最后一个 finalize，表示现有路径上可能省掉的区间，**不是实现后的
端到端加速**。预计收益还取决于整批可跳过比例、Graph 分支/同步开销、
压缩成本及候选改变后的接受长度。本轮 N=4 正式计数约 40.5% / 41.5%
逐请求选择 SA（含官方预热），尚未测量“整批全部通过”的比例。

##### 融合与其他方向

| 方向 | 已有证据 | 结论 |
| --- | --- | --- |
| 融合前缀 eq/AND/where | C4 相关 kernel 合计约 0.031 ms/轮 | 可减少小 kernel，但不是主要开销 |
| 跳过无用的全词表投影 | 每轮 7 次，合计约 5 ms | 比仅隐藏 SA 查询更有价值；运行时跳过仍需分支支持 |
| 量化或改变 draft LM head | 当前实测为大词表 FP16 投影 | 只列候选，尚未验证后端、权重共享与接受质量，不能直接推荐参数 |
| 自动报告中的 GDN→MoE fusion | 规则可把 sigmoid GDN 错认成共享专家门控 | 不采用这一自动归因修改算子 |

建议先用两个 Graph（前缀及余下 MTP）在 C1 做提前退出的计时原型，量化
GPU→CPU 决策同步是否抵消收益；值得继续时再实现 GPU conditional subgraph，
避免 host 同步。target verify 与后续 draft-extend 必须保留。上述 trace 对应
完整 MTP 版本，不能把前缀保护或 SA side-stream 等同于提前退出。

C1 新采集证据目录：
`/root/rivermind-data/validation/draft-cost-20261016/profile-native-c1`。
本地逐阶段机器可读结果与离线归因脚本保存在本次工作目录的
`.tmp/hybrid-spec/accept-regression/draft-cost-analysis.json` 和
`analyze-draft-cost.py`；完整 trace 保留在对应 A100 profile 目录。
该阶段只做 trace 测量；随后实现的提前退出原型与验收单独记录如下。

#### 并发 1 两段 Graph 原型

新增默认关闭的 `draft_graph_mode`，不修改 SA CUDA 状态布局或 v3 扩展名：

- `full`：原有完整 Graph。
- `split_no_skip`：两段 Graph，中间读取 GPU mask，但始终执行尾段；仍按同一
  前缀策略选择 SA，作为拆图及同步成本的控制组。
- `split`：前两个 token 均相同且 SA 原本有效时跳过尾段；不一致则继续尾段。

实验限定 device history、Qwen3.5 MTP、TP=1、topk=1、greedy、K>2、
`num_neural_gate_tokens=2`、`max_requests=1`；启动同时要求
`--max-running-requests 1 --cuda-graph-max-bs-decode 1`。memory saver、
rejection sampling、hot-token 映射和 draft-extend DSA seed 均不开放。
仅更新插件无法驱动旧 worker：需要新增的 draft split Graph API v1。

runner 通过可选 extension backend 包装，在现有 MTP body 生成第二个候选后
结束 prefix capture、开始 tail capture，不复制一份模型循环。第一个 neural
候选来自前一轮 draft-extend，因此首段只做一次 inner forward。首段预先把
完整 SA 候选写入结果矩阵，再覆盖已选择的前两个位置。成功时返回的仍是
完整 K=8 候选，target 仍执行原生 9-token verify；没有重新验证历史匹配前缀。
回退时尾段直接消费首段留下的 hidden state，不重算首个 forward。

两段使用相互独立、与 target/extend 不共享的私有 Graph pool，防止可选尾段
及其跨 Graph 中间张量被别的 replay 覆盖。命中后恢复首段增加的位置计数，
其余输入按原生 execute 每轮重新 staging。verify、接受前缀反馈及 draft-extend
全部保留，不依赖被跳过的 MTP 临时 KV；draft-extend 按接受结果填充后续状态。

本地 CPU 140 passed；最终源码在 A100 的 CPU/native/Gloo 回归为
147 passed、1 skipped，CUDA 13 passed，覆盖连续命中、首/第二 token
不一致、资格关闭、跳过后再回退、位置恢复与真实双 Graph pool 的重放。
CPU harness 执行实际 `draft_forward` body，确认边界位于第一次 inner forward
之后；原有默认路径与 mRoPE 递推用例保持通过。

##### 原型 trace

| 模式 | 捕获 decode 轮数 | 每轮 inner forward 数 | GPU 草稿跨度 |
| --- | ---: | --- | --- |
| split | 9 | 7 轮为 1 次，2 轮为 7 次 | 命中均值 0.9800 ms；回退 7.0267 ms |
| split_no_skip | 9 | 全部 7 次 | 均值 7.0014 ms |
| full | 9 | 全部 7 次 | 均值 6.4248 ms |

split 的本次 profile 共观察到 21 次 inner forward（7×1+2×7），不是只替换
token 后继续运行全部模型。该窗口的命中比例不同于正式 50 请求总体，不能
用 7/9 作为负载的长期命中率。回退的两段间隙约 0.45 ms；控制组约
0.57 ms，均为带 profiler 的观察值，不能直接作为无 profiler 的同步成本。
诊断字段 `decision_wait_ns` 还包含未完成的首段 GPU 工作，不能把其约
0.97–0.98 ms 的每轮均值都当成额外同步开销。

##### 服务对照协议

同一 A100 PCIe 40GB、Qwen3.5-35B-A3B-GPTQ-Int4、TP=1、K=8，所有模式
均限定并发 1、最大运行请求 1；target FP16/gptq_marlin、MTP unquantized、
相同 conv/attention backend、seed=42，overlap 与 chunked prefill 保持开启。
官方 `sglang.benchmark.serving`，同一 shuffle 后的 50 条原始 sa-spec edits
completion 提示，输出上限 2048，保留 EOS。三组 SA 的最小匹配均为 4，
neural 前缀检查均为 2，不能混入前文 N=4 或 min_match_len=16 的结果。
split、full 与原生还各重复前 20 条，用于观察同进程重复性。
正确性另用官方 `sgl-eval` CoT 模板的同一 GSM8K 256 题，输出上限 2048、
num_threads=1；按题目 ID 检查题干/标准答案，记录双向正误变化及截断。

实验目录：`/root/rivermind-data/validation/split-graph-20261016`。

##### 原型服务性能

以下均为官方客户端报告的整体统计，未使用 profiler 耗时计算吞吐或 TPOT。

| 模式 | 接受长度（含 bonus） | 输出 token/s | 平均 TPOT ms | 输出 token 数 |
| --- | ---: | ---: | ---: | ---: |
| 原生 MTP，无 SA | 6.0273 | 305.0391 | 3.2073 | 79888 |
| MTP+SA，full | 5.9431 | 300.6637 | 3.2194 | 75184 |
| MTP+SA，split_no_skip | 5.9130 | 294.7471 | 3.3304 | 87844 |
| MTP+SA，split | 6.0465 | 342.3072 | 2.8664 | 84830 |

本轮 split 相对原生吞吐增加约 12.2%，TPOT 降低约 10.6%；相对 full 吞吐
增加约 13.9%，TPOT 降低约 11.0%。split_no_skip 相对 full 吞吐降低约
2.0%、TPOT 增加约 3.4%。因此本轮观测到同步/拆图有成本，但没有抵消跳过
MTP 的收益。不能把接受长度和输出数的差异全部排除为无关因素，也不能把
全部改善归因给一次纯同步开销的精确差值。

split 正式计数（包含官方预热）为 14074 次 prefix、6044 次 skip、8030 次
tail，skip 比例 42.94%，与实际 SA 选择计数一致。控制组 14910 次 prefix
全部执行 tail，skip 为 0，虽然约 40.2% 的轮次仍选择 SA 候选。这两个
模式都进行每轮一次 CPU mask 读取，区别在于是否提交后续六次 MTP forward。

同进程重复前 20 条，原生/full/split 完全相同的文本仅 2/1/2 条，共同
前缀字符数中位数为 374.5/410.5/886。输入长度向量相同；这一重复性问题
在原生与 full 上亦存在，不能归因于提前退出。长输出轨迹不同，且仅做一组
固定 workload 对照，因此结果应视为该配置的实验观察，不能宣称所有负载
稳定加速或逐 token 等价。

##### 原型正确性

| 模式 | 正确 / 256 | 准确率 | 请求错误率 | 截断题数 |
| --- | ---: | ---: | ---: | ---: |
| 原生 MTP | 248 | 96.875% | 0 | 3 |
| MTP+SA，full | 248 | 96.875% | 0 | 4 |
| MTP+SA，split | 249 | 97.266% | 0 | 3 |

full 答对而 split 答错的题目为 0，反向为 1（gsm8k-161，两者均正常
stop，未截断）。原生答对而 split 答错亦为 0，反向为 1（gsm8k-163）。
两组比较的原始生成文本完全一致分别为 95/256、100/256。该样本未观察到
正确性下降，但不据这一题差异宣称模型质量提升或严格等价。这是 256 题
原型验收，不能冒充前文 1319 题的全量结果。

四组流水线和最终回归均已完成，测试服务已停止。结构化结果见
[split-graph-a100.json](split-graph-a100.json)，包含测试协议、数据哈希、
官方统计、分支计数、重复性与逐题质量对照。完整 trace、命令、原始生成、
实际 benchmark 源码快照和最终验证日志保留在上述 A100 实验目录；本机亦
下载原始证据归档。当时的两段 Graph 属于默认关闭的单并发原型，未实现 GPU
conditional node、独立 SA stream 或多请求压缩重排。后续并发 4 的逐请求
尾段压缩与独立对照见[并发 4 实验](split-graph-c4.md)。

<a id="record-2"></a>

## 记录 2：arctic-overlap-4060/README.md

来源：[arctic-overlap-4060/README.md](arctic-overlap-4060/README.md)。源文件 SHA256：`87abb4477bae2ebdfed4c876ade629f0b0286daee572477f44586634681e3e70`。

### Arctic overlap / chunked prefill RTX 4060 验证

最终完整回归：**735 passed、2 skipped（245.46 秒）**。单卡环境跳过 NCCL
2/8 卡用例；测试结束显存为 0 MiB。

本轮基于 `bc44c2aa40ef88e3c7aed761744975c39d1511d1`，在
`/workspace/arctic-overlap-20261017/repo` 独立 worktree 验证。
先 git fetch 已发布分支，再上传本次补丁；沿用已编译的官方 Arctic suffix
扩展，固定 upstream `9d4643881608178f0f58a33d1b61919ca246997f`。

完整结果、环境与源码 SHA256 见 `metadata.json`，原始输出见 `regression.txt`。
测试前的探索性中断运行不计入通过数量；最终完整运行覆盖最终源码。

本次新增覆盖：

- CPU 尚无 prefill anchor / 尚未提交 verify 输出时，使用 GPU bonus 和接受路径；
  不等长接受、全局 node index、padding、batch 重排和 CPU 追赶后不重复追加。
- 分块中间行丢弃临时 bonus；同批其他 final 行正常索引，最后一块包含完整前缀。
- 重预填充、取消旧 ticket、异常快照、idle、结束回收、同 rid 新请求和容量回退。
- GPU 私有尾部与 CPU 确认共享树隔离；停止截断后只保留最终 response。
- 官方合并缓存与本实现分离 local/global 树的候选、匹配长度选优对照。
- 反馈版本/index/count/token 异常回退，GPU 进度计算剩余预算，历史超容量回退。
- MTP/EAGLE3/STANDALONE 的 overlap/chunked 配置组合；已有 full/split/no_skip
  gate、GPU 动态行号、compact tail、DSV4 形状模拟和 TP 回归保留；
  EAGLE3/STANDALONE compact 用例保留原生 Triton metadata/index kernel。

环境：RTX 4060 Laptop 8 GB，WSL2，Python 3.10.18，Torch 2.8.0+cu128，
Triton 3.4.0。命令（工作机仓库根目录）：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/arctic-suffix-20261017/ArcticInference:/workspace/arctic-overlap-20261017:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests \
  -q --tb=short -rs --junitxml=../regression.xml
```

轻量导入辅助沿用 [保存快照](dsv4-lease-4060/lean_cuda_plugin.py.txt)。
真实服务不要加载该辅助。使用真实官方 CPU C++ 后缀树、GPU 路径打包、pinned
D2H/event、H2D 和 CUDA Graph；新增调度用例为构造的 prefill/verify batch，
神经模型和 child attention 仍为合成计算。不能据此声称真实调度器/权重 serving、
DSV4 TP8 NCCL、模型质量或吞吐已经验收。

内网移除 `--disable-overlap-schedule`，采用默认 chunk size 或显式设置
`--chunked-prefill-size 2048`；仍不可启用 `--enable-mixed-chunk`。
同样支持关闭两者作为对照。CPU 树每轮仍需接受路径 D2H、消费等待和查询，
是否有净收益需实测。完整验证矩阵建议见 [接入指南](arctic-suffix.md)。

<a id="record-3"></a>

## 记录 3：arctic-suffix-4060/README.md

来源：[arctic-suffix-4060/README.md](arctic-suffix-4060/README.md)。源文件 SHA256：`4e9401f58047ecd5e06aedc9c6becff972fe64a2310e53e6f8b440bf6c3115c2`。

### Arctic 官方后端 RTX 4060 验证

结果：完整回归 **663 passed、2 skipped（337.74 秒）**，再通过 46 项新增
Arctic+dsv4（19.56 秒），合计 709 个不重复通过用例。CUDA 455 项、CPU/host/Gloo
254 项。测试结束显存 0 MiB，无 pytest 进程。

隔离 worktree：`/workspace/arctic-suffix-20261017/repo`，基于本分支
`53fdad7f27aa888dd5b440dc284f4ea2725120e4`。先 clone/pull 已发布代码，再上传本次
补丁验证。Arctic 从官方仓库 clone，固定
`9d4643881608178f0f58a33d1b61919ca246997f`，仅按官方 CMake 编译 suffix 模块，
没有修改上游源码，也没有安装/替换 vLLM 或 Torch。

- `metadata.json`：环境、每轮命令/结果、最终本仓库与上游源码 SHA256。
- `regression.txt`：完整回归原始输出。
- `dsv4.txt`：补充的 Arctic+dsv4 split 矩阵原始输出。
- pytest 轻量导入辅助沿用 [已保存快照](dsv4-lease-4060/lean_cuda_plugin.py.txt)。

环境：Python 3.10.18、Torch 2.8.0+cu128、Triton 3.4.0、nvcc 12.0、
GCC 12（SA CUDA 扩展）、GCC 13（官方 Arctic CPU 扩展）、nanobind 2.9.2。
验证使用真实 `SuffixDecodingCache`、C++ tree、CPU 历史/H2D 与实际 CUDA Graph；
模型和 child attention 的执行仍为合成计算。EAGLE3/STANDALONE compact 用例
保留原生 Triton 多步 KV metadata/index kernel 验证。

完整命令（工作机仓库根目录）：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/arctic-suffix-20261017/ArcticInference:/workspace/arctic-suffix-20261017:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests -q --tb=short -rs
```

首次完整回归后，仅增加 `arctic_dsv4` 参数组合，单独运行该新增矩阵。
共享 gate 的 `_synchronize`/`select_step` 与基线 DeviceOverride 的 AST 对照相同，
仅将行数取得方式从 `len(self.slots)` 改为 `self.tokens.shape[0]`，供 staged CPU
候选复用。Gloo TP1/2/4/8 额外覆盖 staged gate；NCCL 2/8 卡在单卡机器上跳过。

这是运行时功能回归，不是完整真实权重/serving、TP8 NCCL、模型质量或吞吐验收。
真实服务不要加载 lean_cuda_plugin。安装与使用见 [接入指南](arctic-suffix.md)。

<a id="record-4"></a>

## 记录 4：arctic-suffix.md

来源：[arctic-suffix.md](arctic-suffix.md)。源文件 SHA256：`6cdf31220d80720947c73db1dc55e9217a0a9b79c22796ef5984edc1b588b2aa`。

### ArcticInference SuffixDecoding 后端

通过 `backend="arctic"` 使用 ArcticInference 官方 `SuffixDecodingCache` 和 C++
后缀树，替换默认 `gpu_sa` 检索。无需将 Arctic 的 vLLM worker 移植到 SGLang；
MTP/EAGLE3/STANDALONE 保留原生 neural draft、target verify 和提交路径。

#### 范围与语义

- 支持 `full`、`split`、`split_no_skip`。gate=0…K、目标词表映射、TP 共识、
  GPU 稳定行号及紧凑 tail Graph 与 GPU SA 共用；split API 仍为 v5。
- 使用 Arctic 原生单链 (`use_tree_spec=False`) 查询，至多 K 个候选，保留其
  频次、概率阈值和匹配长度驱动的候选规则。原生 score 用于其 local/global
  树内部选择，不替代本项目的 gate，也不作为 neural proposal 概率。
- 当前 verify 布局固定 K。默认 `partial_drafts="fallback"`：不足 K 整轮回退；
  原有可选 `zero` 策略仍表示补零草稿并经 target verify，不等于变长验证。
  未接入 Arctic 的树状候选、动态 verify 宽度或 score 驱动的原生混合调度。
- `history_mode="cpu"` 表示 CPU 后缀树，现在允许 overlap 与 chunked prefill。
  可移除 `--disable-overlap-schedule`，并设置 `--chunked-prefill-size 2048`
  或采用模型默认分块值。仍不支持 `--enable-mixed-chunk`。
- 私有树历史由 GPU prefill bonus 和 verify 的 `accept_index/accept_lens`
  补齐，不依赖可能滞后的 `req.output_ids`。Graph 外将每批接受路径和版本打包，
  异步拷贝到 pinned CPU 内存；下一次查询消费时等待对应 CUDA event。
  query/H2D 仍在 Graph 外完成，split Graph 内只执行 gate/共识/选择。
  历史必须等于 GPU `seq_lens+1`，版本/反馈异常、无槽位或超容量时按行回退。
- 分块 prefill 的中间行不保留临时采样 bonus；最后一块启动前快照完整
  `get_fill_ids()` 前缀，包含缓存和先前分块。重预填充撤销旧反馈，异常取消只
  清理对应请求代际。GPU 私有尾部不代表最终交付输出。
- 自身 prompt 私有；开启 `enable_global_pool` 时只有已确认 response 进入
  Arctic 全局树，活跃 response 可供其他请求检索。私有树和共享树分别使用
  官方 cache，按 upstream 的 score 规则选优（同分优先 local）；共享查询使用
  空 local 的内部查询键，不占 donor 槽。GPU 尚未经过 CPU 停止裁剪的尾部只
  进入私有树，绝不进入共享树。完成时丢弃 local prompt，
  最多保留 `max_shared_requests` 个完整 response，按 FIFO 淘汰。
  `max_requests` 只约束活跃请求，完成 donor 不占活跃槽。
- 每次仅追加确认增量；改写/撤回历史会删除旧树并递增 generation。完成时
  从最终 CPU 输出重新对账，包括最后一轮尚未被下次 draft 消费的 token，
  排除停止条件裁掉的尾部。异常结束、弱引用失效、flush 和槽位复用清理旧状态。
- Arctic CPU 树不消耗 GPU SA 状态池；仍有接受路径 D2H、消费等待、CPU 查询、
  候选 H2D 和 split Graph 的成本。启用共享池还会维护独立的已确认响应树。
  允许 overlap 不等于 CPU 查询与 GPU 解码完全重叠，也不保证性能改善；
  GPU SA 的吞吐结果不能套用于 Arctic。

Triton draft attention 支持普通 EAGLE/MTP、EAGLE3 和 STANDALONE；DSV4 MTP
继续使用原生 dsv4 draft attention。其他模型、并行和采样限制同主 README。
`max_history_tokens` 限制整个请求历史，`arctic_max_tree_depth` 控制树深/查询上下文；
`shared_context_tokens` 属于 GPU SA donor 选择，Arctic 使用其自己的树查询规则。

#### 安装

Arctic 为可选依赖，默认安装、导入和 GPU SA 运行不需要它。可安装
`pip install -e './experimental/hybrid-spec[arctic]'`，需保证 upstream 原生 suffix
扩展可导入。缺少扩展时启动明确报错，不会静默使用其他后端。

本轮验证固定官方源码
[`9d4643881608178f0f58a33d1b61919ca246997f`](https://github.com/snowflakedb/ArcticInference/tree/9d4643881608178f0f58a33d1b61919ca246997f)。
为保持现有 Torch/CUDA serving 环境，4060 仅编译官方 CPU suffix 扩展；
没有安装整个 Arctic/vLLM 栈或替换 Torch。完整依赖构建策略由 upstream 管理。
在服务使用的 Python 环境运行下列示例（`ARCTIC_SOURCE` 为新目录）：

```bash
ARCTIC_SOURCE=/实际目录/ArcticInference
# 已存在源码时使用独立 checkout，避免覆盖其他工作的修改。
git clone https://github.com/snowflakedb/ArcticInference.git "$ARCTIC_SOURCE"
git -C "$ARCTIC_SOURCE" checkout --detach 9d4643881608178f0f58a33d1b61919ca246997f
python -m pip install cmake nanobind==2.9.2 numpy
cmake -S "$ARCTIC_SOURCE/csrc/suffix_decoding" -B "$ARCTIC_SOURCE/build-suffix" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$ARCTIC_SOURCE/arctic_inference/suffix_decoding"
cmake --build "$ARCTIC_SOURCE/build-suffix" -j 2
export PYTHONPATH="$ARCTIC_SOURCE:$PWD/experimental/hybrid-spec/src:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from arctic_inference.suffix_decoding import SuffixDecodingCache'
```

后缀模块来自上游 Apache-2.0 源码，本仓库不 vendor。官方 CPU 构建需要支持
C++20 的编译器、Python 开发头文件和 nanobind。

#### 启动

在已可运行的原生 neural-spec 命令上设置：

```bash
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/arctic-suffix-split.json"
# 保留 --speculative-algorithm EAGLE / EAGLE3 / STANDALONE
# --disable-overlap-schedule --chunked-prefill-size -1
# --speculative-num-steps 8 --speculative-num-draft-tokens 9
# --speculative-eagle-topk 1
# --max-running-requests 8 --cuda-graph-max-bs-decode 8
# --speculative-draft-attention-backend triton
# DSV4 MTP 上一项保持 dsv4
```

仅把 `backend` 改成 arctic 而继续配置 `history_mode=device` 会报错，必须同步
设置上述 CPU 历史/调度选项。默认 GPU SA 配置不变。

模板选用 K=8/G=2，`arctic_max_spec_factor=2.0` 用于在匹配较短时容许更多候选；
这属于策略参数，需按负载校准。其他 Arctic 参数默认值：max depth 64、factor 1.0、
offset 0.0、min token probability 0.1。更改 K 时同步修改 JSON 与服务 steps/token
参数，并保持 G≤K。`enable_global_pool=false` 可单独评估请求内后缀树。

#### 调度验证

新的 overlap/chunked 生命周期回归与原始接入证据分开记录，见
[4060 调度验证](arctic-overlap-4060/README.md)。测试使用真实官方后缀树、CUDA
接受路径拷贝与事件、构造的 prefill/verify batch；神经模型和 child attention
计算仍为合成执行。DSV4 真实权重 TP8、真实调度器长输入及性能需在内网验收。

建议内网以相同 JSON、K/G 和输入集，对 native/full/split 分别比较：
两项关闭、仅 overlap、overlap+chunked。输入长度需超过 chunk size，并覆盖并发
1/4/8、长短请求交错、EOS/stop/max_new_tokens、取消/重用/重预填充及共享池开关。
保持 TP8、相同 `--speculative-draft-attention-backend dsv4`，交替启动顺序并重复 3 次。

#### 初次接入验证（历史记录）

4060 完整回归已通过 663 项，跳过两项需要 2/8 卡的 NCCL 用例；另增加
Arctic+dsv4 矩阵单独通过 46 项，两轮合计 **709 passed、2 skipped**。
原始日志、源码哈希和详细范围见
[4060 证据](arctic-suffix-4060/README.md)。

```bash
python -m pytest experimental/hybrid-spec/tests/test_arctic.py -q -rs
python -m pytest experimental/hybrid-spec/tests -q -rs
```

`test_arctic.py` 要求真实官方 C++ 扩展；没有安装时会跳过，不能把跳过计为通过。
4060 的验证环境已安装该扩展，运行时还明确检查它可导入。
Arctic compact CUDA 用例使用真实 CPU 查询和 H2D staging，覆盖 G=0/1/2/3/K、
K=1/2/3/8、B=1/8/16、混合命中和 padding；比较 full/split 草稿链及 tail 工作量。
神经模型和 child attention 执行仍是合成计算，不能据此声称真实权重/serving
或 TP8 NCCL、吞吐、质量已验收。

<a id="record-5"></a>

## 记录 5：compact-verify-4060/README.md

来源：[compact-verify-4060/README.md](compact-verify-4060/README.md)。源文件 SHA256：`778af3708c35ba748702c7224e168fdb218e99f66ea1781ab159c1f910290121`。

### Compact verify — RTX 4060 组件验证

最终完整回归：**813 passed、2 skipped，264.24 秒**；两项跳过分别为 NCCL 2 卡/8 卡测试。
结束后 RTX 4060 显存为 **0 MiB**。本地 CPU 回归另有 259 项通过。

最终结果、环境与源码 SHA256 见 `metadata.json`，完整输出见 `regression.txt`。
仅最终完整运行计入通过数量；探索性失败和中断运行不计入。

基线为 `049db08e4d`，工作机独立 worktree：
`/workspace/compact-verify-20261018/repo`。先 fetch 已发布分支，再应用本轮补丁。
本地源码、配置和测试与工作机逐文件校验；提交前核对全部 54 项 SHA256。

覆盖：

- SA=8，neural=1/2/3/8；gate=0…8，包括 gate=3/neural=2、gate=8/neural=1。
- 实际 worker draft_forward、原生 GPU 动态行号和 split graph：MTP、EAGLE3、
  映射词表、STANDALONE、hidden_states=None，以及官方 Arctic suffix 来源。
- 混合 batch、全 SA、全 neural、重排、缩批、Graph padding，batch=1/2/4/8/16。
- 原生 target 打包 kernel 的 anchor、候选、position、KV 写入列，以及保留原完整 KV plan。
- 固定地址 metadata 刷新、同一档位内长度/顺序变化、不同 token 档位切换，
  4/9/13 等非整窗口大小；模拟 8-token 对齐的真实单 GPU 图回放。
- 原生 Triton attention 与独立 PyTorch softmax attention 数值对照，含 prefix KV、
  ragged query、空 padding 行和反复切换旧图。Graph 外缓冲区由测试显式持有。
- 原生 greedy verify 和 commit kernel：拒绝位置、bonus、全局 accept index、
  seq_lens 更新、draft-extend select_index；补零尾部即使碰巧匹配 target，也不能越过有效链。
- split 的 host neural 行数复用于 verify 选档；不能复用时读回单个总数。
  诊断区分逻辑 token、实际 target 行数和未压缩窗口，padded 对照计入实际图 padding。
- 保留已有 SA 生命周期、共享池、overlap/chunked、CPU/Gloo 和单卡 CUDA 回归。

复现（工作机仓库根目录）：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/arctic-suffix-20261017/ArcticInference:/workspace/compact-verify-20261018:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests \
  -q --tb=short -rs --junitxml=../regression-v6.xml
```

轻量导入辅助沿用[已保存快照](dsv4-lease-4060/lean_cuda_plugin.py.txt)，
仅用于合成模型测试；真实服务不要加载。新 compact verify 用例在测试内隔离完整
serving 依赖；CUDA Graph、打包/映射、接受/提交以及 attention kernel 为原生实际执行。

这不是完整 SGLang 服务或 DSV4 权重验证。神经层使用合成模型；DSV4 child metadata
由形状/地址模拟检查，实际 DSV4 压缩 KV 和 TP8 NCCL 未在这张单卡上执行。
8 对齐测试证明单 GPU 的档位/打包行为，不能当作多卡 SP/NCCL 验收。
真实模型 greedy 输出、DSV4 TP8、长上下文压缩边界、取消/retract 与性能由内网补测。

内网配置与对照方法见[接入说明](../README.md)。保持 SA8/neural3/gate1 相同策略，
比较 padded 与 compact，另外保留原 SA8/neural8 基线；覆盖 c=1/4、gate=3/neural=2、
全命中/无命中/混合 batch、上下文 127/128/129 边界和停止截断，并交替顺序重复测量。

<a id="record-6"></a>

## 记录 6：draft-algorithms-split-20261017.md

来源：[draft-algorithms-split-20261017.md](draft-algorithms-split-20261017.md)。源文件 SHA256：`90bb0a5f9624853a27d25dc0170d08429d77aa192824fb66a1986a7ecd048927`。

### EAGLE3 / STANDALONE split 接入

本次基于 `7ee73928b35779ae9391969c6227b04f1bce2b45`，将 EAGLE/MTP、EAGLE3、
STANDALONE 的 split 与 split_no_skip 统一到 GPU 动态行号压缩和 gate=0…K。
默认仍为 full，不修改原生算法身份和 target verify/commit。

#### 实现

- 移除 split 只允许 `EAGLE` 和两种 MTP 架构的拦截；仍要求 greedy、topk=1、
  单机 TP、device 历史及受支持的模型配置。
- EAGLE3/reduced-vocab 草稿的每次 local argmax 都先经过原生 hot-token 映射，
  再进行 gate、SA 选择及 verify-chain 写入。prefix 与 compact tail 使用相同目标词表。
  split 的 dense chain 不再因存在 token map 而退回 token-list 路径。
- STANDALONE 不依赖 target hidden state；compact tail 的 seed 可以为 None，
  不对其分配 staging 或执行 index_select。EAGLE3 则按实际 prefix 输出形状收集，
  不假设 seed 的拼接宽度与后续 recurrent hidden width 相同。
- 所有路径共享 gate 和 GPU 行号 kernel：G=0/1 不运行前缀 inner forward，
  G≥2 运行 G−1 次；G=K 没有 tail；G>K（例如 K=2/G=3）在配置阶段报错。
- Triton 多步 attention 和原生 dsv4 使用现有的独立 metadata/Graph pool；
  V4 的逐步写槽、Qwen3-MoE fused KV-store 的连续写地址约束保留。
- split API 升为 v5，worker、runner 和插件必须同时更新；通用 API v2、SA 布局 v3 不变。

仍拒绝其他 draft attention（例如 FlashInfer）、translated KV、KV canary、
memory saver、draft-extend DSA seed；其他 SWA/线性 attention 的模型范围未扩张。
取消 batch≤4 不保证任意批次的显存或性能，每个前缀仍捕获 1…B 大小的 tail。

#### 4060 验证

完整回归：**505 passed，2 skipped，227.19 秒**；其中 CUDA 267 项，
CPU/host/Gloo 238 项。两项跳过为需要 2/8 卡的 NCCL 用例。结束后显存 0 MiB。

在工作机隔离 worktree 中先验证补丁，再提交推送。测试原始证据见
[4060 日志和源码哈希](draft-algorithms-split-4060/README.md)。

CUDA compact 矩阵包括 Qwen 风格 MTP、V4 风格 MTP、EAGLE3（有/无 token map）、
STANDALONE；每种覆盖 split/split_no_skip、B=1/8/16、K=1/2/3/8、G=0/1/2/3/K。
同一捕获 Graph 的多次 replay 变化命中行、raw batch 和 padding；除独立预期链外，
每轮还与相同候选和 gate 的 full worker 完整草稿链逐行比较，验证跳过行没有 tail 计算。

EAGLE3/STANDALONE 的 138 个 compact 用例额外使用仓库原生的
TritonMultiStepDraftBackend 元数据方法、build_inner_fb_view 和
`generate_draft_decode_kv_indices` CUDA/Triton kernel；紧凑批的 ragged KV 读取
与按请求表直接求和的独立 dense 参考比较。原生 topk、动态行号 kernel、CUDA Graph
capture/replay、gather/scatter 和私有池在 GPU 上实际执行。

模型、child attention 执行以及完整 serving 的导入边界仍用轻量替身；
不声称跑过真实权重、完整 EAGLE3/STANDALONE 服务、模型精度、吞吐或 TP8 NCCL。
这些仍需在具备模型和完整 SGLang 环境的机器验收。

#### 运行

在已可运行该 checkout 的原生 EAGLE3 或 STANDALONE 命令上：

```bash
export PYTHONPATH="$PWD/experimental/hybrid-spec/src:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/device-draft-split.json"
# 保留 --speculative-algorithm EAGLE3 或 STANDALONE
# --speculative-draft-attention-backend triton
# --speculative-eagle-topk 1
# --speculative-num-steps 8 --speculative-num-draft-tokens 9
# --max-running-requests 8 --cuda-graph-max-bs-decode 8
```

模板采用 K=8/G=2/B=8。更改 K 时同步更改 JSON 与两个 speculative steps/token 参数，
并保持 G≤K。DSV4 仍使用原有 dsv4 模板和 backend，不改为 triton。

完整依赖环境直接运行 `python -m pytest experimental/hybrid-spec/tests -q -rs`。
4060 的 lean_cuda_plugin 只用于隔离测试导入，不加载到真实服务。
做模型对照时固定原生/full/split 的参数，交替顺序重复运行；观察混合批次下的
`mixed_ct`、`tail_request_ct`、接受长度和输出正确性，性能结果独立记录。

<a id="record-7"></a>

## 记录 7：draft-algorithms-split-4060/README.md

来源：[draft-algorithms-split-4060/README.md](draft-algorithms-split-4060/README.md)。源文件 SHA256：`011ff130ea5426781edf37073808d3e6c3b11c6bfa13ce73110e27c4ce3b1976`。

### RTX 4060 split 三算法回归证据

结果：**505 passed，2 skipped，227.19 秒**。CUDA 267 项（含 230 个 compact
用例），CPU/host/Gloo 238 项。测试结束显存 0 MiB，无 pytest 进程。

工作机：`ai-workstation`，NVIDIA GeForce RTX 4060 Laptop GPU 8 GiB，WSL2。
隔离目录 `/workspace/dsv4-algorithms-20261017/repo` 从已推送的 `7ee73928b3` 创建，
上传本次源码补丁后验证，验证通过并核对源码哈希后才提交推送。

- [metadata.json](draft-algorithms-split-4060/metadata.json)：软件栈、命令、测试结果、测试源码 SHA256。
- [regression.txt](draft-algorithms-split-4060/regression.txt)：完整回归的原始 pytest 输出。
- 轻量 pytest 导入辅助沿用 [前轮快照](dsv4-lease-4060/lean_cuda_plugin.py.txt)，
  其 SHA256 同时记录在 metadata.json；只影响 compact CUDA 用例的 serving 导入。

完整命令（工作机仓库根目录）：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/dsv4-algorithms-20261017:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests -q --tb=short -rs
```

230 个 compact CUDA 用例使用精确的仓库 worker 函数体、原生 topk、稳定行号 kernel、
实际 prefix/tail Graph 和 full 对照。138 个 EAGLE3/STANDALONE 用例额外加载
原生 Triton 多步 metadata 方法、build_inner_fb_view 与 KV index kernel，
将紧凑 ragged KV 读取与独立 dense 参考比较。
模型和 child attention 执行仍是合成计算；完整 SGLang 导入边界由辅助脚本隔离。
其余 CUDA 测试不使用这个导入替身，包含真实 GPU SA 查询、反馈与生命周期。

不是完整权重/serving 的模型验收，不包含 TP8 NCCL、模型质量或吞吐结果。
2/8 GPU NCCL 测试在单卡机器上跳过；Gloo TP1/2/4/8 仅验证 CPU 共识。
新配置模板在本地通过 HybridConfig.from_file 校验。内网真实模型直接运行测试，
不要启用这里的 lean_cuda_plugin。

<a id="record-8"></a>

## 记录 8：dsv4-lease-4060/README.md

来源：[dsv4-lease-4060/README.md](dsv4-lease-4060/README.md)。源文件 SHA256：`2654a3730773db84be6219154b243b7b282f3575e8b6778e9b3eaec23175b8ec`。

### RTX 4060 回归证据（2026-10-17）

测试对象：`a3a0d112b6ff02d48a57855afb7d68e3b964bf30`。
工作机从 `https://github.com/ggg-s/sglang.git` 的
`codex/hybrid-suffix-speculation` 分支 clone/pull，测试源码无未提交修改。

- [原始最终输出](dsv4-lease-4060/regression.log)：218 passed、2 skipped，24.24 秒。
- [元数据](dsv4-lease-4060/metadata.json)：精确命令、环境、版本、源码与辅助文件 SHA256。
- [轻量导入辅助原样快照](dsv4-lease-4060/lean_cuda_plugin.py.txt)：仅用于 8 项合成模型 compact
  CUDA 测试；其他测试不走该 fixture 的替换分支。

CPU/Gloo 181 项、真实单卡 CUDA 37 项；NCCL 2 卡和 8 卡两项因只有一张卡而跳过。
8 项 compact 测试的模型/attention 计算本身是测试中的合成计算；辅助 fixture
另外替换未执行的 `FullCudaGraphBackend` 标记类与 attention 容器导入边界，
绕开完整 serving 依赖。原生 topk Triton kernel、ForwardContext、插件的
prefix/tail capture、GPU gather/scatter、私有 Graph pool、所有 mask 与 padding
检查仍实际执行。其余 29 项 CUDA 用例没有这些额外替换，包含 4 项新增 lease
回收及 Graph 查询恢复测试。

不代表真实 DSV4 attention、DSV4 模型数值正确性、TP8 NCCL 或性能已验收。
nvcc 12.0 与 PyTorch 的 CUDA 12.8 runtime 版本不同，记录为本次实际通过的
独立 kernel 测试环境，不据此替代内网的完整软件栈。

原始工作机目录结构：

```text
/workspace/dsv4-lease-validation-20261017/
  repo/                     # GitHub clone
  venv310/                  # 独立测试环境
  lean_cuda_plugin.py       # 本目录快照的原始位置
  metadata.json
  regression-a3a0d11.log
```

在该目录结构中复现：

```bash
cd /workspace/dsv4-lease-validation-20261017/repo
export PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH
export PYTHONPATH=/workspace/dsv4-lease-validation-20261017:experimental/hybrid-spec/src
export CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests -q --tb=short
```

辅助文件保留为 `.py.txt` 原样快照；复制到原始目录并命名为 `lean_cuda_plugin.py` 后使用。
辅助文件按原始位置解析 `repo/python`；在其他位置复现时，应保持这一同级布局。
完整 SGLang 环境的内网验收不加载此辅助文件，命令见
[验收步骤](dsv4-lease-lifecycle-20261017.md)。

验证结束后检查：测试进程已退出，GPU 显存 0 MiB；没有启动模型服务。

<a id="record-9"></a>

## 记录 9：dsv4-lease-lifecycle-20261017.md

来源：[dsv4-lease-lifecycle-20261017.md](dsv4-lease-lifecycle-20261017.md)。源文件 SHA256：`03a27d5da10f010a43e3662724c6d234fa0534c01dab971eaa66fbd6cac57ef1`。

### DSV4 SA lease 回收修复与内网验收

#### 源码结论

`58ae30ac69` 的 `begin_prefill` 用 `len(states)` 限制活跃 lease 数。
正常 decode 完成会调用 `BaseSpecWorker.note_request_finished`，插件随即
`close_request(..., retain=True)`：活跃 lease 从 `states` 删除，已安装的历史
才进入独立 `shared` 队列。因此 `states` 并非正常混入了所有 global-pool 来源。

实际漏口在 `SchedulerBatchResultProcessor.process_batch_result_prefill`：
请求若在 prefill 的第一个 token 就完成，该路径释放 KV，却没有调用 decode 的
finish hook。插件留下 lease，即使请求对象后来被销毁，也没有回收弱引用对应的槽。
这与 health/warmup 后 `max_requests=1` 永久失效的现象一致；具体内网遗留请求
身份没有远端证据，本次不声称已复核。c=4 同样会逐步丢失可用容量。

修复在每次非 idle 的 `begin_prefill` 分配前、锁内清理 finished/dead lease，
复用已有 `close_request`：pending build 被取消/丢弃，已安装且仍有请求对象的
完成历史按原规则保留到 shared。保留活跃容量限制，避免借用 shared 预留槽；
不在逐 token decode 路径增加 CPU 扫描，也不增加 GPU 同步。

新增 10 项回归：max_requests=1/4、global pool 开/关、历史已安装/未安装、
连续六次换入、存活请求保留、容量溢出、槽唯一性、generation 递增及弱引用失效。
修复前 10 项全失败，修复后全通过。

混合 SA/neural decode 的静态检查未发现新的确定错误：最终 gate/候选 TP 共识
在选择 compact tail 前完成；dispatch 排除 padding；每个 tail 大小独立维护
metadata/KV 计划，逐步写地址、hidden state、位置和原行 scatter 均有现有测试。
现有 CPU recurrence 通过不能替代真实 DSV4、CUDA capture 和 TP8 NCCL 验收。
本次不改 mixed-tail 算法，也不以吞吐或输出长度差异证明正确性。

历史参考：`model-pr-history-knowledge/sglang/deepseek-v4/README.en.md`，
包括 #26239 的 MTP Graph/metadata 约束；以当前 `deepseek_v4_backend.py`、
`compact_graph.py`、`device_extension.py` 和 scheduler 调用链为修复依据。

#### 本次工作机结果

修复提交 `a3a0d112b6ff02d48a57855afb7d68e3b964bf30` 已由 ai-workstation
从 GitHub clone/pull 后验证。RTX 4060 Laptop 8GB、WSL2/Linux、Python 3.10.18、
PyTorch 2.8.0+cu128、Triton 3.4.0、nvcc 12.0、GCC/G++ 12：
**218 passed、2 skipped，24.24 秒**，其中 CPU/Gloo 181 项、CUDA 37 项。
单卡跳过 NCCL 2/8 卡两项。结束后无测试进程，GPU 显存为 0 MiB。

8 项 compact CUDA 测试使用额外的轻量导入辅助：替换未执行的完整 Graph delegate
标记类，以及测试本来就要替换的合成 attention 容器边界；从该提交直接加载原生
topk Triton kernel 和 ForwardContext。SA CUDA kernel、stream、Graph capture/replay、
gather/scatter 均实际运行。这是合成模型边界的 GPU 契约测试，不是完整服务、真实
DSV4 attention 或 TP8 模型验收；其余 29 项 CUDA 测试不使用此边界替换。
新增 4 项 CUDA lease 用例在 max_requests=1/4、global pool 开/关下，各连续换入
6 批请求并重放查询，逐项检查命中 mask 与完整候选链。

原始日志、环境/源码哈希和辅助文件见 [4060 证据](dsv4-lease-4060/README.md)。
工作机运行目录：`/workspace/dsv4-lease-validation-20261017/`。

用户提供的 macOS Python 崩溃报告对应 21:53 的本地 Gloo 测试子进程。
同次测试日志为 `uv_bind: operation not permitted`，报告栈在
`gloo::transport::uv::Device` 调用 abort/SIGABRT；这是沙箱禁止本地 socket
绑定触发的进程退出，不是这轮 DSV4 CUDA 崩溃。沙箱外本地 Gloo TP8 已通过；
之后本次完整套件移到上述 Linux 工作机，避免重复触发 Mac 弹窗。

#### GPU / TP 回归

在完整更新的仓库根目录执行，插件和 SGLang 必须来自同一源码版本：

```bash
export PYTHONPATH="$PWD/experimental/hybrid-spec/src:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest experimental/hybrid-spec/tests -q -rs
# 8 张卡可见时明确执行；单卡跳过不能算通过。
python -m pytest 'experimental/hybrid-spec/tests/test_tp.py::test_tp_consensus_and_graph_staging[nccl-8]' -q -rs
```

RTX 4060 可以验证 SA CUDA kernel、辅助 stream、Graph replay、所有 mixed mask、
padding、gather/scatter 和 generation 隔离；单张 4060 不能验证 TP8 NCCL，
也不能运行 DSV4-Flash TP8 模型。test_cuda 的模型/attention 边界是合成计算，
即使通过也不代表真实 DSV4 attention 已通过。

#### 复用上一轮 500 条数据生成同参数对照

以下脚本读取已确认能运行的 plan，沿用模型路径、数据和服务参数。
生成 max_requests=1/c=1 与 max_requests=4/c=4 两份计划；各三轮，轮换
native/full/split 的启动顺序。full/split 的插件参数仅 graph mode 不同。

```bash
export BASE_PLAN=/workspace/validation-runs/dsv4-split-20261017-191157/plan.json
export RUN_DIR=/workspace/validation-runs/dsv4-lease-$(date +%Y%m%d-%H%M%S)
python - <<'PY'
import copy, json, os
from pathlib import Path

source = json.loads(Path(os.environ['BASE_PLAN']).read_text())
out = Path(os.environ['RUN_DIR'])
out.mkdir(parents=True, exist_ok=False)
native = next(s for s in source['servers'] if 'hybrid_config' not in s)
hybrid = next(s for s in source['servers'] if 'hybrid_config' in s)
config = hybrid['hybrid_config']
assert config['num_drafts'] == 8
assert config['num_neural_gate_tokens'] == 2
assert config['partial_drafts'] == 'fallback' and config['enable_global_pool']

def replace_option(command, flag, values):
    while flag in command:
        i = command.index(flag)
        j = i + 1
        while j < len(command) and not command[j].startswith('--'):
            j += 1
        del command[i:j]
    command.extend([flag, *map(str, values)])

for capacity in (1, 4):
    plan = copy.deepcopy(source)
    plan['concurrencies'] = [capacity]
    plan['servers'] = []
    for repeat, order in enumerate((('native','full','split'),
                                    ('full','split','native'),
                                    ('split','native','full')), 1):
        for mode in order:
            server = copy.deepcopy(native if mode == 'native' else hybrid)
            server['name'] = f'{mode}-r{repeat}'
            command = server['command']
            replace_option(command, '--max-running-requests', [capacity])
            replace_option(command, '--cuda-graph-max-bs-decode', [capacity])
            replace_option(command, '--cuda-graph-bs-decode',
                           [1] if capacity == 1 else [1, 2, 4])
            replace_option(command, '--speculative-draft-attention-backend', ['dsv4'])
            if mode != 'native':
                server['hybrid_config'].update(max_requests=capacity, draft_graph_mode=mode)
            plan['servers'].append(server)
    (out / f'plan-c{capacity}.json').write_text(json.dumps(plan, indent=2))

prompts = json.loads(Path(source['workloads'][0]['dataset']).with_name(
    'sa-spec-prompts.json').read_text())
(out / 'smoke-prompts.json').write_text(json.dumps(prompts[:32]))
print(out)
PY
```

先验证不 flush 的 c=1 泄漏场景。终端 A 使用上面生成的 split 配置启动：

```bash
python - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ['RUN_DIR'])
plan = json.loads((out / 'plan-c1.json').read_text())
server = next(s for s in plan['servers'] if s['name'] == 'split-r1')
config = out / 'smoke-config.json'
config.write_text(json.dumps(server['hybrid_config'], indent=2))
os.environ['HYBRID_SPEC_CONFIG'] = str(config)
print('URL:', server['url'], flush=True)
os.execvpe(server['command'][0], server['command'], os.environ)
PY
```

终端 B 设置相同的 RUN_DIR，BASE_URL 使用终端 A 打印的地址，等服务 ready 后执行：

```bash
export BASE_URL=http://127.0.0.1:39921 # 替换为 plan 的实际 URL
for i in $(seq 1 8); do
  curl -fsS "$BASE_URL/generate" -H 'Content-Type: application/json' \
    -d '{"text":"Hello","sampling_params":{"temperature":0,"max_new_tokens":1}}' >/dev/null || break
done
python experimental/hybrid-spec/benchmarks/compare_servers.py \
  --record "$BASE_URL" --prompts "$RUN_DIR/smoke-prompts.json" \
  --concurrency 1 --max-tokens 2048 --output "$RUN_DIR/smoke-c1.json"
```

此 smoke **不要加 `--flush-cache-before-run` 或手动 flush**。比较保存的
`server_info_before/after`：预期 retrieval_requests 有正增量，split skip_ct
有增长。全零说明未通过本轮已知 SA 可命中数据的激活检查，不能继续报告收益。
结束终端 A 服务后再跑正式计划（runner 自行启动、停止每一臂）：

```bash
python experimental/hybrid-spec/benchmarks/run_workload_plan.py \
  --plan "$RUN_DIR/plan-c1.json" --output-dir "$RUN_DIR/c1"
python experimental/hybrid-spec/benchmarks/run_workload_plan.py \
  --plan "$RUN_DIR/plan-c4.json" --output-dir "$RUN_DIR/c4"
```

c=4 至少检查：无请求失败/挂起/CUDA 错误，`mixed_ct > 0`，tail histogram
确实出现 1/2/3 行，`skip_request_ct > 0`。计数应满足
`request_ct = skip_request_ct + tail_request_ct`、
`prefix_ct = skip_ct + tail_ct = sum(tail_batch_histogram.values())`，以及
`tail_request_ct = sum(size * count)`。用每次服务启动后的同一统计窗口比较。
full/split 的接受质量应相近；真实模型逐 token 差异须另设正确性对照，不能由
greedy 长度差、吞吐或 CPU 测试推导等价。

<a id="record-10"></a>

## 记录 10：dynamic-compact-ab-4090-20261018/README.md

来源：[dynamic-compact-ab-4090-20261018/README.md](dynamic-compact-ab-4090-20261018/README.md)。源文件 SHA256：`862d6acdd1020846ac2e5d0070ba956732d5d961034f1114184c5eba9ad13e03`。

### RTX 4090：原生、固定 compact、动态 compact 吞吐对照

三轮交替顺序测量已完成：**54个阶段、5,184/5,184请求、1,327,104输出token**，
计时区间全部CUDA Graph回放。当前配置动态裁剪相对固定compact为负收益，三次配对均下降。
建议本负载保留固定compact；不能据此外推DSV4 MTP/TP8。

下表是预热后128请求段的三轮算术均值，每轮固定32,768输出token；实际验证行包含anchor和图padding。
完整标准差、首64请求段和逐轮数值见[report.md](dynamic-compact-ab-4090-20261018/report.md)、[summary.csv](dynamic-compact-ab-4090-20261018/summary.csv)与[summary.json](dynamic-compact-ab-4090-20261018/summary.json)。


|c|臂|吞吐tok/s|mean TPOT ms|实际验证行数|
|---:|---|---:|---:|---:|
|1|native|406.20|2.2652|33597|
|1|fixed|709.71|1.2067|34054|
|1|dynamic|679.30|1.2699|34060|
|4|native|1257.85|2.8937|33597|
|4|fixed|1914.87|1.8182|34077|
|4|dynamic|1826.31|1.9224|33964|
|16|native|3218.08|4.4060|33711|
|16|fixed|4153.48|3.3397|34721|
|16|dynamic|3893.17|3.5918|34569|

动态相对固定（按相同repeat配对求百分比，再取三轮均值）：

|c|吞吐|TPOT|实际验证行|逐轮吞吐变化|
|---:|---:|---:|---:|---|
|1|-4.28%|+5.24%|+0.02%|-4.98%/-3.21%/-4.66%|
|4|-4.62%|+5.73%|-0.33%|-4.87%/-4.22%/-4.78%|
|16|-6.27%|+7.55%|-0.44%|-6.75%/-6.76%/-5.29%|

模型为 Qwen3-0.6B target + 同权重 STANDALONE draft，FP16、TP1、Triton attention。
原生 K8；固定和动态均 SA8/neural3/gate1、split、global pool，使用相同的 72 个
有限 target token Graph 档位。配置默认 online min_samples/explore_interval=32。
此处不能外推 DSV4 MTP 或 TP8 的速度。

三轮启动顺序：native/fixed/dynamic；fixed/dynamic/native；dynamic/native/fixed。
并发顺序每轮轮换，各臂一致。每档先 flush KV、SA 历史和在线统计；启动阶段
64 条，随后稳定阶段 128 条，两组是互不重合的 edits 请求，平均输入约427/422 token。
每请求 greedy、ignore_eos、固定256输出，不请求 logprobs。两阶段分布不同，
不能把 cold/stable 差异单独归因于在线学习。吞吐与 TPOT 用官方 serving benchmark；
模型启动、编译、预热、诊断、进程导入和每阶段前的休眠均不在计时区间。

benchmark-only CPU 钩子为三臂直接计数 target 验证行，包含 anchor 和 Graph padding。
无逐轮 tensor readback；实际档位直方图、原生 dense 行数和 compact 原有计数器交叉校验。
同时报告每32768个稳定阶段输出 token 的实际验证总行数，以及每输出 token 的行数。

前置尝试单独保留，不计入最终三轮：

- flush-cache 接口成功响应是文本，测试脚本已兼容。
- 官方 benchmark 的逐请求字段需显式 `--output-details`，补上后才做完整长度/失败检查。
- 原生 STANDALONE 连续补入请求时发生真实 CUDA OOB：prefill state 不含 hidden_states，
  draft-extend Graph 却含 hidden_states，新旧请求合并后行数不足，filter_batch 越界。
  三臂统一采用 `standalone-extend-hidden.patch`：STANDALONE Graph 使用 NULL capture，
  回放保留 None；MTP/EAGLE hidden state 仍 clone。新增生产函数级 capture/replay/merge/filter
  回归；旧源码负向对照为预期的2失败/2通过，修复后4通过。
- 详细调试曾遇到 kernel-API level3 对 ForwardBatch repr 在 Graph 内同步的问题；
  改用 level1 与 launch blocking 仅用于定位，所有调试开关都不在正式测量中。

远端目录：`/root/rivermind-data/benchmarks/dynamic-compact-ab-20261018/`。
源码基线 c39a01d74d，正式测量仅统一叠加上述一份 STANDALONE 修复。
`run.py`、`dynamic_bench_counter.py`、`analyze.py` 是复现与统计脚本。


固定方案相对原生的平均吞吐提升约74.7%/52.2%/29.1%（c1/4/16），这是SA、split、
来源上限与compact组合的整体效果，不能全部归于target压缩。
动态在自身轮次的来源预算上确实裁掉约0.64%/1.66%/1.66%的逻辑行，但接受长度由
约8.13降至8.09/8.06/8.02，请求验证轮次相应增加。每输出token的实际验证量几乎没减少，
并承担了在线选择、计时、探索等开销；这里没有单独profile来分摊各项耗时。
该负载draft与target同权重，固定方案接受率高，可省验证量很少。
“预热后”是先64、后128请求的协议阶段名，不保证所有上下文成本档位已充分收敛。

修复最终4060回归：**857 passed、2 skipped，279.03秒**；两项跳过为NCCL2/8卡。
源码散列见standalone-hidden-metadata.json，完整输出见regression-standalone-hidden-v2.txt。
negative-control.txt中的2失败是旧源码对照的预期失败；当前修复对应4项均通过。
本轮源代码修复和回归用例保留在本地工作区，未提交推送。

原始证据evidence.tar.gz包含九次启动的完整命令、每阶段官方原始JSONL（逐请求长度、
文本、ITL、失败记录）、前后诊断、逐档计数、遥测、配置、数据及根因诊断日志。
下载后的SHA256已核对，formal54阶段中无失败、无eager回退；前置失败不混入正式统计。
GPU温度29～55°C，服务全部停止，最终显存1MiB。远端原始文件亦保留在上述目录。

<a id="record-11"></a>

## 记录 11：dynamic-compact-ab-4090-20261018/report.md

来源：[dynamic-compact-ab-4090-20261018/report.md](dynamic-compact-ab-4090-20261018/report.md)。源文件 SHA256：`3fa05368b32e4b166cecd32a088663b73901392c2815f9725fab9471d93d496b`。

### RTX4090 dynamic compact benchmark

All rows are means of three runs; parenthesized values are sample standard deviations. Target rows include anchor and CUDA Graph padding.

#### stable

|c|arm|output tok/s|mean TPOT ms|actual target rows|rows/output|
|---:|---|---:|---:|---:|---:|
|1|native|406.20 (0.40)|2.2652 (0.0022)|33597|1.0253|
|1|fixed|709.71 (3.77)|1.2067 (0.0050)|34054|1.0392|
|1|dynamic|679.30 (3.24)|1.2699 (0.0058)|34060|1.0394|
|4|native|1257.85 (2.39)|2.8937 (0.0040)|33597|1.0253|
|4|fixed|1914.87 (8.36)|1.8182 (0.0076)|34077|1.0399|
|4|dynamic|1826.31 (6.97)|1.9224 (0.0047)|33964|1.0365|
|16|native|3218.08 (7.56)|4.4060 (0.0103)|33711|1.0288|
|16|fixed|4153.48 (9.13)|3.3397 (0.0054)|34721|1.0596|
|16|dynamic|3893.17 (26.71)|3.5918 (0.0271)|34569|1.0550|

|c|dynamic vs fixed throughput|TPOT|target rows/output|throughput changes in run0/1/2|
|---:|---:|---:|---:|---|
|1|-4.28%|+5.24%|+0.02%|-4.98%/-3.21%/-4.66%|
|4|-4.62%|+5.73%|-0.33%|-4.87%/-4.22%/-4.78%|
|16|-6.27%|+7.55%|-0.44%|-6.75%/-6.76%/-5.29%|

#### cold

|c|arm|output tok/s|mean TPOT ms|actual target rows|rows/output|
|---:|---|---:|---:|---:|---:|
|1|native|406.74 (0.30)|2.2601 (0.0024)|16740|1.0217|
|1|fixed|716.82 (1.64)|1.1936 (0.0014)|17049|1.0406|
|1|dynamic|685.67 (3.20)|1.2574 (0.0068)|17044|1.0403|
|4|native|1315.50 (2.78)|2.7054 (0.0053)|16740|1.0217|
|4|fixed|1969.69 (26.49)|1.7340 (0.0169)|17038|1.0399|
|4|dynamic|1882.73 (10.85)|1.8276 (0.0062)|17025|1.0391|
|16|native|3325.00 (22.10)|4.1238 (0.0213)|16740|1.0217|
|16|fixed|4654.92 (32.87)|2.7285 (0.0050)|17205|1.0501|
|16|dynamic|4153.97 (31.20)|3.1165 (0.0031)|17185|1.0489|

|c|dynamic vs fixed throughput|TPOT|target rows/output|throughput changes in run0/1/2|
|---:|---:|---:|---:|---|
|1|-4.35%|+5.34%|-0.03%|-4.15%/-4.77%/-4.12%|
|4|-4.41%|+5.41%|-0.08%|-3.60%/-4.24%/-5.38%|
|16|-10.76%|+14.22%|-0.12%|-10.49%/-10.91%/-10.89%|


<a id="record-12"></a>

## 记录 12：dynamic-rows-gate-20261017.md

来源：[dynamic-rows-gate-20261017.md](dynamic-rows-gate-20261017.md)。源文件 SHA256：`b485a5387ce62cb10f03dc9605668df17cd364ff2280f638ff8ce306a7e7aa9e`。

### GPU 动态行号与可配置 split gate

#### 行为与边界

split 不再预枚举 `mask -> rows`。前缀完成 SA 查询和最终 TP 共识后，一个
Triton kernel 顺序扫描真实请求行，写出未命中的原始行号和一个 int32 行数。
每个 256 行 tile 内做前缀和，tile 间累加偏移；不做原子 append，保证 TP
各 rank 在相同 mask 下生成相同的原始顺序。padding 不参与选择。

行号缓冲区与 count 在捕获前分配，各大小 tail Graph 的 rows 是同一缓冲区
前缀的别名。运行时仅 CPU 读取 count 以选 Graph，无完整 mask D2H、无行号 H2D，
也不创建新的行号 tensor。路由存储随 B 线性增长；已去掉配置、can_run、
Graph 构造和 histogram 中的 batch≤4 硬编码。

每种捕获前缀仍为 1…B 行尾段分别捕获 Graph，并持有独立 metadata、暂存和
私有池。去掉硬上限并不消除这部分捕获开销/显存增长；请先用 B=8，再测 B=16。
本次不声称更大 batch 或动态行号一定提升吞吐。

`K=num_drafts=--speculative-num-steps` 是草稿 token 数，包含 draft-extend
已给出的首候选。`G=num_neural_gate_tokens` 必须是整数且 `0 <= G <= K`。

| K | G | 共享前缀 inner forward 次数 | 未命中行的 tail inner forward 次数 |
|---:|---:|---:|---:|
| 8 | 0 | 0 | 7 |
| 8 | 1 | 0 | 7 |
| 8 | 2 | 1 | 6 |
| 8 | 3 | 2 | 5 |
| 8 | 8 | 7 | 0 |
| 2 | 2 | 1 | 0 |
| 1 | 0 或 1 | 0 | 0 |
| 2 | 3 | 启动校验报错 | 不允许静默截短 gate |

G=0 时首候选起就可选 SA；G=1 只检查已有首候选；G≥2 时，先保留 G 个 neural
token，全部匹配才允许使用后续 SA。任一步失败都保持剩余链的 neural 递推。
G=K 有效但无可省略尾段，此时只回放 prefix，不构造/调度 tail，也不读回 count。
所有 gate 保持一次最终候选与 gate mask 的 TP 共识；G=0/1 没有 inner forward
可用于查询重叠。prefix/metadata 的原生 step id、hidden state 与位置恢复都随 G
移动，target verify 与原生 draft-extend 流程不变。

G=K 或 K=1 时 `skip_ct` / `skip_request_ct` 表示没有调度尾段，不能解释成 SA
命中；SA 来源统计仍看 `hybrid_spec`。`split_no_skip` 在存在尾段时计算全部实际行，
G=K 时同样只有 prefix。

worker、runner 与插件使用 split Graph API **v4**，版本不一致会启动报错。
SA CUDA 存储布局保持 v3。旧配置 G=2、B=1/4 保持可用。

#### 回归范围

CPU 测试直接执行当前 worker 的 `draft_forward` 函数体，覆盖 G=0/1/2/3/K、
K=1/2/8、命中/回退/混合、完整候选 materialization 与位置递推。
CPU compact recurrence 检查逐步 KV/metadata；dispatch 测试包含 B=1/8/32。

GPU 行号测试覆盖 B=1/8/33/257/4097、跨 tile、空实际批、padding、全命中、
全 neural、混合与 no-skip。GPU compact Graph 测试使用当前 worker 函数体、
原生 topk kernel、GPU 行号扫描和真正的 Graph capture/replay，覆盖 B=1/8 及 B=16/G=3、
K=1/2/3/8、G=0/1/2/3/K、Qwen/V4 元数据边界、非连续 request pool 顺序、
动态小批及私有池 allocator 压力。模型和 attention 计算仍为合成边界，不能替代
真实 DSV4 attention、权重或 TP8 NCCL 的内网验收。

#### 工作机验证结果

RTX 4060 Laptop 8GB / WSL2：**363 passed、2 skipped，44.76 秒**。
其中 CPU/Gloo 234 项，CUDA 129 项；Gloo 的 TP=1/2/4/8 均检查 G=0/1/2/3/8
的最终 gate 共识。CUDA 包含 B=16/G=3 的紧凑 Graph，以及跨 256 行 tile 的
稳定行号扫描。两项跳过是需要 2/8 张 NVIDIA GPU 的 NCCL 测试。
运行结束无测试进程，GPU 显存 0 MiB。

92 项 compact CUDA 用例使用合成模型/attention 和轻量导入辅助，执行精确
worker 函数体、原生 topk、动态行号 kernel 与实际 Graph；不作为真实模型通过。
软件栈为 Python 3.10.18、Torch 2.8.0+cu128、Triton 3.4.0、nvcc 12.0、GCC 12。
完整环境、命令、源码哈希及原始日志见 [4060 证据](dynamic-rows-gate-4060/README.md)。
工作机在提交前的隔离工作区验证，提交前核对全部 14 个修改 Python 文件 SHA256
与测试副本完全一致；没有先推送未验证的代码。

#### 内网验收

在已可运行 DSV4-Flash TP8 的环境更新 SGLang 与插件、重启服务后执行：

```bash
export PYTHONPATH="$PWD/experimental/hybrid-spec/src:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest experimental/hybrid-spec/tests -q -rs
python -m pytest 'experimental/hybrid-spec/tests/test_tp.py::test_tp_consensus_and_graph_staging[nccl-8]' -q -rs
```

不要加载 4060 的轻量导入辅助到真实内网服务验证中。先将现有 split 配置另存，
改为 B=8、G=3（保留当前模型、量化、global pool、gate 以外所有参数）：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('experimental/hybrid-spec/configs/device-dsv4-mtp-split-global-pool.json')
config = json.loads(p.read_text())
config.update(max_requests=8, num_neural_gate_tokens=3)
Path('/tmp/dsv4-split-b8-g3.json').write_text(json.dumps(config, indent=2))
PY
export HYBRID_SPEC_CONFIG=/tmp/dsv4-split-b8-g3.json
# 在已有成功的 DSV4 TP8 原生命令上覆盖：
# --max-running-requests 8
# --cuda-graph-max-bs-decode 8
# --cuda-graph-bs-decode 1 2 4 8
# --speculative-draft-attention-backend dsv4
```

基准用同一批 edits，分别在 c=1/4/8 运行。检查请求全部完成、无 CUDA/NCCL
错误/挂起；c=8 确实出现大于 4 行的 tail histogram，以及混合行进入紧凑尾段。
计数窗口内应满足 `request_ct=skip_request_ct+tail_request_ct`、
`prefix_ct=skip_ct+tail_ct=sum(histogram.values())`，
`tail_request_ct=sum(size*count)`。

随后逐一测试 G=0/1/2/3/8，并包含 K=1/G=0、K=2/G=1、K=2/G=2 的边界。
K=2/G=3 应在启动时拒绝。full/split 性能对照必须固定同一个 G 与 K，仅改变
graph mode；不同 gate 的接受质量不能混在一起解释成裁剪收益。
先短跑激活，再用相同 500 条、输出 cap=2048、至少三轮轮换顺序复测。

<a id="record-13"></a>

## 记录 13：dynamic-rows-gate-4060/README.md

来源：[dynamic-rows-gate-4060/README.md](dynamic-rows-gate-4060/README.md)。源文件 SHA256：`615ded2c2962752573ce348f3d5b080a3f10e972f7fef243b6c041e9dec41df5`。

### 动态行号 / gate：RTX 4060 证据

测试源码以 `993fd124c5` 为基线，在工作机隔离 checkout 应用当前源码补丁，
完成验证后才提交。14 个修改 Python 文件（含新 kernel）的精确 SHA256 保存于
[metadata.json](dynamic-rows-gate-4060/metadata.json)，提交前逐项与本地源码核对一致。

[最终原始输出](dynamic-rows-gate-4060/regression.txt)：**363 passed、2 skipped，44.76 秒**。
CPU/Gloo 234 项，单卡 CUDA 129 项；跳过的两项为 NCCL 2 卡与 8 卡。
RTX 4060 Laptop 8GB、WSL2、Python 3.10.18、Torch 2.8.0+cu128、Triton 3.4.0、
nvcc 12.0、GCC/G++ 12。nvcc 与 Torch CUDA runtime 版本不同，记录为这次实际
通过的独立功能测试环境，不据此替代内网完整 serving 软件栈。

92 项 compact Graph 用例覆盖 B=1/8 的 K/G 矩阵及 B=16/G=3；执行实际 worker
函数体、原生 topk Triton kernel、当前 GPU 行号扫描与生产 gather/capture/replay/
scatter。模型/attention 计算是合成边界；额外导入辅助仅替换未执行的 full Graph
delegate 标记类型与测试本来就要替换的 attention 容器导入。
其余 37 项 CUDA 不使用这个 fixture 的替换分支，包含 10 项行号扫描用例。

辅助文件与上一轮的[原样快照](dsv4-lease-4060/lean_cuda_plugin.py.txt)完全相同，
SHA256 也记入元数据；仅其所在的独立目录改变。它从 `repo/python` 直接加载
原生 topk 和 ForwardContext，actual worker body 由测试读取同一 checkout。
普通内网完整环境不要使用该辅助文件。

原始工作机命令（venv 复用上一轮的独立环境）：

```bash
cd /workspace/dsv4-dynamic-20261017/repo
export PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH
export PYTHONPATH=/workspace/dsv4-dynamic-20261017:experimental/hybrid-spec/src
export CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests -q --tb=short -rs
```

辅助文件位于 `/workspace/dsv4-dynamic-20261017/lean_cuda_plugin.py`，
原始日志为该目录的 `regression.log`。新位置复现时保持辅助文件与 `repo/` 同级。
验证后测试进程退出，GPU 显存 0 MiB，没有启动模型服务。

这轮未运行 DSV4 权重、真实 dsv4 attention、TP8 NCCL 或端到端吞吐/正确性评测；
[内网验收步骤](dynamic-rows-gate-20261017.md)继续覆盖这些项目。

<a id="record-14"></a>

## 记录 14：dynamic-verify-4060-20261018/README.md

来源：[dynamic-verify-4060-20261018/README.md](dynamic-verify-4060-20261018/README.md)。源文件 SHA256：`15a84997ce6e2b909aa488c4ba3b53c1e21efa17f9056ca6166611fe0aa4f287`。

### 在线动态 compact verify — RTX 4060 验证

最终完整回归：**853 passed、2 skipped，277.63 秒**。跳过项为 NCCL 2 卡和
8 卡测试；结束后 GPU 显存 0 MiB。仅最终 v8 运行计入通过数量。

`metadata.json` 保存环境、基线 `9b88ffa222`、补丁 SHA256 和 25 项源码散列；
`regression.txt` 为完整输出。工作机独立目录：
`/workspace/dynamic-verify-fast-20261018/repo`。

覆盖 target 反馈标签、截断尾部、首个拒绝、neural/local/shared 来源、冷桶与探索、
完整周期异步计时、flush/idle/prefill、逐轮 GPU/CPU 选择与计数一致性，以及
候选位置超过 256 时的回退。新增六项初始化回归覆盖 draft 子类跳过 target 构造器的捕获流程，以及 attention backend 创建后的能力检查。混合批次、重排/缩批、gate=3/neural=2、
gate=8/neural=1、CUDA Graph token 档位切换沿用原生打包、验收、提交及 attention
算子执行；神经模型层为合成层。官方 Arctic suffix 和真实 Gloo 1/2/4/8 进程组也在回归内。

复现：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/arctic-suffix-20261017/ArcticInference:/workspace/dynamic-verify-fast-20261018:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests \
  -q --tb=short -rs --junitxml=../regression-v8.xml
```

轻量导入辅助沿用[已保存版本](dsv4-lease-4060/lean_cuda_plugin.py.txt)，仅用于
合成模型测试。此次实际运行 Python 3.10.18、Torch 2.8.0+cu128、Triton 3.4.0。

`overhead.py` / `overhead.json` 是融合实现的 TP1 合成选择+反馈微测：batch=1/4/16
中位数约 232/323/226 µs，包含最终总数读回，不含 target 模型或 TP 通信。
这是 v5 专项之后的初步测量；后续版本调整了配置校验和 serving 初始化流程，微测涉及的选择/反馈算子未变化。
不能据此推算服务吞吐、H100 或 TP8 收益。

完整 SGLang 服务、真实模型 greedy 输出、DSV4 压缩 KV 与 TP8 NCCL 不在本次组件
验收范围。4090 服务验证单独记录；DSV4 TP8 仍需内网运行。
配置与测量口径见[动态验证说明](../DYNAMIC_VERIFY.md)。

<a id="record-15"></a>

## 记录 15：dynamic-verify-4090-20261018/README.md

来源：[dynamic-verify-4090-20261018/README.md](dynamic-verify-4090-20261018/README.md)。源文件 SHA256：`4af33cc948947cd1014631f9a37ee171072093b9ff58d6fd49b35365d9f29999`。

### 动态 compact verify — RTX 4090 真实服务

Qwen3-0.6B 同时作为 target 和 STANDALONE draft，FP16、TP1、Triton attention。
GPU 标称 RTX 4090，实际显存 49,140 MiB；环境 Torch 2.14.1+cu130、Python 3.12。
基线 `9b88ffa222` + 本轮补丁；25 项最终源码散列见 `summary.json`。

五组各完成 53 条请求、7,552 个输出 token，总计 **265/265**：

| 配置 | 验证轮数 | 成本样本 | 验证标签 | 混合批次 | 实际 token 图档数 |
|---|---:|---:|---:|---:|---:|
| 原生 STANDALONE K8 | — | — | — | — | — |
| 固定 compact SA8/neural3/gate1 | 165 | — | — | 84 | 28 |
| 动态 compact SA8/neural3/gate1 | 167 | 161 | 6,573 | 95 | 25 |
| 动态 SA8/neural2/gate3 | 203 | 197 | 6,408 | 0 | 22 |
| 动态 SA8/neural1/gate8 | 298 | 292 | 5,938 | 0 | 26 |

覆盖 c=1/4/16；c16 连续三波，请求输出长度交替为 96/128/160/192，包含缩批、
重排与来源混合。普通动态组使用默认冷桶/探索参数 32；两个 gate 边界组调为 8。
gate3/neural2、gate8/neural1 分别实际接受 777/767 次 SA 草稿；本负载这两组
未形成同一步混合，强制混合边界由 4060 原生算子回归补齐。
服务使用 `--disable-overlap-schedule --chunked-prefill-size -1`；本次真实服务不覆盖 overlap/chunked。

动态与固定 compact 的输出 token **53/53 完全一致**。所有 compact 组相对原生
均为 52/53 一致，唯一首分歧在 c4 的第 2 行、第 3 个输出 token：We / The。
额外重复固定 compact 并导出 top-5：二者在该路径 FP16 分数完全相同，
logprob 均为 -1.8312536478042603；640 个补查 token 均达到对应 target 最大分数。
Top-k 在同分元素间的排列不能当作 argmax 的 tie-break 规则。
独立 Transformers FP32 eager 参考更偏向 We，top-2 margin 为 0.01766777。
这些证据支持数值/同分分支的解释，不能据此宣称原生与 compact 逐 token 恒等。
原始证据见 `target-argmax-audit.json`、`divergence-reference.json` 和归档内 top-5。

真实启动发现并修复两处旧 compact 接入问题：draft runner 绕过父构造器时缺少
`ragged_verify_mode`；插件在 attention backend 尚未初始化时读取其能力。
4060 最终完整回归包含对应六个 CPU 流程用例，共 853 passed、2 skipped。

`evidence.tar.gz` 保存每组命令、配置、原始响应、服务日志、诊断及实际执行脚本。
历史 `run.py` 含最初的恢复旧服务逻辑；按用户后续要求，旧 Qwen3.6 主模型和
MTP 权重已删除、恢复取消，释放 24,465,334,272 字节。不要执行该历史入口。
`validate_service.py` 是整理后的五组复现脚本，只管理其自行启动的测试服务；
先准备对应源码、依赖和模型，确认 39923 端口空闲。模型散列见 `model-hashes.json`。
结束后测试服务全部停止，4090 显存 1 MiB；小模型保留。

这是正确性与调度 smoke。记录了单次请求耗时，但没有公平吞吐基准或性能收益结论。
DSV4 MTP 压缩 KV、TP8 NCCL、真实多卡档位对齐及吞吐仍需内网验证；EAGLE3/MTP
本轮由组件回归覆盖，不能由 STANDALONE 服务替代其端到端验收。

<a id="record-16"></a>

## 记录 16：global-pool-6000d-throughput/report.md

来源：[global-pool-6000d-throughput/report.md](global-pool-6000d-throughput/report.md)。源文件 SHA256：`a4ece7d74e0711ff9a9056ef0bbb9cb759acf24a3d7b3d21d68ef6594a821bea`。

### RTX 6000D 跨请求 SA 吞吐对照

本轮结论：单独开启 full 全局池，重复请求基本持平或小幅波动，不同代码请求慢约 7%–9%。
在相同 split 模式下比较开/关池，共享历史对重复请求额外提升约 2.6%–6.4%，对不同代码请求则降低约 6.1%–8.7%。
因此 split 相对 full 基线的总提升应与共享历史的独立收益分开看待。

Qwen3.6-35B-A3B-FP8，单卡 TP=1，K=8，gate=2，min_match_len=4。
全局池保留 16 个完成请求，4 个活跃槽位，H=4096，外部上下文上限 128。
每个测量点 80 请求，固定生成 128 token，greedy、ignore_eos=True；KV 前缀缓存统一关闭。
官方 sglang.benchmark.serving 流式客户端，burst 闭环负载，并发 1/4。无指定 SLA。
三轮测量取中位数，范围为三轮最小/最大值；收益为相同轮次相对 full-off 的比例。
full-off 保留本请求 SA 和 MTP，仅关闭跨请求全局池。

unique-cold：80 个不同的真实代码编辑输入，空池起步。
repeat-cold：其中 16 个输入各重放 5 次，空池起步，建池成本包含在计时内。
repeat-warm：同样的 80 请求，之前通过 16 个真实生成请求预先建池，预热不计入主表。
重复场景是刻意允许内容复用的对照，不能代表普通生产流量的重复率。
未将数据集参考答案放入输入或历史；原始输入索引、长度和哈希见 provenance.json。

| 负载 | 并发 | 配置 | 轮数 | output tok/s 中位数 [范围] | 对 full-off 收益 [范围] | TTFT ms | TPOT ms | SA decode 行占比 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repeat-cold | 1 | full-global | 3 | 337.98 [334.11, 339.25] | +0.1% [-1.1, +0.5] | 58.61 | 2.52 | 71.7% |
| repeat-cold | 1 | full-off | 3 | 337.69 [337.64, 337.84] | +0.0% [+0.0, +0.0] | 57.73 | 2.53 | 53.2% |
| repeat-cold | 1 | split-global | 3 | 396.79 [389.37, 398.29] | +17.5% [+15.3, +17.9] | 54.21 | 2.11 | 71.7% |
| repeat-cold | 1 | split-off | 3 | 382.83 [382.63, 382.88] | +13.3% [+13.3, +13.4] | 58.22 | 2.17 | 53.2% |
| repeat-cold | 4 | full-global | 3 | 770.79 [751.07, 773.89] | +0.0% [-2.0, +0.4] | 96.65 | 4.37 | 71.2% |
| repeat-cold | 4 | full-off | 3 | 770.44 [766.59, 771.05] | +0.0% [+0.0, +0.0] | 95.36 | 4.41 | 53.7% |
| repeat-cold | 4 | split-global | 3 | 798.70 [789.43, 806.57] | +3.7% [+3.0, +4.6] | 94.94 | 4.23 | 70.8% |
| repeat-cold | 4 | split-off | 3 | 780.19 [771.62, 786.42] | +1.8% [+0.2, +2.0] | 96.30 | 4.36 | 53.1% |
| repeat-warm | 1 | full-global | 3 | 338.03 [337.10, 342.21] | +0.4% [-0.2, +1.3] | 58.60 | 2.52 | 73.5% |
| repeat-warm | 1 | full-off | 3 | 337.73 [336.73, 337.82] | +0.0% [+0.0, +0.0] | 57.74 | 2.53 | 53.2% |
| repeat-warm | 1 | split-global | 3 | 397.99 [395.95, 404.57] | +18.2% [+17.2, +19.8] | 54.12 | 2.10 | 73.5% |
| repeat-warm | 1 | split-off | 3 | 382.96 [382.32, 383.83] | +13.4% [+13.2, +14.0] | 58.30 | 2.17 | 53.2% |
| repeat-warm | 4 | full-global | 3 | 785.42 [771.39, 799.68] | +2.0% [-0.5, +4.3] | 101.54 | 4.28 | 74.4% |
| repeat-warm | 4 | full-off | 3 | 770.36 [766.50, 775.52] | +0.0% [+0.0, +0.0] | 95.30 | 4.41 | 53.7% |
| repeat-warm | 4 | split-global | 3 | 822.64 [810.89, 829.24] | +6.8% [+4.6, +8.2] | 98.31 | 4.10 | 74.7% |
| repeat-warm | 4 | split-off | 3 | 779.07 [771.55, 786.54] | +1.4% [+0.2, +1.6] | 96.20 | 4.37 | 53.1% |
| unique-cold | 1 | full-global | 3 | 305.52 [304.71, 305.83] | -8.8% [-9.0, -8.7] | 60.40 | 2.82 | 54.3% |
| unique-cold | 1 | full-off | 3 | 334.88 [334.85, 335.10] | +0.0% [+0.0, +0.0] | 58.08 | 2.55 | 50.8% |
| unique-cold | 1 | split-global | 3 | 344.75 [344.33, 344.78] | +2.9% [+2.8, +3.0] | 55.88 | 2.48 | 54.3% |
| unique-cold | 1 | split-off | 3 | 377.60 [377.28, 378.24] | +12.7% [+12.7, +13.0] | 58.47 | 2.20 | 50.8% |
| unique-cold | 4 | full-global | 3 | 706.14 [705.57, 708.95] | -7.0% [-7.6, -6.8] | 97.08 | 4.86 | 55.6% |
| unique-cold | 4 | full-off | 3 | 760.69 [759.43, 763.70] | +0.0% [+0.0, +0.0] | 95.56 | 4.48 | 52.8% |
| unique-cold | 4 | split-global | 3 | 716.04 [710.51, 725.64] | -5.9% [-7.0, -4.4] | 95.76 | 4.82 | 54.4% |
| unique-cold | 4 | split-off | 3 | 768.38 [756.45, 772.27] | +1.2% [-1.0, +1.5] | 94.54 | 4.43 | 51.5% |

TTFT/TPOT 列为每轮均值的中位数；p99、接受长度、split 分支计数见 summary.json。
SA 占比是采用检索候选的 decode 请求行比例，不是端到端请求命中率。
不把吞吐变化直接解释为模型准确率变化。完整输出文本对照见 output-parity.json。
文本对照共 4320 对请求，其中 636 对文本不同。

repeat-warm 的 seed_inclusive_tok_s_median 将 16 个建池请求和 80 个测量请求一起计入。
模型启动、Graph capture、JIT 和通用预热不计入吞吐；每轮负载前 flush，暖池场景随后再建池。
原三组顺序执行，随后补 split-off 控制；每组内部随机排列场景与并发顺序，温度、功耗、时钟记录在每点 diagnostics。
summary.json 的 global_pool_gain_pct_* 比较相同 Graph 模式下开/关全局池，避免把 split 自身的收益归于共享历史。
这是固定配置与固定长度的吞吐对照，没有搜索最佳参数，也没有测生产请求分布或任务准确率。

完整启动命令及配置见 plan.json、*-command.json、*-config.json。
官方原始结果为 *-r*.jsonl；每请求延迟、文本、错误和缓存统计均保留。

补充 split-off 控制曾因容器进程重置中断；中断前的 11 个点另行保留并排除，完整三轮重新测量。
原三组 54 个点在中断前已完成；GPU UUID、驱动、源码和参数均核对一致，中断原因未确认。

本地交付保存了汇总、逐轮数据和 normalized.jsonl（内含每点启动命令、请求命令和配置）。
完整原始响应、延迟、日志、源码哈希、帮助输出及被排除的中断数据保存在远端：
`/root/rivermind-data/validation/global-pool-throughput-20261017-evidence.tar.gz`。
本地副本：[原始证据归档](/private/tmp/sglang-global-pool-throughput-6000d-evidence.tar.gz)。
SHA-256：`d40b1d043b90c98f63f98c6d2c81202db391813323ff8411b890ba371b327e72`。
有效点数、最终显存和源码核对见 verification.json；中断说明见 interruption.json。

<a id="record-17"></a>

## 记录 17：global-pool.md

来源：[global-pool.md](global-pool.md)。源文件 SHA256：`f310434a139a931fd1218196e0a8c83ce0ebdbdbaffe7a9498ecbfdd457f471d`。

### 跨请求 SA 全局池

日期：2026-10-17。开发基线：`2d53a3d2d2`，分支 `codex/hybrid-suffix-speculation`。

#### 参考源码

本地 Baseten `sa_spec` 固定提交 `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969` 的
`src/api.cc` 和 `src/api.cu` 仍按 request ID 分配独立 SA；全局分配器不是跨请求搜索。

本轮重点读取 TensorRT-LLM 官方 main（核查时为
`7ebcf4debcac29b3f6225cb4affff3ecf05514aa`）的：

- [`suffix_automaton.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/tensorrt_llm/_torch/speculative/suffix_automaton.py)：`SAConfig.enable_global_pool`、池大小与活跃/保留槽位、`_allocate_slot` FIFO 淘汰、`remove_request` 跳过未完成导入的槽位、`extend_global`。
- [`suffixAutomatonKernels.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomatonKernels.cu)：先更新全部 SA，再只读跨槽位查询；匹配长度、本请求优先、续写长度的排序。
- [`sa_enhancer.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/tensorrt_llm/_torch/speculative/sa_enhancer.py)：当前类名为 `SADraftEnhancer`，独立 stream 更新/查询，最后统一覆盖 neural draft。
- [`suffixAutomaton.h`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomaton.h)：`lookupWithSuffix` 的外部后缀遍历，以及匹配落在来源末尾时沿 suffix link 回退到可续写的更短匹配。

读取文件的 SHA-256 分别为
`109d0b38cbc43a30555b8e6aa0a7d788ae6ac44111bfbfc3b1d563fc53a125c0`、
`6e5c79686d357f3aac2faad3c4f5a5801e7833bf7eb21a059520c8cd0a504abd`、
`45941e22f67b0304280970a0657f07de7f55752d1b47d767f28ba767face1848`。
末尾回退参考头文件的 SHA-256 为
`3b5e12900a38b2e973ffea127e679168ad5a196ff763bca69a1b9e2dcfc1ecb0`。
实现独立编写，没有复制或运行时依赖 TensorRT-LLM 源码。

已有 Qwen3.5 PR 历史记录和本分支 `OPTIMIZATIONS.md` 一并复核。历史 PR 仅用于提示
native verify、MTP 递推和 split 的验证边界；本轮不修改模型 forward、位置或 KV 提交。

#### 本轮实现和差异

- 开关默认关闭，仅支持 device 历史。`max_requests` 限制活跃 SA；额外
  `max_shared_requests` 限制完成历史。后者固定预留并在完成时 FIFO 淘汰，
  与 TRT 在总池分配压力下才淘汰的策略不同。
- 搜索已导入的活跃槽位和保留槽位。先消费所有发布的反馈，随后只读搜索；
  无跨 block 的边更新边读，不在 scheduler 回调里启动 CUDA。
- 每个来源保持独立 SA，无跨请求拼接。跨来源上下文上限独立配置；本请求原查询不截断。
  最长匹配若没有后续 token，沿 suffix link 查找可续写的更短后缀。
  先按现有 `partial_drafts` 资格过滤，再比较匹配长度；相同长度优先本请求，
  外部来源优先更长续写，最后按较新 generation 决定。这不宣称逐项复制 TRT 的选择策略。
- 完成历史携带 CPU 确认的最终长度。最后增量可截断后追加；已经索引的丢弃尾部不能原地
  回滚，因此放弃该来源。超容量、无版本匹配、旧 generation 均回退。
- Graph 使用固定地址的槽位、generation、最终长度输入，来源变化也会使 staging key
  失效。flush 清除所有来源，ID 重用不继承旧所有权。
- 继续使用本分支的 forward stream、逐位置候选选择、neural gate、TP 共识和原生
  target verify。没有引入 TRT 的 side stream 或循环结束统一覆盖行为。
- 跨槽位查询参考 TRT 改为每请求一个 block、线程并行扫描来源、共享内存归约。
  block 为 32/64/128 线程，超过 128 个来源由线程分步扫描；比较键为匹配长度、
  未截断续写长度、发布索引，保证跨线程同分结果与原串行扫描一致。
  相同匹配仍优先本请求，完整度资格先于排序判断。
  更新/查询仍在原 forward stream 按 kernel 边界排序，可用于 full 和 split。

池的作用域是同一 worker 的所有 opt-in 请求，包含 prompt 和 target-confirmed 输出。
无租户隔离或持久化。启用方必须允许这些请求之间共享内容。

#### 验证

后续并行扫描版本已在 RTX 6000D 完整回归：182 passed、1 skipped（76.60 s），
其中 26 个 CUDA 用例全部通过。新增来源并行归约的独立文本 oracle、20/64/257 来源、
跨线程与跨步同分排序、两种 partial 策略、Graph 重放测试。
串行/并行微基准的 12 组输入、候选 token 和 mask 完全一致。
full 与 split 的同日吞吐对照及完整证据见
[并行扫描专项记录](sa-parallel-6000d-throughput/report.md)。

以下保留并行化之前的全局池实现验收和开/关池对照。

测试覆盖外部后缀与独立穷举 oracle 的差分、完成/未导入/重复关闭、FIFO、
稳定活跃 batch 下的 Graph 来源更新、flush、padding、旧 generation、最后 verify
反馈的幂等消费、跨文档边界、活跃请求互查和停止尾部截断。

本地 CPU、原生 C++、真实 draft 递推函数体：153 passed。
A100 PCIe 40GB，Torch 2.14.1+cu130，最终完整包回归：175 passed、1 skipped（103.85 s，包含重编译）；
唯一跳过项为需要双卡的 NCCL，用 Gloo 的 1/2/4 rank 共识测试通过。
19 个 CUDA 用例全部通过，包含已有 full/split/compact 回归和 4 个新全局池用例。
补充外部来源的续写长度排序断言后，两个相关 Graph 用例再次通过。
最终全量回归包含来源末尾的 suffix-link 回退断言。命令和源码哈希保存在
[`global-pool-a100.json`](global-pool-a100.json)。

##### 真实服务短对照

沿用 A100 上已验证的 `Qwen3.5-35B-A3B-GPTQ-Int4`、FP16/gptq_marlin、
unquantized MTP、TP=1、K=8、Triton draft/GDN 后端。native 源码为 `972aa16098`，
与本轮 `2d53a3d2d2` 基线仅差文档。本轮插件通过独立 PYTHONPATH 加载，
使用临时端口 39981 和隔离的 CUDA 编译缓存。28 个生产源码文件 SHA-256
与本地工作区逐项一致；重新加载包含最终 suffix-link 回退的新 CUDA 库后再测。

配置为 full、greedy、neural gate=0、2 个活跃槽位 + 2 个保留槽位、H=4096，
跨请求上下文上限 128。先完成一个最长 256 token 的代码生成请求，
再对相同 prompt 请求 64 token；随后 flush，再用相同 prompt 请求 64 token。

| 观测 | 保留首请求后的第二次请求 | flush 后的新请求 |
| --- | ---: | ---: |
| 检索来源的 decode 请求行数增量（并发 1） | 7 | 1 |
| 本样本返回文本 | 与右侧一致 | 与左侧一致 |

三次请求均成功。该差异支持保留历史在实际 worker/完成回调/Graph replay 中生效，
而不只是构造人工 CUDA 输入得到候选。flush 后仍有 1 次命中来自新请求自身的重复，
不应将其误报为池未清空。原始响应、配置和累计计数见
[`global-pool-a100-smoke.json`](global-pool-a100-smoke.json)。临时服务完成后停止，
A100 显存回到 0 MiB。这里只有一组短输入对照，不证明一般 token 等价、任务准确率
或吞吐收益；没有运行新的 GSM8K 或正式性能基准。

##### RTX 6000D 复验

2026-10-17，在 `hz01-ssh.gpuhome.cc:30167` 的单卡 RTX 6000D（SM120，85,651 MiB）
继续验证。Torch 2.14.1+cu130、CUDA Toolkit 13.0；当前工作区源码上传到独立目录，
5,616 个 Python/runtime 与插件源码文件逐项 SHA-256 一致。原服务器 checkout
`53e1300` 保持干净，服务通过 PYTHONPATH 使用本轮 `2d53a3d2d2` 基线及未提交改动。
这次短功能复验选择了 Python TreeCore fallback，没有据此测量性能。

完整包回归为 **175 passed、1 skipped，76.26 s**，包含首次 SM120 CUDA 编译。
19 个 CUDA 用例全部通过；唯一跳过项仍为双卡 NCCL，Gloo 1/2/4 rank 通过。

实际模型使用本机已有的 `Qwen3.6-35B-A3B-FP8`，TP=1、K=8、两个活跃槽位、
两个完成历史槽位、H=4096、上下文上限 128。沿用上述短代码 prompt，每个配置独立
启动服务，先完成最长 256 token 的请求，再请求 64 token，flush 后再请求 64 token。
除 full/global-pool 的最初三请求复验外，其余配置再并发发送两个最长 128 token 的请求。
共 18 次生成全部成功。

| 模式 | neural gate | 保留历史后命中 | flush 后命中 | 并发 2 |
| --- | ---: | ---: | ---: | --- |
| full，全局池开启 | 0 | 7 | 0 | 初次短复验未运行 |
| full，全局池关闭 | 0 | 0 | 0 | 通过 |
| split_no_skip，全局池开启 | 2 | 7 | 0 | 通过 |
| split，全局池开启 | 2 | 7 | 0 | 通过 |

各配置三个串行请求的完整 output token IDs 与关闭全局池的对照一致；两个 split
配置的并发请求也逐项与该对照一致。这里只覆盖这一个短输入，不能推导一般 token
等价或任务准确率。

split 在第二个请求结束时已有 7 次 tail skip；全部请求结束后累计 14 个 batch、
21 个请求行跳过尾部，而 split_no_skip 的 skip 为 0。并发 split 本样本的 mixed_ct=0，
真实服务没有覆盖“一行命中、一行回退”的混合路径，该路径仍以组件测试为证据。
full 模式继续执行 MTP；这里的 skip 证据仅属于显式启用的 split 模式。

最初补测脚本误设 split gate=4，被已有配置校验拒绝；修正为要求的 gate=2 后完成上述
两轮验证，没有修改生产实现。原始配置、启动脚本、token 响应、哈希和计数索引见
[`global-pool-6000d.json`](global-pool-6000d.json)。远端完整日志、脚本和源码归档保存在
`/root/rivermind-data/validation/global-pool-20261017/`。临时服务已停止，显存回到 0 MiB。
上述短功能复验没有吞吐基准、GSM8K 或真实多卡 NCCL 结果。

##### RTX 6000D 吞吐对照

后续在相同 GPU 和模型上使用官方 `sglang.benchmark.serving` 流式客户端，对比
full/关池、full/开池、split/开池，并补充 split/关池以分离两项优化的收益。
统一 K=8、gate=2、4 个活跃槽位、16 个完成历史槽位（开池时）、H=4096；
所有配置关闭 KV 前缀缓存。每个测量点为 80 请求、固定生成 128 token，
并发 1/4，各重复三轮。使用真实代码编辑数据，另将其中 16 个输入各重复五次，
分为空池起步和通过真实请求预先建池两种情况，未导入参考答案。

full 模式开启全局池后，不同代码请求的吞吐中位数降低约 7%–9%；重复请求冷池
基本持平，暖池有小幅且波动的变化。split 的改善依赖负载，不能把其全部收益
归于跨请求共享。补充 split/关池控制后，跨请求共享本身在重复请求冷池的吞吐
中位增益为 3.7%/2.6%，暖池为 3.7%/6.4%（并发 1/4），而不同代码请求为
-8.7%/-6.1%。这些百分比为相同轮次比值的中位数；各组绝对吞吐中位数的比值
可能略有不同。

共 72 个有效测量点、5,760 个请求、737,280 个输出 token 全部完成；split/开池
覆盖了 2,493 个真实 mixed batch。四组控制、轮间范围、TTFT/TPOT、候选接受长度和
分支计数见[吞吐报告](global-pool-6000d-throughput/report.md)。临时服务均已停止，
最终显存为 0 MiB，原服务器 checkout 保持干净。补充控制曾因容器进程重置中断，
中断前 11 个点另行保留并排除，随后完整三轮重跑；原三组 54 个点已在中断前完成。

较大样本中，不同配置相对 full/关池的 4,320 对输出有 636 对文本不同，与上面的
单个短输入对照不同。这包括 split/关池的差异，不能全部归因于共享池。这些测量
只说明固定输入和输出预算下的吞吐表现，不能作为一般输出等价或任务准确率证明；
差异索引和样本保存在报告目录的 `output-parity.json`。本轮没有修改生产实现。

模型输出一致性、真实多卡 NCCL 和端到端性能与组件测试分开验收，
不沿用旧 C1/C4 性能数字作为本轮证据。

本轮增加 CUDA 算子参数和 kernel，布局仍为 v3；更新源码后必须重启服务并完成
JIT 重新编译，不能沿用上一轮纯 Python 改动“不需重编译”的说明。

<a id="record-18"></a>

## 记录 18：gpuhome-gate1-c16-profile-20261018/history-notes.md

来源：[gpuhome-gate1-c16-profile-20261018/history-notes.md](gpuhome-gate1-c16-profile-20261018/history-notes.md)。源文件 SHA256：`0523eeaa7c05eaf0ecfec83c1ac4f2b1d23c1e09a9505354b2af33e42ea4b80e`。

Read model-pr-history-knowledge/sglang/qwen35/README.en.md and query results. #20864 removed H2D in Qwen3.5 SpecV2; inspect remaining per-forward metadata/binding copies in pinned049 rather than assume old host paths still apply. #27869 records batch-invariant numerics work; these captures do not certify output equivalence. Read current skill fuse-overlap-catalog and overlap-catalog: draft topk1 and Mamba scatter/commit fusions already exist; TP1 has no NCCL opportunity. Generic alt-stream and persistent-head precedents are analogies, not enabled paths here. Source-contract resource referenced by history skill was absent in earlier session; actual pinned source and measured kernel signatures are used.

All serving snapshots are git show049db08e4; concurrent local working-tree changes were excluded. Automatic graph-internal CPU source mapping is unreliable and its raw triage is explicitly unreviewed. Direct graph-launch scopes, actual projection signatures, and Nsight phase markers drive the report.

<a id="record-19"></a>

## 记录 19：gpuhome-gate1-c16-profile-20261018/report.md

来源：[gpuhome-gate1-c16-profile-20261018/report.md](gpuhome-gate1-c16-profile-20261018/report.md)。源文件 SHA256：`12fe0f44436805edf404c91be9940180644f1864286a855ac0498aad2d8f0d2e`。

### split gate=1：16并发 Hybrid / 原生MTP profile对照

结论：压缩确实省了一部分MTP主体工作，但7次全词表head几乎不变，neural图只快约1–1.4ms。共同的target verify约40ms，使剩余收益成为1%级。没有发现足以把整秒同步等待当成额外开销的证据。

模型palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4；4090 48GB、TP1、EAGLE内置MTP、正确unquant draft、K8、hybrid split gate1。两边服务上限16、Mamba槽80、KV24576、mem-fraction0.9。代码固定049db08e4，未修改服务源码或精度。

主窗口与上一轮benchmark相同：128条不同Edit、每条256token、并发16。Torch48个scheduler步骤，图级Nsight32步；部分步骤是prefill。带栈6步仅用于图外调用来源。额外C1/C4是同一输入池的子集及长输出窗口，不是控制完整输入分布的并发扫点。

#### 实际raw B=16的neural图（Torch）

| 项目 | 原生MTP | Hybrid |
|---|---:|---:|
| 图样本数 | 19 | 21 |
| neural图batch | 16.00 | 7.10 |
| GPU时间跨度 | 11.779ms | 10.408ms |
| 7次head kernel SUM | 7.503ms | 7.410ms |
| 其余kernel SUM | 4.643ms | 3.464ms |

图跨度与kernelSUM不可直接相减成严格串行分解。head签名来自实际checkpoint权重[248320,2048]的单独映射；head运行FP16，约1.017GB/步。7步有自回归依赖，当前已经是一次图replay，不存在把7步无损合并为一次大GEMM的条件。

#### 没有prefill、前后均raw B=16的完整周期

| Nsight图级测量 | 原生MTP | Hybrid |
|---|---:|---:|
| 严格筛选周期数 | 3 | 7 |
| 平均周期GPU时间窗 | 57.112ms | 56.472ms |
| neural图跨度 | 11.681ms | 10.646ms |
| target verify图跨度 | 40.093ms | 40.342ms |
| draft extend图跨度 | 2.404ms | 2.416ms |
| 图外GPU kernel SUM | 2.738ms | 2.740ms |
| 图间无GPU活动 | 0.174ms | 0.274ms |

Torch独立筛选14/16个原生/Hybrid纯周期，对应neural跨度11.737/10.429ms，支持相同成本结构。Nsight严格周期样本少，约0.1ms差异只作描述，不宣称统计显著。图包络、kernelSUM和空隙不能随意相加。

#### 同步与持续负载图外开销

| Nsight持续窗口 | 原生MTP | Hybrid |
|---|---:|---:|
| 完整verify周期 | 20 | 21 |
| 图间无GPU活动比例 | 7.80% | 8.40% |
| 图外kernel SUM / verify周期 | 5.852ms | 5.144ms |
- 原生 stream同步累计CPU等待519.98ms，其中无GPU活动约1.49ms。
- Hybrid stream同步累计CPU等待1028.96ms，其中无GPU活动约3.54ms。

Hybrid额外scalar读回来自compact_graph.replay_compact里的num_neural.item。动态binding/prefill的device_extension._view还会创建slots/generations GPU tensor并产生阻塞上传；这一调用点已由stack确认。等待期间GPU忙不等于同步完全无害：它仍可能推迟后续CPU准备，需看同步后的enqueue间隙。当前样本不能把所有GPU空隙归咎于这一个调用。

持续窗口的neural FP16 MoE kernelSUM均值从原生约2.83ms/verify降到Hybrid约1.77ms/verify；这说明主体计算确实省了一部分。相反，neural head约7.5→7.4ms，剩余固定成本仍大。图外约2.7ms中，多数是两边共有的Mamba状态提交，逐轮小kernel不是一个被证实的新大瓶颈。

持续窗口还含prefill、batch merge、free-list/索引准备。两边窗口的prefill次数和请求组成不同，不能用整段时间差解释上一轮约-1%的吞吐均值。上一轮三次无profiler均值：Hybrid1434、原生1449tok/s，范围重叠，仍判定基本持平。

#### 为什么低并发更明显

1. 单请求被SA选中，本轮剩余7个MTP步骤全部跳过，7次neural head也一起省掉；16并发中只要还有一个未匹配行，head就仍需逐步执行。行数减半并不会把head耗时减半。
2. 高并发的共同target verify已经约40ms；单批neural图只省1ms左右，摊到完整周期只能留下约1–2%的空间，prefill/替换和小额调度开销就能进一步稀释。
3. 早期低并发样本/窗口更有利：旧有效C1 Nsight窗口20轮只有1次tail，即19轮完全跳过；旧C4同窗口20轮有13次tail。新的128请求benchmark中全跳过比例是C4约9.2%、C16约0.7%。这些范围不同，不能混为统一命中率或将旧单窗口直接换算为benchmark增益。
4. 本次新C1子集窗口也没有复现旧窗口的高命中：47个完整周期里45次tail。低并发有更好的“整轮省掉head”条件，但收益是否明显仍取决于提示词、生成位置和SA命中率。

#### Kernel / source表

| kernel组 | 原生kernelSUM占比 | Hybrid kernelSUM占比 | 已核验来源 |
|---|---:|---:|---|
| other kernels (multiple operators; not assigned to one source) | 50.8% | 49.9% | 多个算子，未强行归因；详见top_kernels JSON |
| neural FP16 MoE | 3.4% | 2.2% | python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py |
| neural full-vocab lm_head | 9.0% | 9.3% | source-snapshots/logits_processor.py |
| target GDN verify | 12.5% | 13.1% | source-snapshots/fused_sigmoid_gating_recurrent.py |
| target GPTQ Marlin MoE | 21.5% | 22.6% | source-snapshots/marlin.py |
| Mamba state commit | 2.7% | 2.8% | source-snapshots/mamba_state_scatter_triton.py |

分母为完整持续窗口内GPU kernelSUM，包含并行重叠，不是GPU墙钟占比。head只统计neural图中已用真实权重签名匹配的投影，其他GEMM没有误归到head。

#### Overlap / 优化机会表

| 优先级 | 项目 | 证据与下一步 |
|---|---|---|
| P1 | Independent draft-only low-precision/quantized head | RawB16 compact meanB7.10 still pays7.41ms head vs native7.50ms; head is fixed-bandwidth-like cost. Derive a separate draft head because current target and MTP share the module/parameter. Benchmark acceptance and end-to-end throughput. Keep target head unchanged. Derived quantized weights add memory alongside the original target head; no optimization was applied. |
| P2 | Avoid unused gate1 full-batch draft attention metadata | eagle_draft_cuda_graph_runner.execute always calls init_forward_metadata_out_graph at line738; captured run_once calls in_graph at line506. gate1 returns from seed before any full-batch MTP forward; compact tail builds its own metadata. Candidate guard for compact gate1 only. Check KV/index/input-buffer dependencies and padding before skipping. Count outside metadata kernels and validate real model output/acceptance. This source-backed opportunity is not yet an A/B result. |
| P2 | Persistent binding buffers and asynchronous upload | device_extension._view constructs device tensors for slots/generations on new batch bindings and prefill; mapping shows blocking copy/streamSync at this call site. Consider owned double-buffered pinned staging and stable GPU addresses. Preserve generation/lifetime ordering. No causal claim that this explains the1% benchmark difference; most wait spans overlap GPU work. |
| P3 | GPU count decision barrier | replay_compact num_neural.item introduces one scalar D2H/StreamSync per decode. The measured no-GPU part of all stream syncs is tiny relative to host wait sum. Measure downstream enqueue gaps as well as wait spans. Avoid claiming the whole host wait as recoverable time; conditional GPU dispatch is a larger design change. |
| capacity | Compact graph pool/buffer reuse | Same-cap16 startup: hybrid draft_decode graph allocations4.594GiB, native0.027GiB. Share only proven mutually exclusive workspace/buffers; prefix hidden/history output lifetime must stay protected. This is primarily capacity work, not a measured steady-state speedup. |
| shared model | Target verify and GDN state traffic | Both modes spend about40ms in target verify plus roughly2.7ms outside kernels in pureB16 cycles. Separate model-kernel work from hybrid-specific work. Existing ReplaySSM spec path is a candidate for an independent correctness/performance study; no unmeasured speedup is claimed. |

#### Fuse-pattern表

| 模式 | 当前状态 | 判断 |
|---|---|---|
| Draft topk1 postprocess | Existing2pass partial/reduce argmax with position/draft writes | Several microseconds/step; does not avoid reading1.017GB head weights. Low priority relative to projection. |
| Mamba state scatter/commit | Existing fused multi-layer scatter and verify commit outputs | About2.4ms/verify outside cost at high batch is shared. Do not propose a per-layer fusion that already exists. |
| Compact-tail where/copy | Seven masked selections/copies retained after topk1 already writes drafts | For real compact split rows, retrieval mask is false; a guarded no-hot-token/no-skip cleanup may remove redundant work. Small GPU-time budget; preserve split_no_skip behavior. |
| Metadata glue graph | Existing target decode metadata glue path; draft runner deliberately disables it | Distinguish target existing glue capture from skipping unused gate1 full-batch draft metadata. No blanket flag recommendation. |

#### 解释边界与产物

- No source/model/precision change. Current split gate1 K8 service restored healthy. Concurrent local working-tree changes were excluded.
- Profile workloads use the aiohttp streaming driver; the unprofiled throughput runs use the official Rust client. Instrumentation and arrival timing differ; profiles explain cost structure and do not replace the three-repeat throughput result.
- C16 primary is the actual128x256 continuous workload. PureB16 cycles are filtered from it, not separate simultaneous16 long-output waves. Failed/drained wave probes are retained as diagnostics only.
- Torch pureB16 samplesH16/N14; graph-level Nsight strict pure samplesH7/N3. These are descriptive cost samples, not repeated throughput measurements or statistical proof of a1% regression.
- KernelSUM, GPUgraph envelopes, kernelUNION, CPUwait spans and GPUidle are different measurements and must not be added indiscriminately.
- Continuous windows differ in number of prefill/replacement events; direct whole-window subtraction is not a causal performance budget. Keep the previous three-repeat unprofiled result as throughput evidence.
- The existing analyzer three-table output is saved as unreviewed because graph-internal CPU source reconstruction can misattribute GEMMs. This report uses direct graph roles and actual head signatures.
- No NCU DRAM counters, task-quality benchmark or quantified quantized-head speedup.
- 主要trace：hybrid-g1/mixed-c16、native-mtp/mixed-c16；图级报告及SQLite在hybrid-g1-nsys/c16、native-mtp-nsys/c16。
- 长输出/单波采样触发延后准入并进入排空，已排除，保存在drain-window-diagnostics与nsys-first-attempt-diagnostic。主要结论只使用实际128x256持续负载。
- 服务已恢复split gate1 K8，实际服务上限16；健康与32token冒烟见final-audit.json。

<a id="record-20"></a>

## 记录 20：gpuhome-gate1-concurrency-20261018/report.md

来源：[gpuhome-gate1-concurrency-20261018/report.md](gpuhome-gate1-concurrency-20261018/report.md)。源文件 SHA256：`3571ba4d2a7d4d613a122864a7fc4e3ba45a13b4d2e5ac18f758c68c526e34c4`。

### gate=1 大并发测试

RTX4090 48GB，palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4，TP1，内置MTP/EAGLE，split gate=1，K=8；远端分支固定049db08e4。

128 条不同 Edit 请求，每条固定输出256 token，三次无 profiler 重复。

| 并发 | Hybrid tok/s | 原生 MTP tok/s | Hybrid 增益 | Hybrid 相对并发4 | Hybrid 平均 TPOT | Hybrid p95 TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 782.6 | 754.7 | +3.7% | 1.00× | 4.68 ms | 6.54 ms |
| 8 | 1109.8 | 1102.0 | +0.7% | 1.42× | 6.59 ms | 9.19 ms |
| 12 | 1308.5 | 1305.5 | +0.2% | 1.67× | 8.33 ms | 11.67 ms |
| 16 | 1434.0 | 1449.0 | -1.0% | 1.83× | 9.97 ms | 13.90 ms |

| Hybrid 并发 | 平均原始 batch | 非空 tail 平均 B | SA行占比（按轮） | 全跳过轮次占比 | mixed 轮次占比 | Neural 接受长度含bonus |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3.86 | 2.04 | 52.2% | 9.2% | 83.6% | 5.35 |
| 8 | 7.56 | 3.65 | 52.2% | 1.1% | 95.9% | 5.35 |
| 12 | 11.14 | 5.38 | 51.9% | 0.5% | 97.0% | 5.31 |
| 16 | 14.12 | 6.78 | 52.3% | 0.7% | 95.3% | 5.31 |

#### 每次吞吐原始结果

| 模式 | 并发 | 三次 tok/s |
|---|---:|---|
| hybrid-g1 | 4 | 776.48, 781.92, 789.30 |
| hybrid-g1 | 8 | 1110.15, 1102.50, 1116.60 |
| hybrid-g1 | 12 | 1296.34, 1313.28, 1315.83 |
| hybrid-g1 | 16 | 1407.97, 1436.03, 1458.02 |
| native-mtp | 4 | 757.14, 753.62, 753.41 |
| native-mtp | 8 | 1107.03, 1101.06, 1097.98 |
| native-mtp | 12 | 1299.76, 1305.90, 1310.98 |
| native-mtp | 16 | 1455.96, 1446.92, 1444.21 |

#### 延迟对照

| 模式 | 并发 | 平均TTFT | p95TTFT | 平均TPOT | p95TPOT | p95请求完成延迟 |
|---|---:|---:|---:|---:|---:|---:|
| hybrid-g1 | 4 | 108.62ms | 148.10ms | 4.68ms | 6.54ms | 1771.05ms |
| hybrid-g1 | 8 | 138.67ms | 238.56ms | 6.59ms | 9.19ms | 2467.11ms |
| hybrid-g1 | 12 | 174.87ms | 348.41ms | 8.33ms | 11.67ms | 3143.13ms |
| hybrid-g1 | 16 | 219.64ms | 426.91ms | 9.97ms | 13.90ms | 3817.21ms |
| native-mtp | 4 | 106.16ms | 137.05ms | 4.86ms | 6.62ms | 1791.79ms |
| native-mtp | 8 | 134.54ms | 215.38ms | 6.68ms | 9.19ms | 2471.02ms |
| native-mtp | 12 | 166.28ms | 320.82ms | 8.34ms | 11.79ms | 3154.55ms |
| native-mtp | 16 | 207.68ms | 333.64ms | 9.95ms | 13.62ms | 3697.05ms |

#### 口径与限制

- 128 distinct Edit requests per timed run; fixed256 output tokens and temperature0; three repetitions. Throughput uses measured requests only. Diagnostics include four warmup requests and batch drain.
- Identical pinned049db08e4 build; correct built-in unquantized MTP head and GPTQ4 target; K8 throughout. Same actual serving cap16, decode graph max16, mem-fraction0.9, Mamba slots80 and KV tokens24576 for native and hybrid. SSM remainsFP32.
- Mean of per-run p95 latency values, not a pooled p95. Counter-derived batch sizes average all rounds and may be lower than full-concurrency steady-state batches.
- The raw Rust accept_length field comes from server avg_spec_accept_length (lifetime metrics); do not interpret its per-run copies as independent measured-run acceptance. Hybrid per-path acceptance below uses counter differences.
- No profiling active during timing runs. Broader workload than previous four Edit prompts; do not directly compare throughput with that earlier table.
- Sequential hybrid and native service phases leave possible order/thermal confounding. Each phase rotates concurrency order across repeats.
- This measures performance and basic successful generation, not task accuracy or guaranteed text equivalence.

#### 并发容量

- 初次请求32并发时默认状态池仅32槽，实际被限制到6；该启动不计入性能测试。
- FP32状态下，32并发的主状态与投机中间状态估算约27GiB，再加模型权重约24.65GiB，尚未计算KV和图就超过可用显存。
- 正式配置：有效并发上限16、Mamba状态槽80、KV容量24576、mem-fraction-static0.9；保留FP32 SSM、FP16 conv、K8。
- 所有正式并发档位均在同一上限16配置下运行，已验证服务实际容量16。
- hybrid-g1：启动后可用3.207GiB，draft decode图4.594GiB。
- native-mtp：启动后可用7.774GiB，draft decode图0.027GiB。

#### 结果解读

- 已测档位中，Hybrid最高吞吐为并发16：1434.0tok/s。
- 并发8/12/16相对原生MTP的均值增益约+0.7%/+0.2%/-1.0%，三次运行范围都有重叠：高并发下未观察到明确净收益。该重叠判断不是统计显著性检验。
- 并发12→16：吞吐提高9.6%，平均TPOT增加19.7%。12是本次网格中值得考虑的吞吐与延迟折中，最优选择仍取决于SLA。
- 随并发增大，按轮统计的SA行占比仍约一半，但几乎每轮都剩下neural请求。压缩后的有效tail batch增大有利于分摊head成本，整轮跳过head的机会则减少。
- 本轮未抓新profile，不能仅凭吞吐与诊断计数断言大batch下各kernel的占比。前次小batch head微基准可以解释趋势，但不是本轮端到端加速的因果分解。
- 扩到32及以上前，需要先处理FP32投机中间状态与compact-tail图的显存占用；当前运行配置只对本次短上下文负载验证。

<a id="record-21"></a>

## 记录 21：model-pr-history-notes.md

来源：[model-pr-history-notes.md](model-pr-history-notes.md)。源文件 SHA256：`4017e0e49707354e72c5ddcc4a4f90b2177b334c5ba18c28d3f0d0876bcfb3c6`。

### Qwen3.6 MTP + SA adaptation evidence

Baseline: `1b0f58030790726b3146f46786a38b23a7f735e8`.

The local Qwen3.5 and Qwen3.6 PR-history dossiers were consulted before changing the model capability gate. Historical evidence guides inspection; it does not establish correctness on this checkout.

- [PR 19391](https://github.com/sgl-project/sglang/pull/19391): Qwen3.5 MTP support for SpecV2; draft-extend must use the V2 path. Current `qwen3_5_mtp.py` handles `is_draft_extend_v2()` and constructs a single full-attention MTP layer.
- [PR 20864](https://github.com/sgl-project/sglang/pull/20864): Qwen3.5 speculative mRoPE and device-side position handling. Current `eagle_worker_v2.py` advances both regular and mRoPE positions; graph staging copies all three mRoPE axes. SA selection must change tokens without changing positions.
- [PR 23486](https://github.com/sgl-project/sglang/pull/23486): Qwen3.6 retains a GDN backbone. The downloaded 35B-A3B FP8 checkpoint declares `Qwen3_5MoeForConditionalGeneration`, with linear/full-attention layers and an embedded MTP head.

Current source verification: `run_eagle_verify` calls `commit_mamba_states_after_verify` using native acceptance indices and lengths. GDN verification writes speculative intermediate states; the commit path selects only the accepted prefix (including the anchor input, not the unprocessed bonus token). The extension keeps this path intact. Qwen's MTP layer is full attention, so it does not introduce a separate recurrent draft-state rollback protocol.

Scope: only the Qwen3.5/3.6 target and `Qwen3_5ForCausalLMMTP` pair, both configured with `--language-only`. Other linear-attention architectures, sliding-window pools, and actual multimodal requests remain rejected. GPU token parity, rejection recovery, graph replay, and measured SA usage remain separate acceptance criteria from these source findings.

#### Plugin performance review (2026-10-16)

Pre-change HEAD: `4030650794`; the supplied Qwen3-14B/0.6B report used
`1b0f580307`. Read the local `model-pr-history-knowledge/sglang/qwen3-core/README.en.md`
implementation coverage and PR #7745 card (Qwen EAGLE3 auxiliary hidden-state
integration), plus the existing Qwen3.5/3.6 notes above. Read the maintained
`AI-Infra-Auto-Driven-Skills/docs/upstream-source-contracts.md` through the skill's
resolved junction target: historical dispatch and trace attribution do not
certify the current runtime or a performance cause.

Decision: optimize the shared plugin consensus/graph boundary, retaining native
model hidden-state capture, token mapping, verification and acceptance paths.
No model-kernel replacement follows from the report's matching GEMM/attention
tables. Current `draft_forward` invokes `prepare` before the first selection;
the scheduler budgets KV after draft construction using currently free GPU
memory. These source observations, rather than historical PR titles, determine
the changes. GPU/NCCL timing and model parity remain deployment acceptance gates.

Direct sa_spec source comparison was rechecked on 2026-10-16 at clean local
HEAD `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969`: `src/api.cc` builds prompt
states on the host and copies used prefixes when requests enter generation;
`src/api.cu` consumes accepted device tokens and performs lookup in one
current-stream kernel; `src/memory.cu` uses asynchronous copies on the active
Torch stream. This supports the existing device-history path, selected explicitly
by `device-k3.json` or `device-sa-spec.json`. The standalone library has no TP
consensus protocol, which is not evidence that SGLang may safely remove consensus
while allowing rank-local slot/version fallback. No new GPU performance claim
is inferred from the reference implementation.

#### Device overhead follow-up (2026-10-16)

Baseline: `2d279d4f34`. Reused the Qwen3-core PR #7745 evidence and current
native draft/verify boundaries above; no model-specific forward changes.
Re-read sa_spec `src/api.cu`, `src/sa_spec/util/flat_graph.hpp` and
`flat_hash_map.hpp`, plus the locally retained TRT-LLM #10951 head
`304ec62dd635cc7ad865c2ae35b4ae02910396d0` `reference-mtp.py` (acceptance,
`sa_spec.extend`, per-step `draft_sampler`, and next-input update).

The reference retains neural forward and selects SA tokens at each sampling
step. The first optimization targets here are our additional six per-verify
PyTorch statistics reductions and repeated stable graph-input copies, not
moving the source-selection point or changing model recurrence. Metrics are
fused into the existing private plugin feedback operator; no new public
`sglang.kernels` entry point or experimental-package relocation is introduced.
Reference hash-based transition lookup versus our linked adjacency scan is a
separate long-history performance risk; changing layout needs its own CPU/GPU
differential validation and length/vocabulary sweep. Source comparison alone
does not establish its contribution to the supplied EAGLE3 slowdown.

#### Hash transition follow-up (2026-10-16)

Baseline: `44637fe669`. Read the same pinned sa_spec `flat_graph.hpp` and
`flat_hash_map.hpp`: inline single edge, then per-node open addressing, growth
by five with a full copy and no allocator reclamation. This implementation uses
an independently written exact `(state, token)` chained hash index with linear
bucket splitting. Up to four outgoing edges stay on a short-list path to avoid
hashing low-diversity histories; larger lists are indexed. The adjacency list
is retained for clone enumeration. Used prefixes remain relocatable between
compact host construction and larger GPU slots, with no device allocation or
full-table rebuild during graph replay. These storage choices are specific to
our bounded runtime and are not claims about sa_spec benchmark equivalence.
See VALIDATION.md for native measurements and the separate CUDA validation gap.

#### Compact inline transitions (2026-10-16)

Baseline: `88f8d971eb`; original-list control: `44637fe669`. Re-read sa_spec
`flat_graph.hpp`: its inline edge avoids an allocated edge record for degree-one
states, and clones copy inline state directly. This implementation extends the
idea to a tagged second target while retaining the bounded hash overflow pool.
Primary labels come from the source's first-position successor; secondary labels
come from the destination's first-position token. Clone retargeting preserves
these labels. The pool bound follows from an injective mapping of non-primary
edges to suffix paths whose last non-primary edge is the mapped edge; the full
string follows primary edges only. Thus there are at most n-1 non-primary edges,
and stored overflow edges are a subset. Tests exercise the tight distinct-token
case, unary/binary zero-pool cases, and transitions to higher degree. This is a
layout-specific argument, not a capacity guess from observed corpora.

Native checksums now canonicalize semantic transitions instead of physical
adjacency addresses. Both original-list and previous-hash A/B measurements are
retained. No model forward, acceptance rule, TP consensus, or scheduling change.

<a id="record-22"></a>

## 记录 22：qwen3-8b-06b-compact-ab-4090-20261019/README.md

来源：[qwen3-8b-06b-compact-ab-4090-20261019/README.md](qwen3-8b-06b-compact-ab-4090-20261019/README.md)。源文件 SHA256：`d50160451f2d302e54f810d790fa59240da738d0eb337008e2467c68d8a19aa4`。

### Qwen3-8B target + Qwen3-0.6B draft：4090 三臂对照

已完成三次独立启动、交替顺序验证：54/54 个阶段、5184/5184 条请求，输出 1,327,104 token。模型分别为独立加载的 Qwen3-8B target 和 Qwen3-0.6B STANDALONE draft，FP16、TP1；GPU 为 RTX 4090 49140 MiB。

稳定阶段（每次每个并发 128 条请求、32,768 个输出 token）的三次均值如下。标准差、每次原始结果及冷启动阶段见 [report.md](qwen3-8b-06b-compact-ab-4090-20261019/report.md) 和 [summary.json](qwen3-8b-06b-compact-ab-4090-20261019/summary.json)。

|并发|臂|吞吐 tok/s|mean TPOT ms|实际 target 验证 token 数|验证 token/输出 token|
|---:|---|---:|---:|---:|---:|
|1|native|182.71|5.2325|42345.0|1.2923|
|1|fixed|208.76|4.5421|39399.0|1.2024|
|1|dynamic|201.37|4.7185|39328.0|1.2002|
|4|native|566.44|6.5788|42435.0|1.2950|
|4|fixed|652.94|5.6991|39507.0|1.2057|
|4|dynamic|619.27|6.0190|38919.7|1.1877|
|16|native|1255.23|11.4925|43380.0|1.3239|
|16|fixed|1442.73|9.9351|40897.0|1.2481|
|16|dynamic|1369.22|10.4299|39880.7|1.2171|

按同一次重复配对计算百分比，再取三次均值：

|并发|固定相对原生吞吐|动态相对固定吞吐|动态相对固定 TPOT|动态相对固定实际验证 token|动态相对固定请求验证轮次|
|---:|---:|---:|---:|---:|---:|
|1|+14.26%|-3.54%|+3.88%|-0.18%|+0.37%|
|4|+15.27%|-5.16%|+5.61%|-1.49%|+0.71%|
|16|+14.94%|-5.10%|+4.98%|-2.49%|+4.56%|

本负载、所测三组中固定 compact 最快。动态裁剪确实减少了实际验证 token，但请求验证轮次增加；合计吞吐更低。这里没有通过 profiler 量化分解动态选择、反馈与额外轮次各自造成的开销。

配置与口径：

- native 为原生 STANDALONE K8，验证宽度 9（含 anchor）。
- fixed/dynamic 均为 SA8/neural3、gate=1、split、global pool、device GPU SA、compact verify；两者使用完全相同的 72 个有限 target Graph token 档位，仅切换 dynamic_verify。
- 每臂独立启动三次，顺序 native→fixed→dynamic、fixed→dynamic→native、dynamic→native→fixed；并发顺序同步轮换为 4/16/1、1/4/16、16/1/4。
- 每个并发清空 KV、SA 历史和校准状态；64 条冷启动阶段请求后接 128 条独立的稳定阶段请求。greedy、ignore_eos，每条输出 256 token，无计时 logprobs。复用上一轮公开 edits 负载；完整 tokenizer 映射、处理流水线及全部负载编码已核对相同。
- 稳定阶段表示协议中的预热后阶段，不保证所有稀有校准桶都已收敛。冷、稳定阶段数据不同，不能把二者差异完全归因于学习。
- 实际验证 token 数包括 target verify 的 anchor 与 CUDA Graph padding，不包括 prefill 和 neural draft forwards。逐阶段官方输出长度、错误字段、计数器及 Graph 档位直方图已校验。
- 原生与混合组的差异同时包含 SA、split、neural 上限和 compact，整体收益不能单独归因于 compact；本轮也不是原生最优 K 的搜索。
- 这是 TP1 的真实大小模型 STANDALONE 验证，不代表 DSV4 MTP TP8 性能。

源码与产物：

三臂均使用 c39a01d74dcf953c5e9ff7f2fc0b4ed379ea7399，加上上一轮已验证的同一 STANDALONE draft-extend hidden-state 修复；该修复保持 STANDALONE 隐藏状态为空，并保留其他路径的 clone 语义。源码没有因性能结果而调整。补丁尚未提交推送。

公开模型来源及权重 SHA256 见 download-integrity.json、公开文件清单见 evidence.tar.gz；精确命令、源码提交、依赖版本、全部逐请求输出、before/after diagnostics 和 GPU telemetry 也在 evidence.tar.gz 中。

远端目录：/root/rivermind-data/benchmarks/qwen3-8b-06b-compact-ab-20261019/；主机 hz01-ssh.gpuhome.cc:30149。测试服务已停止，最终 GPU 状态见 completion.json。

<a id="record-23"></a>

## 记录 23：qwen3-8b-06b-compact-ab-4090-20261019/report.md

来源：[qwen3-8b-06b-compact-ab-4090-20261019/report.md](qwen3-8b-06b-compact-ab-4090-20261019/report.md)。源文件 SHA256：`49d0c4979494fd23e4a14da77ba8548867d9182bce844f8550cf6d8e41bc0412`。

### RTX4090 dynamic compact benchmark

All rows are means of three runs; parenthesized values are sample standard deviations. Target rows include anchor and CUDA Graph padding.

#### stable

|c|arm|output tok/s|mean TPOT ms|actual target rows|rows/output|
|---:|---|---:|---:|---:|---:|
|1|native|182.71 (0.11)|5.2325 (0.0026)|42345|1.2923|
|1|fixed|208.76 (0.30)|4.5421 (0.0083)|39399|1.2024|
|1|dynamic|201.37 (1.19)|4.7185 (0.0320)|39328|1.2002|
|4|native|566.44 (0.23)|6.5788 (0.0067)|42435|1.2950|
|4|fixed|652.94 (1.64)|5.6991 (0.0132)|39507|1.2057|
|4|dynamic|619.27 (1.71)|6.0190 (0.0238)|38920|1.1877|
|16|native|1255.23 (5.43)|11.4925 (0.0439)|43380|1.3239|
|16|fixed|1442.73 (2.54)|9.9351 (0.0214)|40897|1.2481|
|16|dynamic|1369.22 (8.33)|10.4299 (0.0398)|39881|1.2171|

|c|dynamic vs fixed throughput|TPOT|target rows/output|throughput changes in run0/1/2|
|---:|---:|---:|---:|---|
|1|-3.54%|+3.88%|-0.18%|-3.63%/-3.94%/-3.05%|
|4|-5.16%|+5.61%|-1.49%|-5.37%/-4.95%/-5.15%|
|16|-5.10%|+4.98%|-2.49%|-4.54%/-5.69%/-5.06%|

#### cold

|c|arm|output tok/s|mean TPOT ms|actual target rows|rows/output|
|---:|---|---:|---:|---:|---:|
|1|native|184.00 (0.13)|5.1958 (0.0027)|21024|1.2832|
|1|fixed|217.34 (0.35)|4.3521 (0.0087)|19324|1.1794|
|1|dynamic|211.45 (1.05)|4.4835 (0.0208)|19290|1.1773|
|4|native|567.82 (8.21)|6.5126 (0.0641)|21006|1.2821|
|4|fixed|673.64 (8.65)|5.4846 (0.0583)|19349|1.1810|
|4|dynamic|634.64 (4.35)|5.8176 (0.0356)|19180|1.1707|
|16|native|1221.80 (38.12)|10.9536 (0.4216)|21357|1.3035|
|16|fixed|1430.09 (71.10)|9.4268 (0.5533)|20154|1.2301|
|16|dynamic|1393.77 (5.28)|9.6231 (0.0466)|19903|1.2148|

|c|dynamic vs fixed throughput|TPOT|target rows/output|throughput changes in run0/1/2|
|---:|---:|---:|---:|---|
|1|-2.71%|+3.02%|-0.18%|-2.16%/-3.36%/-2.61%|
|4|-5.78%|+6.08%|-0.87%|-4.85%/-6.74%/-5.76%|
|16|-2.37%|+2.32%|-1.25%|+3.59%/-6.07%/-4.62%|


<a id="record-24"></a>

## 记录 24：qwen3-native-k3-ar-4090-20261019/README.md

来源：[qwen3-native-k3-ar-4090-20261019/README.md](qwen3-native-k3-ar-4090-20261019/README.md)。源文件 SHA256：`34168613b1d945f92d0af8ca63e086ed9e39079b08defaef2bbd6bc150d84761`。

### Qwen3-8B target / Qwen3-0.6B draft：原生 K3 与 AR 单次补测

New K3 and AR each run once. K8/fixed/dynamic use exactly previous repeat 2, not their three-repeat means. Same hardware, source, weights, dataset, precision, concurrency order and common scheduling flags. Runs are not contemporaneous; differences near historical run-to-run variation are inconclusive.

原生 K3：3 步起草、4 个验证位置；普通自回归 AR：只加载 target，不启用投机。固定/动态均为 SA8/neural3、gate=1、split、global pool，保持上一轮配置。

各并发 64 条冷启动阶段后接 128 条独立稳定阶段，greedy、ignore_eos、每条输出 256 token。AR 行数统计 target decode；投机方案统计 target verify；均包含 CUDA Graph padding，不包含 prefill 和 neural draft forward。AR 每条的首个输出 token 来自 prefill，因此 decode 逻辑行数为 255/请求。

共同服务约束：FP16、TP1、Triton、context=2048、max-running-requests=16、max-total-tokens=32768、Graph max bs=16、mem-fraction-static=.75、disable-overlap-schedule、chunked-prefill-size=-1。这里只比较这些约束下所测 AR/K3/K8 的最优，不是全参数调优结论。

#### stable

|并发|方案|来源|吞吐 tok/s|TPOT ms|实际验证 token|AR decode 行数|
|---:|---|---|---:|---:|---:|---:|
|1|ar|本次 1 次|56.07|17.6659|0|32640|
|1|native-k3|本次 1 次|130.87|7.4041|35828|0|
|1|native-k8|上轮第 3 次|182.66|5.2328|42345|0|
|1|fixed|上轮第 3 次|209.10|4.5331|39399|0|
|1|dynamic|上轮第 3 次|202.72|4.6821|39362|0|
|4|ar|本次 1 次|207.24|18.7544|0|32640|
|4|native-k3|本次 1 次|440.54|8.6399|35804|0|
|4|native-k8|上轮第 3 次|566.44|6.5828|42435|0|
|4|fixed|上轮第 3 次|654.70|5.6850|39507|0|
|4|dynamic|上轮第 3 次|621.01|5.9980|38855|0|
|16|ar|本次 1 次|708.39|20.5430|0|32640|
|16|native-k3|本次 1 次|1080.41|13.6213|36228|0|
|16|native-k8|上轮第 3 次|1259.57|11.4522|43380|0|
|16|fixed|上轮第 3 次|1445.60|9.9124|40897|0|
|16|dynamic|上轮第 3 次|1372.51|10.4090|40018|0|

|并发|最佳所测原生|固定吞吐增益|动态吞吐增益|固定 TPOT 变化|动态 TPOT 变化|
|---:|---|---:|---:|---:|---:|
|1|native-k8|+14.48%|+10.98%|-13.37%|-10.52%|
|4|native-k8|+15.58%|+9.63%|-13.64%|-8.88%|
|16|native-k8|+14.77%|+8.97%|-13.45%|-9.11%|

#### cold

|并发|方案|来源|吞吐 tok/s|TPOT ms|实际验证 token|AR decode 行数|
|---:|---|---|---:|---:|---:|---:|
|1|ar|本次 1 次|56.08|17.6620|0|16320|
|1|native-k3|本次 1 次|131.33|7.3834|17848|0|
|1|native-k8|上轮第 3 次|184.13|5.1929|21024|0|
|1|fixed|上轮第 3 次|217.51|4.3488|19324|0|
|1|dynamic|上轮第 3 次|211.83|4.4737|19293|0|
|4|ar|本次 1 次|206.65|18.7983|0|16320|
|4|native-k3|本次 1 次|443.81|8.5698|17844|0|
|4|native-k8|上轮第 3 次|572.69|6.4733|21006|0|
|4|fixed|上轮第 3 次|678.69|5.4512|19349|0|
|4|dynamic|上轮第 3 次|639.61|5.7773|19187|0|
|16|ar|本次 1 次|699.12|20.7307|0|16320|
|16|native-k3|本次 1 次|1025.76|13.3518|18008|0|
|16|native-k8|上轮第 3 次|1242.38|10.7346|21357|0|
|16|fixed|上轮第 3 次|1464.60|9.1326|20154|0|
|16|dynamic|上轮第 3 次|1396.94|9.5909|19891|0|

|并发|最佳所测原生|固定吞吐增益|动态吞吐增益|固定 TPOT 变化|动态 TPOT 变化|
|---:|---|---:|---:|---:|---:|
|1|native-k8|+18.13%|+15.04%|-16.25%|-13.85%|
|4|native-k8|+18.51%|+11.68%|-15.79%|-10.75%|
|16|native-k8|+17.89%|+12.44%|-14.92%|-10.65%|

新基线各只跑一次，固定、动态和 K8 复用上一轮同一个单次结果；这些运行不在同一时间段。原始输出、命令、计数和版本随 evidence.tar.gz 保存；上一轮三次汇总仅作为上下文参考，未混入主表。

<a id="record-25"></a>

## 记录 25：sa-parallel-6000d-throughput/report-table.md

来源：[sa-parallel-6000d-throughput/report-table.md](sa-parallel-6000d-throughput/report-table.md)。源文件 SHA256：`6fafb6b9327b9580f7185d126341e2372e7bef36d6a32ba852e28adcdd1b1ad3`。

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

<a id="record-26"></a>

## 记录 26：sa-parallel-6000d-throughput/report.md

来源：[sa-parallel-6000d-throughput/report.md](sa-parallel-6000d-throughput/report.md)。源文件 SHA256：`bd6353598f76b171b47cf62ce134669f8087f03e3bc7a3ffa9aea9d887c8a8a5`。

### RTX 6000D：跨请求 SA 并行扫描对照

日期：2026-10-17，分支 `codex/hybrid-suffix-speculation`，基线
`2d53a3d2d2` 加本分支既有全局池实现。仅将串行跨槽位查询替换为并行查询。

#### 吞吐结果

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

#### 查询微基准结果

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

#### 实现与 split 兼容性

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

#### 正确性与编译资源

RTX 6000D（SM120）完整包测试：**182 passed、1 skipped，76.60 s**，包含首次编译。
唯一跳过项需要双卡 NCCL；Gloo 1/2/4 rank 通过。CUDA 测试从 19 项增加至 26 项。

新增测试用独立 CPU 文本搜索 oracle 比较 20/64/257 个来源、两种 partial 策略、
随机历史、长匹配、代际失效、失效来源、padding、发布顺序变化和 Graph 重放；
另构造不同线程、同一线程跨步的完全同分候选，检查固定发布顺序。
已有 final-feedback、活跃行更新屏障、停止尾部、full/split/compact 回归全部通过。

`cuobjdump --dump-resource-usage`：新查询 44 registers/thread、3072 B shared memory、
0 B stack、0 B local；原串行查询 40 registers/thread、0 B shared。
布局、Python/C++ 接口和槽位容量均不变。精确源码差异见 `parallel-query.patch`。

#### 微基准方法

每个槽位 384 token，外部上下文 128，K=8、min_match=4、H=1024。
20/64/257 槽位，batch=1/4，两个构造负载：随机不同历史；外部来源含完整共同后缀和续写。
随机 token 来自 32 词表，旨在检验扫描开销，不代表真实代码 token 分布。

串行与并行分别加载各自源码和编译缓存，逐项核对输入哈希、候选 token 和 mask。
12 组完全一致。每个 Graph 包含 8 次幂等查询，预热后 5 轮 CUDA Event 计时，
每轮重放 25 次；总查询时间包含 finalize、own-query 和 shared-query 三个 kernel。
另用 Torch Profiler 采集每组 40 次 shared-query kernel 的 GPU 持续时间，报告中位数。
profiler 与吞吐测量分开运行。完整 traces 在证据归档中。

#### 真实模型方法

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

#### 归档与复现

完整原始证据：[本地归档](/private/tmp/sa-parallel-6000d-20261017-evidence.tar.gz)。远端持久副本：
`/root/rivermind-data/validation/sa-parallel-6000d-20261017-evidence.tar.gz`。源码、逐请求原始结果、24 组 profiler traces、
启动命令和参数、种子、日志均在归档内；不含模型权重。两版源码各 5,616 个记录文件
及归档内全部文件哈希已在本地再次核对。详情见 `archive.json`、`verification.json`。

归档 SHA-256：`47ae0f1a7d3beab2ab34a290aa6f1e017ce7e7178d236f3f1de3fbcc581d48b1`。临时服务已停止，最终显存 0 MiB，原远端 checkout 保持干净。

复现顺序：分别设置串行/并行源码 PYTHONPATH 和独立 TORCH_EXTENSIONS_DIR，
先运行 `microbenchmark.py`，再用 `run_comparison.py` 调用归档中的 `run_benchmark.py`，
最后运行 `analyze.py`。精确参数在每个 `*-driver-command.json` 和子目录 `plan.json`。
组合配置示例为 `configs/device-qwen35-mtp-split-global-pool.json`，已核对与实测 JSON 完全一致。

<a id="record-27"></a>

## 记录 27：split-graph-c4.md

来源：[split-graph-c4.md](split-graph-c4.md)。源文件 SHA256：`76d720ecaf4e81f30826fb521ab862fd382c8185ee9fd9622e1d0e7d41d14a89`。

### 并发 4：按请求压缩 MTP 尾段的 A100 实验

已实现混合批逐请求跳过：SA 命中行不执行 MTP 尾段。但本次并发 4 对照中，
相对完整 MTP+SA，裁剪版吞吐下降约 **2.4%**、平均 TPOT 增加约 **2.7%**，
未体现额外性能收益。相对原生的约 1% 吞吐涨幅小于重复波动，不能视为稳定加速。
正确性评测按用户要求不运行；下面的 CUDA 回归测试只验证图调度与数据搬运契约。

#### 实现

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

#### 使用范围

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

#### 对照协议

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

#### 正式性能结果

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

#### Trace：为何少算请求行仍没有额外收益

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

#### 验证与证据

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

<a id="record-28"></a>

## 记录 28：split-profile-6000d/report.md

来源：[split-profile-6000d/report.md](split-profile-6000d/report.md)。源文件 SHA256：`056bab4687bfffc73b884fc314f134717c1d7c2b9d7867ea2a5a96f2ecc326a8`。

### RTX 6000D：split 模式的 trace 瓶颈分析

结论：**仍有优化空间。split 内优先关注 draft LM head 的权重读取；整体推理的最大
GPU 开销在 target verify 的 MoE。继续压缩尾段行数或只改 argmax，预期收益较小。**

#### 采集范围

2026-10-17，源码提交 `671931d83a1e2461f16ea0d4e61dadaec88dd178`。
单卡 RTX 6000D、Qwen3.6-35B-A3B-FP8、TP=1、K=8、gate=2、全局池开启，
4 活跃槽位 + 16 保留槽位、H=4096、外部上下文 128。启动命令和配置分别保存在
`server-command.json`、`split-config.json`，与上一轮吞吐实验保持相同模型执行设置。

仅抓两组短窗口，没有重跑完整吞吐矩阵：

- C1 暖池：先真实生成 16 个代码编辑请求，再重放其中一个 prompt。
- C4 混合批：同样生成 16 个请求建池，再运行第 17–20 个不同 prompt。

两种 batch shape 都先预热，已经执行超过 10 个 decode 步骤。采集在所有请求都返回
首 token 后启动，每组实际覆盖 9 个 decode 步骤，verify 的 batch 分别固定为 1/4。
每请求固定生成 256 token、greedy、ignore_eos=True；使用先前相同的真实代码数据，
没有读取参考答案。256 token 用于保证采集窗口内请求不提前结束，不作为新吞吐结果。

使用未修改的服务源码及内置 profiler API，CPU+GPU、with_stack=False、record_shapes=False。
5,616 个运行源码文件哈希与上一轮实测版本一致；没有给生产代码加诊断标记。
启动时的 JIT、prefill、Graph capture 均在窗口之外。服务完成后停止，显存回到 0 MiB。

#### 阶段分解

以下是累计 kernel GPU 时间占比，**不是 CPU scope 时长，也不是端到端 wall-time 占比**。
采用 `cudaGraphLaunch` correlation 将 kernel 关联到内置 `draft`、`draft_extend`、
`step[VERIFY ...]` 范围；draft 中根据 prefix-decision 前后的 Graph launch 区分 prefix/tail。
全部 GPU kernel 都找到了相应 CUDA runtime/driver launch。不同 stream 上可有重叠，
所以累计 kernel 时长可以略大于窗口跨度。

| 阶段 | C1 暖池 GPU 占比 | C4 混合批 GPU 占比 |
| --- | ---: | ---: |
| target verify | 79.34% | 72.52% |
| prefix（含 SA 查询及一次 MTP forward） | 9.16% | 5.89% |
| MTP tail | 3.53% | 15.49% |
| draft-extend | 6.96% | 4.30% |
| 其余与元数据 | 1.01% | 1.80% |

9 步内，C1 有 8 步跳过尾段；C4 仅 2 步整批跳过，另外 7 步的尾段为 1 行 × 5、
2 行 × 2。C4 有 27/36 个请求行跳过尾段（75%），但 7/9 的批次仍要执行六次后续 MTP。
这是该短 trace 的事实，不是长时间请求命中率估计。

#### Kernel table

| Kernel / 家族 | C1 GPU 占比 | C4 GPU 占比 | 当前源码与结论 |
| --- | ---: | ---: | --- |
| Triton `fused_moe_kernel` | 29.54% | 41.81% | `python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py`；最大整体优化目标，主要在 target verify |
| BF16 全词表 LM head | 16.26% | 20.72% | `LogitsProcessor._compute_lm_head`；包含 prefix、tail、target 和 draft-extend，不能全部归于尾段 |
| SM120 FP8 dense GEMM | 12.59% | 8.19% | 当前线性投影已有 FP8 路径；与 BF16 head 分开统计 |
| MoE activation | 9.15% | 5.05% | Triton MoE 中的 `silu_and_mul` / `act_and_mul` |
| fused GDN recurrent update | 5.15% | 5.29% | `srt/layers/attention/linear/kernels/gdn_triton.py`；已经是融合路径 |
| layer norm | 3.91% | 4.47% | `_layer_norm_fwd_1pass_kernel` |
| 跨请求 SA query | 3.55% | 2.13% | 每步约 0.620 / 0.609 ms；已使用并行来源扫描 |
| FP8 group quantization | 3.27% | 1.46% | 多个算子的量化调用，不能全部归于同一 MoE epilogue |
| draft argmax/finalize | 0.03% | 0.06% | 已有 split argmax + token/position finalize；单独继续优化它的预算很小 |

完整三表工具输出见 `c1-warm-triage.txt`、`c4-mixed-triage.txt`。
其叶级 Python source location 因 Graph 回放没有 stack 映射而显示 unresolved；本报告通过
Graph launch correlation 做阶段归因，不宣称已经取得 eager/formal 两 trace 的叶级映射。

#### LM head 为什么是 split 的重点

直接读取本机 safetensors header 确认 `lm_head.weight` 为 **BF16 `[248320,2048]`**，
共 1,017,118,720 字节（约 1.02 GB），不是 FP8 权重。
结合该尺寸、当前 `torch.matmul` 路径与 vocab 对应的 GPU grid，定位 head：
GEMV grid `[15520,1,1]`，或相应 BF16 GEMM grid `[8,970/485/243,1]`。
相同 GEMV 名称的小矩阵没有计入 head。

| C4 尾段行数 | 回放样本数 | 每次尾段累计 kernel ms | 其中六次 head ms | head 占尾段 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 5.582 | 4.630 | 82.94% |
| 2 | 2 | 5.979 | 4.597 | 76.87% |

单次 head 约 0.77 ms，1 行与 2 行接近；缩行只减少了一部分计算，仍需处理同一个大投影矩阵。
“权重流量主导”是依据矩阵体积与行数/耗时关系作出的推断；本轮没有 Nsight Compute 的
DRAM 吞吐率计数，不把该推断写成硬件带宽测量结论。

优先验证 **draft-only 的独立低精度或专用小 M 投影路径**。当前 MTP 可以直接共享 target
的 head 对象（`qwen3_5_mtp.py::set_lm_head_from_target`），实施时需保持 target 投影不变。
低精度 draft 会影响候选和 gate 通过率，后续小对照需同时看接受长度及吞吐。
这是候选方案，尚未实现或测得新收益。

#### Overlap-opportunity table

| 优先级 | 窗口 | 证据与预算 | 依赖边界 / 建议 |
| --- | --- | --- | --- |
| 1（split 内） | SA 查询与 neural prefix | SA 当前串行处于 prefix，约 0.61 ms/步，占累计 GPU 2.1%–3.6% | 可以研究与首个 MTP forward/head 重叠；必须等待 accepted history 就绪，并在 gate 判定前汇合。需处理 stream 所有权与 Graph 依赖，不能仅新增 stream |
| 2 | mask 返回后到下个 GPU kernel | 中位数 C1 54 μs，C4 434 μs；C4 包含构造/上传尾段行号等调度工作 | 优先预分配行号 staging 或按 bitmask 预建映射；保留实际原始 batch 与当前所有权规则。该间隙受 profiler 放大，须用短的无 profiler 对照验证 |
| 低优先级 | “移除约 1.8 ms CPU 同步” | prefix-decision 中位数 C1 1.773 ms、C4 1.832 ms，但 mask D2H 本身仅约 0.384 μs | 大部分 API 等待覆盖 prefix GPU 工作；不能把 1.8 ms 全部当作可省 CPU 成本 |

mask 的 GPU 拷贝完成到 CPU decision 返回，中位数仅 20.6 / 27.9 μs。
C1 的主要阻塞表现为 stream synchronize；C4 的 pageable mask 回读主要阻塞在 memcpy API。
两者都需要按实际时间线解释，不能只比较 API 名称或累计 `decision_wait_ns`。

这是 single-trace 阶段归因；表中的 overlap 是待验证的依赖分析，尚无新增重叠实现的吞吐证据。

#### Fuse-pattern table

| 模式 | 本轮证据 | 当前实现核查与处理 |
| --- | --- | --- |
| MoE activation + FP8 quant | C4 有 550 对同一 Graph 内直接相邻的 activation/quant，累计占 GPU 4.83% | `triton_utils/fused_moe.py` 先 activation，再按 block_shape 做 quant；有融合候选。需核对 sorted rows、padding、scale 布局，不可直接套用其他量化格式的融合核 |
| draft argmax + position/token finalize | 累计 GPU 仅 0.03% / 0.06% | `kernels/ops/speculative/topk1.py` 已融合 finalize；继续单独压缩此段优先级低 |
| GDN sigmoid gate + recurrent update | 5.15% / 5.29% 对应 fused GDN kernel | 已融合。原始工具将该 gate 名称误配到 “shared expert append”，手工核查后拒绝这个自动归因 |

已核对技能的 fuse/overlap catalog 和当前源码。原始工具部分候选路径仍为迁移前目录，
这里使用当前 `sglang.kernels.ops` 入口，不把泛化匹配或旧路径视为当前 backend 可直接启用的证据。

#### 下一步的最小验证

1. 先对实际 BF16 head 的 M=1/2/4 做小规模投影实验，比较专用路径与独立 draft-only
   低精度路径；保留 target 权重，记录 argmax 与接受长度变化。
2. 同时可以独立验证 SA 与首个 neural forward 的重叠，预期预算限定在每步约 0.61 ms。
3. 若目标是更大的整机收益，再处理 target verify 的 MoE kernel/config 或 activation/quant
   融合；它覆盖的 GPU 预算大于 split 调度本身。

本轮只做采集和诊断，没有修改生产实现，也没有把 trace 中的时间推算成已实现的吞吐提升。
原始 trace、输入/输出、种子、工具源码和元数据保存在完整证据归档中。

#### 原始 trace 与复现

- [c1-warm trace](/private/tmp/split-profile-final-20261017/split-profile-6000d-20261017/formal-c1-warm/formal-c1-warm-TP-0.trace.json.gz)
- [c4-mixed trace](/private/tmp/split-profile-final-20261017/split-profile-6000d-20261017/formal-c4-mixed/formal-c4-mixed-TP-0.trace.json.gz)

[完整证据归档](/private/tmp/split-profile-6000d-20261017-evidence.tar.gz)（含原始三表、采集工具、请求与响应、配置、种子和哈希）。
远端持久副本：`/root/rivermind-data/validation/split-profile-6000d-20261017-evidence.tar.gz`。
SHA-256：`1a9f07cad3929d2f8d0eb669371745b0bb227c4ba8c732fc6144bb3af4c7baec`。52 个归档文件哈希已在本地逐项核对。

以归档根目录为输入运行 `analyze_phases.py ROOT --output phase-summary.json` 可重建阶段统计；
`profiler-tools/scripts/analyze_llm_torch_profile.py --framework sglang --input TRACE` 可重建原始三表。

<a id="record-29"></a>

## 记录 29：gpuhome-gate1-mixed-tail-20261018/report.txt

来源：[gpuhome-gate1-mixed-tail-20261018/report.txt](gpuhome-gate1-mixed-tail-20261018/report.txt)。源文件 SHA256：`65731ff3ba43af3c8e03ceab044ba0f45a97ec2622b03637fdb930ab7c2f60aa`。

Gate=1 多并发 mixed batch neural tail 与全词表 lm_head 分析（2026-10-18）

结论：压缩后的小batch仍需每步读取整个输出头。当前主要成本是7次全词表投影，继续压缩行数、补齐batch或融合argmax都不能消除它。多并发优先考虑有效请求摊销与独立draft输出头的字节压缩；缩短neural深度需要接受长度分布支撑。

范围
RTX4090 48GB，远端049db08e4，正确独立unquant MTP配置、EAGLE、FP16、split gate1、K8、topk1、drafttokens9、最大并发4。gate1复用draft-extend给出的seed，首token比对前没有额外inner MTP forward；SA未命中行随后执行7步neural tail。
本轮同四条Edit rewrite，预热每条至少32token，再采64 scheduler steps；每条固定1024输出，ignoreEOS，temperature0。窗口内64次VERIFY均为B4，覆盖不同neural压缩大小。初次trace只有CPU事件，GPU计时判无效并保留；同配置重启后抓到95803GPU kernel和64次query，下面只使用有效trace。

实际 tail 大小与成本
64轮：all-SA跳过tail 5轮；mixed 57轮；全neural 2轮。非空tail平均B=2.390。
| neural tail B | 次数 | tail GPU首末跨度ms | 7次lm_head kernel SUM ms | head占tail kernel SUM | top1两阶段总ms |
| --- | --- | --- | --- | --- | --- |
| 1 | 14 | 8.536 | 7.275 | 84.6% | 0.0212 |
| 2 | 10 | 9.118 | 7.320 | 79.3% | 0.0223 |
| 3 | 33 | 9.487 | 7.343 | 76.4% | 0.0233 |
| 4 | 2 | 9.754 | 7.361 | 70.6% | 0.0248 |

从B4压到B1，tail跨度仅从约9.75降到8.54ms，输出头仍约7.3ms。真实head在整个采样窗口中与其他kernel不重叠的时间约432.08ms / 1967.85ms（22.0%）；这是该长窗口的GPU时间归因，不是端到端加速上限或预测。kernel SUM、跨流UNION与首末跨度分开计算，不能把重叠duration相加为延迟。
每个非空tail是一次CUDA Graph replay，图内包含全部7步；没有7次主机graph launch。Tail B从实际_topk1_partial_argmax gridX取值，投影由真实checkpoint的eager shape映射与kernel/grid/block精确匹配，每个tail恰好7个head kernel。

真实权重的独立输出头微基准（CUDA Graph，5×50次，ms/次）
| B | FP16投影 | SGLang top1后处理 | 投影+FP32物化+后处理 |
| --- | --- | --- | --- |
| 1 | 1.0360 | 0.0050 | 1.0409 |
| 2 | 1.0460 | 0.0049 | 1.0550 |
| 3 | 1.0496 | 0.0049 | 1.0605 |
| 4 | 1.0536 | 0.0049 | 1.0650 |
| 8 | 1.0610 | 0.0051 | 1.0809 |
| 16 | 1.0739 | 0.0068 | 1.1066 |

权重lm_head.weight=[248320,2048]，checkpoint BF16→runtime FP16，逻辑大小1017118720字节（约1.017GB）。隐藏输入随机FP16；权重来自真实模型。服务在微基准期间空闲，未和服务请求并跑。该测量只说明输出头，不代表完整MTP/target verify或服务吞吐。
B1/B4 projection约1.036/1.054ms；B16仍约1.074ms。按逻辑权重字节/时间约0.95–0.98TB/s，说明小B下权重流量是主要解释；未采NCU，不能把它写成实测DRAM带宽或已证明的硬件峰值。NVIDIA全连接层指南也说明很小batch容易受带宽限制：https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
因此不要把已经SA命中的行重新塞回tail来“提高GEMM效率”；它们没有新的有效neural工作。真正增加有用请求数（更高服务并发）与补假行不同。也不能把同一请求的7个自回归步骤直接合并成B7矩阵乘，因为每个输入依赖上一轮argmax。

接受长度：短Edit与长延续不能混用
此前256-token gate1/B4基准（3次，含warmup+计时）neural平均接受长度约6.305，含bonus；SA行比例62.4%，all-skip轮次30.6%。hist包含batch排空，不能把所有B1 tail都当raw B4的mixed。
本轮完整1024-token生成的neural均值约4.723，该指标包含64步窗口之后的延续和排空，不是窗口内的精确接受分布。长trace中mixed57/64不能外推为短Edit或生产分布。
只有均值，无法知道第5/6/7/8个proposal实际有多少价值。短负载neural均值6.3，直接把neural缩到4可能增加verify次数而变慢。下一轮应先采P(accept>=k)直方图，再比较neural8/6/4；SA长度仍保持8。

表1：Kernel / 源码归因
| 部分 | 证据 | 源码 / 置信度 |
| --- | --- | --- |
| Compacted neural tail | 64 C4 query rounds,59 full tail launches;14xB1,10xB2,33xB3,2xB4. No incomplete tail launches. | compact_graph.py:115 _capture_tail; stage() gathers unmatched rows, one graph loops7 MTP inner forwards; gridX of _draft_topk1_partial_argmax_kernel reveals actualbatch. Actual graph/kernel metadata, no guess from requested concurrency. |
| Tail full-vocab lm_head | Exact kernel/grid/block matched from actual checkpoint eager projection mappings [B,2048]×[2048,248320]; each tail launch has exactly7 matched head kernels. | qwen3_5_mtp.py:257 logits_processor; logits_processor.py:970 _compute_lm_head normal FP16 torch.matmul. High: isolated actual weight mapping + formal graph signatures. |
| Top1 + positions + draft-chain store | 7 partial argmax kernels per tail; argmax+finalize ~0.021–0.025ms per entire7-step tail. | kernels/ops/speculative/topk1.py:74 draft_topk1_postprocess; two-stage reduction already active. Direct kernel identity/grid. |
| Draft/target head sharing | Current EAGLE path shares target embedding/head, and can share the lm_head module. | eagle_worker_v2.py:319 init_lm_head/:327 maybe_share_target_lm_head; qwen3_5_mtp.py:170 set_embed_and_head/:182 set_lm_head_from_target. Pinned source; any draft-only head change must detach the target-shared module/parameter. |

表2：多并发优化优先级 / 重叠机会
| 优先级 | 目标 | 依据 | 下一步与边界 |
| --- | --- | --- | --- |
| P0 workload sweep | Increase useful neural rows per weight read via real serving concurrency8/16, subject to memory/latency | Projection B16 1.074ms versusB1 1.036ms; the head alone amortizes almost16 useful rows at nearly the same latency. Current server max_running_requests4 caps this opportunity. | Sweep B4/8/16 on the same48GB GPU, preserving prompt/output workload and SLA; record actual compact tail histogram, KV/Mamba capacity and target verify cost. Only isolated head B8/16 measured here. Whole-serving throughput, memory fit and latency at8/16 are untested. Do not add fake/padded SA rows merely to enlarge GEMM. |
| P1 principal kernel target | Reduce bytes read by a separate draft-only lm_head | Current FP16 head has1.017GB logical weights per projection; all B1–4 still pay ~1.04–1.05ms, seven times per nonempty tail. | Prototype a draft-only low-precision/weight-quantized head or covered-vocabulary head with fallback, keeping target verifier and expert quantization unchanged. Compare proposal top1 agreement, acceptance histogram, actual head latency and end-to-end serving. No quantized head benchmark or speedup measured. Current draft shares target lm_head; never mutate the shared head in place. Low precision can change draft acceptance, and restricted vocab needs coverage/mapping validation. |
| P1 algorithm ablation after acceptance histogram | Separate neural depth from SA length; test neural6 vs8 before neural4 | In short256-token Edit benchmark neural mean acceptance is6.305 includingbonus; long1024 continuation is4.723. Fixed7 inner-head calls can waste work, but shorter proposals can also increase expensive target verifies. | Collect per-neural-row accepted-length survival curve P(accept>=k), stratified by compact batch/position; keep SA8 and compare neural8/6/4 with explicit valid draft lengths/masking. Current remote049 has a fixed common8 draft width; simply setting globalK4 also shortens SA and is a different experiment. Mean acceptance alone is insufficient to choose a depth. No candidate implemented. |
| P3 | Further row-compaction, padding or GEMV+argmax fusion | B1→B4 only changes projection1.036→1.054ms; postprocess ~5us, while one head reads ~1GB. Tail already runs all7 steps inside one graph. | Retain compaction to save real MTP body and state traffic; consider producer/consumer fusion only after weight traffic/depth issues. One more copy/argmax fusion cannot remove full-weight reads. Independent sequential token steps cannot be batched together without changing the drafting algorithm. |

保守的固定调度算术场景：若neural8→6，少做2/7输出头，可在该窗口少约123.5ms head工作（窗口约6.3%）；neural8→4少约246.9ms（12.5%）。这些数固定了轮数/接受/verify成本，实际不满足，因此只是成本预算，不是吞吐预测。必须用真实接受长度和无profiler端到端结果裁决。

表3：Fusion / 已有实现检查
| 模式 | 状态 | 证据 | 判断 |
| --- | --- | --- | --- |
| 7 neural steps in one CUDA Graph | Already active | One compact-tail graph launch,7 repeated projection+top1 pairs. | No per-step CPU graph-launch problem to solve here. |
| lm_head projection + FP32 logits + argmax | Candidate but low arithmetic/traffic payoff | Standalone postprocess ~0.005ms; full-chain minus projection ~0.005–0.033ms acrossB1–16. | May save output writes/launches, but not the1.017GB weight read. Not the primary target. |
| Neural-row gather/scatter | Already graph-captured | stage() index-selects and final drafts.index_copy_ are inside compact tail graph. | Compaction is useful but remaining tail cost is dominated by projections; do not attribute all non-head time to compaction. |
| Draft-only quantized/restricted-vocab head | Unimplemented candidate | Measured projection floor acrosssmallB, fullvocab248320; target/drafthead sharing is explicit. | First detach a separate derived head; preserve target output and verify token-map/quant scales and acceptance. |

一个必须处理的实现细节：EAGLE init_lm_head会把target的head绑定到draft，Qwen3.5MTP甚至可直接共享lm_head模块。做draft-only量化或缩表必须创建独立派生head，保留target原head；不能原地量化共享对象，也不能再次给未量化MTP experts误套GPTQ。
接下来建议先做B8/16容量与吞吐扫点（配置层、未验证可容纳），并并行评估独立draft head低精度原型；算法方向先补neural接受直方图，再测neural6，避免直接以均值裁断neural4。更大的收益目标是减少权重读取总字节/次数，而不是继续打磨几微秒的argmax。

验证与证据
本轮新增：有效64步GPU trace、真实head的5×50微基准和带shape/stacks的eager映射；未修改/部署优化源码，也未宣称候选的吞吐提升或输出等价。源码保持049db08e4，服务保持gate1/K8；重启后PID 29291，最终health200与32token生成通过。
本地analysis.json、valid-tail-analysis.json、lm-head-microbench.json和lm-head-kernel-signatures.json提供原始统计；evidence.tar.gz保留首次CPU-only诊断、有效GPU trace、完整请求/响应、日志、脚本和pinned源码快照。远端 /root/rivermind-data/benchmarks/gate1-mixed-tail-20261018。

<a id="record-30"></a>

## 记录 30：gpuhome-gate1-profile-20261018/report.txt

来源：[gpuhome-gate1-profile-20261018/report.txt](gpuhome-gate1-profile-20261018/report.txt)。源文件 SHA256：`2198db06857c0b12d6d7b8817d7607ab1a3269fb67b1a93864dd863136eee204`。

Gate=1 CUDA Graph 图外 kernel 与同步分析（2026-10-18）

结论：先做现有 metadata glue graph 开关的 A/B；第一个源码优化候选是 verify/load_batch 的静态缓冲区准备和 small-copy 主机侧计划构建。Prefix .item() 确实同步，但其主机等待约99%以上与已有GPU活动重叠，不能把等待总时长当成可优化收益。

环境与采样
gpuhome-4090-30175，RTX4090 48GB，TP1；codex/hybrid-suffix-speculation @049db08e4d468f8d5b4ba026f07678c356310d51；palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4，正确独立unquant MTP配置，EAGLE，FP16，K8/topk1/drafttokens9，split gate1。
同四条Edit rewrite输入[515,396,457,491]tokens；C1用首条515token，C4用四条；temperature0/ignoreEOS。每请求至少完成32token后开始采样。Torch formal8 scheduler steps、mapping4steps，均保持CUDA Graph开启；mapping启用Python栈和shape只做归因。Nsight另起同配置进程，以graph粒度采两个独立20-step窗口，先C1再C4。Torch输出512token/request，Nsight768；初始prefill不在采样窗口。

低开销Nsight实测（图外kernel，graph内部node不展开）
| 并发 | 活动窗口ms | 活动union ms | 活动间空隙ms/% | 图外kernel次数 | 图外kernel SUM ms | graph次数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 291.792 | 261.144 | 30.647 / 10.50% | 1361 | 4.810 | 61 |
| 4 | 552.241 | 537.179 | 15.062 / 2.73% | 1381 | 14.851 | 73 |

C1约68次图外kernel/轮，C4约69次。计算本身短，真正可研究的是发射接缝。10.50%/2.73%只是这些steady窗口的活动间空隙，不是端到端加速预测；graph envelope内部空隙不可见。
Kernel SUM、GPU活动UNION与wall窗口不同：跨流重叠时间不应相加为延迟。Nsight graph envelope也不能代表SM忙碌率。

Torch正式trace核查（8轮，逐node采集有开销）
| 并发 | kernel总数 | 图外kernel次数 | 图外kernel SUM ms | kernel SUM ms | kernel UNION ms | 活动窗口ms | 无活动比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9145 | 545 | 1.860 | 118.448 | 101.192 | 125.820 | 19.49% |
| 4 | 11357 | 553 | 5.909 | 215.886 | 209.699 | 218.453 | 3.92% |

Torch C1空隙19.49%，而Nsight另一次低开销窗口为10.50%，不能把19.49%直接当无profile上限。NVIDIA说明graph粒度开销较低、node粒度可能产生明显开销；两次窗口的请求进度/分支混合也不同，不能把差值全部归因于profiler。https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html

表1：图外kernel与源码归因（20轮Nsight，基于graph-on映射）
| 类别 | C1次数 / GPU SUM ms | C4次数 / GPU SUM ms | 源码 |
| --- | --- | --- | --- |
| attention-metadata | 400 / 0.470 | 400 / 0.468 | triton_backend.py:508 _fill_kv_indptr_and_indices/:585 _update_target_verify_buffers/:682 _update_draft_extend_buffers/:735 _init_forward_metadata_out_graph |
| Mamba-state-commit | 280 / 2.868 | 280 / 12.728 | hybrid_linear_attn_backend.py:1505 update_mamba_state_after_mtp_verify → mamba_state_scatter_triton.py:695 scatter_mamba_states_after_mtp_verify |
| sampling-and-tree | 180 / 0.520 | 180 / 0.694 | eagle_utils.py:157 build_tree_kernel_efficient/:741 eagle_sample/:384 verify_tree_greedy_func |
| overlap-result-exchange | 160 / 0.332 | 160 / 0.329 | overlap_utils.py:433 _resolve_spec_extras/:544 resolve_seq_lens_cpu/:602 publish/:627 stash |
| verify-preparation | 100 / 0.149 | 80 / 0.106 | decode_cuda_graph_runner.py:1377 load_batch → cuda_graph_buffer_registry.py:388 fill_from/:47 _grouped_foreach_copy_ → small_copy.py:50 try_small_copy; also Mamba tracking/mrope preparation |
| draft-extend-preparation | 80 / 0.177 | 80 / 0.185 | eagle_worker_common.py:115 prepare_for_draft_extend; eagle_worker_v2.py:1116 _draft_extend_for_decode |
| hybrid-prefix-decision-and-replay | 20 / 0.023 | 20 / 0.024 | compact_graph.py:294 replay_compact → record.num_neural.item(); GPU row_compaction.py:24 compact_neural_rows |
| hybrid-plugin | 20 / 0.072 | 20 / 0.066 | hybrid_spec/backends/gpu_sa/__init__.py:141 stage_feedback |

归因流程：graph id=0 / graphNodeId=None确认图外 → GPU correlation匹配CUDA API → graph-on mapping CPU op/Python父栈 → pinned源码。Nsight和Torch使用不同demangler，按operation/grid/block及相邻五个launch上下文匹配。通用copy/index算子共享名字，表中分组是上下文匹配归因，不是跨run的external-id直接连接。C4另有20次vectorized_gather_kernel未匹配到确切源位置，保留未知。其余零散draft preparation/other见nsys-summary.json。

同步开销：实际源头和暴露部分
| 并发 | StreamSync次数 | 主机等待ms | 等待内无GPU活动ms | EventSync总ms | GraphLaunch平均us |
| --- | --- | --- | --- | --- | --- |
| 1 | 20 | 163.604 | 0.985 | 0.067 | 69.55 |
| 4 | 20 | 404.817 | 2.034 | 0.068 | 134.36 |

所有映射到的cudaStreamSynchronize来自 compact_graph.py:294 replay_compact → num_neural.item() → aten::_local_scalar_dense。C1约99.40%、C4约99.50%的主机等待与GPU活动envelope重叠。移除同步并不能移除其等待的prefix/前序GPU计算。EventSynchronize来自batch_result_processor.process_batch_result_decode，每次约3.3us；Torch末尾cudaDeviceSynchronize来自profiler_manager._stop_profile，是采样收尾，不列入稳态瓶颈。
Nsight20轮全部D2H合计：C1 880字节/77.4us GPU copy，C4 3280字节/69.8us GPU copy；H2D分别328/1312字节。DtoD约0.9/3.6MB。4字节count属于延迟/依赖问题，不是PCIe带宽耗尽。TP1没有多卡NCCL结论。

按下一个GPU消费者分类的空隙（只作接缝定位，不作因果加速估算）
| 下一消费者类别 | C1空隙ms | C4空隙ms |
| --- | --- | --- |
| verify-preparation | 10.255 | 3.200 |
| attention-metadata | 6.544 | 2.904 |
| copy | 4.659 | 2.129 |
| graph | 4.017 | 1.854 |
| sampling-and-tree | 3.551 | 1.632 |
| Mamba-state-commit | 0.167 | 0.376 |

C1 verify-preparation下一消费者前约10.255ms空隙，其中9.283ms在其关联launch API进入之前；多次约220us空隙结束于registry.fill_from发射的_small_copy_kernel。attention metadata前另有6.544ms。这里可能包含更早的Python处理、其他API、调度/OS；未做CPU采样，因此不能把整段空隙归给某个Python函数。

表2：优化与重叠优先级
| 优先级 | 目标 | 证据 | 下一步与边界 |
| --- | --- | --- | --- |
| P0: smallest first experiment | Existing metadata glue graph, currently disabled | Attention metadata20 outside launches/round, C1 next-consumer gaps6.544ms/20steps. Existing MetadataGlueGraph in049 captures this exact device prep. | A/B SGLANG_ENABLE_METADATA_GLUE_GRAPH=1 with gate1/K8 unchanged; verify real capture/no fallback, outside-kernel count reduction, unprofiled throughput and acceptance/output/state correctness. Experimental/defaultFalse. CurrentEAGLE,TBOoff,pdmuxoff,noLoRA and padding-free C1/C4 satisfy visible routing conditions; hybrid metadata snapshot/replay correctness remains to be tested. Extra graph launches consume part of the saved dispatch cost. No speedup measured. |
| P1: first source optimization | load_batch / registry fill / small-copy HOST preparation | C1 gaps before verify-preparation consumers10.255ms/20steps;9.283ms precede entry into their correlated launch API. Repeated ~220us gaps end at _small_copy_kernel launched by registry.fill_from. | Cache per-mode/shape slot traversal and copy layout with guards; reduce repeated views/shape/stride extraction and O(N²) alias-range validation in try_small_copy when tensor identity/layout/pointer contracts permit. Consider writing producers directly into fixed slots. Small-copy GPU fusion already active. These gaps include Python and other host work before the consumer; no CPU sampling was collected, so do not assign the full10.255ms to one function or call it removable latency. Dynamic source pointers/padding/source_fn/post_fill behavior must remain correct. |
| P2 | Sampling/tree/commit device-only slices | Sampling/tree9 launches/round, C1 preceding gaps3.551ms/20steps. Mamba commit14 launches/round and most outside GPU duration. | After metadata A/B, consider a fixed-shape post-verify glue graph; reuse int32 commit indices/source rows across already-fused SSM and conv scatter wrappers. Sampling/acceptance and cache rollback determine subsequent inputs; preserve device dependencies. Existing fused_commit_track_indices and fused scatter must not be reimplemented as supposedly missing features. |
| P3 for current workload | Prefix item()/device-conditional graph | NsightC1 StreamSynchronize163.604ms, but no-GPU portion0.985ms; C4404.817ms versus2.034ms. All mapped stream syncs are the prefix scalar read. | Keep as a later device-conditional control experiment if targeted metadata changes leave a measurable exposed seam. ~99.4%/99.5% of these host spans overlap GPU activity envelopes. Conditional graphs do not erase the producer computation; graph-level envelopes hide internal gaps. Event sync ~3.3us/call is not a leading target. |

现有开关的具体位置：environ.py:1420默认False；decode_cuda_graph_runner.py:496创建MetadataGlueGraph，:1548在raw_bs==padded_bs且TBO/pdmux/LoRA关闭时使用；metadata_glue_graph.py定义warmup、capture、forward_metadata恢复和失败回退。当前环境变量未设置，visible条件EAGLE/TBOoff/pdmuxoff/noLoRA满足；本轮没有打开该开关，也没有测得它的吞吐收益。
已有glue只覆盖attention metadata，不覆盖registry.fill_from。因此它是最低改动的首个A/B，但不能假设它会消除全部10.5%空隙。也不能捕获某一请求的动态FB源指针后对后续请求盲目复用。

表3：Fusion模式检查
| 模式 | 当前判定 | 证据 | 建议 |
| --- | --- | --- | --- |
| Attention arange/cumsum/arithmetic/index packing → metadata glue graph | Existing opt-in path, not active | runner/metadata_glue_graph.py; decode_cuda_graph_runner.py:496,:1548; environ.py:1420; 20 outside kernels/round. cumsum of one/four rows still invokes CUB init+scan; static qo_indptr arange is rebuilt each replay. | First test existing graph; only then consider specialized fused metadata builder or constant reuse for fixed(bs,width), preserving dynamic KV lengths/indices. |
| Many FB copies → _small_copy_kernel | Already fused; host preparation remains | cuda_graph_buffer_registry.py:47,:388; kernels/ops/memory/small_copy.py:50; Grouped Triton kernel exists and executes. try_small_copy rebuilds sizes/strides/pointer ranges and performs nested alias checks before launch. | Investigate cached, guarded copy plans/fast path; do not claim one-kernel copy fusion is absent. |
| Verify commit index math + Mamba scatter | Core index math and gather/scatter already fused | kernels/ops/speculative/eagle.py:158 prepare_verify_commit_outputs; spec_utils.py:883 fallback; mamba_state_scatter_triton.py:233,:695; Observed _verify_commit_outputs_kernel (prepare_verify_commit_outputs) already prepares accepted/track indices; fused_commit_track_indices exists as a fallback, not asserted active in this capture. Masked SSM/conv scatter spans all layers. Wrappers still perform casts and source-row creation for normal and track destinations. | Deduplicate wrapper preparation or graph compatible commit slice. C4 bulk copy is larger, but kernel sums overlap other work and are not additive savings. |
| Residual+RMSNorm; sigmoid+delta-rule | Already active graph-internal kernels | layernorm.py; kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py; Formal traces retain FusedAddRMSNorm and fused_sigmoid_gating_delta_rule_update_kernel. | No missing-fusion claim; graph-internal source needs prior eager mapping, not graph replay external IDs. |

建议验证顺序：先在gate1/K8不变时测metadata glue off/on；确认没有capture fallback，并比较图外调用数、足够长且多次重复的无profiler吞吐、接受长度和输出/状态正确性。若收益有限，再对load_batch/copy-plan作更细CPU采样或加轻量范围标记，然后实现带稳定性guard的专用准备路径。Mamba wrapper去重和device conditional graph排在其后。
优化验证至少覆盖固定B1/B4和混合/变批次、padding，避免只在捕获时地址和mask相同的单个请求上成立；cache rollback/track状态与原实现比较。任何数值/输出等价或吞吐加速均需独立实测，不能由profiler表格推出。

证据与服务
原始Torch traces四份、Nsight reports两份及SQLite、完整请求/响应、调用栈映射、同步父栈和pinned源码均保留；automatic-c1-unreviewed.txt仅是工具线索，其graph内部attention→small_copy/GDN→stash误归因已否决。
后续源码分析仅使用049db08e4快照；本地正在进行的未提交改动未部署。模型与gate1配置保持；采样完成后Nsight进程退出，服务PID 27602 继续运行，health200及32token生成通过。
远端 /root/rivermind-data/benchmarks/gate1-profile-20261018；本地analysis.json/torch-summary.json/nsys-summary.json为机器可读数据。evidence.tar.gz保留原始文件与脚本。

<a id="record-31"></a>

## 记录 31：gpuhome-profile-4090-20261018/report.txt

来源：[gpuhome-profile-4090-20261018/report.txt](gpuhome-profile-4090-20261018/report.txt)。源文件 SHA256：`838a6edb8d411a8a401c28fc8df98a6aacc38efba99f8c5ded7d33908a109c94`。

4090 hybrid speculative decoding 验证与瓶颈分析（2026-10-18）

结论：当前 split hybrid-spec 有实测收益；进一步优化应优先减少神经草稿的全词表投影和混合批次中的尾部开销。通信带宽没有成为本次 TP1 的主要瓶颈。同步与 CUDA Graph 需要以暴露在 GPU 外的时间衡量。

部署与基线修正
服务器 SSH Manager：gpuhome-4090-30175；RTX4090 48GB，TP1；远端 codex/hybrid-suffix-speculation @049db08e4d468f8d5b4ba026f07678c356310d51。
模型 palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4，内置 MTP/EAGLE（不是外部 EAGLE3），FP16，K8/topk1/drafttokens9，max_running_requests4，context16384，mem_fraction0.8。
发现 checkpoint 的 MTP experts 是 BF16，但继承 GPTQ 时创建量化参数，loader 找不到 w13_weight 后跳过专家权重。仅加 --speculative-draft-model-quantization unquant 仍会重新自动识别 GPTQ。
已用独立 MTP-unquant 配置目录移除量化元数据，并显式传 draft路径与unquant。权重全部符号链接、原模型不变，target 仍 GPTQ4。MTP加载由2.30GB变为3.43GB；原生MTP接受长度从约1.1恢复到约6.5–6.9（独立512token请求）。
同机同负载错误配置→修正：native C1 56.86→315.65tok/s（5.55x），C4 174.65→807.87（4.63x）。这是加载修正收益，不是 hybrid 算法优化收益。旧报告和本轮错误配置参考数据已标记无效，原始数据保留。

无 profiler 的性能对照：output tokens/s，3次均值[min,max]
| 配置 | C1 | C4 | 相对正确native C1/C4 |
| --- | --- | --- | --- |
| no-spec | 161.37 [161.31,161.47] | 459.15 [440.59,468.43] | -48.9% / -43.2% |
| native | 315.65 [302.84,326.06] | 807.87 [804.75,813.66] | +0.0% / +0.0% |
| full-g2 | 310.29 [269.97,335.14] | 812.34 [804.28,825.37] | -1.7% / +0.6% |
| split-no-skip-g2 | 309.54 [307.57,313.36] | 755.65 [720.11,814.02] | -1.9% / -6.5% |
| split-g2 | 383.70 [360.38,408.29] | 878.66 [864.85,893.68] | +21.6% / +8.8% |
| split-g0 | 404.95 [384.75,422.00] | 867.32 [862.08,875.30] | +28.3% / +7.4% |

full只替换草稿，仍执行神经计算，收益接近零。split确实跳过计算。split_no_skip用于隔离拆图开销，不能作为生产收益配置。gate0在C1较快、C4稍慢，本轮没有证据证明应普遍改成0；保留gate2作为当前配置。
这些是四条 Edit rewrite 请求的受控诊断：C1计时只采491-token一条，C4采四条；每请求固定256输出，不代表完整数据集/生产容量。吞吐包含prefill/TTFT；TPOT详见aggregate.json。不同模式可能因数值/输出差异产生接受长度变化，数据没有强制token完全一致。

Kernel与源码归因（均为8轮正式trace的kernel duration SUM，不等于wall time）
| 热点 | native C1 / split C1(ms) | 源码与置信度 | 含义 |
| --- | --- | --- | --- |
| MTP vocabulary head / cuBLAS GEMV, grid62080 | 66.462 / 16.603 | python/sglang/srt/layers/logits_processor.py:970 _compute_lm_head → qwen3_5_mtp.forward; High: eager correlation + FP16 [1,2048]×[2048,248320], same kernel/grid in graph traces. | Native 8 projections/round; sampled split window 2/round. Each FP16 head has ~1.017 GB logical weights; bandwidth interpretation is an inference, not NCU DRAM measurement. |
| GPTQ Marlin MoE | 31.56 / 31.6 | python/sglang/srt/layers/moe/moe_runner/marlin.py:115 fused_experts_none_to_marlin; High: mapped operator sglang::fused_marlin_moe; target is GPTQ, corrected MTP experts are unquantized. | Target verify cost remains; native C4 ~72.55 ms/8 sampled rounds. |
| CUTLASS x2 dense GEMM | 39.75 / 38.9 | python/sglang/srt/layers/quantization/unquant.py:494 apply; Mapped generic dense application; multiple projection sites, not one exclusive layer. | 1648 launches; do not label the entire family MoE activation or LM head. |
| decode attention _fwd_kernel | 7.94 / 8.16 | python/sglang/kernels/ops/attention/decode_attention.py:992 _decode_softmax_reducev_fwd; Mapped; short generic kernel name requires source check. | Sequential attention/verify dependency; overlap feasibility unproven. |
| fused_sigmoid_gating_delta_rule_update_kernel | None / 4.76 | python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:531; High: model/source mapping; this is GDN, not shared-expert append. | Already fuses sigmoid gating and delta rule; automatic shared-expert fusion suggestion rejected. |

映射证据：eager trace 的 cudaLaunchKernel correlation → aten::mm[1,2048]×[2048,248320] → logits_processor._compute_lm_head → qwen3_5_mtp.forward；正式graph trace匹配同名kernel/grid。eager时间未用于性能结论。

通信、同步与重叠机会
| 优先级 | 方向 | 证据 | 建议及边界 |
| --- | --- | --- | --- |
| P1 | Neural draft depth / full-vocabulary projection count | C1 LM-head 64→16 calls; corrected split +21.6% throughput vs native. | Test smaller neural K while preserving longer SA drafts, cost-aware gating, and full-head projection reduction. No such source change deployed or speedup measured here; acceptance and output validation required. |
| P1 | Mixed C4 tail economics | ~62.0% request rows skip, but only30.2% rounds all-skip;48.4% mixed rounds. | Optimize decisions around whether a neural head execution is needed; compare neural-depth/SA-length combinations. Existing GPU row compaction already works. Smaller batch still reads the full vocabulary weight; scheduling must account for TTFT/TPOT. |
| P1 | Graph-external metadata glue | Nsight20-step C1 has1361 graph-external kernel activities (~68/round),4.72ms kernel sum, and30.10ms between-activity gaps. Many kernels are copies,arange,cumsum,indexing and Mamba-state scatter. | Fuse static-shape index/metadata preparation or capture device-only preparation/commit slices using persistent buffers. Family-level eager mapping identifies Triton attention metadata, CUDA graph buffer fill and post-verify Mamba state commit. Exact graph-on launch site needs confirmation before implementation; actual batch length and rollback dependencies must be preserved. |
| P2 | Prefix scalar D2H and host synchronization | Split C1 CPU decision66.41ms/8 spans, only0.72ms has no concurrent kernel (~98.9% overlapped). All D2H348B/8-step window. | Investigate event timing/async metadata and device-side conditional graphs only against actual exposed gaps. Host wait includes GPU computation;16ms/round telemetry is not removable latency. TP1 has no measured NCCL issue. |
| P2 | CUDA Graph replay / inter-graph gaps | Torch split C1 kernel-union gap18.45%, native1.80%; large gaps intersect instrumented cudaGraphLaunch. | Cross-check lower-overhead Nsight graph-level trace before prioritizing replay changes. Node tracing may amplify launch overhead; graph envelopes cannot measure internal node gaps. |
| P3 | Attention/GDN independent-stream overlap | Trace has existing multiple streams and sequential model dependencies. | Only overlap producer/consumer-independent work after dependency proof. Exclusive kernel duration is not automatically overlap headroom; no speculative parallelism claim. |

split gate2 的无profile诊断：C1全跳过轮次51.2%；C4行跳过62.0%，全跳过轮次30.2%，混合轮次48.4%。计数包含benchmark warmup+计时请求。GPU compaction已存在，不能重复推荐为缺失功能。
拆图同步读取的是4字节count。C1 trace所有D2H合计348字节，CPU decision66.41ms中约65.69ms与kernel重叠；不应把计数器的12.77ms/round主机等待解释为同量可优化延迟。本机只有1张GPU，没有NCCL多卡通信验证。

CUDA Graph独立核实
Nsight graph级采样结果：{"gpu_activities": [{"table": "CUPTI_ACTIVITY_KIND_MEMCPY", "count": 262, "sum_us": 247.296}, {"table": "CUPTI_ACTIVITY_KIND_KERNEL", "count": 1361, "sum_us": 4715.022}, {"table": "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "count": 61, "sum_us": 278933.909}, {"table": "CUPTI_ACTIVITY_KIND_MEMSET", "count": 40, "sum_us": 36.448}], "gpu_activity_window_us": 314007.05, "gpu_activity_envelope_union_us": 283902.435, "between_activity_gap_us": 30104.615, "between_activity_gap_pct": 9.587241751419276}
独立采样有61次graph活动、1361次图外kernel活动（kernel sum4.715ms）。GraphLaunch平均67.73us、最大186.13us；最长活动间空隙284us。说明仍有主机调度/元数据开销，但不能把Torch的18.45%视为无profile上限，也不能把187ms StreamSynchronize等待视为新增耗时。Graph envelope内部空隙不可见，两次采样窗口也不同。
Nsight CUDA Graph API：{"cudaGraphLaunch_v10000": {"count": 61, "sum_us": 4131.278, "mean_us": 67.72586885245902, "max_us": 186.125}, "cudaStreamSynchronize_v3020": {"count": 20, "sum_us": 187043.843, "mean_us": 9352.192149999999, "max_us": 10090.909}, "cudaEventSynchronize_v3020": {"count": 20, "sum_us": 64.063, "mean_us": 3.20315, "max_us": 4.068}}
Graph-level activity envelopes hide gaps between nodes inside a graph. This measures between-activity gaps and API overhead, not kernel occupancy or SM utilization. Capture is separate 20-step warmed C1 workload; Torch traces use 8 steps.
NVIDIA文档说明graph node级跟踪可能有明显开销，graph级跟踪开销较低：https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html
Conditional CUDA Graph是可研究方向，不是已验证收益；且graph envelope隐藏内部node间隙：https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-programming-guide/index.html

Fusion候选与已有融合
| 模式 | 判定 | 证据/收益边界 | 验证要求 |
| --- | --- | --- | --- |
| LM-head GEMV + argmax/top1 | Candidate, not implemented | Confirmed full-vocabulary draft head followed by speculative top1 handling. May remove logits traffic/launches; cannot remove ~1GB weight read per projection. Prioritize fewer projections first. | Compare indices, logits rounding and acceptance on corrected MTP; benchmark without profiler. |
| GDN input qkvz + BA projection packing | Dispatch-limited candidate | Pinned qwen3_5.py finalize_fused_in_proj enables packed path for BF16; serving here is FP16 due GPTQ Marlin dtype requirements. Potential FP16 extension, subject to actual shapes and backend compatibility; not a measured speedup. | Preserve conv/SSM dtypes and dual-stream dependencies; compare numerical error and serving timing. |
| Residual add + RMSNorm | Already active | Formal split trace has FusedAddRMSNormKernel ~1.40ms. No new missing-fusion claim. | Do not optimize by naming alone. |
| Sigmoid gate + delta-rule update | Already active; unrelated to shared-expert append | fused_sigmoid_gating_delta_rule_update_kernel, mapped FLA/GDN source. Automatic analyzer shared-expert recommendation is a false attribution and is excluded. | Review actual shared-expert path separately before proposing any append fusion. |
| Generic dense GEMM + MoE activation/quant | Automatic recommendation rejected as unsupported | Analyzer grouped unquantized dense projections and LM head into a broad family. The family duration is not removable MoE activation work. | Need exact producer/consumer, shape, scale layout and rounding evidence. |

自动分析器输出mapping-formal-analysis.txt仅作线索；其中把GDN误标shared-expert append，以及把通用dense/LM-head时间归入MoE activation的建议已人工否决。以上三张表为最终人工校核结论。

验证范围与复现
260 CUDA/plugin tests passed，184 optional-Arctic tests skipped，19 warnings；真实模型完成所有配置的请求与CPU/GPU trace采样。
性能三次计时与profile分开；profile每请求512输出，在每条请求完成至少32token后采8 scheduler steps。graph-off映射采4步；Nsight采20步。所有原始trace和日志保留。
逐字输出对照：{"1": [{"case": "native", "comparisons": [{"equal": false, "common_prefix_chars": 1239, "text_chars": [2478, 2434]}]}, {"case": "full-g2", "comparisons": [{"equal": true, "common_prefix_chars": 2434, "text_chars": [2434, 2434]}]}, {"case": "split-g2", "comparisons": [{"equal": true, "common_prefix_chars": 2434, "text_chars": [2434, 2434]}]}, {"case": "split-g0", "comparisons": [{"equal": true, "common_prefix_chars": 2434, "text_chars": [2434, 2434]}]}], "4": [{"case": "native", "comparisons": [{"equal": true, "common_prefix_chars": 2434, "text_chars": [2434, 2434]}, {"equal": false, "common_prefix_chars": 2118, "text_chars": [2161, 2209]}, {"equal": false, "common_prefix_chars": 45, "text_chars": [2494, 2521]}, {"equal": false, "common_prefix_chars": 823, "text_chars": [1800, 2060]}]}, {"case": "full-g2", "comparisons": [{"equal": true, "common_prefix_chars": 2434, "text_chars": [2434, 2434]}, {"equal": false, "common_prefix_chars": 2118, "text_chars": [2187, 2209]}, {"equal": false, "common_prefix_chars": 45, "text_chars": [2494, 2521]}, {"equal": false, "common_prefix_chars": 1317, "text_chars": [2060, 2060]}]}, {"case": "split-g2", "comparisons": [{"equal": true, "common_prefix_chars": 2434, "text_chars": [2434, 2434]}, {"equal": false, "common_prefix_chars": 2118, "text_chars": [2234, 2209]}, {"equal": false, "common_prefix_chars": 45, "text_chars": [2488, 2521]}, {"equal": false, "common_prefix_chars": 823, "text_chars": [2033, 2060]}]}, {"case": "split-g0", "comparisons": [{"equal": false, "common_prefix_chars": 1239, "text_chars": [2478, 2434]}, {"equal": false, "common_prefix_chars": 2118, "text_chars": [2234, 2209]}, {"equal": false, "common_prefix_chars": 2279, "text_chars": [2523, 2521]}, {"equal": false, "common_prefix_chars": 823, "text_chars": [1917, 2060]}]}]}
以上逐字对照只覆盖四条样本，不能替代完整Edit质量指标。原生MTP与target-only也可能不同；尚未定位为数值差异或其他原因，不宣称所有模式逐token等价。
远端全部证据：/root/rivermind-data/benchmarks/hybrid-profile-4090-20261018；有效结果在correct-draft/。根目录native/full-g2及unquant-flag-only-reference是错误draft配置参考，不可混入有效结果。
本地analysis.json/aggregate.json/trace-statistics.json提供机器可读结果；evidence.tar.gz保留完整日志、实际启动参数、模型配置收据、source snapshots、原始Torch traces和Nsight报告。
没有修改或部署本地未提交的compact-verify/neural-depth代码；本报告未声称候选kernel或CUDA Graph优化已经落地。

最终服务已恢复为正确MTP draft配置 + split gate2；本机SSH转发 http://127.0.0.1:39922 ，health HTTP200，32token生成通过；远端git工作区干净。

<a id="record-32"></a>

## 记录 32：gpuhome-split-gate1-20261018/report.txt

来源：[gpuhome-split-gate1-20261018/report.txt](gpuhome-split-gate1-20261018/report.txt)。源文件 SHA256：`3a35fbfbdfb13c06a9c5f69ba81d2fac715abdc46c9813b4473bdc6206256085`。

split gate=1验证（RTX4090 48GB, 2026-10-18）

同机新测gate2对照和gate1，K8/topk1/drafttokens9、模型/独立unquant MTP draft/FP16/上下文16384/maxrequests4保持一致。远端源码049db08e4，未部署本地未提交代码。

| 配置 | 并发 | 输出tok/s均值[min,max] | TPOT ms | 接受长度(独立512token probe) | SA行比例 | 全跳过轮次比例 |
| --- | --- | --- | --- | --- | --- | --- |
| split-g2-control | 1 | 377.06 [370.33,380.51] | 2.425 | 7.529 | 50.8% | 50.8% |
| split-g2-control | 4 | 899.75 [891.37,912.46] | 3.592 | 6.225 | 62.6% | 30.4% |
| split-g1 | 1 | 395.38 [382.88,407.75] | 2.309 | 7.314 | 51.2% | 51.2% |
| split-g1 | 4 | 918.58 [903.89,927.50] | 3.535 | 6.522 | 62.4% | 30.6% |

相对本轮gate2对照：
并发1：吞吐+4.86%，三次原始值[395.48953210736505, 382.8849963902769, 407.75276515068873]。
并发4：吞吐+2.09%，三次原始值[924.3493741484493, 927.4972181921936, 903.8907399612036]。

计时使用官方Rust client，预热和计时分开，每请求固定256输出，3次重复；并发1使用seed42洗牌后的491token请求，并发4使用四条Edit rewrite请求。接受长度来自另外的512token生成总completion_tokens/spec_verify_ct，不混入吞吐计时。
SA/skip计数范围为每次官方benchmark的warmup+计时请求。
没有启用Torch或Nsight profiler。源码compact_graph.py:63 prefix_width=max(1,gate)，gate1仍至少执行一步神经prefix；本轮没有新增kernel性能归因。

逐字输出对照（gate1对gate2）：
[
  {
    "concurrency": 1,
    "comparisons": [
      {
        "prompt_index": 0,
        "equal": true,
        "common_prefix_chars": 2434,
        "text_chars": [
          2434,
          2434
        ]
      }
    ]
  },
  {
    "concurrency": 4,
    "comparisons": [
      {
        "prompt_index": 0,
        "equal": true,
        "common_prefix_chars": 2434,
        "text_chars": [
          2434,
          2434
        ]
      },
      {
        "prompt_index": 1,
        "equal": false,
        "common_prefix_chars": 2111,
        "text_chars": [
          2195,
          2120
        ]
      },
      {
        "prompt_index": 2,
        "equal": false,
        "common_prefix_chars": 45,
        "text_chars": [
          2497,
          2521
        ]
      },
      {
        "prompt_index": 3,
        "equal": false,
        "common_prefix_chars": 823,
        "text_chars": [
          1869,
          2060
        ]
      }
    ]
  }
]

greedy文本相同仅代表这些样本相同；如不相同，不能据此声称逐token等价。本轮不替代完整Edit质量评测。
gate2先测且使用已有进程，gate1后测且重新启动，时间顺序/进程状态仍是潜在干扰；短样本均值必须结合范围解读。
服务最终保留gate1，接口http://127.0.0.1:39922，实际参数/日志/最终健康与生成收据见service.json、final-server-info.json和final-smoke.json。
