# Qwen3.5 MTP + SA 接受长度回退分析

修改前源码：`588bbaa4239189813eafd64b293e56c91e1d9fc8`。模型为
Qwen3.5-35B-A3B-GPTQ-Int4，单张 A100 PCIe 40GB，TP=1，K=8，原生
MTP 使用 `EAGLE` 算法身份。固定 sa-spec 原始 completion 提示模板、
`glaiveai/code_edits_sample` 数据和 2048 输出上限。

## 独立检查与同位置证据

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

### 当前上游与历史参考的区别

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

## 修复策略与边界

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

## A100 复测

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
两组流水线均完成，测试服务已停止。配置仍属实验用途；独立 SA stream、
MTP 提前退出与多 GPU 新策略验收均未实现或完成。
