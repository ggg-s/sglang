# 验证记录

## RTX 6000D / Qwen3.6 实机验证（2026-10-15）

最终 SWE 复测与回归测试代码：`1121bd1d0e`，分支 `codex/hybrid-suffix-speculation`。

环境：单张 NVIDIA RTX 6000D（85651 MiB），Ubuntu 22.04，Python 3.12.13，PyTorch 2.14.1 + CUDA 13.0，Transformers 5.19.0，Triton 3.8.0。SGLang 从当前分支 editable 安装，Rust 扩展与 SM120 JIT 算子已编译；GPU SA 扩展经过 nvcc 编译和执行。Python 源码以 Git HEAD 为准，安装时生成的包版本仍含旧提交号。

模型为 `Qwen/Qwen3.6-35B-A3B-FP8`，架构是 Qwen3.5 MoE 的 GDN/全注意力混合结构，使用模型自带 MTP。扩展只开放该模型族的 `--language-only` 路径，保留原生 GDN verify/commit 与 mRoPE 位置更新，实际多模态输入在 HTTP 请求校验阶段拒绝。

数据与用途：

- GSM8K：`openai/gsm8k`，固定版本 `740312add88f781978c0658806c59bc2815b9866`；使用标准答案评测正确率，temperature=0、关闭 thinking、最多 2048 输出 token、并发 4。
- SWE：`nebius/SWE-rebench-openhands-trajectories`，固定版本 `35455389ab51bf5e2306bfd436ef72d0f98bf882`，67,074 条轨迹；仅用于服务性能回放，不执行工具、不计算软件任务完成率。固定选取 16 个完整对话前缀，输入 6909–7152 token（平均 7031.25），最多输出 128 token，分别测并发 1/4。
- 两个数据集及 Qwen3.6-35B-A3B-FP8、Qwen3-32B-FP8、Qwen3-0.6B 均位于服务器数据盘 `/root/rivermind-data`。

已确认的边界：

- K=3 时，eager 与 CUDA Graph 的 4 条单请求 greedy 输出均与原生 MTP 逐 token 一致；K=1 Graph 的单请求对照也通过。诊断计数确认各路径中 SA 实际参与并产生被接受的草稿。
- K=1/3 的同一组并发 4 输入均有 1 条 hybrid/native 输出不一致；原生 MTP 自身在并发 1/4 下也有 3 条不同。target-only 与原生 MTP 的逐 token 对照也存在差异。因此**严格逐 token 等价验收尚未通过**，不能用 GSM8K 答案正确率替代这一结论；现有证据还不能把全部差异归因到批大小数值变化或 SA 实现中的某一项。
- 16 题 GSM8K 冒烟：原生 MTP 15/16，MTP+SA 16/16，均无请求失败、无截断。小样本差异不证明质量提升。
- Windows 核心/真实 draft 函数体回归测试：97 passed。LoRA 默认值 `None` 的回归用例已先复现旧实现失败，再验证修复通过。

128 题 GSM8K（同一批题目、标准答案已逐项核对）：

| 配置 | 正确数 / 总数 | 正确率 | 请求错误 | 达到输出上限 |
| --- | --- | --- | --- | --- |
| Target-only / Graph | 125 / 128 | 97.65625% | 0 | 1 |
| 原生 MTP / Graph，K=3 | 125 / 128 | 97.65625% | 0 | 1 |
| MTP+SA / Graph，K=3 | 125 / 128 | 97.65625% | 0 | 1 |

三个配置未答对的 ID 均为 `gsm8k-12`、`gsm8k-93`、`gsm8k-119`，但生成文本与截断题目不完全相同。`gsm8k-paired.json` 保存逐题标准答案、预测、判分、停止原因与题目摘要。结果覆盖固定 128 题，不是完整 1319 题测试集，也不证明逐 token 等价。

性能测量按实际并发预热，包括该批大小的内核；测量的是端到端输出吞吐和平均请求延迟，包含 HTTP、tokenization、prefill 与 decode，不是 TPOT。早期 eager 并发对照只做单请求预热，不能用于公平的加速比结论。正式 SWE 复测对全部 16 个输入预热后清空缓存再计时，每个并发级别独立执行；测量期间允许正常的前缀缓存复用。SA 参数为 `min_match_len=2`、K=3、最多 4 个请求、每请求最多 8192 个历史 token；这是用于验证的固定配置，未进行参数寻优。各配置均为 TP=1、关闭 overlap、上下文上限 8192、静态显存比例 0.8。

| 配置 / Graph | 并发 1 输出 token/s | 并发 1 平均延迟 / s | 并发 4 输出 token/s | 并发 4 平均延迟 / s |
| --- | --- | --- | --- | --- |
| Target-only | 83.46 | 1.445 | 177.74 | 2.574 |
| 原生 MTP，K=3 | 120.25 | 1.005 | 317.00 | 1.480 |
| MTP+SA，K=3 | 101.49 | 1.193 | 149.59 | 3.130 |

本轮 MTP+SA 相对原生 MTP 吞吐下降约 15.6%（并发 1）和 52.8%（并发 4），**没有获得加速**。这是固定 16 条长输入的单轮实测，不是调优后的性能上限或统计置信区间。SA 在两组测量中分别参与 345/712、316/693 个验证请求；SA 轮次的平均接受长度为 2.03/2.10 token，neural 轮次为 3.37/3.37 token（均含 bonus）。这些是不同请求轮次的来源统计，不能当作同一步的反事实对照；尚未用 profiler 分解开销。最终性能以 `swe-controlled/summary.json` 为准，`swe-gsm8k` 内的早期性能使用不同缓存流程，仅保留为参考。

Linux 完整扩展回归：**104 passed，1 skipped**（单 GPU 无法运行双 GPU NCCL 分支），包括实际 CUDA SA 与 Graph replay。原生 `test_eagle_worker_v2_topk1_fastpath.py`：**12 passed，7 subtests passed**，在完整 SGLang 依赖环境中执行。`uv pip check` 检查 224 个包通过；本轮改动的 Python 文件通过 Ruff 语法/未定义名称检查与格式检查。

原始证据目录：服务器 `/root/rivermind-data/validation/hybrid-linear`（命令、服务日志、token 序列、诊断、16 题预测）、`/root/rivermind-data/validation/swe-gsm8k`（128 题正确率及初轮性能）、`/root/rivermind-data/validation/swe-controlled`（最终性能）；当前仅有 TP=1 实机证据，不能外推多 GPU NCCL/TP 行为。

## 首版 Windows 验证历史

日期：2026-10-15。SGLang 基线：`a12814c1585b0c954234f776bba93dbdd04b1151`。

### 当时已运行

环境：Windows，Python 3.11.15，PyTorch 2.14.1+cpu，pytest 9.1.1。

| 检查 | 结果 |
| --- | --- |
| Python 核心与真实 draft_forward 函数体测试 | 57 passed |
| TP 共识与桥接：真实 Gloo 进程组 TP=1/2/4 | 3 passed，含全局 ranks [1,3] 子组 |
| CUDA 共用 C++ 状态机的宿主编译/运行 | 1 passed，Zig 0.16.0，C++17 |
| CUDA 专项测试 | 3 skipped：当前测试环境无 CUDA 版 PyTorch / nvcc |
| 双 GPU NCCL + 实际 SA + Graph replay | 1 skipped：当前环境无双卡 CUDA/NCCL |
| Ruff F401/F811/F821/F823/UP037 | 通过 |
| Ruff format / isort | 通过 |
| `git diff --check` | 通过 |
| 独立 wheel 构建 | 通过，包含 CUDA `.cu` 与共享 `.h` 源码、插件 entry point |

合计 **61 项通过，4 项 CUDA 跳过**。本地隔离虚拟环境为仓库下 `.venv-hybrid-spec311`，构建产物在 `.tmp/hybrid-spec/dist/`，均不作为源码交付。

关键行为证据：

- Python SA 与独立穷举子串搜索 oracle 比较，每次追加后检查最长重复后缀和有效续写。
- CUDA 使用的相同 C++ `Automaton` 经过二元序列长度 1–12 的完整枚举、256 组确定性长序列、512 个重复 token 与容量越界保护测试。
- 直接执行当前仓库的 `draft_forward` 方法体：K=1/2/4，插件关闭、全 neural、全 SA、混合来源，普通列表和快速矩阵路径，初始及后续 hot-token 映射；逐位置断言 verify 链等于下一次模型输入，forward 次数保持 K-1。
- 请求顺序变化、槽位不足、历史超容量、重复历史、版本间隙/冲突、重建、RID 复用和 reset。
- 固定地址 Graph buffer 的 mask 切换和 padding 清除；这是 CPU buffer 语义验证，不能替代 CUDA Graph replay。
- Gloo 多进程：K=1/3，混合/全 SA/全 neural、单 rank 全部回退、候选分歧、非零全局源 rank；同步结果经过实际 GraphBuffers staging 和逐步选择，检查 batch 缩小时 padding 清除。实际 DraftExtension.bind 路径还覆盖单 rank seq_lens 不匹配、请求重排、共识后的统计和状态释放。
- 非 greedy、grammar、beam、watermark 请求在参数校验阶段拒绝；不支持的启动配置明确失败。
- 来源归因中 correct drafts 不含 bonus，accept tokens 包含 bonus。

### 当时尚未验收

首版 Windows 验证阶段没有运行 Linux SGLang 服务，没有通过 nvcc 编译/执行 CUDA 扩展，也没有运行真实 CUDA Graph replay。未完成 MTP、EAGLE3、Standalone 的逐模型 target greedy 对照、真实 KV 生命周期、abort/retract/flush 服务测试或性能基准。不能据此声称任何模型已可用于生产，也不能声称有吞吐/TPOT 收益。

多 TP 已解除配置限制并接入每轮候选共识；Gloo 只验证进程间算法与 CPU 桥接，不能证明 NCCL stream 顺序、真实 TP 模型分片或多卡 Graph 的正确性。`test_tp.py` 的 NCCL 分支已准备好验证实际 GPU SA、共识、Graph replay 和 padding，需在双 GPU 环境执行。

现有 SGLang CI 测试文件 `test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py` 也已补充真实 worker 的覆盖递推测试，使用仓库 `CustomTestCase` 与原有 CI 注册。该完整依赖环境测试未在本机执行，不计入上述通过数。

当时要求部署环境补跑以下命令（最新实测结果见上文）：

```bash
python -m pytest experimental/hybrid-spec/tests -q
```

确认四项 CUDA 测试（含双 GPU NCCL）实际执行而不是跳过，再按 [README.md](README.md) 在 TP=1/2/4 下分别比较 eager / Graph、K=1 / K>1、target-only / 原生 neural-spec / hybrid。`benchmarks/compare_servers.py` 会保存逐 token 一致性和端到端性能结果；真实 TPOT、阶段时间和 stream 依赖需另采 profiler。

首版范围收窄已明确记录：关闭 overlap、固定单链、无 Arctic、无动态预算、不减少 neural forward。同步历史桥接复用已提交的 CPU token，没有直接消费未提交的设备预测。
