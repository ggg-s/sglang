# 验证记录

## Device 历史模式多 TP（2026-10-16）

此前 `history_mode="device"` 的 TP=1 限制来自缺失 Graph 内的 TP 共识，而非 SA 算法只能使用单卡。本轮允许单机 TP≥1，覆盖普通 EAGLE/MTP、EAGLE3、STANDALONE 的共用插件路径，仍保留其他配置限制。

- 每个 rank 继续维护自己的 SA 和异步 host 构建。`DeviceOverride.prepare()` 在消费反馈、查询 SA 后广播组首 rank 的候选与资格，并用 MIN 归约本地一致性。任一 rank 无槽位、失效或候选不同，该行统一回退 neural；没有按本地 mask 跳过通信。
- eager 与 Graph 使用现有 TP 组的 PyNccl 通信器；显式开启并恢复原有通信器状态，全局源 rank 转为组内 root。device 多 TP 缺少 PyNccl 时明确报错；CPU 历史模式沿用进程组路径。TP=1 不增加通信。
- 共识结果写回已有 Graph tokens/mask 存储；verify 统计使用同步后的实际 mask。Graph 的 padding 行也参与固定形状通信，但保持不选择 SA。
- 配置回归覆盖三种算法、TP=1/2/4/8、overlap 开/关；这只是校验与调用路径覆盖，不表示启动了这些模型服务。

本地结果：

| 检查 | 结果与边界 |
| --- | --- |
| 全部插件测试 | **112 passed, 7 skipped**；其中 6 项 CUDA、1 项 NCCL 因硬件不可用跳过 |
| 真实 Gloo TP=1/2/4 | 通过；含全局 ranks `[1,3]` 子组、K=1/3、候选分歧、单 rank 全部回退、请求重排、仅单 rank 释放槽位、Graph 缓冲 padding 和 verify 统计 |
| device CPU 桥接 | 使用可控的查询/构建边界与真实插件、进程组；Gloo adapter 检查 PyNccl 所需的组内 root、开启/恢复和原地通信语义，不能替代 NCCL |
| 负对照 | 禁用 `DeviceOverride.prepare()` 内共识，真实 Gloo 双进程测试按预期断言失败：`MISSING_DEVICE_CONSENSUS_REJECTED_AS_EXPECTED` |
| CUDA/NCCL 回归实现 | 新增 Graph 内实际 PyNccl 通信、真实 host SA 导入/GPU 反馈/查询、同版本不同候选、旧代际、重排和重复 replay 的检查；**本地未执行** |
| 静态与打包 | 修改文件 Ruff 检查、27 个 Python 文件格式检查、`git diff --check`、wheel 构建及四个修改后源码的归档一致性检查通过 |

服务器 `hz01` 本轮 SSH 返回 `ECONNREFUSED`，因此没有执行多卡模型启动、NCCL Graph、overlap 时序或性能验收。CPU/Gloo 通过不能外推 GPU/NCCL 通过，也不能证明多 TP 加速。

## CPU 初始 SA 构建与 prefill 重叠（2026-10-16）

本轮继续完善 opt-in `history_mode="device"`；默认 CPU 历史模式与原生模型验证/停止处理不变。

- prefill 的 AROUND hook 在 target prefill 前快照完整 prompt、分配 pinned CPU 存储并提交单个后台构建线程；纯 CPU `host_builder.cpp` 使用与 CUDA 相同的 `Automaton`，通过 `ctypes.CDLL` 释放 GIL。后台任务不读取请求对象，也不调用 CUDA。
- prefill 成功后只初始化设备代际/预算并暂存 GPU bonus，不等待 CPU 构建。首次 draft 在需要状态时等待 future，异步拷贝 history/states/edges 的已使用前缀及 meta，然后继续原有 GPU 增量更新。数据都是相对索引，无指针重定位、无额外 GPU 全局同步；CPU 构建慢于 prefill 时仍会等待，不能声称该开销总被隐藏。
- pinned 源缓冲由 CUDA 事件持有，事件完成前不释放；prefill 异常会清理对应 lease，abort/flush/retract、RID/槽位复用会取消或丢弃旧结果。旧 ticket 和旧完成任务不能覆盖新 generation。稳定批次继续复用设备槽位映射。
- 同步补强 LoRA 启动回归。当前代码已包含 `3646c5bc3a` 的修复：无适配器时 `None` 和 `False` 都表示关闭。新测试读取本仓库真实 LoRA 字段及 CLI 生成器，覆盖三种算法、TP=1/2/4/8、默认 None、空适配器列表，以及拒绝显式开启/适配器路径。测试隔离了无关 GPU 常量与适配器元数据，没有声称启动真实模型服务。

本地验证（Windows / Python 3.11.15 / CPU PyTorch）：**111 passed，7 skipped，103.26 秒**。跳过的 6 项 CUDA 和 1 项 NCCL 测试不计入通过；新增真实 GPU host 导入/bonus 保留用例已写入，尚未运行。通过项包含：

- 实际 C++ 状态机的穷举/长序列 oracle，以及紧凑 host 状态复制到更大 decode 容量后继续追加的逐状态/逐边一致性。
- 实际 C ABI/DLL 的完整状态、clone、suffix link、转移表与 Python 独立表示对照，以及非法词表 token 检查。
- 无 sleep 的原生 GIL gate、构建未完成时 prefill 仍可执行、只导入一次、异常释放、flush/结束后迟到结果隔离。
- CPU 拷贝替身验证只复制已使用前缀、保留 GPU bonus 元数据及事件完成前持有源缓冲；它不证明真实 CUDA DMA 的正确性。
- 真实 LoRA 字段/CLI 与插件校验器，以及既有 draft 递推、CPU/Gloo TP 回归。

两个反向对照均按预期失败：从 Git 读取 `3646c5bc3a` 的父版本校验器，新的 CLI 测试复现 `enable_lora=False required`；将 native 调用加载器换为持有 GIL 的 `PyDLL`，原生 gate 测试拒绝通过。源码没有为这些对照回退，脚本与日志只保存在忽略目录 `.tmp/hybrid-spec`。

另修复原生测试自身的问题：本机 Zig 0.16 在 `-O2` 下默认定义 `NDEBUG`，会禁用旧 C++ `assert`。现在显式 `-UNDEBUG`，源码另有编译期保护，构建调用也移出 assert 表达式。此前 Windows/Zig 原生测试的“通过”不能视为有效断言证据；本轮已在断言启用后重跑通过。Linux 编译器是否定义 NDEBUG 取决于其参数，不能将该本机发现外推到所有旧测试环境。

修改文件通过 Ruff 未定义名称/导入检查及格式检查；wheel 构建成功，并检查打包包含 host C++、共享头文件、CUDA 源码与 Python host builder。完整回归、反向对照及打包日志分别位于本地 `.tmp/hybrid-spec/host-full.log`、`host-negative-controls.log`、`host-wheel.log`。

**未验收项**：本地无 CUDA 工具链/GPU；本轮未验证新的 CPU+CUDA 扩展组合编译/链接、真实 pinned H2D、Graph replay、prefill 时间线、模型逐 token 等价或吞吐。上轮单 CUDA 源扩展的编译成功不能替代本轮组合扩展的编译。先前并发严格 token 等价失败仍未解决。本轮完成的是可在本地实现和验证的构建/调度/生命周期部分，未报告新的性能收益。

## 对齐本地 sa_spec 的设备历史路径（2026-10-15）

参考仓库：`../sa_spec`，HEAD `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969`；集成参考为 TRT-LLM PR #10951 的 head `304ec62dd635cc7ad865c2ae35b4ae02910396d0`。该实现同样保留 MTP forward，再按匹配长度选择 SA token。不能把“未跳过 neural forward”单独当成我们与参考实现性能差异的原因。参考脚本要求完整代码续写、输出预算 2048、K=8、默认匹配阈值 4，图中接受长度 3.28→4.41；我们的前轮 K=3/阈值 2 与输入、停止方式并未对齐。

代码提交 `3817502934` 新增 opt-in `history_mode="device"`，旧 CPU 路径仍为默认：

- prefill 初始化一次完整 prompt SA，直接使用 GPU bonus；每轮 verify 从 GPU `predict/accept_index/accept_lens` 暂存接受增量，不再重组、比较或上传完整 CPU 输出历史。
- draft 的 `prepare(seq_lens)` 在 Graph 内消费增量、查询 SA、生成选择 mask；eager 使用同一算子。重放没有新反馈时不会重复追加。版本必须与设备 `seq_lens+1` 一致。
- GPU owner generation、CPU 请求对象代际与 padding 槽位隔离旧反馈和槽位复用。GPU 接受前缀仍可能包含 CPU 停止截断前的尾部，交付长度以原生停止处理为准。
- 新路径允许 TP=1 overlap 实验，其他并行方式仍拒绝；默认 CPU 模式继续拒绝 overlap。
- `partial_drafts="zero"` 用来复现参考集成的不足 K 尾部补零；默认 `fallback` 保留整条回退，两者均经过原生 target verify。参考配置在 `configs/device-sa-spec.json`（K=8、阈值 4）。

验证：Windows 核心/真实 draft 函数体测试 98 passed，包括禁止读取 CPU decode 历史、GPU 反馈张量身份以及 Graph 选择 mask 对统计的桥接测试；native/Gloo 测试在沙箱内遇到 Zig 路径错误及多进程超时，移至沙箱外补跑后 **4 passed，1 skipped**（NCCL）。Ruff 检查与格式检查通过。Linux 完整包级回归 **102 passed，6 skipped**；新增 CUDA Graph、部分续写、全局接受索引、重放幂等、旧 generation、padding 测试因无 GPU 未执行。CUDA 13 编译器针对 `sm_120` 成功编译和加载新扩展，`CUDA_BUILD_EXIT=0`。日志在服务器 `/root/rivermind-data/validation/device-sa/check.log`。

**验收限制**：本轮服务器容器中 `/dev/nvidia*` 不存在，`torch.cuda.is_available()=False`、设备数 0。尚未执行新模式真实 GPU kernel、模型 token 等价、overlap 生命周期压力测试或性能复测。旧模式此前的实机通过项不能移作新模式的证据；此前并发严格 token 等价失败仍未解决。

该实现仍保留请求槽位/代际映射的 staging 和 verify 反馈 kernel，且 prompt SA 初始构建在 GPU 上，尚未实现参考项目 host 构建与 prefill 的重叠。因此目前只是朝参考执行路径对齐，不能宣称逐项一致或零开销。后续需在相同输入、K、阈值、停止策略下比较原生 MTP、阈值大于历史容量的 SA、正常 SA；再分别比较 overlap 开关与两种部分续写策略。

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

## code_edits_sample 工作负载复测（2026-10-15）

代码：`6b289bab43`。数据源：[glaiveai/code_edits_sample](https://huggingface.co/datasets/glaiveai/code_edits_sample)，固定版本 `dfdacc8eb57d576d0662645144bdba4cdba6a106`。完整 JSON 共 9,966 条、50,507,443 字节，已下载到 `/root/rivermind-data/datasets/glaiveai/code_edits_sample` 并校验 SHA256。

选择同一批 16 个完整源文件，构造两种性能输入：diff 输出最多 128 token；完整代码输出最多 512 token。原始 `solution` 未输入模型，也未用这轮性能回放评判代码修改正确率。两种输入的平均长度分别为 427.5 和 436.5 token，完整原始代码平均 351.19 token。沿用 TP=1、Graph、K=3、`min_match_len=2`、并发 1/4；所有输入预热后清空缓存再计时。

回查前轮 SWE 采样发现，16 条前缀中 15 条停在 assistant turn index 2（首个 assistant 回答之前），只有 1 条到 index 4。该有限上下文回放包含大量系统/工具定义，不能代表整个 SWE 数据集中“已读取代码后的编辑”阶段。Code edits 直接提供原始代码，同时改变了输入长度和任务形式，因此跨数据集的吞吐差异不能全归因于复制量。

首轮结果（输出 token/s）：

| 任务 | 并发 | 原生 MTP | MTP+SA | 相对变化 |
| --- | --- | --- | --- | --- |
| diff，最多 128 token | 1 | 177.77 | 187.14 | +5.3% |
| diff，最多 128 token | 4 | 410.99 | 447.46 | +8.9% |
| 完整代码，最多 512 token | 1 | 278.51 | 177.64 | -36.2% |
| 完整代码，最多 512 token | 4 | 636.63 | 546.91 | -14.1% |

首轮 diff 平均接受长度从原生的 3.33/3.37 降至 SA 的 2.87/2.90，因此追加了重复验证。完整代码输出的平均接受长度从 3.63/3.67 降至 3.25/3.23；约 53%–57% 的输出 token 位于与原始代码相同的连续片段中（统计长度至少 8 token 的匹配块），最长匹配块为 384 token。存在复制机会不等于当前替换策略能增加每轮接受量。

diff 的追加 3 轮复测反转服务启动顺序（先 hybrid 后 native），输入、参数、预热及清缓存流程相同。吞吐为中位数 [最小值, 最大值]：

| 并发 | 原生 MTP token/s | MTP+SA token/s | 中位数相对变化 |
| --- | --- | --- | --- |
| 1 | 145.39 [136.05, 218.91] | 186.92 [157.42, 187.03] | +28.6% |
| 4 | 273.50 [259.13, 495.17] | 432.72 [287.12, 433.10] | +58.2% |

这组中位数较高，但波动范围明显重叠，第三轮的速度方向反转。原生并发 1 在三轮中生成的 token 序列及平均接受长度完全相同，吞吐仍从 136.05 变到 218.91；hybrid 并发 1 的 token 序列也完全相同。当前不能把中位数差异当作稳定的 SA 加速比，波动原因尚未定位。每个模式仅运行一个服务进程内的 3 轮请求，不是跨机器或多次独立启动的统计验证。

当前实现仍执行原生 MTP 的 draft forward，再由 SA 候选覆盖草稿；K=3 的固定预算最多验证 3 个草稿加 bonus。索引维护、查询、设备拷贝和选择是额外工作。上面的数字不能区分每项开销，尚未做 profiler 归因，也未对匹配门槛或草稿长度寻优。完整代码工作负载中原生接受长度已接近 4，单靠增加接受量的理想空间约为 9%–10%（仅按接受长度粗估），本轮 hybrid 实际接受长度反而下降。

结论：前轮 SWE 前缀采样确实限制了代码复用场景的覆盖；更换为直接含源码的 code edits 后，diff 出现单轮/中位数的名义增益，但未得到稳定收益证据，完整代码输出仍退化。数据集、输入构造、选择策略和运行时成本均需区分，不能把未提速简单归因于 SWE 数据集本身。此前严格并发 token 等价问题仍未解决；本次是性能复测，不是代码修改正确率验收。

原始目录：`/root/rivermind-data/validation/code-edits`，包含输入来源、完整计划、命令、原始输出、SA 计数及 `analysis.json`。追加 diff 重复测试使用相同输入和参数、反转服务顺序，每个并发进行 3 轮，结果保存在 `repeat-diff`。

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
