# SGLang 混合投机扩展首版

本目录实现 [DESIGN.md](DESIGN.md) 的固定长度、请求级来源选择：所有请求照常运行 neural draft，由设备端 mask 在每个位置选择 SA 或 neural token。选定 token 同时进入下一步模型输入和最终验证链。不会减少 neural forward 次数。

这是实验实现。CPU 语义测试与 CUDA/真实模型验收是不同层次；当前验证记录见 [VALIDATION.md](VALIDATION.md)。未测模型不得视为已支持。

新增 opt-in `history_mode="device"` 路径，用于对齐 Baseten `sa_spec` 的 GPU 增量更新与调度方式。prefill 前快照 prompt 并准备 pinned CPU 缓冲，由后台线程使用与 CUDA 相同的 C++ 状态机构建初始 SA；本机 native 调用释放 GIL，与 target prefill/draft-extend 并行。prefill 返回时只暂存 GPU bonus，首次 draft 才消费构建结果，并在 forward stream 上异步拷贝已使用的状态前缀。不再在 GPU 上逐 token 构建 prompt SA，也不在每轮 decode 读取或重建 CPU 输出历史。

verify 直接在设备上暂存 `predict/accept_index/accept_lens`，下一轮 draft 在 CUDA Graph 内消费增量、查询并选择来源。eager 使用相同设备算子。若 CPU 构建未完成，首次 draft 会等待该 future；不调用额外的 GPU 全局同步。异步 H2D 的 pinned 源缓冲保留到拷贝事件完成；异常、abort/flush、retract 和槽位复用会取消或丢弃旧任务结果，旧任务不能向新代际写入状态。

该路径允许单机 TP≥1 的 overlap 配置。多 TP 在每次设备查询后达成候选与资格共识，eager 和 Graph 使用 SGLang 的 PyNccl 通信器，任一 rank 不合格或候选不同则统一回退 neural；TP=1 不增加通信。**尚未完成 GPU/真实模型验收**。本地 CPU/native 测试不能替代 GPU 时间线、H2D 或真实模型验证。旧模式默认 `history_mode="cpu"`，保留关闭 overlap 的限制及现有验证证据；新路径不继承其通过结论。device 路径保留每批槽位/代际映射的 H2D、verify 反馈 kernel 和一次 host 构建/导入，并未证明零开销或性能收益。

`configs/device-sa-spec.json` 给出 K=8、阈值 4、`partial_drafts="zero"` 的参考策略配置。启动时相应设置 `--speculative-num-steps 8 --speculative-num-draft-tokens 9`；测试 overlap 时移除 `--disable-overlap-schedule`。`zero` 复现参考集成对不足 K 个续写的尾部补零；这些仍是草稿，必须经过原生 target verify，不代表输出补零。默认 `partial_drafts="fallback"` 继续整条回退，可用于单独比较该策略差异。阈值设为大于历史容量的数值可测量“维护 SA、全部选择 MTP”的成本。

device 模式以 generation 和 GPU 历史版本检查隔离旧反馈、槽位复用、重排和 Graph padding；prefill/retract 时重建索引。GPU 接受前缀可能包含 CPU EOS/停止截断前的尾部，不能将索引长度当作交付长度；原生停止处理保持不变。支持单机多 TP 实验，仍不开放 chunk prefill、grammar 等其余未支持组合。

## 实现与边界

| 项目 | 当前实现 |
| --- | --- |
| 检索 | 实例私有 GPU SA；device 模式由 CPU C++ 构建初始状态；Python CPU SA 仅作对照 |
| 神经后端接入 | 普通 EAGLE/MTP、EAGLE3、Standalone 的 V2 共用 draft 路径；均需逐模型 GPU 验收 |
| 参数 | NVIDIA CUDA、单机 TP≥1、纯文本、greedy、topk=1、K≥1、验证宽度 K+1 |
| TP | 两种历史模式均按 rank 复制 SA 并达成选择共识；device 模式在查询后、token 选择前执行共识，支持单机多 TP 实验 |
| 执行 | eager 与 CUDA Graph 接口；K=1 走原生无循环内 forward 路径 |
| 历史 | CPU 模式关闭 overlap、从 CPU 已提交历史取增量；device 模式使用 GPU verify 反馈 |
| 生命周期 | 历史重复不追加；历史变化重建；请求对象代际、结束释放、弱引用清理、flush reset |
| 混合线性注意力 | 实验接入 Qwen3.5/3.6 target + 原生 Qwen3.5 MTP，要求 `--language-only`；保留 GDN verify/commit 和 mRoPE |
| 暂不开放 | CPU 模式 overlap、DP/PP/EP/CP、多机、torch.compile、chunk prefill、混合 prefill/decode、随机采样、grammar、LoRA、其他混合线性注意力模型、多模态请求、多层 EAGLE、FROZEN_KV_MTP、动态预算、Arctic |

默认 CPU 历史模式使用同步调度器已提交到 CPU 的输出，只上传确认增量，当前 anchor 已包含在历史中。device 模式使用 GPU verify 的接受前缀，可能包含最终 CPU EOS/停止截断前的尾部；不能将其计数或索引长度当作最终交付 token 统计。两者均不增加逐步 D2H 读取。

最长重复后缀通过 suffix link 查询，续写采用该后缀的最早出现位置。候选不含 anchor。匹配长度与可用续写长度分别记录；默认续写不足、版本不符、无槽位、历史超容量、token 非法或预算不足时整条链回退 neural。`partial_drafts="zero"` 单独复现尾部补零；不会选择较短匹配来填满候选。

GPU SA 使用每请求一个线程维护状态，尚未对大词表和长历史优化。默认模式的 CPU 前缀检查随历史长度增长；device 模式只在 prefill/retract 快照一次历史，后台线程不触碰请求对象或 CUDA。净收益仍需完整推理循环实测。

## 安装

服务运行要求 Linux、与 PyTorch 匹配的 NVIDIA CUDA Toolkit（含 nvcc）、C++ 编译器和 ninja；在已可运行此 SGLang checkout 的环境操作：

```bash
python -m pip install ninja
python -m pip install -e ./python
python -m pip install -e './experimental/hybrid-spec[test]'
```

CUDA 扩展在第一次 worker 构造时通过 PyTorch 编译并缓存。插件注册不初始化 CUDA。包中包含 `.cu` 源码，不依赖本地 sa_spec checkout 或 TensorRT-LLM。

扩展同时链接纯 CPU `host_builder.cpp`，通过释放 GIL 的 `ctypes.CDLL` 调用；CPU 与 GPU 状态仅含相对整数索引，无需指针重定位。device 模式还需要 pinned CPU 状态内存，随同时在途 prompt 长度增长；构建输入及状态上界约为每请求 `80*L+48` 字节（L>0，不含对象开销）。本地原生测试无需 CUDA，可用 C++ 编译器或 Zig 编译同一份 host builder。

SGLang 基线与补丁接口见 [compat/sglang/README.md](compat/sglang/README.md)。仅安装包不能适配未打补丁的其他 SGLang 版本。

## 启动

从仓库根目录运行。`TARGET_MODEL`、`DRAFT_MODEL` 应替换成你已验证能运行原生 Standalone 的同 tokenizer/vocabulary 模型组合。

```bash
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/example.json"
export SGLANG_PLUGINS=hybrid_spec

python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-algorithm STANDALONE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disable-overlap-schedule \
  --chunked-prefill-size -1 \
  --tp-size 1 \
  --port 30000
```

EAGLE3 / 普通 MTP 使用相应原生模型配置，分别保留原生 `EAGLE3` / `EAGLE` 算法身份，其他首版限制相同。先用 `--disable-cuda-graph` 验证 eager，再去掉该参数验收 Graph。

Qwen3.6-35B-A3B-FP8 使用 `Qwen3_5MoeForConditionalGeneration` 架构。将上述算法改为 `EAGLE`，移除外部 draft model 路径，并添加 `--language-only`，使用 checkpoint 内置的 MTP。当前分支的 `--language-model-only` 不支持这个架构。该接入保留原生位置编码和 GDN 状态提交，只替换起草 token；实际多模态请求仍拒绝。启动成功和包级测试不能替代逐 token 对照验收。

多卡时将 `--tp-size 1` 改为 `--tp-size 2` 或 `--tp-size 4`，模型本身也必须支持对应 TP 分片。所有 rank 使用相同插件与 JSON 配置；target/draft 共享原生 TP 拓扑。保持 DP/PP/EP/CP 为 1、单机和 overlap 关闭。多 TP 代码已接通，当前本机只完成 CPU 多进程通信测试，真实 NCCL/模型验收仍待部署环境执行。

`SGLANG_PLUGINS` 是现有插件白名单；如需其他插件，应在该值中保留它们。未设置 `HYBRID_SPEC_CONFIG` 时本插件不注册 hook；做原生基线时取消该变量并重启服务即可。

请求示例：

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Repeat: alpha beta gamma alpha beta gamma", "sampling_params":{"temperature":0,"max_new_tokens":64}}'
```

首版在请求参数校验阶段拒绝非 greedy、grammar、beam search 和 watermark 请求，worker 入口还会复核 greedy/grammar 条件。不将检索分数伪装成采样分布。

LoRA 关闭时，无需也无法通过 `--enable-lora` 显式传入 False；本版本未传该参数的默认值是 `None`。插件接受 `enable_lora` 为 `None` 或 `False` 且无适配器路径，拒绝启用 LoRA 或配置适配器。若仍看到 `enable_lora=False required`，应更新到包含该修复的分支，而非寻找不存在的关闭参数。

## 配置

配置是独立 JSON，不修改 ServerArgs，也不在热路径读取环境变量。示例值只用于展示格式，不代表经验最优阈值。

| 字段 | 含义 |
| --- | --- |
| `num_drafts` | K，必须等于原生 speculative_num_steps |
| `min_match_len` | 启用检索所需的最短历史匹配长度，须显式配置 |
| `max_requests` | 可同时持有 SA 的请求数；无槽位时 neural fallback |
| `max_history_tokens` | 单请求 prompt + 已提交输出容量；超出时 neural fallback |
| `backend` | 服务只能使用 `gpu_sa`；`reference` 用于测试 |
| `history_mode` | `cpu`（默认，旧同步路径）或 `device`（GPU 反馈与 Graph 内增量处理，支持单机多 TP 实验） |
| `partial_drafts` | `fallback`（默认，不足 K 则整条回退）或 `zero`（参考实现的尾部补零草稿） |

状态内存估计：`4 * max_requests * [H + 4*(2H+1) + 3*(3H+1) + 5]` 字节，加上每请求 `40 + 4*(K+1)` 字节的设备反馈元数据。示例约 36 MiB，另有查询与 Graph buffer。状态在 draft worker 构造时分配，早于 target KV 预算/分配；不修改原生 KV 规划。

多 TP 的 SA 状态按 rank 复制，上述容量是每张卡的占用，不随 TP 分片缩小。每轮在当前 forward stream 上先广播 TP 组首 rank 的候选与 mask，再对本地资格及候选一致性做 MIN all-reduce；任一 rank 不满足条件或候选不同，所有 rank 对该行统一回退 neural。CPU 历史模式在 draft 前通过进程组通信；device 模式在 `prepare()` 内完成本地查询后，通过 PyNccl 通信，Graph 将查询、通信和后续选择一起捕获，eager 也使用同一通信器。PyNccl 不可用时 device 多 TP 明确报错。全局 source rank 会转换成 PyNccl 所需的组内 rank。每轮两次设备通信，广播张量为 `8*B*(K+1)` 字节，归约张量为 `4*B` 字节（不代表实际链路流量）；device Graph 中 B 包括 padding。没有逐 token 通信，也不增加 D2H 读回；TP=1 跳过通信。沿用 SGLang 的 TP batch 顺序/形状一致性约定，不能用于不同 rank 独立调度请求。

通信在 `bind()` 中完成，早于 eager 起草或 Graph staging/replay；不会捕获到 draft Graph 中。统计也是每 rank 的相同副本，汇总时只读取 TP 组首 rank，避免重复计数。

状态操作绑定首次实际起草使用的 forward stream，并等待初始化事件；禁止后续跨 stream 使用。同一 worker 的 Graph replay 串行，padding mask 每轮清零。该版本没有 side stream overlap 或并发 replay lease。

## 测试与验收

```bash
# 无需安装 SGLang GPU 依赖的包级语义测试
python -m pytest experimental/hybrid-spec/tests/test_core.py \
  experimental/hybrid-spec/tests/test_draft_recurrence.py \
  experimental/hybrid-spec/tests/test_tp.py -q

# 用 C++ 编译器检查 CUDA 共用的状态机；Windows 可安装 ziglang
python -m pytest experimental/hybrid-spec/tests/test_native.py -q

# NVIDIA CUDA + nvcc 环境必须再运行；检查输出不能有 CUDA skips
python -m pytest experimental/hybrid-spec/tests/test_cuda.py -q

# 两张 NVIDIA GPU + NCCL + nvcc：实际 SA、TP 共识和 Graph replay
python -m pytest experimental/hybrid-spec/tests/test_tp.py -k nccl -q
```

`test_draft_recurrence.py` 从当前 checkout 加载真实 `draft_forward` 方法和选链函数体，只对模型与设备 kernel 边界使用 CPU 替身；它证明覆盖顺序与递推语义，不证明真实模型、attention/KV 或 CUDA kernel 的正确性。

完整验收需要分别比较 target-only 与 hybrid、原生 neural-spec 与 hybrid。启动两个仅端口/插件配置不同的服务，然后运行：

```bash
python experimental/hybrid-spec/benchmarks/compare_servers.py \
  --baseline http://127.0.0.1:30001 --hybrid http://127.0.0.1:30000 \
  --prompts experimental/hybrid-spec/benchmarks/prompts.json \
  --concurrency 1 --max-tokens 128 --output /tmp/hybrid-comparison.json
```

脚本逐 token 比较 greedy 输出，不一致时返回失败，保存延迟、吞吐和请求元数据。分别测试 Graph 开/关、K=1/K>1、并发 1/多请求、长历史与槽位不足、EOS/长度停止/abort/flush/retract。性能报告同时记录模型版本、硬件、长度、并发与运行参数；脚本测量端到端延迟和吞吐，不将其称为 TPOT。真实 TPOT、各阶段耗时和同步成本需 profiler 时间线。

单张 GPU 无法同时容纳两个服务时，先用 `compare_servers.py --record URL --prompts prompts.json --output native.json` 记录原生服务，关闭后启动 hybrid，再用相同参数加 `--reference native.json --output hybrid.json`。提示词摘要、并发和输出上限必须一致；报告中的诊断计数读取在计时区间之外。

`benchmarks/run_model_matrix.py --model-path MODEL --output-dir DIR` 可顺序启动 Qwen3.5/3.6 的原生、hybrid、target-only 服务，覆盖 eager/Graph、K=1/3、并发 1/4。它要求端口未被已有服务占用，保存命令、服务日志、输出和诊断；`--eval-examples 16` 增加原生/混合 Graph 的 GSM8K 小样本评测，需要 `sgl-eval`。冷启动的 SM120 FP8 GEMM 编译可能超过默认 watchdog，脚本将等待上限设为 1800 秒。

SWE 性能提示词可用 `benchmarks/prepare_swe_prompts.py --dataset trajectories.parquet --revision COMMIT --model-path MODEL --output swe-prompts.json` 生成。它保留工具定义和完整对话轮次，默认选取不超过 7168 token 的前缀，并保存轨迹 ID、token 数和摘要。将该文件传给验收工具的 `--prompts`，可重复测量相同工作负载。工具调用只作为文本，不执行，也不据此计算软件任务完成率。

将 SWE 性能与 GSM8K 答案正确率分开评测，可运行：

```bash
python experimental/hybrid-spec/benchmarks/run_model_matrix.py \
  --model-path MODEL --output-dir DIR --prompts swe-prompts.json \
  --performance-only --eval-examples 128 \
  --modes target-graph native-graph hybrid-graph
```

`--performance-only` 只关闭 SWE 的输出 token 对照，GSM8K 仍按标准答案判分；性能部分对全部输入按指定并发预热、清空缓存，再开始计时。单独使用 `compare_servers.py` 时，可用 `--warmup-all --flush-cache-before-run` 启用相同流程。清空缓存要求服务没有其他正在运行的请求，因此这些工具应连接独占的测试服务。三种模式的完整命令、原始输出和 GSM8K 逐题记录都会保留。

TP 验收对 TP=1/2/4 分别重复上述矩阵，原生与 hybrid 服务必须使用相同 TP 值。重点检查候选分歧/某 rank fallback 后的统一 mask、Graph padding 与通信耗时；TP=1 的正确性结果不能替代多卡实测。

代码编辑数据可用 `prepare_code_edits_prompts.py --dataset edits_data.json --revision COMMIT --model-path MODEL --output-dir DIR` 准备同源的 diff/完整代码输入。只使用 `code` 和 `edit`，不把参考 `solution` 放入提示词。`run_workload_plan.py --plan PLAN --output-dir DIR` 可在每个测试服务加载后顺序运行多组提示词与输出上限；计划包含完整服务 argv、hybrid 配置、提示词路径、输出上限和并发。这样可以同时覆盖短 diff 与较长的代码复制场景，无需为每个工作负载重新加载模型。

开发调试可读取 `draft_worker.draft_extension.metrics.snapshot()`，获取按来源统计的验证请求数、正确草稿数、接受 token 数（包含 bonus）。该诊断调用会同步，禁止放入每步热路径。生产路径只在 GPU 上累加。

启用插件时，现有 `/get_server_info` 诊断返回的 worker 内部状态增加 `hybrid_spec` 字段，包含这些累计计数。读取前后差值可确认 SA 是否实际使用及接受情况；读取会同步，应放在计时区间之外。多 TP 时只读取一个 rank 的副本。

## 文件位置

- `src/hybrid_spec/core/`：历史版本、请求槽位和运行时契约。
- `src/hybrid_spec/backends/`：CUDA SA 与 CPU 对照实现。
- `src/hybrid_spec/integrations/sglang/`：能力校验、已提交历史桥接、Graph buffer。
- `src/hybrid_spec/plugin.py`：entry point 装配、完成/reset/verify 通知。
- `tests/`、`benchmarks/`：行为测试和真实服务对照工具。
