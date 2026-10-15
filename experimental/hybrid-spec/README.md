# SGLang 混合投机扩展首版

本目录实现 [DESIGN.md](DESIGN.md) 的固定长度、请求级来源选择：所有请求照常运行 neural draft，由设备端 mask 在每个位置选择 SA 或 neural token。选定 token 同时进入下一步模型输入和最终验证链。不会减少 neural forward 次数。

这是实验实现。CPU 语义测试与 CUDA/真实模型验收是不同层次；当前验证记录见 [VALIDATION.md](VALIDATION.md)。未测模型不得视为已支持。

## 实现与边界

| 项目 | 当前实现 |
| --- | --- |
| 检索 | 实例私有 GPU 后缀自动机；CPU SA 仅用于对照测试 |
| 神经后端接入 | 普通 EAGLE/MTP、EAGLE3、Standalone 的 V2 共用 draft 路径；均需逐模型 GPU 验收 |
| 参数 | NVIDIA CUDA、单机 TP≥1、纯文本、greedy、topk=1、K≥1、验证宽度 K+1 |
| TP | 各 rank 独立维护 SA；起草前广播候选并对选择 mask 达成一致 |
| 执行 | eager 与 CUDA Graph 接口；K=1 走原生无循环内 forward 路径 |
| 历史 | 关闭调度 overlap；从已提交的 `origin_input_ids + output_ids` 生成确认增量 |
| 生命周期 | 历史重复不追加；历史变化重建；请求对象代际、结束释放、弱引用清理、flush reset |
| 暂不开放 | overlap、DP/PP/EP/CP、多机、torch.compile、chunk prefill、混合 prefill/decode、随机采样、grammar、LoRA、混合线性注意力、多模态、多层 EAGLE、FROZEN_KV_MTP、动态预算、Arctic |

此版本比设计稿收窄了历史接入：使用同步调度器已经提交到 CPU 的输出，不增加 D2H 读取，也不直接索引 verify 的预测缓冲。只有确认的增量上传 GPU；当前 anchor 已包含在历史中。EOS/停止后尾部不会进入下一轮索引。GPU verify 的接受统计可能包含最终 CPU 停止截断前的尾部，不能当作最终交付 token 统计。

最长重复后缀通过 suffix link 查询，续写采用该后缀的最早出现位置。候选不含 anchor。匹配长度与可用续写长度分别记录；任意位置不足、版本不符、无槽位、历史超容量、token 非法或预算不足，整条链回退 neural。不会选择较短匹配来填满候选。

GPU SA 使用每请求一个线程维护状态，首版优先实现正确的实例资源与契约，尚未对大词表和长历史优化。历史同步在 CPU 上检查前缀，复杂度与历史长度有关；需要用完整推理循环实测净收益。

## 安装

服务运行要求 Linux、与 PyTorch 匹配的 NVIDIA CUDA Toolkit（含 nvcc）、C++ 编译器和 ninja；在已可运行此 SGLang checkout 的环境操作：

```bash
python -m pip install ninja
python -m pip install -e ./python
python -m pip install -e './experimental/hybrid-spec[test]'
```

CUDA 扩展在第一次 worker 构造时通过 PyTorch 编译并缓存。插件注册不初始化 CUDA。包中包含 `.cu` 源码，不依赖本地 sa_spec checkout 或 TensorRT-LLM。

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

多卡时将 `--tp-size 1` 改为 `--tp-size 2` 或 `--tp-size 4`，模型本身也必须支持对应 TP 分片。所有 rank 使用相同插件与 JSON 配置；target/draft 共享原生 TP 拓扑。保持 DP/PP/EP/CP 为 1、单机和 overlap 关闭。多 TP 代码已接通，当前本机只完成 CPU 多进程通信测试，真实 NCCL/模型验收仍待部署环境执行。

`SGLANG_PLUGINS` 是现有插件白名单；如需其他插件，应在该值中保留它们。未设置 `HYBRID_SPEC_CONFIG` 时本插件不注册 hook；做原生基线时取消该变量并重启服务即可。

请求示例：

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Repeat: alpha beta gamma alpha beta gamma", "sampling_params":{"temperature":0,"max_new_tokens":64}}'
```

首版在请求参数校验阶段拒绝非 greedy、grammar、beam search 和 watermark 请求，worker 入口还会复核 greedy/grammar 条件。不将检索分数伪装成采样分布。

## 配置

配置是独立 JSON，不修改 ServerArgs，也不在热路径读取环境变量。示例值只用于展示格式，不代表经验最优阈值。

| 字段 | 含义 |
| --- | --- |
| `num_drafts` | K，必须等于原生 speculative_num_steps |
| `min_match_len` | 启用检索所需的最短历史匹配长度，须显式配置 |
| `max_requests` | 可同时持有 SA 的请求数；无槽位时 neural fallback |
| `max_history_tokens` | 单请求 prompt + 已提交输出容量；超出时 neural fallback |
| `backend` | 服务只能使用 `gpu_sa`；`reference` 用于测试 |

状态内存估计：`4 * max_requests * [H + 4*(2H+1) + 3*(3H+1) + 5]` 字节。示例约 36 MiB，另有查询与 Graph buffer。状态在 draft worker 构造时分配，早于 target KV 预算/分配；不修改原生 KV 规划。

多 TP 的 SA 状态按 rank 复制，上述容量是每张卡的占用，不随 TP 分片缩小。每轮在当前 forward stream 上先广播 TP 组首 rank 的候选与 mask，再对本地资格及候选一致性做 MIN all-reduce；任一 rank 不满足条件或候选不同，所有 rank 对该行统一回退 neural。每轮两次设备通信，广播张量为 `8*B*(K+1)` 字节，归约张量为 `4*B` 字节（不代表实际链路流量）。没有逐 token 通信，也不增加 D2H 读回；TP=1 跳过通信。沿用 SGLang 的 TP batch 顺序/形状一致性约定，不能用于不同 rank 独立调度请求。

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

TP 验收对 TP=1/2/4 分别重复上述矩阵，原生与 hybrid 服务必须使用相同 TP 值。重点检查候选分歧/某 rank fallback 后的统一 mask、Graph padding 与通信耗时；TP=1 的正确性结果不能替代多卡实测。

开发调试可读取 `draft_worker.draft_extension.metrics.snapshot()`，获取按来源统计的验证请求数、正确草稿数、接受 token 数（包含 bonus）。该诊断调用会同步，禁止放入每步热路径。生产路径只在 GPU 上累加。

## 文件位置

- `src/hybrid_spec/core/`：历史版本、请求槽位和运行时契约。
- `src/hybrid_spec/backends/`：CUDA SA 与 CPU 对照实现。
- `src/hybrid_spec/integrations/sglang/`：能力校验、已提交历史桥接、Graph buffer。
- `src/hybrid_spec/plugin.py`：entry point 装配、完成/reset/verify 通知。
- `tests/`、`benchmarks/`：行为测试和真实服务对照工具。
