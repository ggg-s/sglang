# SGLang 混合投机扩展（实验）

本目录实现 [DESIGN.md](DESIGN.md) 的固定长度、请求级来源选择。默认 `draft_graph_mode="full"` 照常运行完整 neural draft，由设备端 mask 在每个位置选择 SA 或 neural token；选定 token 同时进入下一步模型输入和最终验证链。显式开启 `split` 后，可按配置 gate 跳过 SA 命中请求的 neural 尾段，使用 GPU 动态行号压缩，不再限定 batch≤4。

当前优化、执行流程、配置区别、A100 性能结论和未实现项见 [OPTIMIZATIONS.md](OPTIMIZATIONS.md)。SA 独立按请求存储；device 模式可显式开启跨活跃请求和完成请求的全局检索池。CPU 语义测试与 CUDA/真实模型验收是不同层次；详细验证记录见 [VALIDATION.md](VALIDATION.md)。未测模型不得视为已支持。

新增 opt-in `history_mode="device"` 路径，用于对齐 Baseten `sa_spec` 的 GPU 增量更新与调度方式。prefill 前快照 prompt 并准备 pinned CPU 缓冲，由后台线程使用与 CUDA 相同的 C++ 状态机构建初始 SA；本机 native 调用释放 GIL，与 target prefill/draft-extend 并行。prefill 返回时只暂存 GPU bonus，首次 draft 才消费构建结果，并在 forward stream 上异步拷贝已使用的状态前缀。不再在 GPU 上逐 token 构建 prompt SA，也不在每轮 decode 读取或重建 CPU 输出历史。

verify 直接在设备上暂存 `predict/accept_index/accept_lens`，下一轮 draft 在 CUDA Graph 内消费增量、查询并选择来源。eager 使用相同设备算子。若 CPU 构建未完成，首次 draft 会等待该 future；不调用额外的 GPU 全局同步。异步 H2D 的 pinned 源缓冲保留到拷贝事件完成；异常、abort/flush、retract 和槽位复用会取消或丢弃旧任务结果，旧任务不能向新代际写入状态。

该路径允许单机 TP≥1 的 overlap 配置。多 TP 在每次设备查询后达成候选与资格共识，eager 和 Graph 使用 SGLang 的 PyNccl 通信器，任一 rank 不合格或候选不同则统一回退 neural；TP=1 不增加通信。device 路径已有 Qwen3.5 MTP 的 A100 TP=1 实测，结果见上述优化总览；多 GPU 模型及真实 NCCL Graph 时间线仍待验收。旧模式默认 `history_mode="cpu"`，保留关闭 overlap 的限制。device 路径保留映射变化时的 H2D、verify 反馈 kernel 和 host 构建/导入，不能声称零开销或所有配置稳定加速。

`configs/device-sa-spec.json` 给出 K=8、阈值 4、`partial_drafts="zero"` 的参考策略配置。启动时相应设置 `--speculative-num-steps 8 --speculative-num-draft-tokens 9`；测试 overlap 时移除 `--disable-overlap-schedule`。`zero` 复现参考集成对不足 K 个续写的尾部补零；这些仍是草稿，必须经过原生 target verify，不代表输出补零。默认 `partial_drafts="fallback"` 继续整条回退，可用于单独比较该策略差异。阈值设为大于历史容量的数值可测量“维护 SA、全部选择 MTP”的成本。

`num_neural_gate_tokens` 默认为 0，保留上述 sa-spec 参考策略；取 1…K 时，先检查 SA 与原生 neural 草稿前 N 个 token 是否一致。检查范围内沿用 neural token；全部一致后才允许 SA 替换后续位置，任一步不一致则本轮余下位置均使用 neural。检查和 mask 更新可被 draft CUDA Graph 捕获，不增加模型 forward、逐步 D2H 或额外 TP collective。TP 仍沿用原生 neural token 在各 rank 一致的约定；这一新策略尚需多 GPU 模型验收。verify 统计读取检查后的最终 mask。

`configs/device-qwen35-mtp.json` 是 Qwen3.5 MTP 的实验配置：K=8、匹配阈值 4、前 4 个 token 一致性检查、短候选整链回退。它与参考配置是不同的选路策略，不能把匹配命中率当作性能收益；模型、量化及负载改变后应重新测量。选择依据与 A100 对照见 [接受长度分析](validation/acceptance-regression.md)。该配置仍运行完整 MTP；CUDA 状态布局保持 v3，安装代码后重启服务即可加载新逻辑，不需要重新编译 CUDA 插件。

`draft_graph_mode="split"` 将 EAGLE/MTP、EAGLE3 和 STANDALONE 的 neural draft
按 gate 宽度拆成共享前缀和紧凑尾段。
`num_neural_gate_tokens=G` 支持 0…K：G=0 不检查 neural 前缀，G=1 检查
来自原生 draft-extend 的首 token，G≥2 运行 G−1 次 inner forward 检查前 G 个
候选。命中行直接采用 SA 余下候选，未命中行从第 G+1 个 token 继续生成；G=0
时从首候选起选择 SA/neural，neural 行仍从已有首候选继续。G=K 合法但没有
可省略的尾段；G>K（如 K=2/G=3）启动时报错。K=1/2 均受支持。

GPU 每轮稳定压缩出 neural 行号，排除 Graph padding；CPU 只读回一个行数，
选择相应大小的 tail Graph。所有大小共用预分配行号缓冲区，不枚举 mask 组合，
也不在热路径上传行号。已移除 batch≤4 硬限制；请求容量由 `max_requests`、
`--max-running-requests` 和 decode Graph 配置决定。每个捕获前缀仍为各个可能
尾段大小保留独立 metadata、暂存和私有 Graph 池，扩大批次会增加捕获显存。

支持单机 TP≥1、greedy、topk=1；EAGLE/MTP、EAGLE3 和 STANDALONE
共用 Triton draft attention 的紧凑路径，DeepSeek V4 原生 MTP 使用 dsv4 draft
attention。模型仍需满足下文的纯文本、attention 和词表兼容约束。
SA 查询在独立 stream 执行，在 gate 边界汇合并做一次候选及最终 mask 的 TP 共识。
G=0/1 没有可重叠的 neural inner forward；G≥2 可与前缀计算重叠。
前缀/尾段使用独立私有池；`split_no_skip` 使用同一切分，但让所有实际请求行
执行尚余尾段（G=K 时同样无尾段）。默认仍为 `full`，不支持 memory saver、
KV canary 或 draft-extend DSA seed。worker、runner、插件需同时更新至 split API v5；
SA 状态布局保持 v3。配置示例仍保留 G=2、B=1/4，不代表能力上限。
详见 [动态行号与可配置 gate 验收](validation/dynamic-rows-gate-20261017.md)。

EAGLE3 的 hot-token 映射在 prefix/tail 的 gate 比较和 token 选择前统一转换为
目标词表 ID；SA 候选始终使用目标词表。STANDALONE 保留原生独立 embedding/head，
紧凑 tail 不分配或收集不存在的 recurrent hidden state。EAGLE3 的 hidden state
按实际 prefix 输出的宽度收集，支持 seed 与循环输出宽度不同。

三种算法可共用 `configs/device-draft-split.json`：在各自已可运行的原生命令上，
保留 `--speculative-algorithm EAGLE` / `EAGLE3` / `STANDALONE` 的原始身份，
设置 `--speculative-draft-attention-backend triton`、`--speculative-eagle-topk 1`、
`--speculative-num-steps 8 --speculative-num-draft-tokens 9`；模板的容量为 8，
相应设置 `--max-running-requests 8 --cuda-graph-max-bs-decode 8`。
不支持的 FlashInfer 等 draft attention、translated KV 池仍明确拒绝。
4060 验证范围和复现方法见 [三种算法 split 接入](validation/draft-algorithms-split-20261017.md)。

device 模式以 generation 和 GPU 历史版本检查隔离旧反馈、槽位复用、重排和 Graph padding；prefill/retract 时重建索引。允许实验性开启 overlap 和 chunked prefill：中间块不构建 SA、不接入临时采样 token；最后一块启动前快照完整历史（含已缓存/先前分块的前缀），后台构建与最后一块 prefill 重叠，返回后只暂存真实 bonus。CPU 构建未完成时首次 draft 仍会等待。GPU 接受前缀可能包含 CPU EOS/停止截断前的尾部，不能将索引长度当作交付长度；原生停止处理保持不变。支持单机多 TP 实验，仍不开放 mixed prefill/decode、grammar 等其余未支持组合。chunked prefill 的真实 GPU/模型验收尚未完成。

## Arctic Suffix 可选后端

`backend="arctic"` 使用 ArcticInference 官方 CPU 后缀树，可替换 GPU SA，支持
MTP/EAGLE3/STANDALONE 的 full/split/split_no_skip。配置模板为
`configs/arctic-suffix-split.json`；需 `history_mode="cpu"`、
`--disable-overlap-schedule --chunked-prefill-size -1`。查询和候选上传在 Graph 外，
gate、TP 共识、GPU 动态行号及紧凑 tail 复用同一套实现。

可选启用请求间 response 缓存，prompt 仅在请求内可见；完成时按最终 CPU 输出
对账并独立限制保留 donor 数。当前固定 K、单链模式，未开放 Arctic 树状/变长
verify 或原生 score 驱动混合策略。Arctic 不创建 GPU SA 池，需安装官方原生 suffix
扩展。安装、参数、调度限制和验证范围见 [Arctic 接入](validation/arctic-suffix.md)。

## 实现与边界

DeepSeek V4 原生 MTP 现在有独立接入：只对 `DeepseekV4ForCausalLM` target +
`DeepseekV4ForCausalLMNextN` draft 放行 `is_hybrid_swa`，其余未适配 SWA 模型、
外部草稿和 DSpark 仍拒绝。纯文本 V4 可直接使用；有视觉能力的 checkpoint
需要显式 text-only 配置，实际多模态输入仍会被请求检查拒绝。

V4 的 `split` 与 Qwen 一样按请求裁剪：按配置宽度完成 gate 后，命中行保留
完整 SA 草稿，其余行收集到对应大小的紧凑尾段 Graph，再写回原行。
全部命中时不运行尾段，Graph padding 不进入尾段；例如 4 行中 3 行命中，尾段只跑 1 行。
注意力仍使用当前选定的原生 dsv4 后端及其具体子类。每个紧凑批拥有独立的
请求/KV 计划与逐步 metadata，保留原生 SWA 和 target 压缩/verify 路径。
`tail_request_ct` 与 `tail_batch_histogram` 均按实际进入尾段的行数记录。
SA 查询重叠、全局池和单次 TP 共识继续生效，真实 GPU 模型与吞吐待内网验收。

配置模板为 `configs/device-dsv4-mtp-split-global-pool.json`。在已经可运行的
原生 V4 MTP 命令上设置以下项，并保留原有模型、TP、量化和 KV 参数：

```bash
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/device-dsv4-mtp-split-global-pool.json"
# --speculative-algorithm EAGLE
# --speculative-num-steps 8 --speculative-num-draft-tokens 9 --speculative-eagle-topk 1
# --speculative-draft-attention-backend dsv4
# --max-running-requests 4 --cuda-graph-max-bs-decode 4
```

K=8 是模板值；修改 K 时同时修改 JSON 的 `num_drafts` 和上述两个服务参数，
split 支持 K≥1、gate=0…K。`max_history_tokens=16384` 包含完整 prompt 和输出历史，
容量不足会回退 MTP，按实际上下文调整。worker/runner 的 split API 为 v5。

| 项目 | 当前实现 |
| --- | --- |
| 检索 | 实例私有 GPU SA；device 模式由 CPU C++ 构建初始状态；Python CPU SA 仅作对照 |
| 神经后端接入 | 普通 EAGLE/MTP、EAGLE3、Standalone 的 V2 共用 draft 路径；均需逐模型 GPU 验收 |
| 参数 | NVIDIA CUDA、单机 TP≥1、纯文本、greedy、topk=1、K≥1、验证宽度 K+1 |
| TP | 两种历史模式均按 rank 复制 SA，在采用 SA 后续候选前完成共识；split 将 gate 结果一起归约，支持单机多 TP 实验 |
| 执行 | eager 与 CUDA Graph 接口；K=1 走原生无循环内 forward 路径 |
| 历史 | CPU 模式关闭 overlap、从 CPU 已提交历史取增量；device 模式使用 GPU verify 反馈 |
| 生命周期 | 历史重复不追加；历史变化重建；请求对象代际、结束释放、弱引用清理、flush reset |
| 混合线性注意力 | 实验接入 Qwen3.5/3.6 target + 原生 Qwen3.5 MTP，要求 `--language-only`；保留 GDN verify/commit 和 mRoPE |
| 混合 SWA | 实验接入 DeepSeek V4 target + NextN MTP；split 使用原生 dsv4 注意力的逐请求尾段，其他 SWA 模型仍拒绝 |
| 暂不开放 | CPU 模式 overlap/chunked prefill、DP/PP/EP/CP、多机、torch.compile、混合 prefill/decode、随机采样、grammar、LoRA、其他混合线性注意力模型、多模态请求、多层 EAGLE、FROZEN_KV_MTP、动态预算、Arctic 树状及变长验证 |

默认 CPU 历史模式使用同步调度器已提交到 CPU 的输出，只上传确认增量，当前 anchor 已包含在历史中。device 模式使用 GPU verify 的接受前缀，可能包含最终 CPU EOS/停止截断前的尾部；不能将其计数或索引长度当作最终交付 token 统计。两者均不增加逐步 D2H 读取。

默认 GPU SA 的最长重复后缀通过 suffix link 查询，续写采用该后缀的最早出现位置。候选不含 anchor。匹配长度与可用续写长度分别记录；默认续写不足、版本不符、无槽位、历史超容量、token 非法或预算不足时整条链回退 neural。`partial_drafts="zero"` 单独复现尾部补零；不会选择较短匹配来填满候选。

GPU SA 使用每请求一个线程维护状态，当前 v3 已引入高分叉哈希、双转移内联、增量分桶和低多样性快路径，历史状态预算约 68 字节/token。CPU 微基准的长历史改善不能直接推算 GPU 查询或模型 TPOT。默认 CPU 历史模式的前缀检查随历史长度增长；device 模式只在 prefill/retract 快照一次历史，后台线程不触碰请求对象或 CUDA。

针对根目录性能分析的后续改动与独立归因边界，见 [VALIDATION.md](VALIDATION.md) 顶部。full 模式把 TP 共识放在 draft 的 `prepare()` 中；split 在前缀结束、gate 判定后执行候选和 mask 共识，Graph 可捕获一次精确 MIN 归约；eager 在相同逻辑位置执行。不能根据某个 rank 的本地命中 mask 跳过通信。

device 模式将六项来源统计融合到已有 GPU verify 反馈 kernel，避免每轮单独执行六次 PyTorch 归约及相关逐元素算子；诊断字段和 bonus 口径保持不变。Graph 槽位/代际映射在请求顺序、代际及捕获形状均不变时复用，避免重复清零和拷贝；SA 查询和 TP 共识仍每轮执行。CPU 历史保留原统计实现。当前 A100 回归和服务运行已覆盖设备反馈路径，组件改动的独立性能归因仍应按固定配置对照。

`configs/device-k3.json` 提供与性能报告相同 K=3、阈值 4、整链 fallback 的设备历史配置，容量为 4×8192，当前 v3 设备状态预算约 2.13 MiB/卡（不含 Graph pool 等额外开销），覆盖报告中并发 4、4096 输入 + 256 输出的负载。它同时改变历史模式和容量，仅作为运行配置，不是单变量归因证据；对照时应分别固定容量、固定历史模式。保留旧 `example.json` 以复现 CPU 路径。

## 安装

默认 GPU SA 服务运行要求 Linux、与 PyTorch 匹配的 NVIDIA CUDA Toolkit（含 nvcc）、C++ 编译器和 ninja；在已可运行此 SGLang checkout 的环境操作：

```bash
python -m pip install ninja
python -m pip install -e ./python
python -m pip install -e './experimental/hybrid-spec[test]'
```

CUDA 扩展在第一次 worker 构造时通过 PyTorch 编译并缓存。插件注册不初始化 CUDA。包中包含 `.cu` 源码，不依赖本地 sa_spec checkout 或 TensorRT-LLM。

扩展同时链接纯 CPU `host_builder.cpp`，通过释放 GIL 的 `ctypes.CDLL` 调用；CPU 与 GPU 状态仅含相对整数索引，无需指针重定位。device 模式还需要 pinned CPU 状态内存，随同时在途 prompt 长度增长；构建输入及状态上界约为每请求 `80*L+48` 字节（L>0，不含对象开销）。本地原生测试无需 CUDA，可用 C++ 编译器或 Zig 编译同一份 host builder。

SGLang 基线与补丁接口见 [compat/sglang/README.md](compat/sglang/README.md)。仅安装包不能适配未打补丁的其他 SGLang 版本。

### 更新时遇到 compatibility.py 本地修改

`git pull` 提示本地修改将被覆盖，说明更新尚未执行。先把该文件的改动保存到 stash，再做 fast-forward 更新：

```bash
git diff HEAD -- experimental/hybrid-spec/src/hybrid_spec/integrations/sglang/compatibility.py
git stash push -m "hybrid-spec compatibility before update" -- \
  experimental/hybrid-spec/src/hybrid_spec/integrations/sglang/compatibility.py
git pull --ff-only
```

保留 stash，查看 `git stash show -p` 后再判断是否需要合并回来。新版本已经处理 LoRA 默认值为 None、device 历史的 overlap 和 Qwen3.5/3.6 text-only MTP；旧的临时放宽校验可能已被覆盖。不要直接 `stash pop` 把旧文件逻辑重新覆盖回来，也无需 `reset --hard`。如果还有其他文件冲突，Git 会继续停止更新，按实际差异处理。

成功更新后，若先前不是 editable 安装，重新运行上面的插件安装命令，并重启服务以加载新代码。部署 sa_spec 风格的设备增量路径时，将 `HYBRID_SPEC_CONFIG` 指向 `configs/device-k3.json`（原生 K=3）或 `configs/device-sa-spec.json`（原生 K=8）；`example.json` 继续用于旧 CPU 历史基线。

## 启动

从仓库根目录运行。`TARGET_MODEL`、`DRAFT_MODEL` 应替换成你已验证能运行原生 Standalone 的同 tokenizer/vocabulary 模型组合。

```bash
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/device-k3.json"
export SGLANG_PLUGINS=hybrid_spec

python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-algorithm STANDALONE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --chunked-prefill-size 2048 \
  --tp-size 1 \
  --port 30000
```

EAGLE3 / 普通 MTP 使用相应原生模型配置，分别保留原生 `EAGLE3` / `EAGLE` 算法身份，其他首版限制相同。先用 `--disable-cuda-graph` 验证 eager，再去掉该参数验收 Graph。

上例启用 device 历史、overlap 和 2048 token 的 chunked prefill；不要添加 `--enable-mixed-chunk`。也可以省略 `--chunked-prefill-size` 使用 SGLang 为该模型解析的默认值。复现旧 CPU 历史基线时改回 `configs/example.json`，并恢复 `--disable-overlap-schedule --chunked-prefill-size -1`。做性能归因时分别比较两项都关闭、仅开启 overlap、两项都开启；native 与 hybrid 使用相同调度参数，且输入长度必须超过 chunk size 才实际覆盖分块路径。

更新 CUDA 扩展代码后重启服务，PyTorch 会按源码变更重新编译。先验证 CUDA 算子，再运行无模型的组件微基准：

```bash
python -m pytest experimental/hybrid-spec/tests/test_cuda.py -q
python experimental/hybrid-spec/benchmarks/benchmark_device_overhead.py \
  --batch-sizes 1 4 32 --num-drafts 8 \
  --output /tmp/hybrid-device-overhead.json
```

微基准交替比较原 PyTorch 统计/融合统计，以及原映射拷贝/固定映射缓存，记录各轮 CUDA event 区间与 wall time。反馈两路都包含 pending-state reset；结果只代表 eager 组件开销，包含 host launch gap，不能代替模型 TPOT 或整体加速比。

Qwen3.6-35B-A3B-FP8 使用 `Qwen3_5MoeForConditionalGeneration` 架构。将上述算法改为 `EAGLE`，移除外部 draft model 路径，并添加 `--language-only`，使用 checkpoint 内置的 MTP。当前分支的 `--language-model-only` 不支持这个架构。该接入保留原生位置编码和 GDN 状态提交，只替换起草 token；实际多模态请求仍拒绝。启动成功和包级测试不能替代逐 token 对照验收。

多卡时将 `--tp-size 1` 改为 `--tp-size 2` 或 `--tp-size 4`，模型本身也必须支持对应 TP 分片。所有 rank 使用相同插件与 JSON 配置；target/draft 共享原生 TP 拓扑。保持 DP/PP/EP/CP 为 1、单机；CPU 历史模式关闭 overlap 和 chunked prefill，device 模式允许开启。多 TP 代码已接通，当前本机只完成 CPU 多进程通信测试，真实 NCCL/模型验收仍待部署环境执行。

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
| `enable_global_pool` | 默认 `false`；`device` 模式下跨已发布的活跃请求和保留请求查询 SA |
| `max_shared_requests` | 完成请求保留上限，默认 0；需开启全局池；达到上限按完成顺序 FIFO 淘汰 |
| `shared_context_tokens` | 跨请求查询最多使用当前历史最后多少个 token，默认 128；本请求查询保持原行为 |

状态内存估计：`4 * (max_requests + max_shared_requests) * [H + 5*(2H+1) + 6*(H+1) + 7]` 字节，加上每槽位 `40 + 4*(K+1)` 字节的设备反馈元数据。`configs/example.json` 约 34 MiB，另有查询与 Graph buffer。状态在 draft worker 构造时分配，早于 target KV 预算/分配；不修改原生 KV 规划。

### 可选的跨请求 SA 历史

`configs/device-qwen35-mtp-global-pool.json` 在 4 个活跃请求之外预留 16 个完成请求槽位。
显式设置 `enable_global_pool=true` 后，同一 worker 内已导入初始 SA 的活跃请求也能互查；
`max_shared_requests=0` 表示只共享活跃历史，不保留完成请求。默认关闭，CPU 历史模式不支持。
这会允许后来的请求检索其他请求的 prompt/输出，只应在允许共享这些内容的同一信任范围内开启；
池不按用户或会话分区，不跨 worker、进程或重启持久化。flush 会清除活跃和保留映射。

每个来源仍保持独立自动机，查询不会拼接不同请求。先统一消费已发布槽位的 GPU 反馈，
再进行只读跨槽位查询。每个请求一个 CUDA block，线程并行扫描来源并归约；超过
128 个来源时分步扫描，候选排序不变。来源末尾无续写时沿 suffix link 回退。
保留本请求的候选作为基线，选择满足原有完整度策略的更长匹配。
匹配长度相同时优先本请求；其他来源之间优先较长续写，仍相同则优先较新 generation。
neural 前缀检查、TP 候选共识、target verify 和 split 的现有约束继续生效。

完成回调只修改 CPU 元数据，不在 scheduler 线程启动 CUDA。下一次 draft 在 forward stream
消费最后的 verify 增量（split 在已等待 forward stream 的 SA 辅助 stream 中消费），按完成请求的最终 token 数截掉停止条件丢弃的尾部；若已写入超长尾部，
该来源失效回退。未导入的后台构建不进入池，槽位复用有 generation 校验。
跨请求查询兼容 `split`，要求 device 历史、gate=0…K、单机 TP≥1、greedy 等
现有 split 约束。增加保留槽位不会放宽活跃并发限制，也会增加显存和查询成本。
组合示例为 `configs/device-qwen35-mtp-split-global-pool.json`：K=8、gate=2、
4 个活跃槽位、16 个保留槽位、H=4096，与 6000D 对照中的 split 配置一致。
吞吐取决于历史重复程度，不能从查询加速直接推断整体收益。
并行查询保持 v3 存储布局；更新后重启服务，扩展会按 CUDA 源码变化自动重新编译。
参考来源、实现差异和验证范围见 [全局池记录](validation/global-pool.md)。

转移边参考 sa-spec 的内联设计：每个状态直接保存第一条转移；第二条转移用同一辅助字段的负值编码，第三条起才分配边记录，超过 4 条出边才维护哈希。单一重复历史直接延长链，二值历史完全不使用边池。转移标签由已有历史和 first-position 恢复，clone 与重定向保持这一不变量。额外边数有 `n−1` 上界，预留边池从 `3H+1` 缩为 `H+1`；哈希仍逐桶分裂、精确比较键，CPU 状态可以直接导入较大 GPU 槽位继续扩展。重置不扫描整池。

状态预算约 **68 字节/历史 token**，低于上一版哈希的 108，也低于最初链表的 72。例如 32 × 262144 每 rank 约 **544 MiB**，另加反馈和 Graph buffer。布局版本为 `hybrid_spec_gpu_sa_v3`，拉取后须重启服务。

多 TP 的 SA 状态按 rank 复制，上述容量是每张卡的占用，不随 TP 分片缩小。每轮将 `[tokens, ~tokens, eligibility]` 打包做一次 MIN all-reduce；按位取反逆转 int64 的有符号顺序，因此归约后 `min(tokens) == ~min(~tokens)` 逐位置成立当且仅当所有 rank 的候选完全相同。资格也必须全部为真。任一 rank 不满足条件或候选不同，所有 rank 对该行统一回退 neural；不需要广播或指定源 rank。归约输入为 `8*B*(2K+1)` 字节（不代表实际链路流量），Graph 中 B 包括 padding。沿用 SGLang 的 TP batch 顺序/形状一致性约定，不能用于不同 rank 独立调度请求。

full 模式在 `prepare()` 中、token 选择前通过现有 PyNccl 执行共识；CPU 历史模式在 `bind()` 中只准备本地候选，Graph staging 后由捕获视图完成共识；device 模式先在 Graph 内消费反馈/查询，再做共识。split 先运行配置宽度的 neural 前缀，查询完成后检查 G 个 gate token，将最终 gate mask 一起纳入同一次候选共识。通信只在模型主 stream 执行，各 rank 无条件参与，之后 CPU 决策和紧凑尾段行号一致。eager 使用相同逻辑位置和通信器；多 TP 缺少 PyNccl 时明确报错。没有按本地 mask 分支、逐 token 通信或新增 D2H 读回；TP=1 不创建同步器。统计读取最终共识 mask；每 rank 为相同副本，汇总时只读取 TP 组首 rank，避免重复计数。K=1 沿用原生无 draft Graph 的 eager 入口，不声称通信被捕获。

初始化、导入和反馈写入仍绑定首次实际起草使用的 forward stream，并等待初始化事件；禁止无依赖地跨 stream 使用。split 的查询辅助 stream 由运行时以 `hybrid_sa_query` 角色持有，在 capture 外创建，每次查询先等待 forward stream，前缀 gate 前由主 stream 等待查询完成。Graph 捕获 fork/join 依赖；全部 GPU 读写在进入尾段前汇合。同一 worker 的 Graph replay 串行，padding mask 每轮清零，不支持并发 replay。

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

长历史 A/B 默认分别编译压缩前 `88f8d971eb` 和当前源码；传入 `--baseline-ref 44637fe669` 可对照最初链表版。使用同一基准，输出原始样本、中位数和语义状态校验：

```bash
# CPU 原生状态机；需要 C++ 编译器或 ziglang
python experimental/hybrid-spec/benchmarks/benchmark_sa_history.py \
  --baseline-ref 88f8d971eb --output /tmp/sa-history-cpu.json

# NVIDIA CUDA + nvcc；同进程比较真实 Graph 内消费反馈/查询 kernel
python experimental/hybrid-spec/benchmarks/benchmark_sa_history_cuda.py \
  --baseline-ref 88f8d971eb --lengths 8192 65536 262144 --batch-sizes 1 4 32 \
  --output /tmp/sa-history-cuda.json
```

默认覆盖单 token 重复、2/64/8192 种 token 的伪随机历史。CPU 可用 `--cpu N` 固定到允许使用的逻辑核心，减少大小核调度影响。校验按状态和排序后的转移图规范化，支持不同物理布局。CUDA 每轮追加 9 token，计时前恢复相同历史；这是缓存较热的组件实验，排除导入、反馈生成、TP 通信和模型 forward，不能当成 TPOT。基线 commit 必须存在于本地 Git 历史；服务吞吐仍需固定模型与工作负载另测。

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

### sa-spec 原始模板：内网原生/插件对照

`--template sa-spec` 使用 `sa_spec/bench/make_dataset.py`（参考提交 `1cc7e8c86b19`）的原始 completion 提示词，连同开头、结尾和标签之间的换行；不套 chat template，不添加 system/thinking 指令，不使用 `solution`。按文件顺序取前 N 条，超长直接报错，不截断或跳过样本。额外输出 **`sa-spec-sglang.jsonl`**，供官方 `python -m sglang.benchmark.serving --dataset-name custom` 发压；该格式要求两轮 conversation，因此第二轮使用空字符串占位，只有第一轮原始 prompt 会发送给服务。原来的 `sa-spec-prompts.json`、TRT 格式 `sa-spec-dataset.jsonl` 和 paired 模式继续保留。

下面命令在 Linux 服务器仓库根目录、现有服务的 Python/CUDA 环境执行。替换三个本地路径；`DRAFT` 是 **EAGLE3 头**，不是 Qwen3-0.6B。这里沿用 Qwen3-14B/EAGLE3 做 SGLang 内部 A/B，不声称复现 DeepSeek/TensorRT-LLM 的绝对性能。数据固定快照为 `glaiveai/code_edits_sample@dfdacc8eb57d576d0662645144bdba4cdba6a106`，对应 JSON SHA-256 为 `de3198ef872f268bda76c70ee27b5d184ef3d0ae18f3b68135eb5c2c54ea4c8a`。

```bash
git pull --ff-only
python -m pip install -e './experimental/hybrid-spec[test]'

export DATASET=/实际路径/edits_data.json
export MODEL=/models/Qwen/Qwen3-14B
export DRAFT=/实际路径/EAGLE3-Qwen3-14B
export CUDA_VISIBLE_DEVICES=0,1
export RUN_DIR="$PWD/validation-runs/sa-spec-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
sha256sum "$DATASET"

python experimental/hybrid-spec/benchmarks/prepare_code_edits_prompts.py \
  --dataset "$DATASET" \
  --revision dfdacc8eb57d576d0662645144bdba4cdba6a106 \
  --model-path "$MODEL" --template sa-spec --num-prompts 500 \
  --max-prompt-tokens 14336 --output-dir "$RUN_DIR/data"

python experimental/hybrid-spec/benchmarks/make_sa_spec_plan.py \
  --data-dir "$RUN_DIR/data" --output "$RUN_DIR/plan.json" \
  --model-path "$MODEL" --draft-model-path "$DRAFT" \
  --algorithm EAGLE3 --tp-size 2 --num-drafts 8 \
  --context-length 16384 --concurrencies 1 4 --port 39921

python experimental/hybrid-spec/benchmarks/run_workload_plan.py \
  --plan "$RUN_DIR/plan.json" --output-dir "$RUN_DIR/results"

python -m json.tool "$RUN_DIR/results/summary.json"
```

运行前停止占用所选 GPU 的旧服务，并保持 39921 端口空闲。计划运行器按顺序启动原生、停止原生、启动插件、停止插件，不需手动另开服务。**发请求和计算 TTFT/TPOT 全部调用官方 `sglang.benchmark.serving`，不再使用 `compare_servers.py`。** 两侧共享模型、K、TP、调度、随机种子、输入、greedy 采样和 **2048 输出上限**。显式传入 `--disable-ignore-eos`，使 `ignore_eos=false`，可以自然提前结束；不传 `--apply-chat-template` 或 `--disable-stream`，保持原始文本和流式测量。上下文为 16384，输入最多 14336，给输出预留 2048；超长时同时提高数据预算、计划上下文及实际显存预算，不悄悄过滤。首次可用 `--num-prompts 32` 做冒烟，正式对照再用前 500 条。旧数据目录缺少官方 JSONL 时须重新运行准备脚本；旧计划也须重新生成。

插件配置由计划生成：`history_mode=device`、`min_match_len=4`、`partial_drafts=zero`、槽位数等于最大测试并发、历史容量等于上下文长度。默认使用 SGLang 的 overlap/chunked-prefill 设置；若要匹配旧部署，在生成计划时显式加 `--disable-overlap-schedule --chunked-prefill-size -1`，两侧会一致应用。`--repeats 3` 生成三组并交替原生/插件启动顺序，需加载模型六次；`--num-drafts 3` 可单独做 K=3 对照，不能拿原生 K=3 与插件 K=8 直接归因 SA 收益。

每个并发使用官方 warmup：重复第一个样本，输出最多 32 token，预热请求数等于并发；然后 `--flush-cache` 清空缓存并正式计时。官方 custom loader 会按 `--seed 42` 洗牌，两侧顺序一致；准备阶段已校验输入预算，运行器还检查成功数必须等于请求数，避免静默过滤/失败。`summary.json` 原样收录官方 `output_throughput`、`mean_ttft_ms`、`mean_tpot_ms`、`median_tpot_ms`、`p95_tpot_ms`、`p99_tpot_ms` 等字段。官方 TPOT 对每个输出长度大于 1 的成功请求计算 `(latency-TTFT)/(output_tokens-1)`，再汇总请求级均值/分位数，不是对吞吐取倒数；`accept_length` 采用官方服务统计口径。

SA 来源指标 `sa_fraction`、`sa_mean_accept_length`、`neural_mean_accept_length` 来自发压进程前后诊断差值，**包含官方 warmup**，在结果中以 `hybrid_spec_delta_scope` 标明；计数单位是验证行，不是独立用户请求。它们不参与 TPOT 计算。官方原始 JSONL 保留 `--output-details` 生成的文本、输出长度、TTFT、ITL 和错误；`*-command.json` 保存真实发压 argv，`*-bench.log` 保存官方日志，`*-diagnostics.json` 保存插件诊断，`*-server.log` 保存服务日志。缺少插件诊断或请求未全部成功会报错；不覆盖已有官方结果。流式性能不等于正确性验证，仍需检查生成文本。原始模板字节相同也不代表不同模型的 tokenizer、停止行为或推理后端相同。

如服务已经独立启动，下面是同一官方发压命令（切换服务后换一个结果文件名执行）。不要使用 TRT 格式 JSONL 代替 `sa-spec-sglang.jsonl`：

```bash
python -m sglang.benchmark.serving \
  --backend sglang --base-url http://127.0.0.1:39921 \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name custom --dataset-path "$RUN_DIR/data/sa-spec-sglang.jsonl" \
  --num-prompts 500 --sharegpt-output-len 2048 --sharegpt-context-len 16384 \
  --max-concurrency 4 --request-rate inf --seed 42 --temperature 0 \
  --disable-ignore-eos --warmup-requests 4 --flush-cache \
  --output-details --output-file "$RUN_DIR/official-c4.jsonl"
```

开发调试可读取 `draft_worker.draft_extension.metrics.snapshot()`，获取按来源统计的验证请求数、正确草稿数、接受 token 数（包含 bonus）。该诊断调用会同步，禁止放入每步热路径。生产路径只在 GPU 上累加。

启用插件时，现有 `/get_server_info` 诊断返回的 worker 内部状态增加 `hybrid_spec` 字段，包含这些累计计数。读取前后差值可确认 SA 是否实际使用及接受情况；读取会同步，应放在计时区间之外。多 TP 时只读取一个 rank 的副本。

## 文件位置

- `src/hybrid_spec/core/`：历史版本、请求槽位和运行时契约。
- `src/hybrid_spec/backends/`：CUDA SA 与 CPU 对照实现。
- `src/hybrid_spec/integrations/sglang/`：能力校验、已提交历史桥接、Graph buffer。
- `src/hybrid_spec/plugin.py`：entry point 装配、完成/reset/verify 通知。
- `tests/`、`benchmarks/`：行为测试和真实服务对照工具。
