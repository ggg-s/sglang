# 验证记录

当前机制、优化阶段与跨实验结论的汇总入口见 [优化总览](OPTIMIZATIONS.md)。
下文按阶段保留当时的协议、结果和验收限制。

## DeepSeek V4 SA 与整批 split 接入（2026-10-17，待内网 GPU 验收）

修复两处接入缺口：`validate_model` 将 V4 target/NextN draft 的原生 hybrid-SWA
误判为完全未适配；`SplitDraftGraphBackend` 仅允许 Qwen3.5 MTP。
现在按确切架构对和纯文本条件放行 V4，DSpark、其他草稿、其他 SWA 架构仍拒绝。
split 使用原生 dsv4 多步注意力，不复用 Triton compact tail。全实际请求 gate 通过时
整批跳过；混合/全不命中时运行原始整批尾段，保留 metadata、KV 写地址和请求顺序。
只用 raw_bs 范围的共识 mask 作决策，padding 不阻止整批跳过，也不计入请求数；
尾段直方图仍按实际模型 padded 行数记录。gate=2、查询重叠、单次 TP 共识与
native target verify/draft-extend 保持原有协议。V4 尚未支持逐行裁剪。

历史参考：本地 `model-pr-history-knowledge/sglang/deepseek-v4/README.en.md`
的 #26239 记录指出，V4 多步草稿的 per-step KV 地址已经进入 backend metadata。
随后核对当前 `deepseek_v4_backend.py::DeepseekV4MultiStepBackend`、
`init_forward_metadata_in_graph` 及 `deepseek_v4_nextn.py`：原生 MTP 层使用
`COMPRESS_RATIO_NEXTN_LAYER=0`，target 的压缩/verify 路径继续由原生后端负责。
据此选择保持原生 batch/metadata 的两段 Graph，避免将 Qwen 的紧凑元数据套到 V4。
历史笔记与源码核查均不代表新的 GPU 正确性或性能结果。

本地 core 69 passed（含 V4 正/反向模型校验和 K=3/8 所有小批 mask/padding 的整批调度），
worker recurrence 94 passed，真实 Gloo TP=1/2/4 为 3 passed，总计 166 passed。
无 CUDA 的本机跳过 CUDA/NCCL，不能计为 GPU 通过。新增真实 CUDA Graph 捕获测试
`test_dsv4_whole_batch_graph_capture_keeps_native_rows`，其模型和注意力边界为合成实现，
验证本次 split capture/replay/skip；不验证真实 V4 压缩 KV 或模型精度。
此前的 `is_hybrid_swa` 拒绝及 Qwen-only split 拒绝均可在修复前源码复现。

内网按 README 的 V4 配置与 `--speculative-draft-attention-backend dsv4` 启动，
先验证并发 1/4、TP=1/2/4、全命中/全不命中/混合命中、3→4 padding、请求结束
和槽位复用，再对同一 TP 的 full/原生 MTP 核对输出与接受长度；性能另做短对照。
模板 K=8、验证宽度 9；服务参数必须与 JSON 一致。保留原生可运行的 target
attention、量化与 KV 配置，不使用 Qwen 示例里的 Triton draft attention。

## Split 查询重叠、调度预分配与多 TP（2026-10-17，待 GPU 验收）

仅修改 SA＋MTP 混合投机链路，保留 gate=2、候选排序和 target verify。
split/split_no_skip 在 `hybrid_sa_query` 辅助 stream 消费反馈并查询，
与生成第二个 neural token 的首个 MTP forward 重叠。前两个 gate token
固定使用 neural；主 stream 等待查询完成后更新 gate mask，将候选和最终 mask
打包执行一次 TP MIN 共识，再读取 mask、选择整批跳过或紧凑尾段。
不在辅助 stream 运行 NCCL，不按本地命中结果省略 collective。
候选分歧、任一 rank gate 失败或无有效候选时，所有 rank 对该行统一回退。

最多 4 请求的所有 mask/实际批大小组合在 capture 后预分配行号 tensor，
每轮仅复制已持有的设备行号，消除临时 tensor 创建与行号 H2D。
保留一次 mask CPU 读取；其等待包含未完成的前缀 GPU 工作，不能声称全部消除。
开启单机多 TP 配置校验；DP/PP/EP/CP、多机、最大活跃并发和其他限制不变。
worker、runner、插件需要同时更新至 split Graph API v3，否则启动明确报错，
避免新版插件在旧 worker 中过早读取尚未完成的查询。CUDA SA 源码与状态布局未改。

本地分文件验证：core 61 passed，实际 worker recurrence 94 passed，
真实 Gloo TP=1/2/4 为 3 passed；NCCL 2 卡项因本机无 CUDA 跳过。
覆盖延迟查询完成后的整批跳过、gate 不匹配、所有 compact mask、raw/padded
批大小切换、候选/神经前缀/资格的 rank 分歧和重放后恢复。
Gloo 证明选择协议，CPU 模型边界替代验证 worker 候选链；两者均不替代 CUDA、
NCCL Graph 或真实模型吞吐。新增真实 SA 辅助 stream Graph 测试与 NCCL split
反馈/共识用例，留给内网 GPU 环境执行。本次未测新的吞吐，不报告性能增益。

从仓库根目录，在内网已安装 SGLang/CUDA 依赖的环境中运行：

```bash
export PYTHONPATH="$PWD/experimental/hybrid-spec/src:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest experimental/hybrid-spec/tests/test_core.py -q
python -m pytest experimental/hybrid-spec/tests/test_draft_recurrence.py -q
python -m pytest experimental/hybrid-spec/tests/test_cuda.py -q
python -m pytest experimental/hybrid-spec/tests/test_shared_parallel_cuda.py -q
python -m pytest experimental/hybrid-spec/tests/test_tp.py -q
```

`test_tp.py` 在同一台机器上至少有 2 张可见 NVIDIA GPU 时执行真实 PyNccl
Graph 和原生 SA feedback/query；只有一张卡时该项会 skip，不能记为多 TP 通过。
服务使用 `configs/device-qwen35-mtp-split-global-pool.json`，保持 K=8、gate=2、
Triton draft attention、greedy，分别设置 `--tp-size 1/2/4` 和并发 1/4，
检查整批 SA/全 neural/混合批、3→4 padding、请求结束与槽位复用、flush。
`split_no_skip` 也应跑一次混合批以检查所有实际请求行都进入尾段。

性能只需小对照：使用优化前 `671931d83a` 与当前版本的独立源码目录，
同一模型/TP/配置/提示顺序/输出长度，预热后交替运行重复请求和不同代码请求，
并发 1/4 各至少两轮。检查逐请求输出、接受长度和分支计数，再比较输出 token/s；
改变 TP 时必须同时改变基线服务的 TP。短 trace 检查 SA 查询是否与首个 MTP
forward 重叠、共识是否仍在主 stream、各 rank 尾段大小是否一致。

## 并发 4：混合批逐请求裁剪尾段（2026-10-16）

已实现最多 4 请求的紧凑 MTP 尾段 Graph：整批先检查两个 neural token，
只收集未命中 SA 的行执行剩余 6 次 MTP forward，最后写回原请求行。
A100、TP=1、同一 50 条 edits、2048 输出上限，每种模式重复两轮，均使用
Triton draft attention。原生/full/裁剪的平均吞吐为 837.59/867.38/846.26
token/s，平均 TPOT 为 4.4814/4.2909/4.4057 ms。裁剪相对 full 吞吐
−2.44%、TPOT +2.68%，本次没有额外性能收益；相对原生的约 1% 吞吐差异
小于原生重复波动。约 39.8% 的 decode 请求行跳过尾段，但整批免尾段仅
3.77%（含收尾小批）；小 batch 缩行后 LM head 耗时几乎不变。
按用户要求未做 GSM8K/任务正确性评测。实现、逐轮数据、trace、边界和验证
见[并发 4 实验](validation/split-graph-c4.md)。当前需要 split Graph API v2，
默认仍为 full；CUDA SA 布局保持 v3。

## 并发 1 两段 Graph 提前退出（2026-10-16）

已实现 opt-in `draft_graph_mode=split`：前两个 token 与 MTP 一致后跳过剩余
MTP，不一致则续跑尾段；默认仍为 full。A100、TP=1、K=8、相同 50 条 edits、
2048 输出上限，原生/full/split_no_skip/split 吞吐分别为
305.04/300.66/294.75/342.31 token/s；本轮同步与拆图代价没有抵消收益。
同一 GSM8K 256 题原生/full/split 为 248/248/249，未观察到新增答错；
长输出有重复性差异，不能据此声称严格等价或所有负载稳定加速。
最终 CPU/native/Gloo 147 passed、1 skipped，CUDA 13 passed。完整协议、
trace 和限制见[分析记录](validation/acceptance-regression.md)。CUDA SA 仍为
v3；该轮原型使用 split Graph API v1，当前分支统一要求更新 SGLang
worker/runner 至 split Graph API v2。

## Qwen3.5 MTP 接受长度分析与 A100 对照（2026-10-16）

新增可选 neural 前缀检查；默认关闭，CUDA 扩展布局保持 v3。A100 的原生与
两组 SA 策略、官方 edits benchmark、全量 GSM8K CoT 及重复性限制见
[接受长度分析](validation/acceptance-regression.md)。CPU/native/Gloo 141 passed、
1 skipped，CUDA 11 passed；以下早期记录的“未验证”仅描述当时状态。
该阶段尚未实现 MTP 提前退出；随后原型见上节。独立 SA CUDA stream 与
多 GPU 新策略验收仍待完成。

## 双转移内联：压缩状态并修复低多样性追加退化（2026-10-16）

基线为上一版哈希 `88f8d971eb`，另以原始链表 `44637fe669` 作独立对照。重新检查 sa_spec `flat_graph.hpp` 的单边内联和 clone 直接复制方式，进一步将本实现的第一、第二条转移保留在状态中：

- 状态从 4 列改为 5 列：length/link/first-position/first-target/secondary。secondary=-1 表示空，负值 `~target` 表示第二条内联转移，非负值表示额外边链表。第三条转移才展开边池；总出度超过 4 才建立哈希。单一重复历史直接延长链；二值历史的边记录和哈希桶数量均为零。
- 第一条转移的标签是源状态首次出现位置的下一个 token；第二条标签是目标状态首次出现位置的 token。clone 复制 first-position 和内联转移；重定向保留 first-position，所以不用另存 token 标签。负值仅编码状态 ID，未缩小词表或历史索引位宽。
- 边池从 `3H+1` 缩为 `H+1`，不是按语料经验缩容。证明：任一非第一转移可拼接一条 root→source 路径及目标的“只走第一转移”尾路径，得到以该边为最后一条非第一转移的后缀。确定性保证映射单射；完整字符串全走第一转移，因此非第一转移不超过 `n−1`。实际边池记录还是其子集，因为第二条转移也可内联。全不同 token 的长历史达到该上界，测试同时检查数组尾部哨兵。
- 状态预算约 **108 → 68 字节/token（−37.0%）**；相对原始链表 72 字节/token 还减少约 5.6%。32×262144 的每 rank 状态池 **864 → 544 MiB**；32×16384 为约 34 MiB。反馈/Graph buffer 另计。CPU 与 CUDA 共用状态机，仍按有效前缀导入，Graph 内不分配、不重建整张哈希表。扩展名升为 `hybrid_spec_gpu_sa_v3`，更新后需重启。

CPU 使用 i5-13500HX、Windows、Clang 21.1.0 `-O3`，固定逻辑 CPU 2；每组一次预热、七次样本中位数。xorshift32 seed=17，追加阶段为 1024 token 的每 token 均值。checksum 改为按语义状态和排序后的转移图计算，避免把不同存储布局误判为不一致。全部 12 组旧/新语义校验一致。

下面对照**原始链表**，而不只对照已发生退化的上一版哈希：

| 历史 token | token 种类上限 | 构建 ms：原始 → 紧凑 | 追加 ns/token：原始 → 紧凑 | 追加加速比 |
| --- | --- | --- | --- | --- |
| 8192 | 1 | 0.046 → 0.026 | 5.76 → 3.71 | 1.55× |
| 8192 | 2 | 0.272 → 0.301 | 31.05 → 31.05 | 1.00× |
| 65536 | 1 | 0.358 → 0.234 | 5.18 → 3.52 | 1.47× |
| 65536 | 2 | 2.602 → 2.239 | 41.41 → 39.75 | 1.04× |
| 262144 | 1 | 1.308 → 0.844 | 5.08 → 3.12 | 1.63× |
| 262144 | 2 | 16.209 → 11.942 | 92.38 → 61.91 | 1.49× |

本轮六组低多样性追加均持平或更快，消除了此前明显的 decode 追加回退。**不宣称所有阶段无回退**：8K 二值 CPU 构建仍测到 0.272 → 0.301 ms，约 +10.7%。256K、8192 种 token 的高分叉追加仍比原始链表快约 166×；相对上一版哈希在不同组合有小幅涨落，不能用极端高分叉加速比推算模型 TPOT。

两组完整原始样本（包括非加速项目）：[对照上一版哈希](validation/sa-compact-native-vs-hash.json)、[对照原始链表](validation/sa-compact-native-vs-original.json)。微基准默认 baseline 为 `88f8d971eb`，支持 `--baseline-ref 44637fe669`；CPU 可用 `--cpu N` 控制核心亲和性。

验证：`test_native.py` + `test_core.py` **42 passed**，`test_cuda.py` **10 skipped**。覆盖二值穷举、2/3/5 字母表的独立 Python 完整图对照、随机 clone、内联目标重定向、哈希碰撞、host→大槽位续写、32K 极限扇出、单/二值零边池、长重复后新增分支、int32 大 token、容量与哨兵。CUDA 用例扩展为单/二值/高分叉三种长历史导入后 Graph 更新；本机无 CUDA/nvcc，**CUDA 编译、GPU 性能和真实 TPOT 仍未验证**。服务器先运行 CUDA 差分测试，再运行 README 中的同进程 Graph A/B，并固定模型/头/TP/K/调度参数做服务对照。

## 转移边哈希与长历史微基准（2026-10-16）

基线 `44637fe669`；参考 sa_spec 固定快照 `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969` 的单边直达、多边哈希设计。本实现独立实现了以下布局，未复制参考库的分配器或改变候选语义：

- 出边数 ≤4 时保留短链，第 5 条边时才建立精确 `(state, token)` 哈希索引。保留邻接链供 clone 枚举；碰撞比较完整键，不会误把哈希相同当成转移相同。
- 每增加一条被索引的边，分裂一个桶；避免 decode 中整表扩容。桶链长度仍有碰撞相关的最坏情况，不能声称严格 O(1)。根节点和高分叉节点不再全量扫描出边。
- `edges` 从 3 列增至 6 列，新增 source/hash-next/bucket-head；桶列与邻接列互不覆盖。`meta` 从 5 列增至 7 列，保存 hash-base/indexed-count。桶数量不超过已分配边数，所有索引均为槽内相对位置，且不依赖最大历史容量。CPU 构建、GPU eager 更新、Graph 内反馈消费共用 `automaton.h`。
- 仍只导入有效前缀，reset 不清整池；未索引边字段初始化，脏槽复用不会看到旧桶。扩展名升为 `hybrid_spec_gpu_sa_v2`，需重启服务。状态约从 72 增至 108 字节/历史 token：32 × 262144 每 rank 约 864 MiB，另加反馈与 Graph buffer。

第一版所有分叉都索引，二值历史出现明显回退，因此最终版本增加了 ≤4 出边的快路径。仍未完全消除低多样性负载的成本，下面保留全部结果。

本地 **CPU**：i5-13500HX、Windows、Zig/Clang 21.1.0、C++17 `-O3`。两版使用同一 harness，xorshift32 seed=17；每个长度/词表组合预热一次，测五次取中位数。构建不含分配，追加是在相同长历史后追加 1024 token 的每 token 均值；不是单次 decode kernel 延迟。新旧所有语义状态与转移的 checksum 一致。原始样本见 [sa-history-native-cpu.json](validation/sa-history-native-cpu.json)。

| 历史 token | token 种类上限 | 构建 ms：旧 → 新 | 追加 µs/token：旧 → 新 | 追加加速比 |
| --- | --- | --- | --- | --- |
| 8192 | 1 | 0.041 → 0.058 | 0.0050 → 0.0076 | 0.65× |
| 8192 | 2 | 0.313 → 0.375 | 0.0358 → 0.0421 | 0.85× |
| 8192 | 64 | 1.317 → 0.505 | 0.1802 → 0.0727 | 2.48× |
| 8192 | 8192 | 47.602 → 0.388 | 11.1363 → 0.0500 | 222.73× |
| 65536 | 1 | 0.369 → 0.512 | 0.0081 → 0.0073 | 1.11× |
| 65536 | 2 | 2.736 → 3.906 | 0.0491 → 0.0575 | 0.85× |
| 65536 | 64 | 13.428 → 6.126 | 0.2693 → 0.0978 | 2.76× |
| 65536 | 8192 | 1128.460 → 7.358 | 17.7081 → 0.1201 | 147.42× |
| 262144 | 1 | 1.653 → 2.035 | 0.0059 → 0.0084 | 0.70× |
| 262144 | 2 | 15.957 → 22.210 | 0.0734 → 0.1213 | 0.61× |
| 262144 | 64 | 120.606 → 34.952 | 0.8536 → 0.2271 | 3.76× |
| 262144 | 8192 | 5941.910 → 31.802 | 22.6168 → 0.1488 | 151.97× |

高分叉合成历史支持“消除线性查找瓶颈”的判断；它不能证明真实代码编辑请求会获得相同加速。单 token/二值历史仍存在额外存储和维护成本，256K 二值追加 0.0734 → 0.1213 µs/token；容量与收益需按实际工作负载选择。接受长度没有因这项数据结构改动而增加的依据。

验证：`test_native.py` + `test_core.py` **42 passed**，`test_cuda.py` **8 skipped**。原生 C++ 覆盖二值穷举、随机 clone、强制碰撞、逐步 hash/邻接一致性、compact host 导入较大槽后继续追加、满容量及脏槽重用；Python 对照独立 SA 的 links/first-position/edges，验证异步导入所有权与 GIL 释放。新增 CUDA 用例覆盖 8K 以上历史导入、Graph 内追加分桶及全语义状态对照。Ruff 修改文件检查、两个微基准 `--help`、`git diff --check` 通过。

**本机无 CUDA/nvcc，未编译或执行 CUDA 改动；没有 GPU 加速比或模型 TPOT 结论。** `benchmark_sa_history_cuda.py` 在同一进程编译基线和当前版本，测真实 `consume_query_kernel` 的 Graph replay：每轮追加 9 token，默认 B=1/4/32，计时前恢复相同历史；验证各行完整语义状态。这个缓存较热的组件测试排除 H2D 导入、反馈生成、TP all-reduce 和模型 forward，仍需配合固定模型、头、TP、K、调度和负载的服务 A/B。

## 对照 sa-spec 的 device 热路径优化（2026-10-16）

修改前 HEAD：`2d279d4f34`。直接对照本地 sa_spec 固定快照和已保存的 TRT-LLM #10951 集成，确认参考路径仍执行 MTP forward，在每步 sampler 选择 SA token 并将其用于下一步输入。没有将“保留 neural forward”误判为本实现独有的开销。

本轮实现两项优化，均限 device 历史模式：

1. **统计融合到已有反馈 kernel。** 原 `Metrics.record()` 每次 verify 单独执行六次 reduction，另有 dtype 转换、减法、乘法、stack 和计数累加。现在同一个 `feedback_kernel` 在收集接受路径时累加六个 int64 计数，每行更新所属来源的三项计数。统计发生在 slot/generation 校验之前，无 SA 槽或旧代际的 neural 行也计入；prefill 不计入 verify 统计。计数缓冲首次在消费 stream 分配，诊断仍显式同步读回，CPU 历史保留原实现。
2. **固定请求映射不重复 staging。** 由 worker 构造的视图携带不可变的 host slot/generation key，连同 raw/padded batch size 检查。全部不变时跳过 slot 清零和两次 metadata 拷贝；重排、槽位复用、padding 形状变化均重新 staging。没有 host key 的可变视图仍每次拷贝。只缓存映射，不缓存 SA 查询、选择 mask 或 TP 共识结果。

保留一次精确 TP all-reduce。参考库没有本实现的 rank-local fallback 协议，不能据此去掉共识。也未改变 K、阈值、候选、接受采样或 neural forward 次数，因此本轮目标是降低每轮成本，不声称提高接受长度。

仍有值得单独测量的差异：sa_spec 转移边使用内联单边和哈希表，本实现使用链表扫描；高出度节点的查找成本随边数增加。容量扩大、host 构建、GPU 更新延迟需按历史长度和词表多样性扫描。没有原始 trace 或 GPU 测量证明它是当前 EAGLE3 退化的主因，本轮不混入 SA 布局重写。

本地结果：`test_core.py` **38 passed**；`test_tp.py` **3 passed, 1 skipped**（真实 CPU Gloo TP=1/2/4、子组；NCCL 跳过）；`test_cuda.py` **7 skipped**。新增 CUDA 差分用例覆盖融合统计、无槽位行、过期代际、prefill 不计数。CPU 缓冲测试覆盖不变映射零写入、重排、代际变化、padding 缩小再扩大及可变视图。Ruff 修改文件检查、微基准 `--help` 和 `git diff --check` 通过。**本机无 CUDA/nvcc，未编译或执行修改后的 CUDA kernel，没有 GPU 加速比。**

新增 `benchmarks/benchmark_device_overhead.py`：同进程交替测试原统计/融合统计、原映射 staging/固定映射复用，默认 B=1/4/32、K=8，每组 5 次、每次 200 轮，输出每次及中位数 stream/wall 微秒。两路反馈均包含相同的 pending 清空操作；它只隔离组件 eager 开销，不能代表端到端吞吐、GPU kernel 纯执行时间或 CUDA Graph 内耗时。安装插件后运行 README 命令，先执行 CUDA 差分测试，再做微基准及固定模型/数据/调度参数的 native/hybrid A/B。

## Device overlap + chunked prefill 接入（2026-10-16）

修改前 HEAD：`b39fab6873`。device 历史原已允许 overlap；本轮放开其 chunked prefill 校验并补齐分块生命周期。CPU 历史仍要求 `--disable-overlap-schedule --chunked-prefill-size -1`，mixed prefill/decode 继续拒绝。

`Scheduler` 在构造本轮 `ScheduleBatch` 时保存 `chunked_req`，标识尚未完成 prefill 的请求。扩展以该批标记判断中间块，而不读取 overlap 结果处理会修改的 `req.inflight_middle_chunks`：

- 中间块不构建 SA、不占槽、不消费临时采样 token；若该请求因 retract 重新 prefill，先关闭旧代际，避免旧 build/feedback 污染后续请求。
- 同批的其他 final-chunk 请求正常构建。不能用批级 `contains_last_prefill_chunk` 跳过整批，也不能删除中间行后错用原始采样下标；无效行保持 slot=-1。
- 最后一块开始前按该行完整 `seq_lens_cpu` 快照历史，包含 radix 命中的前缀和之前各块；仅一次后台构建，与最后一块 target/draft prefill 重叠。首次 draft 才导入完整状态并消费真实 bonus，若 host build 尚未完成仍会等待。
- 全部行均无可用 SA 槽位时，prefill 返回钩子不读取 `next_token_ids`，兼容中间块跳过采样的情况。

本地回归：修改前，新增分块场景暴露中间块占槽、临时反馈和重算旧状态未被丢弃的问题；修改后 `test_core.py` **37 passed**，`test_draft_recurrence.py` **72 passed**，`test_cuda.py` **6 skipped**（本机无 CUDA）。这覆盖分块/非分块后台构建、最后块与中间块共批、完整历史快照、槽位复用及旧回调隔离。Ruff 修改文件检查与 `git diff --check` 通过。

尚未在 GPU 上验证模型输出、NCCL/Graph 或吞吐。部署时使用 `history_mode="device"`，移除 `--disable-overlap-schedule`，将 `--chunked-prefill-size -1` 改为 `--chunked-prefill-size 2048`（或省略以使用模型默认值），保持 `--enable-mixed-chunk` 不启用。固定模型/头、TP、K、JSON、请求和采样参数，native 与 hybrid 分别比较：两项关闭、仅 overlap、overlap + chunked prefill。输入应跨至少两个 chunk，同时覆盖短长请求共批、prefix cache 命中、abort/retract；检查 token 对照、SA 使用率和接受长度后再测 TTFT/TPOT。短于 chunk size 的请求不能用于证明分块路径正确或有收益。

## 性能报告独立复核与共识路径改进（2026-10-16）

输入为根目录 `hybrid-spec性能瓶颈分析一.md`，报告基线 `1b0f580307`；本轮修改前 HEAD 为 `4030650794`。本地只有分析 Markdown，没有报告引用的原始 CPU/GPU trace，以下区分源码事实、实验观察和待证假设。

### 独立判断

1. **可以确认 CPU 历史路径存在额外开销，但不能确认它解释全部退化。** 当前 CPU `DraftExtension.bind()` 原先每批调用共识，且完整历史拼接、词表检查、前缀比较、不可变 tuple 追加均随历史增长。报告中的“合成负载通信次数相同”与“无条件通信导致合成负载下降”不能仅凭汇总表相互推出；GPU kernel 累计时间也不是 CPU 临界路径时间。
2. **eager 通信不等于全设备/主机同步，更不必然破坏已有 Graph。** 原生 draft、verify、draft_extend 原本就是分别 replay；图外算子可以在同一 stream 上排队。是否产生长 host gap、跨 rank 到达偏差、通信等待，需要 CPU launch、GPU 时间线和所有 TP rank 的相关性证据。`record_param_comms` 和 u32 kernel 名称本身不足以证明“打断整段 Graph 导致 TPOT 翻倍”。
3. **不采用按本地 `use_retrieval` 跳过共识。** 一张卡无槽位/版本不符、另一张卡命中时，参与集合通信的 rank 数或顺序会不一致。对 GPU mask 做 `.any().item()` 还会增加 D2H 等待。保留所有 rank 的同序共识和统一 fallback。
4. **不直接把 SA 分配挪到 KV 规划之后。** 当前 `Scheduler.init_model_worker()` 先构造 draft，再规划 KV；`KVCacheConfigurator._profile_available_bytes()` 从当前空闲显存减 runtime slack，已经驻留的 SA 因而计入预算。若以后延迟分配，需要在 KV 规划时显式预留相同空间。Graph 固定地址也不能证明不同进程/配置之间不受布局影响。
5. **缩小配置的实验有混杂因素。** 32×16384 改成 1×256 同时改变分配量、可索引长度和被跟踪请求数；并发 4 时只跟踪 1 个请求，不能把恢复都归因于内存布局。0.73/0.79 约为原生吞吐的 92%，而恢复损失比例是 `(0.73-0.52)/(0.79-0.52)` 约 78%，两者口径不同。
6. **当前分支已有可复用的增量设备模式。** `DeviceOverride.prepare()` 已把反馈消费、SA 查询与 TP 共识放进 draft Graph；不需要重新设计一套 SA 状态机。它消除每轮完整 CPU 历史检查，但仍有 SA 更新、查询、反馈与通信开销，也不减少 neural forward。接受长度降低时，消除这些开销仍不保证获得加速。

### 本轮代码

- 两阶段“广播候选 + MIN 一致性”改为一次精确 MIN 归约 `[tokens, ~tokens, eligibility]`。按位取反逆转有符号 int64 顺序，逐位置比较最小值与最大值即可确认所有 rank 候选一致；不使用可能碰撞的哈希，也没有负号在 INT64_MIN 上的溢出。不再需要 source/root rank。小消息的集合通信次数由每批 2 次降为 1 次，归约输入从两种张量改为 `8*B*(2K+1)` 字节；这不是实测性能提升声明。
- CPU 历史模式 `bind()` 只准备本地候选；新增 `TPOverride.prepare()`，通过已有原生 draft 消费入口执行共识。GraphBuffers 绑定同样的视图，因此 Graph 可捕获通信，eager 使用相同语义。两种模式都复用现有 PyNccl；缺少可用通信器时 CUDA TP 启动报错。
- CPU 路径待统计视图引用 Graph 的最终选择 mask，避免把共识前的本地命中当成实际 SA 使用；padding 清理保留有效行，并支持对同一别名视图重复 staging。
- 新增 `configs/device-k3.json`：K=3、阈值 4、整链 fallback，容量 4×8192（约 2.25 MiB/卡），覆盖报告的并发/长度。旧 example 保留，避免无法复现旧路径。这个配置同时改变两项因素，不能用于单变量归因。
- 未改变模型层、原生 target verify、接受采样、KV 池或停止逻辑。K=1 沿用原生没有 draft Graph 的 eager 入口，不声称被捕获；TP=1 不创建同步器。

### 部署环境的必要对照

保持同一模型、GPU、TP、K=3、阈值 4、fallback、输入、停止方式、缓存流程和 overlap 配置，分别执行：

| 对照 | 唯一变化 | 要回答的问题 |
| --- | --- | --- |
| 修改前/后 CPU 历史，均 32×16384 | 本轮共识实现 | 一次归约及 Graph 捕获减少多少开销 |
| 修改后 CPU/device，均 32×16384 | 历史模式 | 完整 CPU 历史维护占多少临界路径 |
| device，32/4 请求槽，均 H=16384 | 预分配槽位数，实际并发均 4 | 空闲容量是否影响性能，避免只跟踪一个请求 |
| device，4×16384/4×8192 | 每槽容量，全部实际历史 <8192 | 分配足迹是否独立影响吞吐 |
| 上述关键配置，Graph 开/关 | Graph | 通信捕获的净收益；不能只比较 kernel 累计占比 |

分别测合成未命中与重复代码命中场景；预热覆盖真实并发，每组多于 4 个请求并重复独立运行、交替顺序。无 profiler 的端到端 TTFT/TPOT/输出吞吐用于性能结论；单独抓所有 TP rank 的 CPU/GPU 时间线定位 gap、通信等待、SA 更新/查询。记录实际请求数、输出 token 数、SA 选中率和接受长度。若新旧序列/长度不同，应同时报告，不能只报吞吐比。

本地验证（Windows / Python 3.11.15 / PyTorch 2.14.1+cpu，按文件单独运行）：

| 检查 | 结果 |
| --- | --- |
| `test_core.py` | 34 passed |
| `test_draft_recurrence.py` | 72 passed，执行当前原生 draft 函数体；K=1/2/4、eager 选择语义、矩阵/列表、hot-token 与 mRoPE |
| `test_tp.py` | 3 passed，1 skipped；真实 Gloo TP=1/2/4 和 `[1,3]` 子组；无 GPU/NCCL，相关用例跳过 |
| `test_native.py` | 4 passed，实际 C++ 状态机与 host builder |
| `test_cuda.py` | 6 skipped，无 CUDA |

合计 **113 passed，7 skipped**。新增/扩展覆盖精确共识（int64 极值、同和但不同候选、混合 mask、单 rank 回退）、CPU 历史 eager/Graph 缓冲桥接、捕获前全 false 到运行时命中、请求重排、padding、固定存储、同视图重复 staging 和共识后统计。CPU 缓冲测试不等于实际 Graph replay；双 GPU NCCL 分支包含真正的 Graph 捕获，但本机未执行。

Gloo 在沙箱内的临时目录创建/进程通信失败；改用工作区临时目录并在沙箱外运行后通过。native 编译也在沙箱外完成。没有 CUDA/NCCL 实机性能结果，不宣称恢复报告中的 34%/51% 损失。

修改的 Python 文件通过 Ruff 导入顺序及 F401/F811/F821/F823/UP037 检查；27 个 Python 文件格式检查和 `git diff --check` 通过。新增 JSON 经 `HybridConfig` 实际解析，设备状态估计为 2,359,712 字节/卡。

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
