# Split compact 的 conditional graph 实验接入

在已有 split 配置中增加 `"conditional_draft_graph": true`，默认关闭。
模板为 [device-draft-split-conditional.json](configs/device-draft-split-conditional.json)。
它与 `draft_graph_mode="split"` 配合；不接受 `full` 或 `split_no_skip`。
现有 Triton / dsv4 attention、KV、greedy、单机 TP 等兼容约束继续适用。

可再设置 `"conditional_draft_graph_dispatch": "switch"`，使用一个原生
SWITCH 替代平衡 IF 树；省略时为 `"if"`，保留原有行为。
模板为 [device-draft-split-switch.json](configs/device-draft-split-switch.json)。
SWITCH 要求 CUDA runtime 和 nvcc 工具链均 >=12.8，以及 Ninja、兼容的
C++ 编译器与 PyTorch `CUDAGraph(keep_graph=True)` / raw graph 接口；
首次启用会在捕图前 JIT 编译插件内的小型 CUDA 适配器。

可额外设置 `"conditional_draft_graph_joined": true`，把 prefix、行压缩和
SWITCH 放进同一张主图；仅支持 SWITCH，默认关闭。
模板为 [device-draft-split-joined.json](configs/device-draft-split-joined.json)。

TP>1 的条件图使用按通信 group 缓存的专用 PyNccl communicator，显式设置
`graphUsageMode=1`，在 KV 预算前预留通信资源，要求实际加载的 NCCL >=2.29，且未设置
`NCCL_GRAPH_MIXING_SUPPORT` 覆盖该配置。prefix/tail 捕获和预热期间，模型
collective 与 SA 共识统一路由到该实例；退出捕获作用域后恢复普通 communicator
及其通信后端，普通图和 eager 路径的默认初始化保持不变。条件图在每个 group
上仅允许同一执行流串行 replay；先释放 Graph，再销毁 group 及专用 communicator。

## 执行方式

普通 split：

```
prefix replay → GPU 压缩行 → num_neural.item() → CPU 选择 tail replay
```

开启后：

```
prefix replay → GPU 压缩行 → conditional tail graph replay
                                  └→ GPU 按 num_neural 选择固定形状的尾段
```

默认分段模式保留 prefix 和行压缩的独立启动。
仅开启 draft 条件图时，移除的是中间选择 tail 的 `.item()`；最终 compact verify 确定 target Graph
token 档位时原有的 host 读回仍可能存在，不能据此宣称整轮无同步。

每个 tail 大小先准备自己的 metadata / 暂存并预热，再直接捕获到条件节点内，
不额外保留一套普通 tail Graph。使用 PyTorch `CUDAGraph` 的
`begin_capture_to_if_node` / `end_capture_to_conditional_node` 接口，
与其 `torch.cond` CUDA Graph 适配器相同；不要求整个模型能通过 Dynamo 编译。
该接口依赖具体 PyTorch 版本，当前按仓库的 PyTorch 2.14.1 接入。

以上是 IF 路径。SWITCH 将每个确定性 tail 单独捕获为
`CUDAGraph(keep_graph=True)`，然后把它作为一个 SWITCH body 中的 child
graph node；0 是空分支，1..B 对应精确行数。一个 CUDA kernel 直接读取
int32 `num_neural`、设置条件 handle，并累计 histogram/mixed 统计，省去
IF 树的多次比较、handle 设置及独立统计 kernel。
捕图期间通过 CUDA API 把 SWITCH 插入 PyTorch 的 parent graph，更新其
依赖，使所选 tail 等待 setter 完成。热路径没有 Python 选分支或嵌套 replay。
全部 leaf Graph 对象及其 private pool 保留到 parent reset 后才释放；
不能只留下 CUDA raw pointer，否则临时 tensor 的地址可能被重新分配。
仅适用于已有的确定性 greedy 推理 tail，不支持含随机采样的 child graph。
SWITCH 生产路径在行压缩 kernel 内写入 `live_bs`，不再单独 `fill_`；
histogram/mixed 仍逐步准确累计。直接调用 dispatcher 而未经过行压缩时，
`replay(raw_bs)` 保留 fill，防止缩批使用旧 batch。prefix 仍未合图；
不承诺减少 Graph 启动数或显存。

joined 模式会保留 prefix 的 raw graph/private pool，把它作为新主图的
child node，在同一主图中捕获行压缩并原生创建 SWITCH；不能把含条件节点的
旧 SWITCH 图直接克隆成 child graph。主图开始有一个 4-byte batch 写入节点，
每轮通过 `cudaGraphExecMemsetNodeSetParams` 更新它的整数值，然后只 replay
一次。行压缩从固定 GPU 地址读取实际 `raw_bs`，循环受捕图容量 B 限制，
仅选取 `row < raw_bs` 的有效行；统计也使用这个实际 batch。
更新参数只影响之后的 launch，已排队的 launch 保留原值；没有复用 pinned
host scalar 的覆写竞态，没有额外 GPU 读回。输入节点更新成本包含在计时中。
这减少 host 提交次数，但增加图内 batch 写入和 prefix child 的设备开销，
是否划算仍以完整一轮 wall 测量为准。主图 template/node handle、prefix 和
各 leaf 的 Graph/private pool 都保留到 cleanup。

`num_neural=0` 时跳过所有 tail；其余大小按平衡 IF 树选一条分支，
避免每轮对所有大小逐个执行条件判断。压缩行号仍稳定排序，只包含实际请求，
不包含 Graph padding。每个大小仍保留固定形状的 attention metadata / buffer；
条件图另有自己的内存池，不能保证显存减少。

TP 沿用 prefix gate 结束后的候选及最终 mask 共识；所有 rank 按相同大小顺序
捕获所有分支，再按共识后的行数运行同一条 collective 序列。
条件体内的 NCCL / custom all-reduce 兼容性仍需在实际部署组合上验收。
捕获失败会报错，不允许某个 rank 独自回退；应停掉全部 rank、关闭选项后重启。

## 服务器验证

需要 NVIDIA CUDA >=12.4、兼容驱动和提供上述绑定的 PyTorch 构建。
ROCm、CPU 或缺少绑定时明确报错。不要启用 PyTorch allocator 的
`graph_capture_record_stream_reuse:True`，PyTorch 条件捕获暂不支持该组合。

在 GPU 服务器仓库根目录安装插件及测试依赖：

```bash
python -m pip install -e './experimental/hybrid-spec[test]'
```

先运行 STANDALONE 的真实 CUDA Graph 递归测试（合成模型、原生 KV metadata
和 topk / 行压缩，覆盖 changing mask、padding、零尾段及不同来源上限）：

```bash
python -m pytest experimental/hybrid-spec/tests/test_cuda.py \
  -k 'conditional and standalone and not arctic' -q
```

两张卡再运行独立的 PyNccl 条件分支测试。它刻意让 rank 的本地资格不同，
验证达成共识后不会走不同的 collective 分支，覆盖非二次幂 batch、padding
及反复切换 tail 大小；不加载模型：

```bash
python -m pytest experimental/hybrid-spec/tests/test_tp.py \
  -k conditional -q
```

TP1 已在 RTX 4060 / Torch 2.14.1+cu130 验证，覆盖 20 项 conditional
合成模型用例及 B=1/4/16/32 的 FP16 GEMM 调度探针；
结果、首次失败和限制见 [4060 测试记录](validation/conditional-graph-4060-20261019/README.md)。
IF 节点、统计与启动的消融见 [4060 开销分析](validation/conditional-graph-overhead-4060-20261019/README.md)：
B=32 时当前 IF 树是主要新增成本，消除 `.item()` 仍可能回退。
后续 [SWITCH 实测记录](validation/conditional-switch-4060-20261019/README.md)
包含接入方式、659 项回归及 IF/SWITCH 开销对照。
[统计/batch 融合记录](validation/conditional-switch-fused-4060-20261019/README.md)
包含当前 SWITCH 的 673 项回归及与原 SWITCH、`.item()` 的整轮交错对照。
[合图记录](validation/conditional-joined-4060-20261019/README.md) 包含 joined
模式的 700 项回归、固定 batch 与循环变长的三路完整轮对照。
TP2 条件 collective 和真实模型服务仍待验收；skip 不代表验证通过。
在已可运行的原生模型命令上设置：

```bash
export SGLANG_PLUGINS=hybrid_spec
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/device-draft-split-conditional.json"
```

模板保持 K=8、gate=2、容量 8；原生命令需匹配
`--speculative-num-steps 8 --speculative-num-draft-tokens 9`、topk=1、
`--max-running-requests 8 --cuda-graph-max-bs-decode 8`，普通模型使用
`--speculative-draft-attention-backend triton`。已有配置可直接只增加开关，
模型路径、TP、gate 等按原有命令保持一致。

## 观测和 A/B

分支直方图、mixed 数等在 GPU 累计；`/get_internal_state` 的诊断 hook
读取时才同步并合并到 `hybrid_spec_split_graph`。
`conditional_draft_graph=true` 标识该模式，`decision_wait_ns` 应保持 0。
CPU 侧只即时维护 `prefix_ct` / `request_ct`，不要直接把进程内的
`split_graph_counters` 当作条件模式的完整分支统计。
做性能测量时不要持续轮询诊断接口，以免读回本身影响结果。

用同一模型、负载、batch、gate 和 target Graph 配置，分别关闭/开启开关，
重启服务后比较输出、接受长度、吞吐、TPOT、启动显存和 trace。
应检查 `hybrid_prefix_decision` 对应的 `.item()` 空闲是否消失，以及条件判断
本身的 GPU 开销。此版不保证减少 Graph 启动次数；4060 调度微测未显示稳定加速。

可与 `dynamic_verify=true` 组合；接受率和预算策略不变。
开启 conditional 会改变共享固定成本，因此离线耗时表必须按新配置重跑；
默认关闭时仍兼容此前没有该字段的旧表。
IF 与 SWITCH 使用不同的耗时表指纹，切换 dispatch 后也需要重跑离线表。
融合版 SWITCH 的指纹另含 revision=2，旧 SWITCH 的表也会被拒绝。
joined=true 也进入耗时表指纹，不能沿用分段 SWITCH 的表。

## GPU 选择 compact target 档位（第二步）

增加 `"conditional_verify_graph": true`，同时保留
`"conditional_draft_graph": true` 和 `"verify_mode": "compact"`。
target 分派使用 CUDA SWITCH，要求 CUDA toolkit >=12.8；draft 可以继续使用
IF 或 SWITCH。开关默认关闭，旧配置与旧耗时表指纹保持兼容；开启后耗时表指纹
包含这个字段，不能直接沿用旧表。

TP 对最终长度达成共识后，GPU 归约 M 并对原有 target token 网格做 lower-bound。
CPU 只保留按 live batch 全窗口计算的入场上界，不读取选中的 G。
一个 target 分支负责自己的打包、KV 地址翻译、attention 元数据准备、G 行模型
forward 及原生 compact-to-dense 散射。每个已有 token 档位只捕获一次模型，
没有按 batch×档位增加模型 Graph，也没有用 dense 执行代替 compact。
分支复用 target 临时内存池，散射结果保存在池外的共享输出缓冲中。

target 与 draft 条件图共用第一步的专用 NCCL communicator 生命周期。
target leaf 内不记录原有的 external read-done event；parent replay 后在主流
记录完成事件，保守覆盖所有共享 KV 元数据读取，避免把 event 节点带进条件体。

当前接入 Triton 和 DSV4，包括 online C128 MTP。条件 verify 的 C128 planner
直接从 GPU 实际长度构造原生 CompressPlan，不需要 CPU 长度镜像。
普通 prefill、decode 和等长 verify 继续使用原有元数据路径。
继续沿用 compact 原有的 topk=1、无 Mamba /
mRoPE / LoRA / DP / PP / PDMux / two-batch-overlap 限制。

compact 计数器在 GPU 累计，诊断查询才读回，flush 不重置这个累计值。
动态价格向量按 context / 网格缓存，只在首次使用或该 context 的 EMA 更新后上传。
采样计时若需要 G 标签，会先异步复制到独立 pinned host 标量；仅在完成事件
query 成功后读取 CPU 标量，不为标签在 decode 中等待 GPU。

短回归入口：

```bash
PYTHONPATH=experimental/hybrid-spec/src python -m pytest -q \
  experimental/hybrid-spec/tests/test_conditional_verify.py \
  experimental/hybrid-spec/tests/test_compact_verify.py \
  experimental/hybrid-spec/tests/test_switch_graph.py
```

组件测试覆盖实际 SWITCH、原生 ragged/scatter kernel 与简化 GPU 模型/KV 边界；
不等于 DSV4 真模型或 TP=2 的端到端验证，不据此宣称吞吐收益。

2026-10-20 短回归记录：

- 本地 CPU：`test_core.py`、`test_conditional_verify.py`、`test_compact_verify.py`、
  `test_compact_tail.py`、`test_switch_graph.py` 合计 188 passed / 63 CUDA skipped。
- 远端 `hz01-ssh.gpuhome.cc:30120`：单卡 RTX 4090 48GB，
  PyTorch 2.12.1+cu130 / CUDA 13.0。
- `test_conditional_verify.py`：9 passed，21.11 秒（包含首次编译）。
  检查实际 target 行数只等于选中 G、升降档、缩小 batch、KV 写入、散射输出、
  GPU 计数及异步计时标签；模型与 KV translator 是确定性的 GPU 边界替身。
- 原生 compact prepare（三种模式）、ragged Triton attention 独立参考对照、
  已有 SWITCH（排除 joined）：13 passed，7.48 秒。
- 未启动服务、未下载权重、未测吞吐；单卡机器未执行 TP=2，未验证 DSV4 真模型。

DSV4 online C128 补充：

- GPU planner 保持原生 16-byte CompressPlan ABI；实际 verify token 之外的计划
  标为 invalid，padding 请求不写在线状态。
- 按每条请求的真实长度和 packed offset 保存逐前缀状态，支持 9-token 窗口；
  提交仍使用实际接受后的序列长度，跨 128-token 边界保持原有重置语义。
- 条件 target 启动时重新设置 native commit 的 Python 标记，GPU 计划负责标记
  pending；提交后清空 pending，兼容连续 replay 及神经尾段被跳过的下一轮提交。
- 没有关闭 online compress，也没有为了适配 C128 恢复等长 target 执行。
- 新增 `test/registered/kernels/ops/attention/test_online_c128_ragged.py`：远端
  5 passed / 2 dtype subtests passed（3.53 秒）。覆盖真实 DSV4 元数据入口、
  FP32/BF16 GPU 与 CPU 参考、128 边界、padding、9-token 状态银行和连续捕图 replay。
  隔离 harness 替代包初始化、测试框架和未使用的 legacy TVM/JIT 导入边界；
  实际 planner、Triton 内核、controller 和提取的 DSV4 元数据方法均执行生产代码。
  完整 DSV4 模型、原生 compressor 数值链和 TP=2 仍未验证。
