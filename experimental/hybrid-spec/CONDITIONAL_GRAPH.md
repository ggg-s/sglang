# Split compact 的 conditional graph 实验接入

在已有 split 配置中增加 `"conditional_draft_graph": true`，默认关闭。
模板为 [device-draft-split-conditional.json](configs/device-draft-split-conditional.json)。
它与 `draft_graph_mode="split"` 配合；不接受 `full` 或 `split_no_skip`。
现有 Triton / dsv4 attention、KV、greedy、单机 TP 等兼容约束继续适用。

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

这版保留 prefix 和行压缩的独立启动，没有把整个 draft 合成一张图。
移除的是中间选择 tail 的 `.item()`；最终 compact verify 确定 target Graph
token 档位时原有的 host 读回仍可能存在，不能据此宣称整轮无同步。

每个 tail 大小先准备自己的 metadata / 暂存并预热，再直接捕获到条件节点内，
不额外保留一套普通 tail Graph。使用 PyTorch `CUDAGraph` 的
`begin_capture_to_if_node` / `end_capture_to_conditional_node` 接口，
与其 `torch.cond` CUDA Graph 适配器相同；不要求整个模型能通过 Dynamo 编译。
该接口依赖具体 PyTorch 版本，当前按仓库的 PyTorch 2.14.1 接入。

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
