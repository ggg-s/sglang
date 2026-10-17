# GPU 动态行号与可配置 split gate

## 行为与边界

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

## 回归范围

CPU 测试直接执行当前 worker 的 `draft_forward` 函数体，覆盖 G=0/1/2/3/K、
K=1/2/8、命中/回退/混合、完整候选 materialization 与位置递推。
CPU compact recurrence 检查逐步 KV/metadata；dispatch 测试包含 B=1/8/32。

GPU 行号测试覆盖 B=1/8/33/257/4097、跨 tile、空实际批、padding、全命中、
全 neural、混合与 no-skip。GPU compact Graph 测试使用当前 worker 函数体、
原生 topk kernel、GPU 行号扫描和真正的 Graph capture/replay，覆盖 B=1/8 及 B=16/G=3、
K=1/2/3/8、G=0/1/2/3/K、Qwen/V4 元数据边界、非连续 request pool 顺序、
动态小批及私有池 allocator 压力。模型和 attention 计算仍为合成边界，不能替代
真实 DSV4 attention、权重或 TP8 NCCL 的内网验收。

## 工作机验证结果

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

## 内网验收

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
