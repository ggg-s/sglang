# DSPARK split 调度开销优化

日期：2026-10-20。基线为 `e04810c639`。仅优化 gate=0 split 的 host 调度和缓冲分配，保留既有 gate、SA/neural 候选选择、static verify 和非 folded 接受/提交策略。

## 触发数据

用户提供的 500 条结果：native DSPARK 544.49 tok/s，full+SA gate=1 为 537.89 tok/s，split+SA gate=0 为 515.72 tok/s。split 的 `proposal_ct=183122`、`skip_request_ct=93901`、`neural_request_ct=89221`、`all_sa_ct=93901`、`mixed_ct=0`。这些计数表明跳过 proposal 已生效；不能仅由跳行比例推导节省的 GPU 时间。

full 与 split 的 gate 不同，接受长度也不同，二者差异不能全部归因于 split 调度。本次不调整 gate 或候选，不把组件调度微测当成端到端收益证据。

## 变更

- 原 `(~use_retrieval).nonzero()` 在 CUDA 上需同步以确定输出长度。随后，非空 `rows.cpu()` 再同步读回行号；空行号不产生第二次有效数据回读。
- 新实现使用 worker 自有的 pinned CPU mask 缓冲，阻塞拷贝一次固定长度 mask，然后在 CPU 判断全 SA / 全 neural / 混合。没有 CUDA nonzero；全 SA / 全 neural 不构造或传输行号。混合批才生成 CPU 行号，并以 `non_blocking=True` 上传后执行原行选择。
- 全 SA 直接借用 SA 查询候选，避免 clone。原生静态验证只读取该输入，接受反馈在下一轮查询之前完成；混合批使用独立可复用缓冲，保持原 SA 候选不变。
- anchor/mask block、greedy mask、temperature 与混合候选缓冲在 worker 初始化时分配。每轮只更新本批 anchor；不再每轮分配 `full_like` 和两个 `ones`。
- 默认 full 路径不分配这些新缓冲，保持原行为。没有新环境变量或全局缓存。

这仍是 host 分支：一次 mask 回读等待没有消除。混合批原生 CPU 长度和 KV 计划选择仍在，条件图及 folded 接受策略没有扩展。

## 短回归

本地 `test_dspark.py`：**20 passed / 8 CUDA skipped**。已有原生 proposal/KV 选行、混合批合并、TP 否决边界与完整 static verify/commit 接线继续通过。新增 B=1 命中/未命中反复切换检查，覆盖 row 0 未命中、候选与 anchor 无残留、借用输入不被改写及元数据缓冲复用；同一用例提供 CPU/CUDA 两个参数。

Ruff 仓库检查项与格式、`git diff --check` 通过。

按用户指定改到工作站 **RTX 4060 Laptop 8GB**，复用已有 PyTorch **2.14.1+cu130**、CUDA **13.0** 和 driver **592.82**。独立目录 `/workspace/dspark-split-overhead-20261020/source`，上传 48 个源文件并核对 SHA256，未改动已有工作区或安装依赖。

远端 `test_dspark.py`：**28 passed，60.15 秒**，其中 **8 项真实 CUDA 用例**，没有跳过。覆盖 4 项原生 proposal/KV 选行及 eager/folded CUDA Graph 组合、1 项 B=1 反复命中切换、3 项真实 GPU SA 构建/查询/接受反馈。TP 否决继续使用可控共识边界，不代表多卡 NCCL 验证。

测试后没有改动生产实现或测试代码；微测脚本从 CPU-only profile 扩展为 CPU/CUDA profile，以取得 runtime 同步调用记录。最终 `manifest.json` 保存实际文件哈希（含最终脚本）、测试用例数和环境；`source-manifest.json` 是最初上传清单。远端全部结束后 GPU **0 MiB / 0%**。

## 有界微测

`check_dispatch.py` 比较基线和当前生产 `propose`，B=1、K=5，分别测全 SA 与全 neural。神经结果是预生成的确定性缓冲，不执行模型、SA 查询、target verify 或网络请求。每组 warmup 20 次、5 轮各 200 次，交替测量顺序，报告中位数和每轮值；另外各 CPU/CUDA profile 4 次，统计 nonzero、clone、临时张量与 CUDA runtime 同步调用。计时在 profiler 之外进行。

最终带 CUDA runtime 记录的复测（`dispatch-cuda.json`）：

| 分支 | 优化前中位数 | 优化后中位数 | 调度耗时下降 |
|---|---:|---:|---:|
| 全 SA | 145.54 μs | 49.42 μs | 66.0% |
| 全 neural | 120.95 μs | 27.75 μs | 77.1% |

每次调用的 profile 事件数（原始记录为各 4 次调用的合计）：

| 事件 | 全 SA 前 → 后 | 全 neural 前 → 后 |
|---|---:|---:|
| `aten::nonzero` | 1 → 0 | 1 → 0 |
| `cudaStreamSynchronize` | 1 → 1 | 2 → 1 |
| `cudaMemcpyAsync` | 3 → 2 | 2 → 1 |
| `aten::clone` | 1 → 0 | 0 → 0 |
| `aten::full_like` | 1 → 0 | 0 → 0 |
| `aten::ones` | 2 → 0 | 0 → 0 |

每段 profile 结束还记录到一次 `cudaDeviceSynchronize`，属于测量边界，不计为每轮 proposal 开销。`cudaMemcpyAsync` 包括 device 内拷贝，不能全部解释为 D2H。这验证了全 neural 分支少一次 host 等待，全 SA 仍一次等待，但移除了 nonzero、候选 clone 和临时初始化；不是所有分支都减少一次同步。

首轮 CPU-only profile 的计时也保留在 `dispatch.json`：全 SA 190.65 → 61.32 μs，全 neural 111.85 → 27.90 μs。两轮都观察到调度成本下降；绝对耗时存在 WSL/系统调度波动，不设单次微秒阈值，也不把这些结果外推到内网 GPU。

```bash
git show e04810c639:experimental/hybrid-spec/src/hybrid_spec/integrations/sglang/dspark.py > /tmp/dspark-before.py
PYTHONPATH=experimental/hybrid-spec/src python \
  experimental/hybrid-spec/validation/dspark-split-overhead-20261020/check_dispatch.py \
  /tmp/dspark-before.py
```

这是调度组件微测，不能换算真实模型的 TPOT 或吞吐。本轮没有真实 DSPARK 权重服务、完整 DSV4 attention/压缩 KV 数值验证或吞吐实验；gate=0 的接受长度损失和非 folded 接受/提交的成本仍存在。

`evidence.tar.gz` 包含测试日志/XML、两次微测原始 JSON 与 profiler stderr、基线源码、源文件哈希和环境清单。SHA256：`9db7837f027909ca1d54dde0f754f91607e99724a855daf264dca5cc3204b5d7`。
