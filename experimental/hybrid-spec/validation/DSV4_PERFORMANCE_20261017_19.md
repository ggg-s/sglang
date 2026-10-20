# DeepSeek-V4-Flash 混合投机解码（MTP + Suffix SA）H100 实测报告

> 实验时间：2026-10-17～2026-10-20；整理更新：2026-10-20。沿用原文件名，保留既有文档链接。
>
> 数据来源：用户提供的内网实测汇总；本次补齐离线标定、edits c1/c16/c32、SWE 与 decode 指标，未重跑 GPU 实验，也未从内网产物重新计算逐请求统计。百分比按展示值核对，原始全精度数值可能有舍入差异。各轮使用历史版本，不代表当前分支 HEAD 的一次统一验收。
>
> 入口：[全部测试总报告](TEST_RESULTS.md)；[完整历史记录](TEST_RESULTS_DETAILS.md)。本报告区分固定长度、动态预算、逻辑验证 token 与实际 Graph 执行量。

## 目录

- [1. 主要结果](#summary)
- [2. 实验方法与口径](#conditions)
- [3. Edits：500 条性能对照](#edits)
- [4. SWE：线上负载复现](#swe)
- [5. Decode 阶段速度对照](#decode)
- [6. 工程约束与问题记录](#constraints)
- [7. 配置建议与适用范围](#conclusions)
- [8. 产物、版本与待核对项](#provenance)

<a id="summary"></a>

## 1. 主要结果

| 场景 | 观测结果 | 比较范围 |
|---|---|---|
| edits c1，target-only → native MTP K8 | 166.16 → 270.16 tok/s，+62.6% | MTP 本身的收益 |
| edits c1，padded SA8/neural3/gate1/global-pool | **430.74 tok/s**，TPOT **2.183 ms** | 相对当轮 native K8 +59.4%；相对 target-only +159.2% |
| edits c1，compact 对 padded | 416.16 对 430.74 tok/s，−3.4% | TP8 下该组小 batch 对齐不利；不能外推到所有并发 |
| edits c8～32，fixed compact | 相对 native K8 **+48.6%～+61.6%** | 已测 compact 臂中 fixed 优于 dynamic；未覆盖同并发 padded 对照 |
| edits，加载离线表的 dynamic compact | 比 fixed compact 慢 **3.0%～5.2%** | c1/c16/c32；少验证 token 尚未转化成净收益 |
| SWE c8/c16，SA8 对 baseline | 同批次 RPS **约 +37% / +28%** | baseline、MTP8、SA8 三臂实验 |
| SWE c8/c16，SA8 对 MTP K3 | 同批次 RPS **约 +4%～+6%** | K3、K8、SA8 三臂实验；K3 比 K8 更强 |
| SWE c80 | SA 的 TPOT 降低约 25%，E2EL 增加约 6% | KV 压力下系统吞吐下降，decode 改善没有带来端到端收益 |

edits 的 K/gate 网格中，混合 SA 的 K8 组合优于对应 K3 组合；**native MTP 则是 K3 优于 K8**，不能概括为“edits 一律用 K8”。gate1 在所测 full/split、K3/K8 组合中优于 gate2；Arctic suffix 与 GPU SA gate1 的吞吐基本持平。

## 2. 实验方法与口径

<a id="conditions"></a>

### 2.1 环境与负载

| 项 | 配置 |
|---|---|
| 硬件 | 8×NVIDIA H100 80GB HBM3，TP=8，驱动 580.105.08 |
| 模型 | DeepSeek-V4-Flash，FP8 权重；`SGLANG_DSV4_FP4_DEQUANT=1`，反量化后用户报告为 **35.84 GB/卡** |
| 软件 | SGLang `codex/hybrid-suffix-speculation` + `experimental/hybrid-spec` 插件；各轮提交另列 |
| edits | `glaiveai/code_edits_sample`，500 条；历史快照 `dfdacc8e`，输入 153～903 token，输出上限 2048 |
| edits 工具 | hybrid-spec harness 调用官方 `sglang.benchmark.serving`；greedy、overlap 开启，历史配置 chunked prefill=8192 |
| edits 重复 | 用户最新说明：`--repeats 1`、顺序执行；主表 c1，并发测试另标 c8/c16/c32 |
| SWE | `swe_bench_openai_new_matched_v6`，4219 请求 / 459 user group |
| SWE 工具 | `/workspace/perf-benchmark` dynamic 模式，`per_user_replay=true`、`active_user_ratio=1.0`，c8/c16/c80，每点名义时长 900s |

输出 2048 是上限，不是每条实际输出长度。不同工具、提交和实验批次分别比较，不把各轮行数相加作为独立样本数。

### 2.2 配置术语

| 名称 | 含义 |
|---|---|
| K | `num_drafts` / `--speculative-num-steps`，草稿步数 |
| gate / G | `num_neural_gate_tokens`，SA 与 neural 前 N 个 token 的一致性检查 |
| neural3 | `num_neural_drafts=3`；neural 回退最多 3 步，SA 仍可 8 步 |
| full | `draft_graph_mode=full`，完整 neural draft |
| split | `draft_graph_mode=split`，拆分 gate 前缀和尾段，跳过 SA 命中行的 neural 尾段 |
| static | `verify_mode=static`，SA/neural 等长，target 验证完整窗口 |
| padded | `verify_mode=padded`，接受链按来源长度截断，target 仍验证完整逻辑窗口 |
| compact | `verify_mode=compact`，按有效 token 打包 target 验证 |
| fixed | `dynamic_verify=false`，验证上限按来源/模式固定，不在线缩短 |
| dynamic | `dynamic_verify=true`，compact 内按在线接受统计选择更短前缀 |
| global-pool | `enable_global_pool=true`，跨请求及完成请求共享 SA 历史 |

**fixed 与 padded 是两个轴。** 本文的 fixed compact 始终指 `verify_mode=compact` + `dynamic_verify=false`，不能改写成 padded。

| verify_mode | dynamic_verify=false | dynamic_verify=true |
|---|---|---|
| static | 等长完整窗口 | 当前配置校验不支持 |
| padded | 来源限长、完整验证窗口 | 当前配置校验不支持 |
| compact | **fixed compact** | **dynamic compact** |

配置语义核对自 [config.py](../src/hybrid_spec/config.py)；动态流程见 [在线校准与动态验证](../DYNAMIC_VERIFY.md)。SA fraction 是被选用 SA 候选的 decode 请求行占比，不是独立请求命中率；accept 是服务报告的平均每轮接受长度，包含项以原始统计定义为准。

### 2.3 正确性与统计边界

用户报告，同一 target-only 服务连续跑两遍，32 条中有 21 条文本不同，总输出 token 相差 1 个。当前栈存在输出不可逐 token 复现的现象，严格文本恒等不能单独作为投机正确性判据；这也不能替代接受规则、KV 状态或任务质量验证。本次汇总没有新增质量验收结果。

SWE 客户端 `total_cached_tokens=0` 与服务端约 86% prefill 命中率并存。用户将其归因于 OpenAI 响应未回传 `cached_tokens`；本报告按服务端统计记录命中率，未独立核对远端响应与客户端记账源码。

<a id="edits"></a>

## 3. Edits：500 条性能对照

<a id="mtp"></a>

### 3.1 MTP 本身的价值

c1，native MTP 为 K8。

| 配置 | 吞吐 tok/s | mean TPOT ms | 相对 target-only |
|---|---:|---:|---|
| target-only，无投机 | 166.16 | 5.921 | — |
| native MTP K8 | 270.16 | 3.598 | 吞吐 +62.6%，TPOT −39.2% |

<a id="c1-k8"></a>

### 3.2 K8 / c1 配置全景

native 在不同轮次约 270.2～270.6 tok/s；相对 native 一列保留用户报告的当轮舍入结果，不将它们统一除以一个跨轮基线。

| 配置 | 吞吐 tok/s | TPOT ms | accept | SA fraction | 相对 native K8 |
|---|---:|---:|---:|---:|---:|
| native MTP | 270.2～270.6 | 约 3.59 | 约 2.97 | — | — |
| full + SA，gate0 † | 358.41 | 2.651 | 4.219 | 27.7% | +32.5% |
| split + SA，gate0 † | 371.45 | 2.552 | 4.222 | 27.7% | +37.3% |
| full + SA，gate1 | 381.51 | 2.477 | 4.506 | 36.1% | +41.0% |
| full + SA，gate2 | 358.74 | 2.649 | 4.224 | 27.8% | +32.6% |
| split + SA，gate1 | 395.32 | 2.384 | 4.502 | 36.0% | +46.1% |
| split + SA，gate2 | 369.74 | 2.565 | 4.226 | 27.8% | +36.7% |
| GPU SA split gate1，Arctic 轮复测 | 395.52 | 2.383 | 4.503 | 36.0% | +46.1% |
| Arctic suffix，gate1 | 393.85 | 2.412 | 4.546 | 36.7% | +45.6% |
| Arctic suffix，gate2 | 359.53 | 2.658 | 4.219 | 28.8% | +32.9% |
| static SA8/neural8，gate1 | 394.53 | 2.389 | 4.500 | 36.0% | +46.0% |
| **padded SA8/neural3，gate1** | **430.74** | **2.183** | 4.427 | 35.7% | **+59.4%** |
| compact SA8/neural3，gate1 | 416.16 | 2.259 | 4.389 | 34.3% | +54.0% |

† 最新报告将 gate0 两行标为 `device-sa-spec`、`partial=zero`；gate1/2 与 neural3 使用 `partial=fallback` + global-pool，因此 gate0 与其他行并非只改变 gate。早期汇总曾将同组 358.41 / 371.45 标为 gate2/global-pool/max_requests=4，split accept 为 4.221；保留此标签冲突，最终以原 command/config 为准。

split 相对 full 的吞吐在 gate1 为 +3.62%，gate2 为 +3.07%。Arctic gate1 相对同轮 GPU SA 为 −0.42%，接受长度略高；CPU 查询/H2D 抵消接受收益是待 profile 验证的解释，不能由该吞吐表直接分解开销。

<a id="k8-k3"></a>

### 3.3 K8 与 K3

| 方案 | K8 吞吐 tok/s | K3 吞吐 tok/s | K3 accept |
|---|---:|---:|---:|
| native | 270.56 | 350.34 | 2.877 |
| full gate1 | 381.51 | 370.30 | 3.296 |
| full gate2 | 358.74 | 350.07 | 3.110 |
| split gate1 | **395.32** | 367.85 | 3.297 |
| split gate2 | 369.74 | 342.65 | 3.108 |

K3 的 split 相对 full：gate1 −0.66%、gate2 −2.12%；可节省的尾段更短，拆图方案的净收益为负。K3 full gate2 相对 native K3 −0.08%，基本持平。

在此等长 K/gate 网格内，K8 split gate1 比 K3 full gate1 快约 6.8%。全表最优仍为 §3.2 的 padded SA8/neural3；它相对历史 native K3 350.34 约 +22.95%，这是跨轮参考，不能替代同期配对实验。

<a id="gate"></a>

### 3.4 Gate 对照

| K | full gate1 相对 gate2 | split gate1 相对 gate2 | SA fraction gate1 / gate2 |
|---:|---:|---:|---|
| 8 | +6.35% | +6.92% | 约 36.0% / 27.8% |
| 3 | +5.78% | +7.36% | 约 53.0% / 40.5% |

这里按展示值重算；K8 full 的旧汇总为 +6.34%，差异来自舍入。gate1 在这四组中均较快，支持该负载优先用 gate1，尚不是跨模型最优结论。

<a id="verify"></a>

### 3.5 Padded 与 compact：c1 / SA8/neural3/gate1

| 模式 | 吞吐 tok/s | TPOT ms | accept | 逻辑 target token 相对 dense |
|---|---:|---:|---:|---|
| **padded** | **430.74** | **2.183** | 4.427 | 无压缩，每请求完整 9-token 窗口 |
| compact | 416.16 | 2.259 | 4.389 | −36.5% |

compact 相对 padded 吞吐 −3.38%，TPOT +3.48%。用户报告 TP8 下 compact 的 4/9-token 档位对齐为 8/16，该 c1 比例下实际 target token 比 dense 对照多 19.3%。**逻辑 token 减少不能直接换算为实际计算节省。** 19.3% 的精确分母、dense 对照的物理 padding 与图直方图仍待原始 diagnostics 核实；不将该比例外推到 c8～32。

两臂接受长度、SA 比例也略有变化，本对照衡量端到端配置效果，不是相同候选轨迹下的纯算子差值。

### 3.6 离线耗时表初始化：c1/c16/c32（ecd8ef51a3）

离线标定覆盖 batch 1/16/32 × context 512/1024/2048，统计验证 token 数与 verify 耗时的关系；正式 dynamic 服务通过 `dynamic_verify_cost_table_path` 加载 `verify-costs.json`，接受率继续在线统计、成本由在线样本修正。标定目录见 §8；本批没有独立的“仅离线表、关闭在线修正”消融。

| c | 方案 | 吞吐 tok/s | TPOT ms | accept | 相对当轮 native K8 |
|---:|---|---:|---:|---:|---:|
| 1 | native | 269.93 | 3.601 | 2.968 | — |
| 1 | **fixed compact** | **413.06** | **2.280** | 4.348 | **+53.0%** |
| 1 | dynamic compact | 391.41 | 2.406 | 4.345 | +45.0% |
| 16 | native | 1497.45 | 10.456 | 2.970 | — |
| 16 | **fixed compact** | **2300.89** | **6.656** | 4.295 | **+53.7%** |
| 16 | dynamic compact | 2232.09 | 6.835 | 4.182 | +49.1% |
| 32 | native | 2214.46 | 14.003 | 2.986 | — |
| 32 | **fixed compact** | **3289.69** | **9.268** | 4.452 | **+48.6%** |
| 32 | dynamic compact | 3180.56 | 9.462 | 4.268 | +43.6% |

| c | dynamic 相对 fixed 吞吐 | accept 变化 |
|---:|---:|---:|
| 1 | −5.24% | −0.07% |
| 16 | −2.99% | −2.63% |
| 32 | −3.32% | −4.13% |

用户报告 c16/c32 dynamic 的整轮逻辑验证 token 约少 8%，但接受长度也下降；c1 逻辑 token 几乎相同，仍慢 5.2%。探索、成本查询和计时可能贡献控制开销，具体占比尚无本批 trace 归因。加载离线表后，dynamic 仍未超过 fixed；这些结果不证明在线 EMA 或离线表某一项单独造成回退。

<a id="concurrency"></a>

### 3.7 c8/c16 并发补充（c39a01d74d）

| c | 方案 | 吞吐 tok/s | TPOT ms | accept | 相对当轮 native K8 |
|---:|---|---:|---:|---:|---:|
| 8 | native | 1009.40 | 7.740 | 2.938 | — |
| 8 | **fixed compact** | **1631.27** | **4.666** | 4.333 | **+61.6%** |
| 8 | dynamic compact | 1577.54 | 4.836 | 4.283 | +56.3% |
| 16 | native | 1489.15 | 10.496 | 2.951 | — |
| 16 | **fixed compact** | **2316.97** | **6.652** | 4.339 | **+55.6%** |
| 16 | dynamic compact | 2250.37 | 6.791 | 4.234 | +51.1% |

本批 dynamic 相对 fixed 吞吐：c8 −3.29%、c16 −2.87%；TPOT +3.64% / +2.09%，accept −1.15% / −2.42%。用户报告逻辑 token 相对 dense 的节省为 dynamic 约 41%、fixed 约 37%，不能直接当作物理 Graph 耗时降幅。

`mixed_ct` 在 c8/c16 分别为 23,883 / 37,200，c1 恒为 0，说明真实混合批出现。native 从 c1 约 270 到 c8 1009 tok/s，约 3.7 倍；padded c1 与 compact c8 的比较涉及 verify 模式变化，不作为固定配置的并发扩展比。

<a id="swe"></a>

## 4. SWE：线上负载复现

### 4.1 工作集与运行边界

模型权重 35.84 GB/卡，`mem-fraction-static=0.7` 下用户报告 KV 池约 1.99M tokens。按每活跃用户约 46k tokens 粗估：c80 约 3.68M，c8/c16 约 368k / 736k。前者超过池容量，与用户报告的排队/抢占吻合；实际驻留仍受前缀共享、请求长度分布和调度影响，不能把乘法估算视作精确占用。

c80 测试中，split capture 80 OOM；将 capture 上限降到 32 又遇到 `total 297 > max grid tier 288`。因此该轮 c80 SA 使用 full 路径。c8/c16 的 split + compact 可运行，服务端 prefill 缓存命中约 86%。这些是所测配置的边界，不是所有 H100 配置的硬性并发上限。

### 4.2 c80：KV 受限时的端到端负收益

| 方案 | 完成请求 | RPS | Completion tok/s | TTFT s | TPOT ms | E2EL s |
|---|---:|---:|---:|---:|---:|---:|
| baseline，无投机 | 950 | **1.01** | **232.5** | **10.8** | 347.5 | **77.5** |
| MTP K8 | 902 | 0.95 | 223.6 | 25.7 | 289.4 | 82.2 |
| full SA，gate1 | 907 | 0.95 | 222.7 | 28.8 | **260.0** | 82.0 |

相对 baseline，MTP/SA 的 TPOT 降低 16.7% / 25.2%，TTFT 增加约 138% / 167%，E2EL 增加约 6%，RPS 降低约 6%。用户报告 KV 抢占、前缀逐出，以及复访时服务端命中 0～58 条；额外草稿/Graph 显存和 prefill 成本可能进一步压缩可用空间，具体分摊需要内存与调度日志。

RPS 保留工具输出；900s 为名义压测时长，不用“完成请求 / 900”覆盖工具的统计窗口。

### 4.3 c8/c16：baseline / MTP8 / SA8 同批次对照

SA8 为 split + compact。批次：`hs-swe-c8c16-20261020-102040`。

| c | 方案 | RPS | TTFT ms | TPOT ms | E2EL s | Completion tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 8 | baseline | 2.162 | 447.6 | 13.81 | 3.629 | 488.4 |
| 8 | MTP K8 | 2.321 | 486.0 | 12.68 | 3.434 | 529.6 |
| 8 | **SA8** | **2.963** | 539.1 | **8.74** | **2.695** | **688.3** |
| 16 | baseline | 2.978 | 531.2 | 22.50 | 5.327 | 627.1 |
| 16 | MTP K8 | 3.068 | 563.4 | 22.12 | 5.153 | 649.9 |
| 16 | **SA8** | **3.808** | 637.2 | **16.01** | **4.197** | **795.7** |

| 对比 | RPS | TPOT | E2EL | Completion tok/s |
|---|---:|---:|---:|---:|
| c8 SA8 / baseline | **+37.0%** | **−36.7%** | **−25.7%** | **+40.9%** |
| c16 SA8 / baseline | **+27.9%** | **−28.8%** | **−21.2%** | **+26.9%** |
| c8 MTP8 / baseline | +7.4% | −8.2% | −5.4% | +8.4% |
| c16 MTP8 / baseline | +3.0% | −1.7% | −3.3% | +3.6% |

c8 SA8 RPS 按展示值 2.963 / 2.162 重算约 +37.05%，一位小数为 +37.0%；原汇总为 +37.1%，本报告摘要使用约 +37%。服务端 prefill 命中率：baseline 86.0%、MTP 85.9%、SA 86.7%。SA 改善 RPS/TPOT/E2EL，同时 TTFT 高于 baseline。

### 4.4 c8/c16：MTP K3 / K8 / SA8 同批次对照

批次：`hs-swe-mtpk-20261020-122930`。本轮没有提供独立 baseline 行，不能与 §4.3 的 baseline 冒充同期配对。

| c | 方案 | RPS | TTFT ms | TPOT ms | E2EL s | Completion tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 8 | MTP K3 | 2.770 | 511.5 | 9.70 | 2.882 | 640.2 |
| 8 | MTP K8 | 2.314 | 495.0 | 12.53 | 3.439 | 528.1 |
| 8 | **SA8** | **2.931** | 540.9 | **8.68** | **2.722** | **680.2** |
| 16 | MTP K3 | 3.665 | 592.8 | 17.70 | 4.343 | 765.3 |
| 16 | MTP K8 | 3.060 | 572.0 | 22.13 | 5.159 | 644.9 |
| 16 | **SA8** | **3.821** | 632.7 | **16.28** | **4.144** | **799.7** |

| 对比 | RPS | TPOT | E2EL |
|---|---:|---:|---:|
| c8 K3 / K8 | **+19.7%** | **−22.6%** | **−16.2%** |
| c16 K3 / K8 | **+19.8%** | **−20.0%** | **−15.8%** |
| c8 SA8 / K3 | +5.8% | −10.5% | −5.6% |
| c16 SA8 / K3 | +4.3% | −8.0% | −4.6% |
| c8 SA8 / K8 | +26.7% | −30.7% | −20.8% |
| c16 SA8 / K8 | +24.9% | −26.4% | −19.7% |

该衍生表按展示值计算；原报告分别给出 c8 K3/K8 TPOT −22.7%、c8 SA8/K3 TPOT −10.4% 和 E2EL −5.5%、c16 SA8/K3 RPS +4.2%，均存在约 0.1 个百分点的舍入差异。实际判断为 **SA8 相对更强的 native K3 再提升约 4%～6% RPS**。

原摘要的 SA8 相对 baseline +35.6% / +28.3%，来自本轮 SA8 2.931 / 3.821 对上一批 baseline 2.162 / 2.978，属于跨批次参考。对外主结论采用 §4.3 同批次约 +37% / +28%，与本节相对 K3 的增量分别陈述。

<a id="decode"></a>

## 5. Decode 阶段速度对照

### 5.1 指标定义

| 指标 | 计算方式与含义 |
|---|---|
| edits 的 TPOT 倒数 | `1000 / mean_TPOT_ms`，平均 token 间隔的倒数；不是逐请求速率的算术平均 |
| SWE `mean_decode_throughput` | 按用户提供的工具定义，为逐请求 `output_len / decode_time` 的算术平均；本地未核对 perf-benchmark 源码 |
| Completion tok/s | 系统输出 token 总数 / 工具统计墙钟时间，受 prefill、排队及请求组合影响 |

沿用原报告“纯 decode”的用途，以下统一称 **decode 阶段速度**。decode_time 若包含抢占或阶段内等待，该指标也会受影响；它不是 GPU kernel 的净计算速率。两种平均方法不等价，不能把“decode 速率与系统吞吐的差”直接量化成 prefill/排队成本。

### 5.2 Edits c1 / K8：TPOT 倒数

| 配置 | mean TPOT ms | TPOT 倒数 tok/s | 相对 native K8 |
|---|---:|---:|---:|
| target-only | 5.921 | 168.9 | — |
| native MTP | 3.592 | 278.4 | — |
| full + SA gate1 | 2.477 | 403.7 | +45.0% |
| split + SA gate1 | 2.384 | 419.5 | +50.7% |
| Arctic gate1 | 2.412 | 414.6 | +48.9% |
| static SA8/neural8 gate1 | 2.389 | 418.6 | +50.4% |
| **padded SA8/neural3 gate1** | **2.183** | **458.1** | **+64.5%** |
| compact SA8/neural3 gate1 | 2.259 | 442.7 | +59.0% |

这里 native TPOT 3.592 来自用户 decode 汇总，§3.1 的另一轮为 3.598，不混用分母。MTP 对 target-only 的 TPOT 倒数约 +64.8%；padded 对 compact 约 +3.5%。

### 5.3 Edits fixed / dynamic compact：TPOT 倒数

| c | native | fixed compact | dynamic compact | fixed 相对 native |
|---:|---:|---:|---:|---:|
| 1 | 277.7 | **438.6** | 415.6 | +57.9% |
| 16 | 95.6 | **150.2** | 146.3 | +57.1% |
| 32 | 71.4 | **107.9** | 105.7 | +51.1% |

单位 tok/s，c1/c16/c32 对应 §3.6 的 `ecd8ef51a3` 批次。另一个 `c39a01d74d` 批次 c8：native 129.2、fixed 214.3、dynamic 206.8，fixed 相对 native +65.9%。这些是单请求速率指标，不能再乘并发数作为精确系统吞吐。

### 5.4 SWE c8/c16：逐请求平均 decode 速率

| c | 方案 | mean_decode_throughput tok/s | 平均每轮接受 token | 相对 baseline |
|---:|---|---:|---:|---:|
| 8 | baseline | 79.16 | — | — |
| 8 | MTP K8 | 87.17 | 2.68 | +10.1% |
| 8 | MTP K3 | 118.88 | 2.63 | +50.2% |
| 8 | **SA8** | **150.05** | **4.12** | **+89.6%** |
| 16 | baseline | 48.55 | — | — |
| 16 | MTP K8 | 50.28 | 2.67 | +3.6% |
| 16 | MTP K3 | 65.61 | 2.63 | +35.1% |
| 16 | **SA8** | **81.51** | **4.11** | **+67.9%** |

用户 decode 汇总同时包含 baseline、K3、K8 与 SA8，跨 §4.3/4.4 的两批测试；未提供各指标到原始 DB 行的逐项映射，比例作为该汇总的参考比较。SA8 相对 K3 为 +26.2% / +24.2%，相对 K8 为 +72.1% / +62.1%；K3 相对 K8 为 +36.4% / +30.5%。后一个 SA8/K8 原汇总为 +62.2%，按展示值重算为 +62.1%。

接受长度由约 2.6 提升到约 4.1，与 decode 改善方向一致；系统 RPS 的收益还受 prefill、请求长度和调度影响，不能仅由接受长度推算。

### 5.5 SWE c80：decode 改善与系统负收益并存

| 方案 | mean_decode_throughput tok/s | 平均每轮接受 token | 相对 baseline |
|---|---:|---:|---:|
| baseline | 3.69 | — | — |
| MTP K8 | 4.68 | 2.66 | 约 +27% |
| full SA gate1 | **5.86** | **4.08** | **约 +59%** |

原报告增益为 +27.1% / +59.0%；展示值重算为 +26.8% / +58.8%，此处使用约数。虽然 decode 阶段速率改善，§4.2 的 E2EL 仍增加约 6%、RPS 下降约 6%，端到端部署选择必须同时看系统指标。

<a id="constraints"></a>

## 6. 工程约束与问题记录

| 问题 | 用户报告的现象 | 适用结论与处理 |
|---|---|---|
| 权重与 KV 预算 | 反量化后 35.84 GB/卡；mem fraction 0.5 约剩 2.7 GB KV | 不按 18.6 GB 估算该运行配置；SWE 使用 0.7，实际余量以启动日志为准 |
| max_requests=1 | c1 下 SA 永久零命中 | 用户报告由 `a3a0d112b6` 修复；旧版本用 4 绕过 |
| compact 的 TP8 对齐 | c1 的 4/9 档位映射 8/16，报告实际 token +19.3% | 本组 c1 优先 padded；高并发需单独看物理 token / Graph 直方图 |
| split capture 80 | 大量尾段图与 compact 档位捕获 OOM | 所测 c80 配置未运行成功；不能推广为所有 c>16 均不可用，edits 已有 c32 结果 |
| capture 上限降至 32 | SWE 出现 `total 297 > max grid tier 288` | 同时满足 live batch、验证 token 档位覆盖和显存约束，不能只降低 capture 上限 |
| dynamic verify | 逻辑验证量下降但接受长度也下降，整轮慢约 3%～5% | 离线成本初始化尚未翻盘；保留 fixed 对照，不把逻辑节省当作实际收益 |
| cached_tokens | 客户端 0，服务端 prefill 命中约 86% | 依据服务端口径记录；客户端记账需另行核对 |
| greedy 不可复现 | target-only 两遍 21/32 文本不同 | 严格文本恒等不足以单独判定正确性，仍需接受规则/KV/质量证据 |
| cwd 导入 | 从 `/workspace` 启动命中旧 checkout | 使用正确仓库目录 `/workspace/hybrid-spec`，核对实际 import 路径 |
| compat/runner 早期缺口 | 过早读 attn_backend、draft runner 缺 ragged_verify_mode | 用户报告已在 `c39a01d74d` 修复；旧版需相应补丁 |

split 对容量 B 捕获多种尾段 batch，显存不能只按模型和 KV 估算；prefix、tail、verify Graph 与工作区也占预算。bs80 的 OOM 不提供各项精确占用，不能直接推断每张 Graph 的显存成本。

<a id="conclusions"></a>

## 7. 配置建议与适用范围

1. **Edits c1：优先 padded + SA8/neural3 + gate1 + global-pool。** 所测最高 430.74 tok/s；本组 compact 比 padded 慢，不推荐为“逻辑 token 更少”而切换。
2. **Edits c8～32：已测 compact 臂中优先 fixed。** 相对 native K8 +48.6%～+61.6%；尚无同并发 padded 对照，不能声称所有 verify 模式中最优。
3. **SWE c8/c16：用 native MTP K3 作更强基线。** SA8 在同轮基础上再提高约 4%～6% RPS；c16 是所测两档中的较高 RPS 点，c8 延迟更低，需结合 SLA 选择。
4. **SWE c80：当前容量约束下不采用本批投机配置。** 降低并发或增加可用 KV 容量；HiCache/host offload 属于后续候选，本批没有实测其净收益。
5. **按负载设置 capture。** 覆盖实际 live batch 和 token 档位，同时验证 Graph 内存。SWE c8/c16 可运行、c80 失败，edits c32 可运行，不能只按并发数给统一开关规则。
6. 对外发布前建议每臂至少 3 次、交替顺序，记录版本/命令、缓存状态、成功率、实际验证量与质量指标；当前多数是单轮或单次 900s 观测。

<a id="provenance"></a>

## 8. 产物、版本与待核对项

### 8.1 内网产物索引

以下路径来自用户提供的运行清单，未在本次整理中逐目录读取。

| 内网路径（根目录 /workspace/validation-runs/） | 内容 |
|---|---|
| `dsv4-sa-20261017-120722/` | 32 条早期冒烟 |
| `dsv4-split-20261017-191157/` | 500 条 native/full-g0/split-g0；早期配置标签冲突见 §3.2 |
| `dsv4-k8-gates-20261017-234114/` | K8 gate 网格 |
| `dsv4-k3-gates-20261018-032320/` | K3 gate 网格 |
| `dsv4-arctic-20261018-175813/` | Arctic 与 GPU SA |
| `dsv4-n3-compact-20261018-220704/` | SA8/neural3 padded / compact |
| `dsv4-targetonly500-20261019-112512/` | 无投机 baseline |
| `dsv4-c8c16-20261019-135510/` | edits c8/c16 fixed / dynamic compact，c39a01d74d |
| `dsv4-offline-calib-20261019-170617/` | verify-costs.json 离线耗时表 |
| `dsv4-c1c16c32-20261019-174934/` | edits c1/c16/c32 native / fixed / dynamic，ecd8ef51a3 |
| `hs-swe-c80-20261019-235850/` | SWE c80 三臂 |
| `hs-swe-c8c16-20261020-102040/` | SWE baseline / MTP8 / SA8 |
| `hs-swe-mtpk-20261020-122930/` | SWE MTP K3 / K8 / SA8 |

用户报告每目录含 `results/summary.json` 或 `*_dynamic.csv/.db`、官方 JSONL、server/bench 日志、config/plan。SWE 每批具体提交及 decode 指标到 DB 行的映射仍以这些文件为准。

### 8.2 版本与证据边界

- edits 历史版本从 `588bbaa4` 演进到 `c39a01d74d`，离线表对照为 `ecd8ef51a3`；不能把全部数值当作同一个提交下的单变量实验。
- 本报告没有给出最新合并图 / conditional SWITCH 在 H100 上的独立 A/B 结果；4060 组件/微测见各自验证记录，不能与本报告的真实模型收益相加。
- 本地其他 DSV4 分析产物属于各自运行范围，不能自动替代此处内网目录的原始记录。本次只核对用户汇总、配置语义与展示值算术，没有新增 trace、Graph 内存或质量归因。
- 保留历史疑点：早期 `decision_wait` 报告“合计 1483s、约 7.2µs/prefix”单位不一致；205,544 次若为 1483s 则约 7.215ms/次，若为 1483ms 才约 7.215µs/次，不用于本次收益解释。

### 8.3 计算约定

吞吐变化为 `(方案 / 基线 − 1) × 100%`，TPOT/E2EL 同理，负值表示延迟降低。原始绝对值完整保留；派生百分比若与原汇总不同，说明来自展示值舍入或跨批次基线。所有比较均指所测组合，不承诺跨模型、并发、上下文和缓存状态保持同样收益。
