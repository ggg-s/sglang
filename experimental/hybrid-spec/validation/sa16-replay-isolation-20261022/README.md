# SA16/MTP8：固定容量 1，隔离 ReplaySSM 的影响

2026-10-22，用户授权按同一服务容量做快照 / Replay 对照。在 A100-PCIE-40GB 上完成四次独立启动、200 个正式请求。

**同样的请求容量 1、Graph 最大 batch 1，快照平均 79.80 tok/s，Replay 平均 75.43 tok/s。Replay 吞吐低 5.47%，TPOT 高 13.89%；接受率没有下降。**

## 对照方法

- 服务器：用户指定 `fj02-ssh.gpuhome.cc:30154`，独立端口 39922。GPU 为 A100-PCIE-40GB，驱动 580.119.02，功率限制 250 W，最大 SM 时钟 1410 MHz。
- Qwen3.5-35B-A3B-GPTQ-Int4 + unquant FP16 MTP，TP1，FP16 激活 / 卷积、**FP32 SSM**，attention / GDN decode / verify 均为 Triton。
- 生产代码固定 `50084814319d2e61eee6499500e9fe1904a2bc8b`，核验 5659 个文件。复用前次只读源目录，未修改生产代码。
- 两组均为 SA16 / neural8、frequency、split / gate=1 / fixed compact、min-match4、16K history、无 global pool；相同 `sa16-n8.json`。
- **max-running=1，decode Graph max batch=1**；Mamba cache=40、实际 KV=131072 token、context=16384、mem-fraction=0.94。每次启动读取实际值并断言，不以命令行意图代替实际容量。
- 唯一模式差异：快照不开 Replay；Replay 开 `--enable-linear-replayssm-spec --linear-replayssm-cache-len 64`。目标验证宽度仍为最大 17，没有缩短 SA 链。
- 仍为同一批 50 条冻结的 [SWE-rebench/OpenHands 真实 Agent 前缀](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)，7587–11996 input token。保留工具定义和历史工具结果，没有重新采样。JSONL SHA256：`14fa95c7d417ae66a0e59dfeb7e9aeea4013334ba474d2a19ea4109f69f004c5`。
- ABBA 顺序：**快照 R1 → Replay R1 → Replay R2 → 快照 R2**；每项独立启动服务，先相同正确性检查、长前缀恢复、8 条最多 256 token 预热，再做 8 条最多 32 token 预热并清缓存。
- 官方 `sglang.benchmark.serving`，C1、temperature=0、seed=42、自然 EOS、输出上限 2048；每项 50 请求、客户端 warmup=0。四项 input_lens 向量逐项相等，before/after 计数排除预热。

ABBA 用于降低运行顺序的影响；每组只有两次独立启动，不提供统计置信区间。没有将这里的固定前缀续写当作执行工具后的 SWE 修复评测。

## 性能结果

下表为两轮算术平均；相对百分比按同轮配对后平均。

| 配置 | R1 / R2 tok/s | 平均 tok/s | TPOT ms | TTFT ms | 平均总时长 s |
|---|---:|---:|---:|---:|---:|
| SA16/MTP8，完整快照 | 79.185 / 80.407 | **79.796** | **5.046** | 496.1 | 45.355 |
| SA16/MTP8，ReplaySSM L64 | 75.439 / 75.416 | **75.427** | **5.746** | 498.2 | 48.630 |
| Replay 相对快照 | 两轮均较低 | **-5.47%** | **+13.89%** | 约 +2.0 ms | — |

两组每项实际输出：快照 3614 / 3624 token，Replay 3716 / 3620 token。吞吐和 TPOT 按客户端真实输出统计，没有强行截成相同长度。外层进程起止时间包含导入开销，不能拿它重算吞吐。

## 接受统计

统计均为本项 50 个正式请求的设备计数差值，再对两轮取平均。

| 指标 | 快照 | Replay |
|---|---:|---:|
| 请求轮次选择 SA | 35.95% | 35.58% |
| SA draft 接受率，不含 bonus | 53.34% | **53.71%** |
| SA 接受长度，含 bonus | 9.534 | **9.594** |
| neural 接受长度，含 bonus | 5.532 | **5.609** |
| 总体接受长度，含 bonus | 6.971 | **7.027** |
| 每 1000 实际输出 token 的 verify 请求行数 | 158.33 | 157.45 |
| 每 1000 实际输出 token 的 target 验证位置 | 1880.37 | 1865.22 |

Replay 的接受长度略高、每输出 token 的验证工作位置略少，但 TPOT 仍更高，**不支持“SA 候选接受率变差导致本轮变慢”的解释**。在固定服务容量后，开启 Replay 路径带来了可观的时间代价。

快照路径保存各候选位置的状态；Replay 用较小的历史记录恢复所需状态，增加了读历史与状态计算。本次可确认这个模式切换的整体代价；没有额外捕获 trace，不能进一步把 0.70 ms 的 TPOT 差值全分摊给某个 GDN 内核。设备接受计数可能包含最终停止截断前的尾部，不能等同实际交付 token；host decision wait 也不能整段当作额外 GPU 耗时。

## 显存与正确性

| 实际检查 | 快照 | Replay |
|---|---:|---:|
| 请求槽 / KV token | 1 / 131072 | 1 / 131072 |
| 常驻 FP32 SSM GiB（日志） | 2.40 | 2.40 |
| 完整 SSM 中间快照 GiB（日志） | 1.99 | 0 |
| Replay 记录 GiB（日志相加，四舍五入） | 0 | 约 0.088 |
| target verify Graph GiB（服务上报） | 0.109 | 1.018 |
| 启动可用显存 GiB | 8.787 | **9.812** |
| 正式请求区间最高观测 GPU MiB | 32102 | **31052** |

四次启动的分配结果一致。Replay 记录节省的空间有一部分被更大的 verify Graph 分配占用；净启动可用显存增加 **1.025 GiB**。这不是八并发的空间差值；上一份八并发报告使用了不同的活动槽和 Graph 容量。GPU 每秒采样给出观测峰值，可能漏掉短时峰值；Graph 上报值也不能直接相加当作整机占用。

每次启动均执行相同提示、thinking=false、temperature=0 的固定 GSM8K 前 20 条 smoke，四次均 **19/20**，0 错误、0 截断；长前缀恢复均命中 **11648 cached token**，两次回复文本相同。200 个正式请求全部成功、输出非空、无长度上限截断、均有闭合 tool_call 标记；没有检查工具参数正确或执行工具。

同组两轮逐字相同输出：快照 44/50、Replay 45/50。量化 MoE 的温度 0 重复输出仍有差异，不宣称逐位等价。正确性检查是 smoke，生产内核边界回归见前次组件及 A100 注册测试报告，本次没有修改内核。

## 和之前 82.41 tok/s 的关系

之前 82.41 的 SA16 配置也是容量 1 / Graph 1、完整快照；输入 SHA 和本次相同。**本轮快照只复现到 79.80，低约 3.18%，没有完整复现旧值。** 本轮快照总体接受长度为 6.971，旧两轮为 7.179 / 7.058，也存在输出与接受轨迹差异。

因此，本轮可以归因的是“当前代码、相同容量下，Replay 比快照慢约 5.47%”；不能把旧 82.41 到前次新 74.54 的全部差距都归给 Replay，也不能把剩余差距直接归给 Graph 容量。前次容量 8 / Graph 8 的 Replay 单并发为 74.54，这次容量 1 / Graph 1 为 75.43，来自不同批次，只作为参考。

本次没有重新启动原生 MTP8，**不重新给出对原生 MTP8 的严格配对收益**。历史 +8.73% 保留为历史配置结果，本次解释了其中一部分变化，并没有重新证明那个收益。

## 复现与证据

- `launch.sh`、`sa16-n8.json`、`plan.json`：固定参数。
- `run.sh` / `run_isolation.py`、`checks.py` / `bench_helpers.py`：启动顺序、相同正确性检查、预热和正式测量。
- `analyze_bench.py`、`audit_results.py`：同输入、零请求错误、实际容量、设备计数、缓存恢复、显存与清理审计。
- `summary.json`、`audit.json`：全部逐项数据及两轮均值；`historical-sa16-reference.json` 仅为旧结果引用，不作为本次计时分母。
- `evidence.tar.gz`：86 个原始文件及 SHA256 清单，**1,551,690 bytes**；本地逐成员校验通过。含冻结输入、来源与许可、完整命令、服务日志、数学预测、测量输出、源码 SHA 和显存/时钟/温度采样。
- SHA256：`530c93b747ca9cc8379bf10f51b595f857e640662ffc64930be54545cb44d1b6`。
- 远端：`/root/rivermind-data/validation/sa16-replay-isolation-20261022`。全部自有测试服务、采样进程结束，39922 端口关闭，GPU 最终 **0 MiB / 0%**，服务器保持运行。
