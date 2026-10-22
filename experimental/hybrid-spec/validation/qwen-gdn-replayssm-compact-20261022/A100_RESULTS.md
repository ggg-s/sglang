# ReplaySSM + GDN compact：A100 完整模型验证

2026-10-22，在用户指定的 `fj02-ssh.gpuhome.cc:30154` 完成。
验证代码：[50084814319d](https://github.com/ggg-s/sglang/commit/50084814319d2e61eee6499500e9fe1904a2bc8b)，远端逐文件核验 5659 个源码文件。

**FP32 状态下，SA16/MTP8 已能在 A100 40GB 保持完整 KV 容量并跑完 8 并发长输入。吞吐没有超过原生 MTP8：单并发低约 1.1%，8 并发低约 6.6%。**

## 测试口径

- NVIDIA A100-PCIE-40GB，PyTorch 2.14.1+cu130、CUDA 13.0、Triton 3.8.0。
- Qwen3.5-35B-A3B-GPTQ-Int4，target 为 GPTQ Marlin，MTP draft 为 unquant FP16；激活 FP16、SSM 主状态 **FP32**、卷积 FP16，TP=1。
- attention / linear decode / linear verify 均为 Triton。decode、target verify、draft-extend 开启 CUDA Graph，prefill Graph 关闭。
- 三组均为 max-running=8、Mamba cache=40、KV=131072 token、context=16384、mem-fraction=0.94、seed=42；每次启动后读取实际容量并断言一致。
- Replay 两组均用 L=64；原生 MTP8 的 Replay 组用于隔离 Replay 影响，没有另行搜索最佳 L。原生 MTP8 为 8 个 draft、验证宽度 9；混合为 SA16 / neural8，frequency / split / gate=1 / fixed compact，最大验证宽度 17。
- 沿用冻结的 50 条 [SWE-rebench/OpenHands 真实 Agent 前缀](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)，46 个仓库，7587–11996 input token，保留工具定义、工具结果和完整对话前缀。没有重新采样；输入 JSONL SHA256 为 `14fa95c7d417ae66a0e59dfeb7e9aeea4013334ba474d2a19ea4109f69f004c5`。
- 使用官方 `sglang.benchmark.serving`，temperature=0，自然 EOS，最多生成 2048 token，并发上限 1/8，各两轮，每项 50 请求。共 **12 项、600 个正式请求**；逐项确认 input_lens 完全相同。
- 各模式先做正确性 smoke、缓存恢复和预热；每项计时前再预热并清缓存。计时客户端 warmup=0，设备诊断采用本项 before/after 差值，排除预热。R1 顺序 C1→C8，R2 顺序 C8→C1。

冻结数据来自官方 rows API 快照，内容 SHA 固定；API 返回值没有绑定 Hub git revision。此处是固定输入的单轮续写吞吐测试，没有执行工具或计算 SWE 修复成功率。

## 性能

下表为两轮算术平均。吞吐包含本批 prefill 和生成；TPOT 为客户端 mean TPOT。相对变化先按同轮配对计算，再取平均。

| 配置 | C1 输出 tok/s | 对原生 MTP8 | C1 TPOT ms | C8 输出 tok/s | 对原生 MTP8 | C8 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
| 原生 MTP8，原快照状态池 | **75.38** | — | 6.013 | **126.05** | — | **36.762** |
| 原生 MTP8 + ReplaySSM | 72.94 | -3.23% | 6.305 | 123.48 | -2.04% | 38.773 |
| SA16/MTP8 + ReplaySSM | 74.54 | **-1.12%** | **5.832** | 117.75 | **-6.59%** | 38.893 |

相比同样开启 ReplaySSM 的原生 MTP8，混合模式 C1 吞吐 +2.19%、TPOT 降低 7.50%；C8 吞吐 -4.64%、TPOT 增加约 0.31%。因此 Replay 的内存收益成立，本次配置的整体吞吐收益不成立。两轮样本不足以证明约 1% 的小差异具有统计显著性。

| 配置 | C1 R1 / R2 tok/s | C8 R1 / R2 tok/s | C1 / C8 TTFT ms（均值） |
|---|---:|---:|---:|
| 原生 MTP8 | 75.886 / 74.880 | 125.725 / 126.378 | 487.5 / 1558.1 |
| 原生 MTP8 + Replay | 72.694 / 73.191 | 123.376 / 123.579 | 493.1 / 1511.7 |
| SA16/MTP8 + Replay | 74.571 / 74.505 | 117.634 / 117.860 | 498.2 / 1611.2 |

这里的并发是请求并发上限，包含自然调度和 prefill，不代表全程 decode batch 恒为 8。吞吐使用客户端正式测量区间，不能用外层进程起止时间重新计算。

## 显存与容量

| 实际启动/采样结果 | 原生 MTP8 | 原生 MTP8 + Replay | SA16/MTP8 + Replay |
|---|---:|---:|---:|
| FP32 常驻 SSM GiB（日志四舍五入） | 2.40 | 2.40 | 2.40 |
| 完整 SSM 中间快照 GiB（日志） | 4.75 | **0** | **0** |
| Replay 记录 GiB | 0 | 0.398 | 0.398 |
| target verify Graph GiB（服务上报） | 0.373 | 0.811 | 1.779 |
| draft decode / split Graph GiB（服务上报） | 0.018 | 0.018 | 2.207 |
| draft-extend Graph GiB（服务上报） | 0.020 | 0.020 | 0.041 |
| 启动后可用显存 GiB（服务上报） | 5.732 | 9.642 | **6.429** |
| C8 正式请求期间观测最高显存 MiB | 35232 | 31230 | **34896** |
| 实际 KV token / 请求槽 | 131072 / 8 | 131072 / 8 | **131072 / 8** |

`nvidia-smi` 每秒采样，所列是观测峰值，可能漏掉短时峰值。Graph 数据是服务各阶段上报的分配变化，不能直接相加当作整机总占用；还有权重、KV、索引、临时工作区和分配器缓存。

同为 MTP8，Replay 后启动可用显存增加 **3.910 GiB**。SA16 的原快照需求约 8.965 GiB，现 Replay 记录实分配约 0.398 GiB，记录部分相差约 **8.567 GiB**；这个差额来自已验证的生产池分配与原布局公式，不是本次重新启动 SA16 原快照服务的整机差值。

SA16 的更多 verify / split Graph 消耗了部分节省空间。实际结果仍是 131072 KV token、8 个请求槽全部保住，最长 11996 token 输入完成了两轮 8 并发，无 OOM、无请求错误。

## 正确性与回归

- A100 完整依赖环境执行新增注册 CUDA 测试：**8 passed + 24 subtests passed**，0 failure/error/skip。JUnit 将子测试计入总数，显示 tests=32，不能说成 32 个独立顶层测试。
- 配置保护、Replay 状态池会计和工作站曾 deselect 的两项 draft-extend 生命周期用例：**9 passed + 20 subtests passed**，0 failure/error/skip。工作站依赖缺口已补验。
- 三组分别通过固定前 20 条 GSM8K、相同显式逐步解题提示、temperature=0、thinking=false 的 `sgl-eval` smoke：均 **19/20**，0 错误、0 截断。
- 三组重复长前缀请求均命中 **11648 cached token**，恢复前后输出文本相同。
- 正式请求均有非空输出、没有触发 2048 token 上限，600/600 包含完整 `<tool_call>…</tool_call>` 标记。这里只检查标记闭合，不等于工具参数正确或任务完成。
- 同模式两轮文本逐字相同的请求数（C1 / C8，各 50）：原生 46/46，原生 Replay 47/44，混合 Replay 46/45。量化 MoE 下本次运行也存在温度 0 的重复差异，不能宣称全模型逐位等价。

数值边界、部分接受、状态折叠、换行、FP16/BF16 激活、FP32 状态及 Graph 的组件对照见 [4060 验证记录](README.md)。GSM8K 20 条仅为 smoke，不能替代大规模质量评测。

## 接受统计与本次结果的含义

| 混合路径正式请求差值 | C1 | C8 |
|---|---:|---:|
| 请求轮次选择 SA 比例 | 35.32% | 35.88% |
| SA 平均接受长度（含 bonus） | 9.67 | 9.61 |
| SA draft 接受率（不含 bonus） | 54.21% | 53.79% |
| neural 平均接受长度（含 bonus） | 5.59 | 5.58 |
| 总体平均接受长度（含 bonus） | 7.03 | 7.03 |
| compact 相对全 17 宽度的验证位置减少 | 30.44% | 30.17% |
| 整批跳过 neural tail 的比例 | 35.32% | 3.70% |

C8 的 SA 选择率、接受率没有明显塌陷，但一批里往往仍有 neural 请求：两轮合计只在 243 次 prefix 中跳过了 9 次 tail。因而 SA 行减少并不等于省掉一整次模型调用。compact 的 30% 节省以全宽 17 为参照，不能拿来直接当作对原生宽度 9 的加速比。

原生 Replay 对照说明重建状态本身也有代价；本轮混合模式 C1 的 TPOT 改善被 prefill、首 token 和调度等端到端成本抵消。C8 还承担宽度 17 的验证及 split 执行成本。上述计数能解释为什么省显存未自动转化为吞吐收益，**没有新 trace，不能据此精确分摊各内核耗时**。诊断中的 host decision wait 与前序 GPU 工作重叠，不能整段加为额外开销；接受计数也可能包含最终 EOS 截断前尾部，不能当作已交付 token 数。

## 证据与复现

启动参数见 `launch.sh`，测量流程见 `run_matrix.py` / `bench_helpers.py`，汇总见 `analyze_bench.py`，审计见 `audit_results.py`。`summary.json` 保存两轮逐项数据与 before/after 设备计数，`audit.json` 保存容量、正确性、恢复及显存观测；原始完整证据在 `a100-evidence.tar.gz`。

- 归档大小 **1,827,115 bytes**，含 122 个原始文件及成员 SHA256 清单；本地逐成员验证通过。
- SHA256：`923264dbc99e56714ac17ee8a49fd4dd21e40491c3188f5189175365c93d83df`。
- 含冻结 workload、数据来源/许可、数学输入/提示、完整测量命令、服务日志、生成结果、精度结果、源码 SHA 清单、硬件采样和测试 XML/退出码；不含模型权重。
- 远端目录：`/root/rivermind-data/validation/qwen-gdn-replayssm-compact-20261022`。
- 自有服务和采样进程均结束，39921 端口关闭，最终 GPU **0 MiB / 0%**；没有关闭租用服务器。
