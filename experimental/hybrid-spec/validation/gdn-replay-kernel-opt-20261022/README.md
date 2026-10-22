# GDN ReplaySSM：A100 kernel profile 与优化

2026-10-22，在用户指定的 A100-PCIE-40GB 上完成定位、修改、数值回归和完整模型复测。**FP32 主状态、D/K 高低位补偿、L64 环形历史及最大 SA16 预算均保留。**

同一批 50 条 SWE-rebench/OpenHands Agent 前缀、相同容量 8 / Graph batch 8，新旧 Replay 各测 C1/C8 两轮，共 400 个正式请求。优化后 C1 吞吐 **+6.70%**、C8 **+3.45%**；启动可用显存进一步增加 **0.907 GiB**。

## 热点在哪里

先在容量 1 / Graph batch 1 下，分别启动快照、旧 Replay，并验证相同 GSM8K smoke 和长前缀缓存恢复。使用同一真实 Agent 长前缀，单独采 prefill；decode 强制续写 1024 token，在至少生成 192 token 后启动 profiler，采 11 步（含 guard）。以下截取首个 target verify GPU 开始到最后一个开始之间的 **10 个完整周期**。

| Kernel 表：平均每周期 GPU kernel 累计时间 | 快照 | 旧 Replay | 优化 Replay 的观测值 |
|---|---:|---:|---:|
| GDN：`fused_sigmoid_gating_delta_rule_update_kernel` / `gdn_replayssm_spec_circular_kernel` | 1.157 ms | **7.246 ms** | **2.143 ms** |
| 注意力相关 `_fwd_kernel` | 20.715 ms | 20.681 ms | 20.796 ms |

快照和旧 Replay 的这两个窗口均执行了 11 张 neural tail Graph，主要算子构成接近；GDN 时间差约 6.09 ms，与周期时间 41.82→47.71 ms 的增加规模接近。热点明确落在 Replay 的 circular verify，而不是 SA 查询或 compact 拷贝。

**优化后那段强制长续写的 SA/neural 比例发生变化，仅执行了 2 张 tail Graph。** 所以 7.246→2.143 ms（约 -70%）是该诊断窗口中的观测，不能据此直接分摊端到端加速，也不能把两个窗口当作严格同形状的配对。固定输入的算子测试及新旧服务的自然 EOS 测量见下文。GPU 累计 kernel 时间、GPU interval union 和周期 wall time分别统计，不能直接相加混用。

| Overlap 机会表 | 证据 | 判断 |
|---|---|---|
| 本次单 trace 未确认新的可隐藏工作 | Graph 回放缺少完整 Python launch-site 映射；CPU prefix wait 的无 GPU 活动部分约 0.2 ms，远小于整个 wait | 不将整段 CPU wait 算为额外耗时；本次收益来自 kernel 配置和 tile 上界 |

| Fuse 机会表 | 当前源代码与工具结果 | 本次处理 |
|---|---|---|
| 无达到工具置信门槛的新融合模式 | Replay kernel 已融合 gate、q/k normalization、历史投影、窗口求解及写环；catalog 未命中新模式 | 保留现有算术融合，调整启动配置和无效计算范围 |

这是单 trace 的保守 overlap 分析，不是 mapping/formal 双 trace 的依赖证明。完整三表输出见 `snapshot-triage.md`、`replay-triage.md`、`optimized-triage.md`，原始路径及 SHA 见 `trace-provenance.json`。

## 为什么这个 kernel 慢、修改了什么

编译资源探针在正式计时结束后独立运行，报告的是 Triton `n_regs` / `n_spills` 字段：

| 相同模型维度，单请求 | warp / stage | BS | 每线程寄存器 | Triton n_spills | shared memory |
|---|---:|---:|---:|---:|---:|
| 旧版，实际 9 或 17 token | 1 / 2 | 32 | 168 | **1356** | 56 KiB |
| 优化，实际 9 token | 4 / 1 | 16 | 255 | **12** | 24 KiB |
| 优化，实际 17 token | 4 / 1 | 32 | 255 | **178** | 32 KiB |

旧配置让一个 warp 承担很大的 tile，产生大量 spill：临时数据放不进寄存器，需要访问 local memory。四个 warp 分摊 tile 后，spill 显著减少。这里不是“每线程寄存器数下降”；实际寄存器数上升、落到 local memory 的部分减少。

另一个问题是，C1 的 compact Graph 只有 9 个总 token 时，旧代码仍按最大草稿 17 向上取整为 32 行。总 token 容量天然也是每个请求窗口的上界，现据此使用 16 行 tile。这个上界来自 Graph 捕获时已知的张量形状，不读取 GPU cursor，也不改变全局 17-token early-flush margin。

生产改动仅在 `gdn_replayssm_spec_decode.py` 的 host launch wrapper：

- 按 packed 总容量收紧 tile 上界。
- 对实测 SM80、FP16 激活、FP32 checkpoint、H16/HV32、K=V=128、L64、最大窗口 9/17、batch≤8 的 compact-history 路径，默认 4 warp / 1 stage；其他形状保留原默认值，显式传入的设置继续生效。
- 不修改 JIT kernel 的数学表达式、TF32 点积设定、低位补偿、状态池布局或 dtype。硬件能力在调用时查询，没有新增可变全局配置。

## 固定输入微基准

Qwen3.5-35B 的 H16/HV32/K128/V128、FP32 checkpoint、FP16 高低位历史、L64，覆盖 B1/B8、实际窗口 9/17、历史 0/16/30，CUDA Graph 计时。216 个配置均通过输出及补偿后 D 记录对照；输出阈值 `rtol=0.02, atol=0.003`，D 记录 `rtol=0.002, atol=0.0002`。

历史 16 的代表值：

| 同输入单层 kernel | 旧配置 μs | 调整后 μs |
|---|---:|---:|
| B1，实际 9 token；收紧 tile + 调整 warp/stage | 156.50 | **20.22** |
| B1，实际 17 token；调整 warp/stage | 165.59 | **58.24** |
| B8，实际 17 token；调整 warp/stage | 803.52 | **255.56** |

单层重复输入会受益于缓存，不代表整机吞吐。B8 全 neural 的总容量仍大于 17，生产路径不能仅凭总容量推出每行都≤9；因此没有把 sweep 中强制 bound=9 的 B8 数字当作实际集成收益。全部配置和同上界/窄上界分项见 `tile-sweep.json`、`tile-summary.json`。

另补采严格同输入的算子 profile：新旧版本共享 q/k/v 和只读 FP32 checkpoint，历史记录以相同内容分别复制，B1 / 历史16 / 最大窗口17，每图30次调用、采10次回放，共300次 kernel。输出及补偿后的 D 记录对照通过，checkpoint 未被 verify 修改。

| 同输入算子 profile，平均单次 kernel | 旧版 | 优化 |
|---|---:|---:|
| 实际 9 token | 168.16 μs | **20.31 μs** |
| 实际 17 token | 161.88 μs | **58.17 μs** |

这组结果避免了完整模型强制续写中 SA/neural 比例变化的影响；仍属于缓存较热的单算子测量。脚本及原始 trace 见 `capture_controlled.py`、`controlled/`，统计见 `controlled-summary.json`。

## 完整模型性能

使用 Qwen3.5-35B-A3B-GPTQ-Int4 target + unquant FP16 MTP、TP1、Triton attention/GDN，SSM 始终 FP32。所有 server 参数逐字段核对一致：capacity=8、Graph max batch=8、Mamba cache=40、实际 KV=131072、context=16384、mem-fraction=0.94、L64、SA16/neural8、frequency / split / gate=1 / fixed compact。

复用相同 50 条真实 Agent 前缀，7587–11996 input token，完整工具定义与历史结果；未重新采样。输入 SHA256：`14fa95c7d417ae66a0e59dfeb7e9aeea4013334ba474d2a19ea4109f69f004c5`。数据来源与许可在证据中的 `workload-provenance.json`。

ABBA 四次独立启动：旧 R1 → 优化 R1 → 优化 R2 → 旧 R2。每次先相同 smoke、缓存恢复和预热；R1 C1→C8，R2 C8→C1，每项 50 请求。官方 serving client，temperature=0、自然 EOS、最多 2048 token，正式计时 warmup=0。下表是两轮均值，收益为同轮配对百分比的平均。

| 容量 8 / Graph 8 | 旧 Replay | 优化 Replay | 配对收益 |
|---|---:|---:|---:|
| C1 输出 tok/s | 75.002 | **80.028** | **+6.70%** |
| C1 TPOT ms | 5.815 | **5.114** | **降低 12.06%** |
| C8 输出 tok/s | 121.209 | **125.393** | **+3.45%** |
| C8 TPOT ms | 39.139 | **37.173** | **降低 4.98%** |

| 分轮输出 tok/s | R1 | R2 |
|---|---:|---:|
| 旧 C1 | 74.918 | 75.086 |
| 优化 C1 | 80.311 | 79.746 |
| 旧 C8 | 121.906 | 120.513 |
| 优化 C8 | 125.959 | 124.827 |

SA 预算、来源策略未变。总体接受长度（含 bonus）C1 为旧 7.028 / 优化 6.960，C8 为旧 7.178 / 优化 7.130；优化组略低仍有吞吐收益。所有 400 条请求成功、输出非空、无 2048-token 截断、均有闭合 tool_call 标记；没有执行工具或计算 SWE 修复成功率。

另有容量 1 / Graph 1 的优化配置两轮检查，共 100 个请求：80.800 / 79.872 tok/s。它与容量 8 测量分开，不混入上表。本次共 500 个正式请求；诊断强制续写请求不参与吞吐统计。没有重测原生 MTP8，不给出新的严格配对“对原生收益”。

## FP32、显存与正确性

| 容量 8 的实际分配与采样 | 旧 Replay | 优化 Replay |
|---|---:|---:|
| 常驻 SSM，FP32（日志） | 2.40 GiB | **2.40 GiB** |
| 完整中间 SSM 快照 | 0 | **0** |
| 高低位 Replay 环形记录 | 约 0.398 GiB | **约 0.398 GiB** |
| target verify Graph（服务上报） | 1.779 GiB | **0.873 GiB** |
| draft split Graph（服务上报） | 2.207 GiB | 2.207 GiB |
| 启动后可用显存 | 6.429 GiB | **7.336 GiB** |
| C8 正式请求观测最高 GPU MiB | 34898 | **34032** |
| 实际 KV token / 请求槽 | 131072 / 8 | **131072 / 8** |

省显存的状态池布局保持不变，Graph 部分占用进一步下降。Graph 值是服务阶段上报的变化，不能相加当作整机总占用；GPU 每秒采样只能给出观测峰值。

最终源码验证：注册数值测试 **3 passed + 6 subtests passed**，相关组件 **44 passed**，0 failure/error/skip。新增真实模型维度及窄 Graph 用例，覆盖 FP16/BF16 激活、FP32 状态、64 轮连续部分接受、请求换行、状态槽与请求槽不同、历史容量折叠和 prefix-cache 跟踪状态。逐轮对照原生递归快照，状态阈值仍为 `rtol=0.002, atol=0.0002`，没有放宽。

完整模型四次容量 8 启动的相同 GSM8K 20 条 smoke：旧两次均 19/20，优化为 20/20、19/20，均无请求错误或截断；这不证明质量提升。四次长前缀恢复均命中 11648 cached token，恢复前后文本相同。量化 MoE 下温度 0 仍有文本差异：同组两轮逐字一致的请求数，旧 C1/C8 为 46/44，优化为 47/42（各 50），不宣称逐位等价。

仓库选定 Ruff 规则、format、isort、注册校验及 diff whitespace 检查通过。没有修改默认 split / gate=1 / fixed compact 选择。

## 源码与证据

基线生产代码为 `50084814319d2e61eee6499500e9fe1904a2bc8b`。最终优化文件哈希见 `final-production-sha256.json`，容量 8 优化组每次启动均核对源码清单。早期 C1 profile 的硬件探测缓存后来改为调用时读取；全部 JIT 函数 AST 不变，证明见 `final-source-ok.json`；最终回归执行最终硬件选择逻辑。

`matrix8/summary.json`、`audit.json` 保存每项数据、设备差值、参数一致性和显存观测；`resources-summary.json` 保存编译资源；`profile-summary.json` 及三份 triage 保存模型 trace 分析。脚本、输入、所有原始命令、预测、日志、XML、源码 SHA 清单、源覆盖包和 trace 均留在原始归档。

原始 trace 和较大的归档保存在本地与远端验证目录，未加入 Git；归档清单及 SHA 随报告提交。远端目录：`/root/rivermind-data/validation/gdn-replay-kernel-opt-20261022`。自有服务和采样进程均退出，39923/39924 端口关闭，GPU 最终 0 MiB / 0%；服务器保持运行。
