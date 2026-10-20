# 条件图、DSV4 C128 与 DSPARK SA 短回归（2026-10-20）

本记录对应 `codex/hybrid-suffix-speculation` 在基线 `0993919fc8` 上的本轮实现。
没有下载权重、启动模型服务或执行吞吐大实验。以下分别标注组件验证和未验证范围，
不能据此推断端到端性能收益。

## 实现

- 条件图使用 group 所有的专用 PyNccl communicator，设置 `graphUsageMode=1`；
  普通 communicator 的默认初始化方式不变。
- opt-in `conditional_verify_graph` 让 GPU 选择 compact target 档位，分支执行
  pack、KV 地址规划、attention 元数据、模型 forward 和 scatter；计数、价格缓存
  与异步计时标签避免新增热路径标量读回。
- DSV4 online C128 使用 GPU 实际长度生成原生 CompressPlan，支持逐前缀状态和
  接受后提交，包括 9-token 窗口、128-token 边界与连续 replay。
- DSPARK SA 使用等长 greedy block，query 后在 native proposal 上执行 gate 和
  一次 TP 共识；原地替换 token 保留 folded sampler/verify 缓冲区别名。
  native target verify、KV 注入与提交保持原流程，接受前缀和 bonus 回写 GPU SA。
  被替换候选不沿用 neural confidence，FutureMap 对应 confidence 携带关闭。

DSPARK 当前要求 `gpu_sa + device history + full draft + static verify`，
不与 confidence 裁剪、compact、split 或动态预算混用，仍执行完整 neural block。
示例 K=7；V4 官方 Flash/Pro DSpark checkpoint 配置 K=5，对齐时使用 verify 窗口 6。

## 已完成验证

远端为单卡 RTX 4090 48GB，PyTorch 2.12.1+cu130，CUDA toolkit 13.0。
使用隔离源码目录和已有 PyTorch/Triton 环境，不安装完整 serving 环境。

| 范围 | 结果 | 说明 |
|---|---|---|
| 条件 communicator CPU 合约 | 12 passed | 生命周期、作用域恢复与配置约束；CPU 不代替 NCCL |
| 本地 shared SA / conditional / compact / tail / SWITCH | 188 passed / 63 CUDA skipped | 条件 target 实现阶段的集合结果 |
| 远端 `test_conditional_verify.py` | 9 passed，21.11 秒 | 首次编译在内；真实 SWITCH、原生 pack/scatter 与确定性模型边界 |
| 远端原生 compact prepare、ragged Triton attention、已有 SWITCH | 13 passed，7.48 秒 | SWITCH 排除 joined；attention 对独立参考 |
| 远端 DSV4 ragged C128 | 5 passed / 2 dtype subtests passed，3.53 秒 | GPU planner/controller、FP32/BF16、跨 128 边界、padding、9-token bank、连续捕图 |
| DSV4 接入后重跑条件 target | 9 passed，5.26 秒 | 原条件图组件保持通过 |
| 本地 DSPARK 专项 | 13 passed / 3 CUDA skipped | gate、回退、TP veto、共享缓冲区、原生 decode 接线、生命周期和配置保护 |
| 本地 shared SA 核心 | 105 passed | `test_core.py` |
| 远端 DSPARK 专项 | 16 passed，77.56 秒 | 含首次 GPU SA JIT；三个 CUDA 用例覆盖接受 1/2/4 token，不把拒绝尾部写入 SA |
| 最终 DSPARK 架构/confidence 保护 | 远端 1 passed，1.74 秒 | 包括缺省无扩展行为及 DSV4 DSpark draft 架构 |

另已完成 Python 静态错误/未使用导入检查与 `git diff --check`。
DSPARK 的 CPU 接线测试执行生产 `_forward_decode` 方法体，model、accept 和 KV kernel
边界使用确定性替身；GPU 用例执行实际 GPU SA 构建、查询和反馈内核。
DSV4 隔离 harness 替代包初始化、测试框架和未使用的 legacy TVM/JIT 导入，
实际 planner、Triton 内核、controller 与提取的 metadata 方法执行生产代码。

## 未验证范围

- 单卡机器没有执行 TP=2 CUDA/NCCL 回归；用户提供的内网 communicator A/B
  属于原始问题证据，不能当作本轮专用 communicator 实现的多卡验收。
- 没有执行完整 DSV4 模型、原生 compressor 数值链或 DSPARK 真模型服务。
- 没有新增 SA 接受率、TPOT 或吞吐结论；DSPARK 接入不等于已经获得加速。

代码入口与复现测试命令见 [条件图说明](../../CONDITIONAL_GRAPH.md) 和
[DSPARK 接入说明](../../README.md#dspark--gpu-sa实验接入)。
