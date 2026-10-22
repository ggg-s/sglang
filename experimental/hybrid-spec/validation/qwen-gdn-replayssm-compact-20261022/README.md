# ReplaySSM + GDN compact：4060 组件验证

2026-10-22：接线及组件验证完成；**完整模型正确性、整机显存和并发 1/8 性能仍未验收**。
原 A100 `fj02-ssh.gpuhome.cc:30224` 多次返回 connection refused；后续验证已获授权改用同主机的 30154 端口。
用户随后授权先在工作站 RTX 4060 验证 CUDA 用例；未替换成小模型报告性能。

## 修改

- 允许已识别 GDN 模型在 Triton decode/verify 下使用 compact / cap-accept ReplaySSM，保留树验证和其他不支持组合的拒绝检查。
- 将 compact 的变长标记传到 ReplaySSM，初始化不属于任何请求的 Graph token 补齐区域，避免后续投影读取未初始化数据。
- FP32 历史重建补读 update/key 的低位补偿。原路径虽保存了低位数据，却只读高位；新增 BF16 激活测试在 eager/Graph 下均重现约 `2.19e-4` 的状态差异。修复后保持原测试阈值通过。
- GDN/KDA 共用分配前窗口检查：长度为正的 2 次幂，且至少为最大验证宽度的两倍。SA16 的宽度为 17，测试配置使用 L=64。

主状态保持 **FP32**，不改变原有默认 TF32 点积路径。卷积及历史记录仍使用激活精度；保留卷积回滚缓存。
默认 split / gate=1 / fixed compact 不变，ReplaySSM 仍通过已有参数显式启用。

## 验证结果

工作站：RTX 4060 Laptop 8 GiB、PyTorch 2.14.1+cu130、CUDA 13.0、Triton 3.8.0。
使用独立目录 `/workspace/qwen-gdn-replayssm-compact-20261022`，未改动共享 Python 环境。

| 检查 | 结果 |
|---|---|
| GDN 专项 | **44 passed，0 skipped**：24 CPU + 20 CUDA |
| 相关 compact / tail / GDN 组件回归 | **168 passed，0 skipped，2 deselected**：97 CPU + 71 CUDA |
| 本地 CPU 组件 | 99 passed，71 CUDA cases skipped |
| 注册检查、仓库选定 Ruff 规则、diff whitespace | 通过 |

专项包含 FP16/BF16 激活、FP32 状态、eager/Graph、64 轮连续变长窗口、请求换行、不同请求槽与状态槽、部分接受、容量折叠和前缀缓存跟踪状态。
逐轮用原生递归中间快照对照有效状态与跟踪状态。状态误差判据为 `rtol=0.002, atol=0.0002`，输出为 `rtol=0.02, atol=0.003`；**不宣称逐位一致**。

工作站只有组件依赖：生产函数体和实际 CUDA 内核原样执行，服务导入及模型对象由轻量 fixture 隔离。
两项 draft-extend Graph 生命周期用例依赖完整服务 delegate，因此明确 deselect；完整注册 CPU 测试文件也仍待完整环境执行。
44 项专项是 168 项回归的子集，不能相加计算独立用例数。

## 状态池显存

按 [Qwen 官方 GPTQ 配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/blob/main/config.json) 的尺寸，执行生产 `MambaPool` 分配代码：30 层 GDN、32 value heads、16 key heads、128×128 状态、40 个常驻槽、8 个活动请求、验证宽度 17、L=64。
仅隔离分配器集成及无关的 cache-copy 预热；张量在 4060 上实际分配。

| 项目 | GiB |
|---|---:|
| 原完整中间快照：公式值，与前次 A100 8.96 GiB 日志一致 | 8.96484 |
| 实际 Replay 记录：D/K/G + D/K 低位 | **0.39757** |
| 两者差额，推算可减少的状态池空间 | **8.56728** |
| Replay 池全部张量，含常驻 FP32 状态、卷积和中间卷积 | 2.93449 |

实际检查 `intermediate_ssm is None`、`temporal.dtype == torch.float32`。
这不包含模型权重、KV、Graph 或服务临时工作区，**不能据此确认 A100 上最终 KV 容量或 8 并发已可用**。
原 8.96 GiB 快照未在 8 GiB 工作站上强行分配。

## 完整模型待办

冻结了前次相同的 50 条 SWE-rebench/OpenHands Agent 输入及内容 SHA；未重新采样。
`launch.sh`、`run_matrix.py`、`performance-plan.json` 已准备，尚未执行：

1. 原生 MTP8，保留原快照路径作为基线。
2. 原生 MTP8 + ReplaySSM，分离 Replay 本身的影响。
3. SA16/MTP8 + ReplaySSM，frequency / split / gate=1 / fixed compact。

统一 FP32 状态，max-running=8、Mamba cache=40、KV 目标=131072、mem-fraction=0.94。
先做完整模型正确性、缓存恢复与同提示 GSM8K smoke；实际容量达标后，再跑并发 1/8、各两轮。
**没有新模型性能数据，也未复用旧数据冒充此次结果。**

社区参考：[GDN ReplaySSM #28695](https://github.com/sgl-project/sglang/pull/28695)、[紧凑历史及生命周期 #35544](https://github.com/sgl-project/sglang/pull/35544)。核心已在当前分支；本次补接线和发现的历史重建问题，没有整段移植其他框架。审计记录见 `history/model-pr-history-notes.md`。

## 证据

- `cuda-summary.json`、`pool-memory.json`：结果与实际分配字节数。
- `gdn-replay.*`、`compact-regression.*`：原始日志、XML、退出码。
- `before-history-residual-*`：修复低位读取前的两项数值失败，阈值未放宽。
- `source-manifest.json`：本地与远端逐一核验的 5659 个源码文件。
- 本地 `cuda-source.tar.gz` + `final-overlay.tar.gz`：组件源文件及最终覆盖包；完整源包留在验证环境，最终覆盖包收录于证据归档。新增 GPU 测试也随代码提交。
- `cuda-evidence.tar.gz`：日志、最终覆盖包和证据清单；本地验证了压缩包及每个成员的 SHA256。
- `local-verification.json`：本地验收记录。归档 SHA256：`38b2649ccf39c3047d6f73abed83c8f69e4e60f4a574a3df1bd66ff3c1d895ba`。

4060 组件阶段的自有测试进程已结束，GPU 最终为 **0 MiB / 0%**。完整模型验证结果将另行补充。
