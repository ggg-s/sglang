# Qwen3.5/3.6 MTP：GDN compact 接入与实测

日期：2026-10-21。基于 `1da8182885` 后的工作树。

实验接入已完成：GDN 组件的输出、中间状态、接受后的状态及下一轮续算与等宽
对照逐项一致；A100 上 Qwen3.5-35B-A3B 的真实服务和 CUDA Graph 可运行。
**逐 token 生成等价尚未验收，也未确认稳定性能收益。** 本次量化 MoE 对照自身
就有输出波动，不能将“服务完成请求”写成完整质量验收通过。

## 实现

- GDN 声明 ragged verify Graph 能力，沿用公共 metadata 的每请求 `qo_indptr`。
- 卷积把紧凑 token 临时放回 `[请求, 最大验证宽度]` 槽位；卷积中间状态保持原
  布局，随后只取回紧凑 token 的结果。Graph 剩余位置进入独立 sink，不写活跃状态。
- GDN 递归按真实请求长度运行；compact 自动走变长 Triton verify，避免等宽
  FlashInfer/CuTe kernel 的 reshape 假设。ReplaySSM fold 同样避开等宽 CuTe 分支。
- 接受前将 logits/hidden states 展开回原请求级索引；原生接受、GDN 状态提交及
  draft-extend 保持原接口。packed batch 更新的 Mamba tracking 计划同步回原 batch。
- 纯文本 mRoPE 直接展开紧凑位置，不再先要求总 token 数整除请求数。
- 仅对 Qwen3.5/3.6 原生 MTP、`--language-only` 放行；其余 Mamba/mRoPE 模型仍拒绝。
  `conditional_verify_graph=true` 暂不支持，使用 host token 档位 dispatch。

普通 Qwen 模板改为 fixed compact（`verify_mode=compact`、`dynamic_verify=false`），
仍为 split/gate=1。原模板的 neural 长度默认等于 SA；需要不同预算时参考新增
`configs/device-qwen35-mtp-compact-verify.json`（SA=8/neural=3/gate=1）。
target full attention 也必须支持 ragged Graph，例如本轮的 Triton；不要直接沿用
不支持该能力的 target attention 后端。所有相关 SGLang 与插件源码需一起更新并重启。

## 组件验证

新测试执行生产 GDN 的卷积与 `forward_extend` 方法和实际 Triton kernel，隔离
未使用的 serving 导入及模型/池对象；不是独立重写 GDN 算法后自测。

- 4060 专项：`test_gdn_compact.py` **21 passed**，其中 16 项 CUDA、5 项 CPU。
  snapshot / ReplaySSM fold、FP16 / BF16、eager / Graph、普通 / 重叠卷积窗口均覆盖。
  同一 Graph 改变 `[9,4,0,0] → [4,9,0,0] → [8,3,1,0]` 长度与请求槽位。
  与等宽路径比较有效输出、逐步状态、部分接受提交、单独 radix tracking 状态，
  再从该状态运行下一轮；断言均为 rtol=atol=0。padding 输入使用 NaN 以暴露污染。
- packed verify / tracking 接口专项：**6 passed**，验证 KV 布局及原 batch 接收更新后的
  tracking 计划。与上述早期 snapshot 专项一起曾得到 **19 passed**，数量有重叠。
- A100 完整依赖环境运行后端能力 CI 测试：**2 passed，4 subtests passed**。
- 本地包级回归：**368 passed / 714 skipped**；该次早于新增 8 项 fold CUDA 参数，
  后续本地专项为 5 passed / 16 skipped，配置/模型检查复验为 114 passed。
- 格式及 diff whitespace 检查通过；改动文件的 E9/F 检查通过，但
  `forward_batch_info.py` 排除该文件原有的 F811 重复 `is_dllm_extend` 定义，未修改旧问题。

最终完整 4060 包级回归：**846 passed / 254 skipped / 0 failed**，没有 deselect。
结果另见 `regression-summary.json` 及 `cuda-evidence.tar.gz`。首次扩大回归的
843 passed / 254 skipped / 3 failed 使用了旧生产依赖和错误的本机 CUDA toolkit；
保留失败日志，不计为通过。最终回归统一完整源码快照、CUDA 13 及 GCC 12。

## A100 真实模型

用户提供的 GPUHome 服务器，A100 PCIe 40GB、TP=1。模型为已存在的
`Qwen3.5-35B-A3B-GPTQ-Int4`，target FP16/gptq_marlin，MTP unquantized；
PyTorch 2.14.1+cu130、Transformers 5.19.0。实际运行库和源码摘要见 `manifest.json`。
独立目录为 `/root/rivermind-data/validation/qwen-gdn-compact-20261021`，
通过 PYTHONPATH 使用源码快照，未覆盖原服务器安装。

共同设置：SA=8/neural=3/gate=1、frequency、split、greedy、最长输出192 token、
target/draft attention 与 GDN 均为 Triton，overlap 与 chunked prefill 开启，
并发上限4，Mamba cache20。compact 与 padded 只改变验证布局。
compact 成功捕获14个 token 档位；服务日志确认 decode 使用 CUDA Graph。

12条固定合成提示，覆盖代码、中英文问答、重复序列与4827-token长输入；
每组先 flush cache，分别并发1/4。按 compact → padded → padded → compact 执行，
每种模式共48个请求，全部完成；长输入实际经过4096-token分块。

### 实际计算位置与耗时

`target_token_ct` 是实际 target 图/forward 的 token 行数，不是所有算子的 FLOPs；
GDN 卷积仍有临时等宽计算和打包成本。接受长度含 bonus，按请求 verify 次数加权。

| 模式/轮次 | 并发 | 输出token | target位置 / 未压缩位置 | 位置减少 | 接受长度 | 整组耗时s |
|---|---:|---:|---:|---:|---:|---:|
| compact 1 | 1 | 1740 | 2209 / 4509 | 51.0% | 3.540 | 9.570 |
| padded 1 | 1 | 1733 | 4482 / 4482 | 0% | 3.543 | 8.296 |
| padded 2 | 1 | 1748 | 4536 / 4536 | 0% | 3.540 | 8.337 |
| compact 2 | 1 | 1748 | 2226 / 4536 | 50.9% | 3.533 | 8.324 |
| compact 1 | 4 | 1738 | 2224 / 4509 | 50.7% | 3.521 | 4.880 |
| padded 1 | 4 | 1748 | 4617 / 4617 | 0% | 3.471 | 4.170 |
| padded 2 | 4 | 1734 | 4482 / 4482 | 0% | 3.538 | 4.073 |
| compact 2 | 4 | 1730 | 2165 / 4410 | 50.9% | 3.584 | 3.864 |

两轮一慢一快、输出也不完全相同，且首轮可能包含额外预热，不能据此声称稳定加速。
这里是功能对照中的粗粒度墙钟记录，不是标准性能验收。

### 输出稳定性与状态恢复

| 对照 | 并发1完全相同请求 | 并发4完全相同请求 |
|---|---:|---:|
| compact 1 vs padded 1 | 9/12 | 6/12 |
| compact 2 vs padded 2 | 11/12 | 8/12 |
| padded 1 vs padded 2 | 7/12 | 8/12 |
| compact 1 vs compact 2 | 8/12 | 6/12 |

为隔离投机验证，padded 下对同一提示 flush 后仅生成首 token，重复10次；
6次输出 `red`，4次输出 `Here`，两者 logprob 常相等或仅差0.015625。
该探针不进入投机 verify，响应也没有 spec verify 统计。因此至少存在独立于
compact 的既有 prefill 数值不稳定，不能将所有差异直接归为新增状态逻辑。
它**不能证明 compact 所有差异均无害**；未完成算子级定位、batch invariance 或
正式 KL/任务质量验收，不能宣称逐 token 等价。原始 top-5 logprob 已保留。

另对4827-token相同长输入连续请求两次：第二次命中4800个 device cache token，
两次输出的160个 token **完全一致**，接受长度均为8.421。
这单独覆盖 Mamba 前缀恢复并继续生成，不能外推为全部提示的质量保证。

Qwen3.6 走同一 Qwen3.5 架构及原生 MTP 适配，**本轮没有3.6 checkpoint实测**。
多 TP、其他量化/attention/GDN 后端与多模态均不在本次真实模型验收范围内。

## 证据

- `summary.json`：逐组计数、耗时、逐 token 比较和探针摘要。
- `manifest.json`：A100 环境及实际源码摘要。
- `a100-evidence.tar.gz`：完整服务日志、96个主对照请求、10次prefill探针、
  2次缓存复用请求、启动脚本、配置、分析脚本、源码覆盖包及后端能力检查日志。
- 4060 回归摘要及原始日志/XML 单独归档。服务与测试进程在完成后释放 GPU；
  不终止用户提供的服务器实例。
