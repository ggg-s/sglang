# DSPARK gate=0 按请求跳过 proposal：短回归

日期：2026-10-20。实现位于当前工作区，默认 full 策略不变。本轮没有下载权重、启动模型服务或执行吞吐实验。

## 实现

新增 opt-in 组合：`gpu_sa + history_mode=device + draft_graph_mode=split + num_neural_gate_tokens=0 + verify_mode=static`。

1. 沿用现有 GPU SA 查询，在调用 proposer 前完成 TP 候选/资格共识。
2. SA 全命中时完全跳过 neural proposal；混合批只为 neural 请求执行原生 `DraftBlockProposer.propose/_run_forward`，随后按原始行号合并 SA/neural 草稿。
3. 请求子集通过浅拷贝建立，不修改原始 scheduler batch、请求列表或 target sampling/penalizer 状态。原生 CPU 长度和 overlap 的 `nxt_kv_lens_cpu` 同步选择。
4. `KVLocPlan.select_rows` 选择已分配/翻译的写地址，保留所有已缓存的 FULL/SWA id，不重复分配或翻译。子计划清空旧读表，由原生 reader 根据所选行重建；完整 target 计划保持不变。
5. 有 SA 行的批次使用独立合并缓冲，不消费 folded accept/commit 的结果，走原生非 folded 验收和 hidden/KV 提交。draft 子批仍可使用原生 CUDA Graph。全 neural 时保留原生 folded 输出别名。已捕获 target 图内的 epilogue 可能仍执行，不能把该策略描述为取消了全部 folded 算子。
6. static target 验证、接受路径和 hidden/KV 提交继续覆盖完整 batch；原接受前缀和 bonus 按原行顺序回写 GPU SA。替换候选不使用 neural confidence。

`hybrid_spec_dspark_draft` 提供累计计数：`proposal_ct`、`request_ct`、`neural_request_ct`、`skip_request_ct`、`all_sa_ct`、`mixed_ct`。这些是逻辑请求行数，不包括原生 Graph padding，不代表实际 FLOPs 或吞吐提升。

当前使用 GPU nonzero 和行号回读选择 CPU 元数据，存在同步及动态分配。没有扩展到条件图、gate≥1 split、split_no_skip、compact/padded verify 或动态预算；这些配置在启动时继续明确拒绝。

## 启用

在已经可运行的原生 DSPARK 命令上使用新模板：

```bash
export SGLANG_PLUGINS=hybrid_spec
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/device-dspark-sa-split.json"
export SGLANG_RAGGED_VERIFY_MODE=static
# 模板为 gamma=5：原生命令保留
# --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
```

其他 checkpoint 需同步修改 gamma、`num_drafts` 和验证窗口；服务最大并发不得超过模板 `max_requests=8`。原 full 模板继续可用。

## 已完成短回归

| 检查 | 结果 | 覆盖 |
|---|---|---|
| 本地 DSPARK 专项 | **19 passed / 7 CUDA skipped** | 默认 full/gate、配置保护、原生 proposal/decode 方法体、static verify/commit、生命周期 |
| 本地相关 shared 配置与诊断 | **30 passed / 79 deselected** | 配置错误、默认 LoRA、split 约束、诊断 |
| 远端 DSPARK 专项 | **26 passed，84.00 秒** | 包含上述 CPU 检查及 **7 项真实 CUDA 用例**；含首次 SA JIT |
| 远端原生 KVLocPlan 单测 | **14 passed，11.07 秒** | 原有虚拟/物理/SWA 写地址、读表和 Graph buffer 合约回归 |
| Ruff 仓库检查项、格式、diff whitespace | **通过** | 无未定义名称/未使用导入/格式错误 |

远端：`hz01-ssh.gpuhome.cc:30120`，RTX4090 48GB，PyTorch **2.14.1+cu130**，CUDA **13.0**，driver **580.119.02**。独立目录 `/root/rivermind-data/validation/dspark-g0-split-20261020/source`，复用之前已上传的冻结源码，再覆盖本次文件。验证代码 SHA256 与本地逐文件一致。完成后 GPU 占用 **0 MiB**。

新 proposal 测试执行生产的 `DraftBlockProposer.propose/_run_forward` 和 `KVLocPlan`；模型/Markov kernel/ForwardBatch 构造为确定性轻量边界。B=4、gamma=3，覆盖两种 `sample_from_anchor` 布局和 eager/folded 输出；同一 proposer 连续运行全 SA、全 neural、交错混合、单 neural、TP 否决后的不同单行，核对真正执行的请求行、物理 KV 写入位置、原始 batch 不变和候选合并顺序。folded CUDA 分支实际捕获/回放轻量模型边界，并检查缓冲区写入与生命周期。4 项 CUDA 参数用例覆盖上述组合。

另外 3 项 CUDA 用例执行真实 GPU SA 构建、查询及接受反馈（长度 1/2/4），确认只追加接受前缀与 bonus，不写入拒绝尾部。原生 decode 接线检查覆盖 full、混合 split、全 SA split，确认 target static 验证和 hidden/KV 提交没有因跳过 proposal 被跳过。

首次远端尝试在读取配置示例 fixture 时失败，尚未运行 CUDA；补齐遗漏的既有 full JSON 后，完整专项全部通过。未修改运行代码来绕过失败。

复现：

```bash
PYTHONPATH=python:experimental/hybrid-spec/src python -m pytest \
  experimental/hybrid-spec/tests/test_dspark.py -q
PYTHONPATH=python python -m pytest \
  test/registered/unit/mem_cache/test_kv_loc_plan.py -q
```

## 边界与证据

没有执行真实 DSPARK 权重服务、完整 DSV4 模型/attention/压缩 KV 数值链或 TP≥2 NCCL；TP 否决测试使用可控共识边界，不是实机多卡证明。轻量 CUDA Graph 与生产方法体检查不能替代完整模型验收。本轮只证明上述组件合约，不给出接受长度、TPOT 或吞吐收益结论。

`evidence.tar.gz` 包含初次 fixture 失败日志、最终 DSPARK 通过日志、原生 KV 单测日志及 CUDA 环境/代码哈希 manifest。SHA256：`57e08daf59be2edf8ca2c7b8b774f70897e544e0d8bdf5a00d8efa0f8bcc531e`。
