# 首版验证记录

日期：2026-10-15。SGLang 基线：`a12814c1585b0c954234f776bba93dbdd04b1151`。

## 已运行

环境：Windows，Python 3.11.15，PyTorch 2.14.1+cpu，pytest 9.1.1。

| 检查 | 结果 |
| --- | --- |
| Python 核心与真实 draft_forward 函数体测试 | 57 passed |
| TP 共识与桥接：真实 Gloo 进程组 TP=1/2/4 | 3 passed，含全局 ranks [1,3] 子组 |
| CUDA 共用 C++ 状态机的宿主编译/运行 | 1 passed，Zig 0.16.0，C++17 |
| CUDA 专项测试 | 3 skipped：当前测试环境无 CUDA 版 PyTorch / nvcc |
| 双 GPU NCCL + 实际 SA + Graph replay | 1 skipped：当前环境无双卡 CUDA/NCCL |
| Ruff F401/F811/F821/F823/UP037 | 通过 |
| Ruff format / isort | 通过 |
| `git diff --check` | 通过 |
| 独立 wheel 构建 | 通过，包含 CUDA `.cu` 与共享 `.h` 源码、插件 entry point |

合计 **61 项通过，4 项 CUDA 跳过**。本地隔离虚拟环境为仓库下 `.venv-hybrid-spec311`，构建产物在 `.tmp/hybrid-spec/dist/`，均不作为源码交付。

关键行为证据：

- Python SA 与独立穷举子串搜索 oracle 比较，每次追加后检查最长重复后缀和有效续写。
- CUDA 使用的相同 C++ `Automaton` 经过二元序列长度 1–12 的完整枚举、256 组确定性长序列、512 个重复 token 与容量越界保护测试。
- 直接执行当前仓库的 `draft_forward` 方法体：K=1/2/4，插件关闭、全 neural、全 SA、混合来源，普通列表和快速矩阵路径，初始及后续 hot-token 映射；逐位置断言 verify 链等于下一次模型输入，forward 次数保持 K-1。
- 请求顺序变化、槽位不足、历史超容量、重复历史、版本间隙/冲突、重建、RID 复用和 reset。
- 固定地址 Graph buffer 的 mask 切换和 padding 清除；这是 CPU buffer 语义验证，不能替代 CUDA Graph replay。
- Gloo 多进程：K=1/3，混合/全 SA/全 neural、单 rank 全部回退、候选分歧、非零全局源 rank；同步结果经过实际 GraphBuffers staging 和逐步选择，检查 batch 缩小时 padding 清除。实际 DraftExtension.bind 路径还覆盖单 rank seq_lens 不匹配、请求重排、共识后的统计和状态释放。
- 非 greedy、grammar、beam、watermark 请求在参数校验阶段拒绝；不支持的启动配置明确失败。
- 来源归因中 correct drafts 不含 bonus，accept tokens 包含 bonus。

## 尚未验收

没有运行 Linux SGLang 服务，没有通过 nvcc 编译/执行 CUDA 扩展，也没有运行真实 CUDA Graph replay。未完成 MTP、EAGLE3、Standalone 的逐模型 target greedy 对照、真实 KV 生命周期、abort/retract/flush 服务测试或性能基准。不能据此声称任何模型已可用于生产，也不能声称有吞吐/TPOT 收益。

多 TP 已解除配置限制并接入每轮候选共识；Gloo 只验证进程间算法与 CPU 桥接，不能证明 NCCL stream 顺序、真实 TP 模型分片或多卡 Graph 的正确性。`test_tp.py` 的 NCCL 分支已准备好验证实际 GPU SA、共识、Graph replay 和 padding，需在双 GPU 环境执行。

现有 SGLang CI 测试文件 `test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py` 也已补充真实 worker 的覆盖递推测试，使用仓库 `CustomTestCase` 与原有 CI 注册。该完整依赖环境测试未在本机执行，不计入上述通过数。

部署环境必须补跑：

```bash
python -m pytest experimental/hybrid-spec/tests -q
```

确认四项 CUDA 测试（含双 GPU NCCL）实际执行而不是跳过，再按 [README.md](README.md) 在 TP=1/2/4 下分别比较 eager / Graph、K=1 / K>1、target-only / 原生 neural-spec / hybrid。`benchmarks/compare_servers.py` 会保存逐 token 一致性和端到端性能结果；真实 TPOT、阶段时间和 stream 依赖需另采 profiler。

首版范围收窄已明确记录：关闭 overlap、固定单链、无 Arctic、无动态预算、不减少 neural forward。同步历史桥接复用已提交的 CPU token，没有直接消费未提交的设备预测。
