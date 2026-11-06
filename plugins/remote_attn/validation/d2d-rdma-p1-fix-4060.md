# D2D RDMA 三项 P1 修复验证

日期：2026-11-06。分支：`codex/remote-attn-pre-layerwise`，基线
`5884814306731be4485f3fa7ef3666e8fbe1979e`。本次修复尚未提交。

**结果：CPU 与 RTX 4060 环境验证通过；真实双机 GPUDirect RDMA 尚未验收。**
本报告覆盖修复后的最终源码，替代旧报告对 transport 身份与完成证明的结论。

## 修复与回归

| 问题 | 修复 | 关键证据 |
| --- | --- | --- |
| 请求协议被当成实际 transport | 初始化后读取自身 classic P2P metadata；只有已验证的单一 TCP/RDMA 才参与协商 | 原生 Mooncake 在 MC_FORCE_TCP=1/0/空串/true、无 HCA 自动发现场景均正确记录 TCP，不开放 GPU；未知格式、TENT/TEV1/SHM 按存在性拒绝 |
| TCP fallback 被自动发现为 RDMA | 共享初始化锁内临时强制 TCP，核实 metadata 后恢复环境；HCA filter 选择也受锁保护 | CPU 有 HCA 分支契约与并发初始化测试；4060 实际 MC_FORCE_HCA=1 仍得到 TCP，环境恢复，已有 engine 保留 |
| 同步 API 内部重试隐藏迟到 DMA | Qwen 和 DSV bulk 统一单次 async submit + 同 ticket status；不调用 native sync convenience API | CPU 拒绝隐藏重试和无效 ticket；真实 CUDA staging 的迟到写发生在取消/恢复被拒绝后，源、目标 MR/backing 仍保留，未完成 entry 不 Local |

握手升级为 **d2d-mooncake-v2**，拒绝 v1；部署必须两端一起升级。
Qwen 的 HANDOFF_FAST_ASYNC=0 和 1 均执行同一单次提交契约。
保留原生先预留再冻结、layer group × token range、逐 entry Local 与 retirement 语义。
失败、超时、断线和取消均不构成 DMA drain 证明，不触发 TCP 重传或 staging 复用。

## 最终验证结果

| 套件 | 数量 | 结果 |
| --- | ---: | --- |
| 既有 CPU/socket/原生预留/TP2 Gloo/P4 契约 | 361 | PASS |
| 协议、metadata、回退、初始化并发与单次提交契约 | 26 | PASS |
| 4060 原生 Mooncake 协议/回退/单次 TCP 提交 | 7 | PASS |
| 真实 CUDA staging，fake NIC/visibility 故障边界 | 11 | PASS |
| Qwen 真实 Mooncake TCP、group/range、取消、状态丢失 | 8 | PASS |
| DSV4/DSV4.1 真实 Mooncake TCP，逐 entry 渐进 Local | 2 | PASS |

合计 **387 项 CPU 测试、28 项 4060 环境用例（其中 21 项 CUDA 用例）**。
配置的 Ruff 静态检查、格式检查和 git diff --check 通过。

4060 上保留真实 CUDA tensor、stream/event、gather/copy/scatter、控制 socket、
entry retirement 与注册所有权实现。fake NIC 用独立 CUDA stream 模拟外部写，
可以在返回失败、执行取消/恢复请求之后才写入原 staging。测试检查整个期间
注册和 backing 未释放，未重试、未复用，未发布错误 Local，原生请求 KV 未回收。
模拟 sync API 另设“失败旧尝试仍挂起、内部重试成功”的实现，修复路径不得调用它。

**这不是 RDMA HCA 完成队列、GPUDirect 可见性或跨机吞吐验证。**
4060 的 MC_FORCE_HCA 用例只验证强制 TCP 的优先级，不能代替有 HCA 双机验收。

## 复现与源码一致性

使用现有 CUDA13 venv：`/workspace/remote-attn-cuda-20261028/venv/bin/python`，
CUDA_HOME=`/workspace/conditional-graph-20261019/cuda13`。
最终隔离快照：`/tmp/d2d-rdma-fix-4060`，日志位于其 `final-results`。

```bash
export PYTHONPATH=python:plugins/remote_attn:test/manual/remote_attn
python test/registered/unit/plugins/test_remote_attn.py
python test/registered/unit/plugins/test_d2d_mooncake_backend.py
python test/manual/remote_attn/mooncake_contract_probe.py --source-root "$PWD" --output /tmp/native-contract.json
python test/manual/remote_attn/rdma_staging_cuda_probe.py --source-root "$PWD" --output /tmp/gpu-staging.json
python test/manual/remote_attn/range_layer_cuda_probe.py --output /tmp/qwen-tcp.json
HANDOFF_D2D_GROUP_SIZE=3 HANDOFF_D2D_TOKEN_RANGE=17 python test/manual/remote_attn/group_layer_cuda_probe.py --source-root "$PWD" --suite dsv-local --output /tmp/dsv-local.json
```

本地 CPU 使用隔离 Torch 和最小 SGLang 顶层导入边界；TP2 worker 同样从源码
加载轻量核心契约，未替换分配、预留、传输完成或回收实现。
[JSON 报告](d2d-rdma-p1-fix-4060.json) 包含测试结果与 102 个测试源码文件的 SHA-256；
已核对验证机文件与本地最终工作树完全一致。
