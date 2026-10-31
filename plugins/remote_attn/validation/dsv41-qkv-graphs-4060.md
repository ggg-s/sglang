# DSV4.1 稳定 QKV 投影入图：本地与 RTX4060 验证

日期：2026-10-31。支持 Remote batch **1/2/4**；启动方式仍为原生 SMG + P/D。
本次不增加开关，不修改缓存、scheduler overlap、optimistic prefill 或 chunk 配置。
原始结果与逐文件源码 SHA256 见 [验证数据](dsv41-qkv-graphs-4060-results.json)。

## 实现边界

SGL 只改 `python/sglang/srt/models/deepseek_v4.py`：

- `MQALayer.prepare_qkv`：提取原 `forward` 的串行/HIP/NPU/CUDA/low-ratio 多流分派。
  普通 eager 和 Local full graph 继续原样调用；不改变原 flags 的计算与选择条件。
- `project_decode_qkv`：稳定的融合/非融合 wqa、wqb、wkv 和 Q norm/RoPE 进入 graph。
  复用原 `_compute_q_a/_compute_q_b`；融合投影只算一次；跨边界仅返回普通 tensor，
  量化 wrapper 留在投影计算内部。限定非 unified、非 mega 的 V4.1 low-ratio decode。
- `update_decode_state`：图外读取本轮 batch 的 SWA、indexer、carry 信息，调用原生 KV
  norm/quantize/store 和 low-ratio sources；已有 KV helpers 仅新增可选 `projected_kv`，
  沿用 `SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE`，避免重复 wkv。

插件每层由三个 private prepare 绑定减少为一个 `prepare_qkv` 绑定；没有继续新增
private prepare hook。metadata/state/attention 都使用当前 live batch，dynamic sparse、
SWA/carry 更新、RPC 和 merge 留图外。Remote state 按主流顺序执行，Local 保留原生
多流和 full graph。原有 GPU proof、TP 退休和 overlap 恢复协议保持不变。

每 bucket 仍为 metadata + 每层 state/attention：**2L+1 breaks，通常 2L+2 segments**。
没有新增 graph break、逐 decode TP 投票或逐请求重新捕获。投影融合策略、KV 精度模式
与边界 ABI 在启动时参与已有 TP 共识，任何 rank 不匹配均在额外捕获之前拒绝。

## 本地

- 完整插件测试 **160 项通过，其中 DSV4.1 59 项**。
- 新增检查融合/非融合 × 两种 KV 精度的 Q、缓存与 carry 一致性，投影不修改动态状态、
  state 不重复 GEMM、融合切片 alias 不破坏 Q、量化 wrapper 不跨桥，以及不支持模式拒绝。
- 原生路由覆盖 HIP/NPU/CUDA/low-ratio/serial 分支；
  `python3 test/manual/remote_attn/verify_dsv41_prepare_dispatch.py` 将提取后的分派 AST
  与基线 `d3e2d0eaa6` 比较，仅允许显式参数名称绑定；通过。
- 真实双进程 Gloo TP2：同配置接受；开关、bucket、ABI、candidate 标签、投影融合或
  KV 精度不一致均由双方拒绝；Remote/Local 图判定不增加逐 decode all_reduce。
- 核心源码预检 **41 hooks / 103 core interfaces**；isort、ruff 和 diff 检查通过。

## RTX4060

用户授权的 ubuntu 工作机 RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130。
隔离目录 `/tmp/dsv41-qkv-20261031`，复用既有 venv，没有安装依赖或修改部署。

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
/workspace/remote-attn-cuda-20261028/venv/bin/python \
  test/manual/remote_attn/dsv41_segmented_graph_probe.py \
  --source-root . --output qkv-unfused-fp32.json
# 另跑 --fused-projection、--bf16-kv-source 及二者组合。
```

四种组合均通过；每组 batch 1/2/4、三个 progressive 阶段共 **18 次 Remote replay、
42 次真实 sparse RPC、8 次 Local full replay**。每组保留四个 native candidate variants
的 16 个 full keys，只捕获 3 个 Remote banks，各 8 段。捕获零 RPC，重放零 Python QKV
投影调用；不同 live cache row 的 SWA 写入和跨步 carry 与独立计算完全一致。
graph/eager 最大绝对误差 **0**。positions 使用原生 int64 类型。

共享 Qwen graph 探针及既有 DSV progressive CUDA 探针也通过；后者保留 pinned H2D、
不完整末页、取消时 GPU 读取保护及 TP 有序逐资源退休，稳定 Remote/Local 额外投票均为 0。

首轮探针的 Local full graph 曾触发 `index_copy_` 越界：测试夹具没有持有 capture-time
metadata tensor，留下 CUDA graph 的悬空地址。补齐与原生 runner 等价的 buffer 生命周期后
重跑上述矩阵通过；未因此修改生产回收逻辑。

## 验证范围

新 core QKV/state 方法从实际源码加载执行；native full/breakable backends、weak tensor
views、CUDA graph capture/replay、GPU tensor 运算、RemoteHub TCP 和 native-byte sparse
reader 实际运行。三层小模型的 linear/norm/store kernels、metadata/indexer/carry planner
为 fixture；共享 Qwen 探针的 attention/RPC 也是 toy 边界。

4060 不支持完整模型所需的 SM100 kernels；没有加载完整 checkpoint、启动 SMG/P/D
服务、验证多 GPU NCCL、整模 token/logprob 一致性或证明 TTFT/ITL/吞吐收益。
上线仍需目标模型和硬件验收。新插件必须配套本分支的 SGL 核心接口，先跑接口预检。
