# Q/O/LSE Mooncake 统一数据面：4060 验证

日期：2026-11-06，分支 `codex/remote-attn-pre-layerwise`，基线
`5884814306731be4485f3fa7ef3666e8fbe1979e`。修改尚未提交。

**405 项 CPU 测试与 35 项 RTX 4060 环境用例全部通过。真实双机 RDMA 未验收。**
本报告记录 Q/O/LSE 完成统一后的验证结果，补齐此前 D2D bulk 修复未覆盖的数据面。

## 覆盖范围

- Qwen3-8B、DSV4、DSV4.1 共用已验证 Mooncake engine、能力交集、TCP fallback 与单次提交契约。
- `te_negotiate` 固定原生 PD identity、backend 和 endpoint；`te_open` 发布注册区。拒绝旧的未协商连接、错误身份及协商后 endpoint 替换。
- GPU RDMA 注册 GPU staging；TCP 注册 pinned Host staging，独立 CUDA copy stream D2H/H2D，逐 region payload+footer 到达证明。
- Q、O/LSE 均经单次 async submit＋同 batch status，不进入 native sync 隐藏重试。
- 注册/注销不确定、Q/O 迟到写、取消、状态异常、回复丢失和 producer event 失败均保留相应 MR/backing/copy_owner；失败请求不复用，源 reader lease 不提前释放。
- 原生预留后冻结、D2D group×token range、逐 entry Local、RequestIO/retirement 和 P4 恢复语义由既有回归继续覆盖。

新 wire `remote-mooncake-v1`，能力契约 `d2d-mooncake-v2`，两端需要一起升级。
配置详见 [Remote Attention Mooncake 数据面](../REMOTE-MOONCAKE-ZH.md)。

## 测试结果

| 套件 | 数量 | 结果 |
| --- | ---: | --- |
| 既有 CPU/socket/原生预留/TP2/P4 契约 | 361 | PASS |
| engine 身份、metadata、TCP 初始化与单次提交 | 26 | PASS |
| 新 Q/O/LSE 协商、取消、异常、迟到写与持久所有权 | 18 | PASS |
| 三模型 Q/O：真实 TCP、混合能力 TCP 回退、GPU staging 与故障 | 21 | PASS |
| DSV Q/O→Local，原生布局及 C4/C128 | 3 | PASS |
| GPU reader 取消、早退拒绝、缓存 pin、安全复用 | 1 | PASS |
| DSV4/4.1 D2D 逐 entry 渐进 Local | 2 | PASS |
| Qwen D2D layer group×token range、取消和状态丢失 | 8 | PASS |

新 Q/O GPU 套件每模型 7 项：真实 Mooncake TCP、混合能力后的真实 TCP、fake NIC GPU
正常路径、Q 迟到写、O/LSE 迟到写、写入后状态异常、O/LSE 挂起时取消。
Qwen 使用实际 FlashInfer 计算，DSV 使用原生 byte layout 与 sparse kernel；均是小型
pool/scheduler fixture，不加载完整模型权重或运行生产 SMG 服务。

fake NIC 使用独立 CUDA stream 的真实 DeviceToDevice memcpy。失败返回后仍可在取消
发生后执行原地址的迟到写，测试确认 backing/MR 仍在、未复用且相关请求/源 KV 的安全
回收证明未被伪造。真实 TCP 路径测试实际注册、发送、接收和 pinned Host 搬运。

**未覆盖真实 HCA/CQ、跨机迟到 DMA、GPUDirect 外部写可见性、多 GPU 完整推理及性能收益。**
这些需要具备目标 GPU/HCA 的双机环境另行验收。通过 4060 不代表吞吐或 Decode P99 已改善。

## 复现与证据

验证机：RTX 4060 Laptop GPU；Python：
`/workspace/remote-attn-cuda-20261028/venv/bin/python`，
CUDA_HOME=`/workspace/conditional-graph-20261019/cuda13`。
隔离快照：`/tmp/qo-mooncake-4060`，结果在 `final-results`。

```bash
export PYTHONPATH=python:plugins/remote_attn:test/manual/remote_attn
python test/registered/unit/plugins/test_remote_attn.py
python test/registered/unit/plugins/test_d2d_mooncake_backend.py
python test/registered/unit/plugins/test_remote_mooncake_data_plane.py
python test/manual/remote_attn/qo_mooncake_cuda_probe.py --source-root "$PWD" --output /tmp/qo.json
python test/manual/remote_attn/range_layer_cuda_probe.py --output /tmp/qwen-range.json
HANDOFF_D2D_GROUP_SIZE=3 HANDOFF_D2D_TOKEN_RANGE=17 python test/manual/remote_attn/group_layer_cuda_probe.py --source-root "$PWD" --suite dsv-local --output /tmp/dsv-local.json
```

本地 CPU 使用隔离 Torch 与轻量 SGLang 顶层导入边界，模型 pool/预留/释放契约未替换。
GPU legacy 合并 launcher 预先初始化 Torch testing helper，避免 fixture 清理 sys.modules
后重复注册 DTensor；不改变运行时实现。

[JSON 报告](qo-mooncake-4060.json) 保存各套件结果和 **104 个实际测试源码文件 SHA-256**；
验证机、归档和本地最终源码逐一一致。配置的 Ruff 检查、相关文件格式检查及
`git diff --check` 均通过。原始本地日志位于 `/private/tmp/qo-tests` 和
`/private/tmp/qo-rdma/final-results`。
